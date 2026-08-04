"""Measure the attention sink at POSITION 1 (the first CONTENT token) for a model, under FULL causal attention
on held-out wikitext. Full-attention eval probes the *learned* tendency to sink (windowing mechanically zeros
pos-0 attention, so we eval unmasked to see whether the model still wants to pile onto its first token).
sink_rate = fraction of heads whose mean attention to the first content token exceeds ε=0.3 (paper metric)."""
import torch, json, argparse, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def eval_seqs(tok, L, n, data="wikitext"):
    if data == "alpaca":                                     # held-out tail of alpaca, packed [query⊕answer]
        ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(50000, 52000))
        buf, seqs = [], []
        for ex in ds:
            q = ex["instruction"] + ("\n" + ex["input"] if ex["input"].strip() else "")
            buf.extend(tok("### Instruction:\n" + q + "\n\n### Response:\n" + ex["output"],
                           add_special_tokens=False).input_ids + [tok.eos_token_id])
            while len(buf) >= L:
                seqs.append(buf[:L]); buf = buf[L:]
                if len(seqs) >= n:
                    return seqs
        return seqs
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    buf, seqs = [], []
    for row in ds:
        if not row["text"].strip():
            continue
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
        while len(buf) >= L:
            seqs.append(buf[:L]); buf = buf[L:]
            if len(seqs) >= n:
                return seqs
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)                # base id or fine-tuned dir
    ap.add_argument("--startup", action="store_true")        # prepend the saved startup soft-prompt
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx_len", type=int, default=512)      # eval context (smaller than train, for output_attentions speed)
    ap.add_argument("--n_eval", type=int, default=16)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--history", type=int, default=0)        # prepend H real history tokens; measure sink at first content token
    ap.add_argument("--data", default="wikitext", choices=["wikitext", "alpaca"])
    args = ap.parse_args()
    dev = "cuda"
    base = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(args.model if os.path.isdir(args.model) else base)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    emb = model.get_input_embeddings()
    prompt, P = None, 0
    if args.startup:
        prompt = torch.load(os.path.join(args.model, "startup_prompt.pt")).to(dev)
        P = prompt.shape[0]

    H = args.history
    seqs = eval_seqs(tok, (H + args.ctx_len) if H else args.ctx_len, args.n_eval, args.data)
    head_sum, block_sum, n = None, torch.zeros((), device=dev), 0
    c0 = P if prompt is not None else H                      # first content key position (after prompts / history block)
    for s in seqs:
        ids = torch.tensor([s], device=dev)
        with torch.no_grad():
            if prompt is not None:
                inp = torch.cat([prompt.to(torch.bfloat16).unsqueeze(0), emb(ids)], dim=1)
                out = model(inputs_embeds=inp, output_attentions=True, use_cache=False)
            else:                                            # full / windowed / history (history prefix is already in `s`)
                out = model(input_ids=ids, output_attentions=True, use_cache=False)
        attns = [a for a in (out.attentions or []) if a is not None]
        per = []                                             # accumulate on GPU; single sync at the end
        for a in attns:
            af = a[0].float()                                # (heads, T, T)
            per.append(af[:, c0 + 1:, c0].mean(dim=1))       # per-head mean attn to first content token
            if c0 > 0:
                block_sum = block_sum + af[:, c0:, :c0].sum(dim=2).mean()   # attn mass on the cold-start block (prompts / history)
        cur = torch.cat(per)
        head_sum = cur if head_sum is None else head_sum + cur
        n_layers = len(attns)
        n += 1
    head_mean = (head_sum / n).cpu()
    block = (block_sum.item() / (n * n_layers)) if c0 > 0 else None   # per-layer mean attn on the cold-start block
    res = {"label": args.label, "n_eval": n, "n_heads": head_mean.numel(),
           "mean_attn_first_content": head_mean.mean().item(),
           "max_head_attn_first_content": head_mean.max().item(),
           "sink_rate_pos1": (head_mean > args.thresh).float().mean().item(),
           "block_attn": block}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[{args.label}] mean_attn_first_content={res['mean_attn_first_content']:.4f}  "
          f"sink_rate(>{args.thresh})={res['sink_rate_pos1']:.3f}  "
          f"block_attn={res['block_attn']}", flush=True)


if __name__ == "__main__":
    main()
