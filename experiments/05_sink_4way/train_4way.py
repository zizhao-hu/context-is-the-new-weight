"""Fine-tune a non-instruction-tuned base LM on long-context text with one of 4 schemes, to compare the
attention sink at POSITION 1.  schemes:
  full             — standard full-causal long-context tuning (baseline; builds the sink)
  windowed         — sliding-window attention of width W (uniform context; early tokens cold-start)
  warmup_windowed  — full-causal for --warmup-steps to accumulate context, THEN windowed
  startup          — windowed + P trainable startup soft-prompts that cover the cold start (our design)
"""
import torch, argparse, os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-0.5B"


def window_mask(T, W, device, dtype):
    q = torch.arange(T, device=device)[:, None]
    k = torch.arange(T, device=device)[None, :]
    allowed = (k <= q) & (k > q - W)                 # sliding causal band of width W
    m = torch.zeros(T, T, device=device, dtype=dtype)
    m.masked_fill_(~allowed, float("-inf"))
    return m[None, None]


def pack(tok, ds, L, n_seq):
    buf, seqs = [], []
    for row in ds:
        t = row["text"]
        if not t.strip():
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        while len(buf) >= L:
            seqs.append(buf[:L]); buf = buf[L:]
            if len(seqs) >= n_seq:
                return seqs
    return seqs


def pack_with_history(tok, ds, L, Hist, n_seq):
    """Continuous-stream packing: each length-L content chunk carries the REAL previous `Hist` tokens as
    history (so the cold start is filled by real prior text, the 'sliding with history' config)."""
    buf, seqs = [], []
    for row in ds:
        if not row["text"].strip():
            continue
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
        while len(buf) >= Hist + L:
            seqs.append((buf[:Hist], buf[Hist:Hist + L]))     # (history, content); contents are contiguous
            buf = buf[L:]
            if len(seqs) >= n_seq:
                return seqs
    return seqs


def fmt_alpaca(ex):
    q = ex["instruction"] + ("\n" + ex["input"] if ex["input"].strip() else "")
    return "### Instruction:\n" + q + "\n\n### Response:\n", ex["output"]


def pack_alpaca(tok, ds, L, n_seq, Hist=0):
    """Pack [query ⊕ answer] instruction examples into length-L chunks; loss-mask = 1 on ANSWER tokens only
    (the instruction-tuned mask). Returns (content, mask) or (history, content, mask) if Hist>0."""
    eos = tok.eos_token_id
    bt, bm, seqs = [], [], []
    for ex in ds:
        q, a = fmt_alpaca(ex)
        qt = tok(q, add_special_tokens=False).input_ids
        at = tok(a, add_special_tokens=False).input_ids + [eos]
        bt.extend(qt + at); bm.extend([0] * len(qt) + [1] * len(at))
        thr = Hist + L if Hist else L
        while len(bt) >= thr:
            seqs.append((bt[:Hist], bt[Hist:Hist + L], bm[Hist:Hist + L]) if Hist else (bt[:L], bm[:L]))
            bt, bm = bt[L:], bm[L:]
            if len(seqs) >= n_seq:
                return seqs
    return seqs


def make_seqs(tok, args, L, W):
    """Unified: returns list of (history, content, lmask). lmask=1 where loss applies (all tokens for wikitext;
    answer tokens only for alpaca)."""
    use_hist = (args.scheme == "sliding_history")
    Hist = W - 1
    if args.data == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        raw = pack_alpaca(tok, ds, L, args.n_seq, Hist if use_hist else 0)
        return [(h, c, m) for (h, c, m) in raw] if use_hist else [([], c, m) for (c, m) in raw]
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    if use_hist:
        return [(h, c, [1] * len(c)) for (h, c) in pack_with_history(tok, ds, L, Hist, args.n_seq)]
    return [([], c, [1] * len(c)) for c in pack(tok, ds, L, args.n_seq)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True,
                    choices=["full", "windowed", "warmup_windowed", "startup", "sliding_history"])
    ap.add_argument("--data", default="wikitext", choices=["wikitext", "alpaca"])   # pretrained text vs instruction QA
    ap.add_argument("--ctx_len", type=int, default=1024)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n_startup", type=int, default=255)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--warmup_steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n_seq", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(dev)
    model.train()
    H = model.get_input_embeddings().weight.shape[1]
    L, W, P = args.ctx_len, args.window, args.n_startup

    prompt = None
    params = list(model.parameters())
    if args.scheme == "startup":
        prompt = torch.nn.Parameter(torch.randn(P, H, device=dev, dtype=torch.float32) * 0.02)
        params = params + [prompt]
    opt = torch.optim.AdamW(params, lr=args.lr)

    seqs = make_seqs(tok, args, L, W)                                                     # list of (history, content, lmask)
    print(f"[{args.scheme}/{args.data}] {len(seqs)} sequences of length {L}; H={H}", flush=True)

    emb = model.get_input_embeddings()
    for step in range(args.steps):
        hist, content, lmask = seqs[step % len(seqs)]
        nh = len(hist)
        windowed_now = (args.scheme in ("windowed", "startup", "sliding_history")) or \
                       (args.scheme == "warmup_windowed" and step >= args.warmup_steps)
        if args.scheme == "startup":
            ids = torch.tensor([content], device=dev)
            inp = torch.cat([prompt.to(torch.bfloat16).unsqueeze(0), emb(ids)], dim=1)   # (1, P+L, H)
            mask = window_mask(P + L, W, dev, inp.dtype)
            logits = model(inputs_embeds=inp, attention_mask=mask).logits[0][P:]          # content positions
        elif args.scheme == "sliding_history":
            ids = torch.tensor([hist + content], device=dev)
            mask = window_mask(nh + L, W, dev, torch.bfloat16)
            logits = model(input_ids=ids, attention_mask=mask).logits[0][nh:]             # content positions
        else:
            ids = torch.tensor([content], device=dev)
            mask = window_mask(L, W, dev, torch.bfloat16) if windowed_now else None
            logits = model(input_ids=ids, attention_mask=mask).logits[0]                   # (L, V)
        tgt = torch.tensor(content[1:], device=dev)
        lm = torch.tensor(lmask[1:], device=dev, dtype=torch.float32)                      # 1 where loss applies
        ce = F.cross_entropy(logits[:-1].float(), tgt, reduction="none")
        loss = (ce * lm).sum() / lm.sum().clamp(min=1)                                     # answer-masked (alpaca) / all (wikitext)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad()
        if step % 50 == 0:
            print(f"  step {step}/{args.steps} windowed={windowed_now} loss={loss.item():.3f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    if prompt is not None:
        torch.save(prompt.detach().cpu(), os.path.join(args.out, "startup_prompt.pt"))
    print(f"[{args.scheme}] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
