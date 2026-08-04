"""Attention-sink measurement for Qwen3.5-9B (8 softmax layers, output_attentions + eager).

Two questions:
 (1) base vs tuned: does instruction/experience tuning amplify the sink? (run with --window 0)
 (2) does a SLIDING WINDOW reduce the sink? Apply a width-W band mask at inference and measure whether a NEW
     sink re-forms at the window's left edge (the 'frontier' = oldest visible key).

Metrics:
 - sink_abs0    = mean attention to absolute key position 0 (the classic sink). Under a window it is masked
                  out for far queries, so it mechanically -> ~0.
 - frontier_sink = mean attention to each query's OLDEST VISIBLE key (key 0 under full attention; the window's
                  left edge i-W+1 under a width-W window), averaged over full-window queries. This is the
                  apples-to-apples sink: full vs windowed. If windowed frontier_sink << full, the window
                  REDUCED the sink; if ~equal, the sink just MOVED to the window edge.
"""
import torch, json, argparse, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

POOL = [
    "WebShop Instruction: i want a long clip-in hair extension which is natural looking and priced lower than forty dollars, then Search.",
    "Page 1 Total results 50 : Clip in Hair Extensions Real Human Hair twenty inch priced thirty nine dollars ninety nine cents, Next.",
    "The instruction asks for a natural looking long clip-in extension under forty, so I should compare prices and click the closest match.",
    "Buy Now is the final action once the selected item matches color length and price constraints from the original instruction.",
    "The quick brown fox jumps over the lazy dog while the morning sun rises slowly above the quiet river valley.",
    "To compute the gradient of the loss with respect to the parameters, apply the chain rule backward through every layer.",
    "Once upon a time in a distant kingdom there lived a sleepless king who counted the stars each restless night.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using the energy captured from sunlight by chlorophyll.",
    "The treaty was signed in the spring of that year, ending a long conflict and redrawing the borders of three nations.",
    "A balanced binary search tree keeps insertion lookup and deletion within logarithmic time by rebalancing after each update.",
    "She poured the coffee, opened the window, and listened to the city waking up beneath a pale grey sky.",
    "Inflation measures the rate at which the general level of prices for goods and services rises over a period of time.",
    "The spacecraft adjusted its trajectory with a brief thruster burn before settling into a stable orbit around the moon.",
    "He argued that the proof was incomplete because the inductive step assumed exactly what it was trying to establish.",
    "The recipe calls for two cups of flour, a pinch of salt, and a tablespoon of honey folded gently into the batter.",
    "Migratory birds navigate thousands of miles using the earth's magnetic field, the position of the sun, and learned landmarks.",
    "The committee postponed the vote after a lengthy debate about funding, scope, and the timeline for the proposed renovation.",
    "Quantum entanglement links the states of two particles so that measuring one instantly constrains the outcome of the other.",
]


def long_inputs():
    return [" ".join(POOL), " ".join(reversed(POOL)), " ".join(POOL[6:] + POOL[:6])]


def measure(tok, model, texts, window=0, max_len=900):
    dev = "cuda"
    pl_abs0, pl_front, n = None, None, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
        T = ids.shape[1]
        if window and T < window + 12:                             # need full-window queries past the band
            continue
        mask = None
        if window:
            q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
            allowed = (k <= q) & (k > q - window)                 # width-W causal band
            mask = torch.zeros(T, T, device=dev, dtype=model.dtype)
            mask.masked_fill_(~allowed, float("-inf")); mask = mask[None, None]
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False)
        attns = [a for a in (out.attentions or []) if a is not None]
        if not attns:
            raise RuntimeError("no attentions returned — need eager attn_implementation")
        # frontier queries: windowed -> i>=W so the oldest-visible edge (i-W+1) is NOT the absolute sink; full -> key 0
        qs = range(window, T) if window else range(1, T)
        abs0, front = [], []
        for a in attns:                                            # a: (1,H,T,T)
            af = a[0].float()
            abs0.append(af[:, :, 0].mean().item())                 # attention to absolute key 0
            fr = [af[:, i, (i - window + 1) if window else 0].mean().item() for i in qs]
            front.append(sum(fr) / len(fr))                        # attention to the OLDEST VISIBLE key
        pl_abs0 = abs0 if pl_abs0 is None else [x + y for x, y in zip(pl_abs0, abs0)]
        pl_front = front if pl_front is None else [x + y for x, y in zip(pl_front, front)]
        n += 1
    if not n:
        raise RuntimeError(f"no inputs long enough for window={window}")
    pl_abs0 = [x / n for x in pl_abs0]; pl_front = [x / n for x in pl_front]
    return {"window": window, "n_inputs": n, "n_softmax_layers": len(pl_abs0),
            "sink_abs0": sum(pl_abs0) / len(pl_abs0),
            "frontier_sink": sum(pl_front) / len(pl_front),
            "per_layer_frontier": pl_front, "per_layer_abs0": pl_abs0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--windows", default="0", help="comma-sep window widths; 0 = full causal")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16,
                                                 trust_remote_code=True, attn_implementation="eager").to("cuda").eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, os.path.join(args.adapter, "adapter")).eval()
    by_window = {}
    for w in [int(x) for x in args.windows.split(",")]:
        res = measure(tok, model, long_inputs(), window=w)
        by_window[str(w)] = res
        tag = "full" if w == 0 else f"W={w}"
        print(f"[{args.label} {tag}]  sink_abs0={res['sink_abs0']:.4f}  frontier_sink={res['frontier_sink']:.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"label": args.label, "adapter": args.adapter, "by_window": by_window}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
