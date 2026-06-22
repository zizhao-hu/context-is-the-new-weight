"""Perplexity vs constant-working-memory budget W, model-agnostic (works for softmax Transformer AND gated
linear attention). The sequence is cut into non-overlapping W-token windows (memory reset every W tokens =
constant working memory of W); we score the last `score_last` tokens of each window (each has ~W tokens of
context). Sweep W to see how each architecture degrades as the working memory shrinks."""
import torch, argparse, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--budgets", default="128,256,512,1024,2048")
ap.add_argument("--n_windows", type=int, default=100)
ap.add_argument("--score_last", type=int, default=64)
a = ap.parse_args(); dev = "cuda"
budgets = [int(x) for x in a.budgets.split(",")]
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
nparam = sum(p.numel() for p in model.parameters()) / 1e9
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
toks = []; need = max(budgets) * a.n_windows
for r in ds:
    if not r["text"].strip():
        continue
    toks += tok(r["text"], add_special_tokens=False).input_ids
    if len(toks) >= need:
        break
print("MODEL %s | params %.2fB" % (a.model, nparam), flush=True)
for W in budgets:
    nll = 0.0; ntok = 0; sl = min(a.score_last, W - 1)
    for w in range(a.n_windows):
        s = toks[w * W:(w + 1) * W]
        if len(s) < W:
            break
        ids = torch.tensor([s], device=dev)
        with torch.no_grad():
            lg = model(ids).logits[0]
        tgt = ids[0, W - sl:W]; pred = lg[W - sl - 1:W - 1].float()
        nll += F.cross_entropy(pred, tgt, reduction="sum").item(); ntok += sl
    print("  budget W=%-5d ppl %.3f   (scored %d tok)" % (W, math.exp(nll / ntok), ntok), flush=True)
print("BUDGET_DONE", flush=True)
