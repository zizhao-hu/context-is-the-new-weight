"""Train-window x deploy-window perplexity matrix for the strict-control fla 1.3B pair (softmax transformer vs
gated gla). One job = one row: train the model under its scheme (base=no train / triangle=full-causal CPT /
windowed=sliding-window CPT at --train_window) then evaluate held-out perplexity at every deployment window
(set Attention.window_size). gla has no softmax Attention layers, so window_size is a no-op -> it returns the
same constant-memory perplexity at every column (the gated reference). Mirrors paper Table 4 on the fla pair."""
import torch, argparse, math
import fla
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--scheme", required=True, choices=["base", "triangle", "windowed"])
ap.add_argument("--train_window", type=int, default=256)
ap.add_argument("--deploy", default="128,256,512,1024,0")     # 0 = full (no window)
ap.add_argument("--steps", type=int, default=800); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--n_seq", type=int, default=2000)
ap.add_argument("--n_eval", type=int, default=48); ap.add_argument("--eval_len", type=int, default=2048)
ap.add_argument("--tag", required=True)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=bf16).to(dev)
attns = [m for m in model.modules() if m.__class__.__name__ == "Attention" and hasattr(m, "window_size")]
print("softmax-attn layers: %d" % len(attns), flush=True)


def packseqs(split, n, L):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n: return out
    return out


if a.scheme in ("triangle", "windowed"):
    twin = None if a.scheme == "triangle" else a.train_window
    for m in attns: m.window_size = twin
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95)); tr = packseqs("train", a.n_seq, a.ctx)
    for s in range(a.steps):
        ids = torch.tensor([tr[s % len(tr)]], device=dev)
        lg = model(ids).logits[0]
        loss = F.cross_entropy(lg[:-1].float(), ids[0, 1:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        if s % 200 == 0: print("  step %d loss %.3f" % (s, loss.item()), flush=True)
    model.gradient_checkpointing_disable()
model.eval(); model.config.use_cache = False
ev = packseqs("test", a.n_eval, a.eval_len)
out = []
for W in [int(x) for x in a.deploy.split(",")]:
    dw = None if W == 0 else W
    for m in attns: m.window_size = dw
    tot = 0.0; ntok = 0
    for s in ev:
        ids = torch.tensor([s], device=dev)
        with torch.no_grad():
            lg = model(ids).logits[0]
        tot += F.cross_entropy(lg[:-1].float(), ids[0, 1:], reduction="sum").item(); ntok += len(s) - 1
    ppl = math.exp(tot / ntok); out.append("deploy%s %.2f" % ("full" if W == 0 else str(W), ppl))
print("RESULT matrix %s | %s" % (a.tag, " | ".join(out)), flush=True)
print("MATRIX_DONE", flush=True)
