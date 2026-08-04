"""CPT on Qwen2.5-0.5B under a *literature* attention mask, then eval held-out ppl under the three
deployment masks (full / sliding / StreamingLLM). Produces the Fig-2 b/c/d rows.
  train_mask:  full        -- triangle / full causal (reference)
               windowed    -- b. Gemma/SWAT: plain causal sliding window W (no sink, no history)
               swaa        -- c. SWAA / StreamingLLM-mask: keep S sink cols + recent window W
               longformer  -- d. Longformer: symmetric window +-W/2 (ATTENDS FUTURE; non-causal -> degenerate
                              for next-token LM; included to quantify that it is mal-designed)
Train + eval in one process (no checkpoint saved). Mirrors qwen_stream.py.
"""
import torch, argparse, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "Qwen/Qwen2.5-0.5B"
ap = argparse.ArgumentParser()
ap.add_argument("--train_mask", required=True, choices=["full", "windowed", "swaa", "longformer"])
ap.add_argument("--window", type=int, default=256)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--n_seq", type=int, default=2000)
ap.add_argument("--score_last", type=int, default=1024)
ap.add_argument("--n_eval", type=int, default=40)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16


def tmask(kind, L, W, S, dev, dt):                       # TRAINING masks (b/c/d)
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]
    causal = k <= q
    if kind == "full":         allowed = causal
    elif kind == "windowed":   allowed = causal & (k > q - W)
    elif kind == "swaa":       allowed = causal & ((k < S) | (k > q - W))
    elif kind == "longformer": allowed = (torch.abs(k - q) <= W // 2)     # symmetric, attends future
    m = torch.zeros(L, L, device=dev, dtype=dt); m.masked_fill_(~allowed, float("-inf"))
    return m[None, None]


def emask(kind, L, W, S, dev, dt):                       # EVAL / inference masks (all causal)
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]
    causal = k <= q
    if kind == "full":        allowed = causal
    elif kind == "sliding":   allowed = causal & (k > q - W)
    elif kind == "streaming": allowed = causal & ((k < S) | (k > q - W))
    m = torch.zeros(L, L, device=dev, dtype=dt); m.masked_fill_(~allowed, float("-inf"))
    return m[None, None]


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)


def pack(split, n, L):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n: return out
    return out


model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95))
seqs = pack("train", a.n_seq, a.ctx)
for step in range(a.steps):
    ids = torch.tensor([seqs[step % len(seqs)]], device=dev)
    lg = model(ids, attention_mask=tmask(a.train_mask, a.ctx, a.window, a.sink, dev, bf16)).logits[0]
    loss = F.cross_entropy(lg[:-1].float(), ids[0, 1:])
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
    if step % 100 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
model.eval()
ev = pack("test", a.n_eval, a.ctx); sl = a.score_last
def ppl(kind):
    nll = 0.0; ntok = 0
    for s in ev:
        ids = torch.tensor([s], device=dev)
        with torch.no_grad():
            lg = model(ids, attention_mask=emask(kind, a.ctx, a.window, a.sink, dev, bf16)).logits[0]
        tgt = ids[0, a.ctx - sl:a.ctx]; pred = lg[a.ctx - sl - 1:a.ctx - 1].float()
        nll += F.cross_entropy(pred, tgt, reduction="sum").item(); ntok += sl
    return math.exp(nll / ntok)
print("RESULT cpt train=%s W=%d sink=%d | full %.3f | sliding@%d %.3f | streaming(s%d+%d) %.3f"
      % (a.train_mask, a.window, a.sink, ppl("full"), a.window, ppl("sliding"), a.sink, a.window, ppl("streaming")),
      flush=True)
print("CPT_DONE", flush=True)
