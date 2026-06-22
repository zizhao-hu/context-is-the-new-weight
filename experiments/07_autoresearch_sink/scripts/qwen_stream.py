"""Qwen2.5-0.5B streaming benchmark (aim 3a): does our windowed TRAINING beat the post-hoc StreamingLLM trick
(built on the standard/triangle scheme)? Train a scheme {base/windowed} then eval held-out ppl on deep positions
under three deployment masks: full causal, sliding-window W (each token sees last W), and StreamingLLM (keep S
initial 'sink' tokens + last W). Base collapses under sliding, recovers under StreamingLLM; windowed-FT should
work under plain sliding (no sink tokens) and match/beat StreamingLLM. Train+eval in one process (no save)."""
import torch, argparse, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "Qwen/Qwen2.5-0.5B"
ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["base", "windowed"])
ap.add_argument("--window", type=int, default=256)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--n_seq", type=int, default=2000)
ap.add_argument("--score_last", type=int, default=1024)
ap.add_argument("--n_eval", type=int, default=40)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)


def mask(kind, L, W, S, dev, dt):
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]
    causal = k <= q
    if kind == "full":        allowed = causal
    elif kind == "sliding":   allowed = causal & (k > q - W)
    elif kind == "streaming": allowed = causal & ((k < S) | (k > q - W))   # StreamingLLM: S sink + last W
    m = torch.zeros(L, L, device=dev, dtype=dt); m.masked_fill_(~allowed, float("-inf"))
    return m[None, None]


def pack(split, n, L):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n: return out
    return out


if a.scheme == "windowed":
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95))
    seqs = pack("train", a.n_seq, a.ctx)
    for step in range(a.steps):
        ids = torch.tensor([seqs[step % len(seqs)]], device=dev)
        lg = model(ids, attention_mask=mask("sliding", a.ctx, a.window, a.sink, dev, bf16)).logits[0]
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
            lg = model(ids, attention_mask=mask(kind, a.ctx, a.window, a.sink, dev, bf16)).logits[0]
        tgt = ids[0, a.ctx-sl:a.ctx]; pred = lg[a.ctx-sl-1:a.ctx-1].float()
        nll += F.cross_entropy(pred, tgt, reduction="sum").item(); ntok += sl
    return math.exp(nll/ntok)
print("RESULT Qwen0.5B-%s | full %.3f | sliding@%d %.3f | streamingLLM(s%d+%d) %.3f"
      % (a.scheme, ppl("full"), a.window, ppl("sliding"), a.sink, a.window, ppl("streaming")), flush=True)
print("STREAM_DONE", flush=True)
