"""Qwen2.5-0.5B sink-spreading analysis (aims 1+2). Train a scheme {base/triangle/windowed/startup} then, with
EAGER attention, capture where attention lands under that scheme's deployment mask: Type-1 = attention received by
pos-0; Type-2 = attention received by low-information punctuation tokens; plus broadcasting entropy of the
per-position attention-received distribution. Windowed training should move mass off pos-0 (Type-1) onto the
distributed Type-2 tokens and raise the entropy (better broadcasting). Train+analyze in one process (no save)."""
import torch, argparse, math, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "Qwen/Qwen2.5-0.5B"
ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["base", "triangle", "windowed", "startup"])
ap.add_argument("--window", type=int, default=256)
ap.add_argument("--n_startup", type=int, default=64)
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--n_seq", type=int, default=2000)
ap.add_argument("--clen", type=int, default=256)
ap.add_argument("--n_eval", type=int, default=16)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)
H = model.config.hidden_size; emb = model.get_input_embeddings()


def wmask(scheme, P, L, W, dev, dt):
    T = P + L
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    allowed = causal if scheme in ("base", "triangle") else (causal & (k > q - W))
    m = torch.zeros(T, T, device=dev, dtype=dt); m.masked_fill_(~allowed, float("-inf"))
    return m[None, None]


def pack(split, n, L):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip():
            continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n:
                return out
    return out


prompt = None
if a.scheme != "base":
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    params = list(model.parameters())
    if a.scheme == "startup":
        prompt = torch.nn.Parameter(torch.randn(a.n_startup, H, device=dev, dtype=torch.float32) * 0.02)
        params = params + [prompt]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95))
    seqs = pack("train", a.n_seq, a.ctx)
    W = None if a.scheme == "triangle" else a.window
    for step in range(a.steps):
        ids = torch.tensor([seqs[step % len(seqs)]], device=dev)
        if prompt is not None:
            inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], 1)
            am = wmask("windowed", a.n_startup, a.ctx, a.window, dev, bf16)
            lg = model(inputs_embeds=inp, attention_mask=am).logits[0][a.n_startup:]
        else:
            am = wmask(a.scheme, 0, a.ctx, a.window, dev, bf16)
            lg = model(ids, attention_mask=am).logits[0]
        loss = F.cross_entropy(lg[:-1].float(), ids[0, 1:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        if step % 100 == 0:
            print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
model.eval()
# punctuation token ids (low-information / Type-2 candidates)
PUNCT = set()
for p in [".", ",", ";", ":", "!", "?", "\"", "'", ")", "(", "-", "\n", " .", " ,", " the", " ="]:
    for t in tok(p, add_special_tokens=False).input_ids:
        PUNCT.add(t)
ev = pack("test", a.n_eval, a.clen)
recv = np.zeros(a.clen); ent = 0.0; npunct_mass = 0.0; tot_mass = 0.0; pos0 = 0.0; n = 0
W = None if a.scheme in ("base", "triangle") else a.window
for s in ev:
    ids = torch.tensor([s], device=dev)
    with torch.no_grad():
        if prompt is not None:
            inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], 1)
            out = model(inputs_embeds=inp, attention_mask=wmask("windowed", a.n_startup, a.clen, a.window, dev, bf16), output_attentions=True)
            atts = [A[0].float()[:, a.n_startup:, a.n_startup:] for A in out.attentions]   # content x content
        else:
            out = model(ids, attention_mask=wmask(a.scheme, 0, a.clen, a.window, dev, bf16), output_attentions=True)
            atts = [A[0].float() for A in out.attentions]
    amap = torch.stack(atts).mean(0).mean(0)              # avg layers, heads -> [q, k]
    r = amap.sum(0).cpu().numpy()                          # attention received per key
    recv += r; pos0 += r[0]
    pr = r / (r.sum() + 1e-9); ent += -(pr * np.log(pr + 1e-12)).sum()
    punct_pos = [i for i, t in enumerate(s) if t in PUNCT]
    npunct_mass += r[punct_pos].sum(); tot_mass += r.sum(); n += 1
recv /= n
sink_pos0 = pos0 / tot_mass                                # fraction of all attention on pos-0 (Type-1)
type2 = npunct_mass / tot_mass                             # fraction on punctuation (Type-2)
print("RESULT Qwen0.5B | scheme=%s | sink_pos0=%.4f | type2_punct=%.4f | entropy=%.3f | top1=%.4f"
      % (a.scheme, sink_pos0, type2, ent / n, recv.max() / recv.sum()), flush=True)
print("SINK_DONE", flush=True)
