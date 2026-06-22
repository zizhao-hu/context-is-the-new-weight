"""Qwen2.5-0.5B attention-MAP capture per scheme (aims 1+2 visual). Train {base/triangle/windowed/startup} then
save the 2D attention map (content-query x key, averaged over layers/heads/eval-seqs) under the scheme's
deployment mask, so the figure shows Type-1 (pos-0 sink column) -> Type-2 (distributed punctuation) spreading."""
import torch, argparse, numpy as np
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
ap.add_argument("--out", required=True)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)
H = model.config.hidden_size; emb = model.get_input_embeddings()


def wmask(scheme, P, L, W, dev, dt):
    T = P + L; q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    allowed = causal if scheme in ("base", "triangle") else (causal & (k > q - W))
    m = torch.zeros(T, T, device=dev, dtype=dt); m.masked_fill_(~allowed, float("-inf")); return m[None, None]


def pack(split, n, L):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n: return out
    return out


prompt = None
if a.scheme != "base":
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    params = list(model.parameters())
    if a.scheme == "startup":
        prompt = torch.nn.Parameter(torch.randn(a.n_startup, H, device=dev, dtype=torch.float32) * 0.02); params += [prompt]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95)); seqs = pack("train", a.n_seq, a.ctx)
    for step in range(a.steps):
        ids = torch.tensor([seqs[step % len(seqs)]], device=dev)
        if prompt is not None:
            lg = model(inputs_embeds=torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], 1),
                       attention_mask=wmask("windowed", a.n_startup, a.ctx, a.window, dev, bf16)).logits[0][a.n_startup:]
        else:
            lg = model(ids, attention_mask=wmask(a.scheme, 0, a.ctx, a.window, dev, bf16)).logits[0]
        loss = F.cross_entropy(lg[:-1].float(), ids[0, 1:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        if step % 100 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
model.eval()
ev = pack("test", a.n_eval, a.clen); P = a.n_startup if prompt is not None else 0
amap = np.zeros((a.clen, P + a.clen)); n = 0
for s in ev:
    ids = torch.tensor([s], device=dev)
    with torch.no_grad():
        if prompt is not None:
            out = model(inputs_embeds=torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], 1),
                        attention_mask=wmask("windowed", a.n_startup, a.clen, a.window, dev, bf16), output_attentions=True)
            atts = [A[0].float()[:, a.n_startup:, :] for A in out.attentions]
        else:
            out = model(ids, attention_mask=wmask(a.scheme, 0, a.clen, a.window, dev, bf16), output_attentions=True)
            atts = [A[0].float() for A in out.attentions]
    amap += torch.stack(atts).mean(0).mean(0).cpu().numpy(); n += 1
np.savez(a.out, attn_map=amap / n, P=P, scheme=a.scheme, window=a.window)
print("RESULT saved %s | scheme=%s shape=%s" % (a.out, a.scheme, (amap / n).shape), flush=True)
print("MAP_DONE", flush=True)
