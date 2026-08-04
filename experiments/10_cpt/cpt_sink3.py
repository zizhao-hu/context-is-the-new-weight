"""Both sink types under ONE full deploy -> mean +/- SEM (apples-to-apples for triangle vs sliding).
Full-causal deploy (pos-0 available at every depth, so 'does the model still use it?' is meaningful):
  p0   = pos-0 sink: deep-query [W,T) attention on col-0, mean over heads (and max head).
  dist = distributed sink: deep-query attention on NON-pos-0 persistent low-info bands
         (band col j>0 with local visible-query mean c[j]=mean_{q in [j,j+W)} A[q,j] > 5/W),
         mean over heads (and max-band head). Identity-agnostic concentration.
No --load_ckpt = base; --load_ckpt <dir> = triangle (a) / sliding (b)."""
import torch, argparse, math, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
ap = argparse.ArgumentParser()
ap.add_argument("--load_ckpt", default=""); ap.add_argument("--W", type=int, default=512)
ap.add_argument("--T", type=int, default=1024); ap.add_argument("--nseq", type=int, default=12)
ap.add_argument("--label", default="model")
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16; NINF = float("-inf"); W = a.W; T = a.T
MODEL = "meta-llama/Llama-3.2-3B-Instruct"; tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(a.load_ckpt or MODEL, dtype=bf16, attn_implementation="eager").to(dev).eval()
def pack(n, Lp):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test"); buf = []; out = []
    for r in ds:
        if r["text"].strip(): buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= Lp:
            out.append(buf[:Lp]); buf = buf[Lp:]
            if len(out) >= n: return out
    return out
q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
MF = torch.zeros(T, T, device=dev, dtype=bf16); MF.masked_fill_(~(k <= q), NINF); MF = MF[None, None]
qdeep = torch.arange(W, T, device=dev)
@torch.no_grad()
def attn(ids): return torch.stack(model(ids, attention_mask=MF, use_cache=False, output_attentions=True).attentions, 0)[:, 0].float()
samps = pack(a.nseq, T); p0m = []; p0x = []; dm = []; dx = []
for samp in samps:
    ids = torch.tensor([samp], device=dev)
    A = attn(ids).reshape(-1, T, T)                                    # [LH,T,T] full deploy
    p0h = A[:, qdeep, 0].mean(-1)                                       # [LH] deep-query pos-0 mass per head
    p0m.append(p0h.mean().item()); p0x.append(p0h.max().item())
    c = torch.zeros(A.shape[0], T, device=dev)
    for j in range(T):
        r1 = min(j + W, T); c[:, j] = A[:, j:r1, j].mean(1)            # local visible-query column mean
    band = (c > 5.0 / W).float(); band[:, 0] = 0.0                      # persistent bands, exclude pos-0
    sm = (A[:, qdeep, :] * band[:, None, :]).sum(-1).mean(-1)          # [LH] non-pos0 band mass per head
    dm.append(sm.mean().item()); dx.append(sm[int(c[:, 1:].max(-1).values.argmax()) ].item())
    del A, c, band, sm; torch.cuda.empty_cache()
f = lambda v: (np.mean(v), np.std(v) / math.sqrt(len(v)))
(p0mm, p0ms), (p0xm, p0xs), (dmm, dms), (dxm, dxs) = f(p0m), f(p0x), f(dm), f(dx)
print("SINK3 label=%s n=%d p0mean=%.4f p0meansem=%.4f p0max=%.4f p0maxsem=%.4f distmean=%.4f distmeansem=%.4f distmax=%.4f distmaxsem=%.4f" % (
    a.label, len(p0m), p0mm, p0ms, p0xm, p0xs, dmm, dms, dxm, dxs), flush=True)
