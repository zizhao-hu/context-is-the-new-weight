"""In-max-context (Lm<=ctx, fixed full-causal forward) held-out ppl for bpe8 checkpoints.
Calibration: a,b,h should reproduce tab:toydeploy's in-ctx column (25, 74, 107).
For register masks also reports the with-registers variant. Run: python incx_bpe9.py --masks a,b,h,i,j
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--masks", default="a,b,h,i,j")
ap.add_argument("--Lm", type=int, default=256)
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--nP", type=int, default=8)
ap.add_argument("--suffix", default="")   # ckpt tag suffix, e.g. _w128_fw for the cc toy run   # must match the training run's nP so the ckpt-tag suffix lines up
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

_z = np.load("/scratch1/zizhaoh/wikitext_llama_bpe_big.npz"); _tr = _z["tr"]; _te = _z["te"]
_uniq = np.unique(np.concatenate([_tr, _te])); K = len(_uniq)
_lut = np.full(int(_uniq.max()) + 1, -1, dtype=np.int64); _lut[_uniq] = np.arange(K)
BOS_ID = K; V = K + 1
test_ids = torch.tensor(_lut[_te], dtype=torch.long).to(dev)

def rope_pos(x, pos, base=10000.0):
    B, H, T, D = x.shape; half = D // 2
    freq = base ** (-torch.arange(0, half, device=x.device).float() / half)
    ang = torch.outer(pos.float(), freq)
    cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
def rope(x): return rope_pos(x, torch.arange(x.shape[2], device=x.device))

class Block(nn.Module):
    def __init__(self, Cd, H):
        super().__init__(); self.H = H
        self.ln1 = nn.LayerNorm(Cd); self.ln2 = nn.LayerNorm(Cd)
        self.qkv = nn.Linear(Cd, 3 * Cd); self.proj = nn.Linear(Cd, Cd)
        self.mlp = nn.Sequential(nn.Linear(Cd, 4 * Cd), nn.GELU(), nn.Linear(4 * Cd, Cd))
class GPT(nn.Module):
    def __init__(self, V, Cd, H, Ln):
        super().__init__(); self.emb = nn.Embedding(V, Cd)
        self.blocks = nn.ModuleList([Block(Cd, H) for _ in range(Ln)])
        self.lnf = nn.LayerNorm(Cd); self.head = nn.Linear(Cd, V, bias=False)
        self.head.weight = self.emb.weight

@torch.no_grad()
def reg_kv(model, prompt):
    x = prompt[None]; P = x.shape[1]
    cm = torch.zeros(P, P, device=dev)
    cm.masked_fill_(~(torch.arange(P, device=dev)[None, :] <= torch.arange(P, device=dev)[:, None]), float("-inf"))
    cm = cm[None, None]; out = []
    for blk in model.blocks:
        B, _, Cd = x.shape; h = blk.H; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, P, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out.append((k, v))
        att = ((rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + cm).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, P, Cd)
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
    return out

@torch.no_grad()
def run(mask, ck, with_regs):
    cfg = ck["cfg"]; W = cfg["W"]; Pn = cfg.get("Pn", 0) or 0
    model = GPT(V, cfg["n_embd"], cfg["n_head"], cfg["n_layer"]).to(dev)
    model.load_state_dict(ck["sd"]); model.eval()
    prompt = ck["prompt"].to(dev) if ck["prompt"] is not None else None
    STREAM = mask in ("e", "f", "g", "h", "i", "f_scal", "f_pref", "f_rpref")
    g = torch.Generator().manual_seed(1234)
    ix = torch.randint(0, test_ids.numel() - a.Lm - 2, (a.n,), generator=g)
    seq = torch.stack([test_ids[i:i + a.Lm + 1].cpu() for i in ix]).to(dev)
    if STREAM: ids, tgt = seq[:, :a.Lm], seq[:, 1:a.Lm + 1]
    else:      ids = torch.cat([torch.full((a.n, 1), BOS_ID, device=dev), seq[:, :a.Lm - 1]], 1); tgt = seq[:, :a.Lm]
    x = model.emb(ids); Pm = 0
    if with_regs and mask in ("h", "j", "k"):  # token-prepend geometries
        x = torch.cat([prompt[None].expand(a.n, -1, -1).to(x.dtype), x], dim=1); Pm = Pn
    Tm = x.shape[1]
    cm = torch.zeros(Tm, Tm, device=dev)
    qi = torch.arange(Tm, device=dev)[:, None]; ki = torch.arange(Tm, device=dev)[None, :]
    cm.masked_fill_(~(ki <= qi), float("-inf")); cm = cm[None, None]
    RIDE = with_regs and mask in ("i", "n", "o", "f_rpref")
    SCALAR = with_regs and mask in ("l", "p", "f_scal")
    PREF = with_regs and mask in ("m", "q", "f_rpref", "f_pref")
    ssc = ck.get("sink_sc"); ssc = ssc.to(dev) if ssc is not None else None
    pkv = ck.get("pkv"); pkv = pkv.to(dev) if pkv is not None else None
    Hh = cfg["n_head"]
    rkv = None
    if RIDE:
        rkv = ([(pkv[li, 0].view(Pn, Hh, -1).transpose(0, 1)[None], pkv[li, 1].view(Pn, Hh, -1).transpose(0, 1)[None]) for li in range(cfg["n_layer"])]
               if mask in ("o", "f_rpref") else reg_kv(model, prompt))   # riding-prefix uses PKV; riding-token uses prompt-derived KV
    RIDE_POS = (torch.arange(Pn, device=dev) - (W + Pn - 1)).float() if RIDE else None
    for li, blk in enumerate(model.blocks):
        B_, T_, Cd = x.shape; h = blk.H; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B_, T_, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if RIDE:
            lc = (rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + cm
            pk = rkv[li][0].expand(B_, -1, -1, -1); pv = rkv[li][1].expand(B_, -1, -1, -1)
            lr = (q @ rope_pos(pk, RIDE_POS).transpose(-2, -1)) / math.sqrt(d)
            att = torch.cat([lr, lc], dim=-1).softmax(-1)
            o = (att @ torch.cat([pv, v], dim=2)).transpose(1, 2).reshape(B_, T_, Cd)
        elif SCALAR:
            lc = (rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + cm
            pad = ssc[li].view(1, -1, 1, 1).expand(B_, -1, Tm, 1)
            att = torch.cat([lc, pad], dim=-1).softmax(-1)[..., :Tm]
            o = (att @ v).transpose(1, 2).reshape(B_, T_, Cd)
        elif PREF:
            pos = torch.arange(Pn, Pn + Tm, device=dev).float()
            qr = rope_pos(q, pos); kr = rope_pos(k, pos)
            pk = pkv[li, 0].view(Pn, Hh, -1).transpose(0, 1)[None].expand(B_, -1, -1, -1)
            pv = pkv[li, 1].view(Pn, Hh, -1).transpose(0, 1)[None].expand(B_, -1, -1, -1)
            pkr = rope_pos(pk, torch.arange(Pn, device=dev).float())
            att = torch.cat([(qr @ pkr.transpose(-2, -1)) / math.sqrt(d),
                             (qr @ kr.transpose(-2, -1)) / math.sqrt(d) + cm], dim=-1).softmax(-1)
            o = (att @ torch.cat([pv, v], dim=2)).transpose(1, 2).reshape(B_, T_, Cd)
        else:
            att = ((rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + cm).softmax(-1)
            o = (att @ v).transpose(1, 2).reshape(B_, T_, Cd)
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
    lg = model.head(model.lnf(x))[:, Pm:, :]
    nll = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1))
    print("INCX mask=%s regs=%s Lm=%d n=%d ppl=%.2f" % (mask, with_regs, a.Lm, a.n, math.exp(nll.item())), flush=True)

_USES_NP = ("h", "i", "j", "k", "m", "n", "o", "q", "f_pref", "f_rpref")
for m in a.masks.split(","):
    tag = m if m in ("a", "b", "c", "d", "j", "k", "l", "m", "n", "o", "p", "q") else "%s_B" % m
    if m in _USES_NP and a.nP != 8: tag += "_p%d" % a.nP
    tag += a.suffix
    ck = torch.load("/scratch1/zizhaoh/bpe8_%s.pt" % tag, map_location=dev, weights_only=False)
    run(m, ck, with_regs=False)
    if m in ("h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "f_scal", "f_pref", "f_rpref"): run(m, ck, with_regs=True)
print("INCX_DONE", flush=True)
