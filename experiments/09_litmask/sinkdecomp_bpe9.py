"""Consistent persistent-column sink decomposition across ALL masks (eval-only, saved ckpts).

One measure for every regime: a content column j is PERSISTENT iff the W queries right after j
give it > thr/W mean attention (the only queries guaranteed to see j under any windowed mask).
Deep-query (pos >= W) attention is then decomposed into DISJOINT classes summing to 1:
  p0   = first sequence column (BOS/pos-0)
  reg  = trainable register columns (h: absolute prefix; i: riding constant-offset block)
  band = persistent content columns EXCLUDING p0 (the distributed / separator sink)
  rest = everything else (ordinary content attention)
Reported for (1) the average over all layers*heads and (2) the strongest-band head (same head
for all classes, so rows stay additive). Run: python sinkdecomp_bpe9.py
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir", default="/scratch1/zizhaoh")
ap.add_argument("--masks", default="a,b,c,d,e,f,g,h,i")
ap.add_argument("--Lm", type=int, default=512)
ap.add_argument("--Bm", type=int, default=16)
ap.add_argument("--thr", type=float, default=5.0)
ap.add_argument("--sinkS", type=int, default=4)
ap.add_argument("--profile", action="store_true")   # also dump per-query-position class profiles (npz)
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
        self.drop = nn.Dropout(0.0)
class GPT(nn.Module):
    def __init__(self, V, Cd, H, Ln):
        super().__init__(); self.emb = nn.Embedding(V, Cd)
        self.blocks = nn.ModuleList([Block(Cd, H) for _ in range(Ln)])
        self.lnf = nn.LayerNorm(Cd); self.head = nn.Linear(Cd, V, bias=False)
        self.head.weight = self.emb.weight

@torch.no_grad()
def reg_kv(model, prompt):
    """pre-RoPE per-layer K/V of the register block (causal self-attn among registers)."""
    x = prompt[None]; P = x.shape[1]
    cm = torch.zeros(P, P, device=dev); cm.masked_fill_(~(torch.arange(P, device=dev)[None, :] <= torch.arange(P, device=dev)[:, None]), float("-inf")); cm = cm[None, None]
    out = []
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
def decompose(mask, ck):
    cfg = ck["cfg"]; W = cfg["W"]; Pn = cfg.get("Pn", 0) or 0
    model = GPT(V, cfg["n_embd"], cfg["n_head"], cfg["n_layer"]).to(dev)
    model.load_state_dict(ck["sd"]); model.eval()
    prompt = ck["prompt"].to(dev) if ck["prompt"] is not None else None
    STREAM = mask in ("e", "f", "g", "h", "i")            # j is IID (BOS-anchored) like a
    g = torch.Generator().manual_seed(999)
    ix = torch.randint(0, test_ids.numel() - a.Lm - 2, (a.Bm,), generator=g)
    seq = torch.stack([test_ids[i:i + a.Lm + 1].cpu() for i in ix]).to(dev)
    ids = seq[:, :a.Lm] if STREAM else torch.cat([torch.full((a.Bm, 1), BOS_ID, device=dev), seq[:, :a.Lm - 1]], 1)
    x = model.emb(ids); Pm = 0
    if mask in ("h", "j", "k"):
        x = torch.cat([prompt[None].expand(a.Bm, -1, -1).to(x.dtype), x], dim=1); Pm = Pn
    Tm = x.shape[1]; S = a.sinkS
    qi = torch.arange(Tm, device=dev)[:, None]; ki = torch.arange(Tm, device=dev)[None, :]
    if mask in ("a", "e", "j", "p", "q"): allowed = ki <= qi
    elif mask == "c":        allowed = (ki <= qi) & ((ki < S) | (ki > qi - W))
    elif mask == "d":        allowed = torch.abs(ki - qi) <= W // 2
    elif mask in ("h", "k"):
        rq = qi - Pm; rk = ki - Pm
        allowed = (ki < Pm) | ((rk <= rq) & (rk > rq - W))
    else:                    allowed = (ki <= qi) & (ki > qi - W)
    msk = torch.zeros(Tm, Tm, device=dev); msk.masked_fill_(~allowed, float("-inf")); msk = msk[None, None]
    RIDE = (mask in ("i", "n", "o"))
    SCALAR = (mask in ("l", "p"))
    PREF = (mask in ("m", "q"))
    ssc = ck.get("sink_sc"); ssc = ssc.to(dev) if ssc is not None else None
    pkv = ck.get("pkv"); pkv = pkv.to(dev) if pkv is not None else None
    H = cfg["n_head"]
    def pref_kv(li, B):
        pk = pkv[li, 0].view(Pn, H, -1).transpose(0, 1)[None]
        pv = pkv[li, 1].view(Pn, H, -1).transpose(0, 1)[None]
        return pk.expand(B, -1, -1, -1), pv.expand(B, -1, -1, -1)
    def pref_kv(li, B):
        pk = pkv[li, 0].view(Pn, H, -1).transpose(0, 1)[None]
        pv = pkv[li, 1].view(Pn, H, -1).transpose(0, 1)[None]
        return pk.expand(B, -1, -1, -1), pv.expand(B, -1, -1, -1)
    rkv = None
    if RIDE:
        rkv = ([(pref_kv(li, 1)[0], pref_kv(li, 1)[1]) for li in range(cfg["n_layer"])]
               if mask == "o" else reg_kv(model, prompt))
    RIDE_POS = (torch.arange(Pn, device=dev) - (W + Pn - 1)).float() if RIDE else None
    Ncont = a.Lm                                                  # content columns (excl. absolute prompt)
    qdeep = torch.arange(Pm + W, Tm, device=dev)                  # deep content queries (rows)
    # column classes in ATT coordinates: [OFFR riding regs | (Pm abs regs) content...]
    rows = []                                                     # (p0, reg, band, rest) per [L,B,h]
    prof = None                                                   # per-query-position class profile [3, Tcontent]
    xx = x
    for li, blk in enumerate(model.blocks):
        B_, T_, Cd = xx.shape; hh = blk.H; d = Cd // hh
        qkv = blk.qkv(blk.ln1(xx)).view(B_, T_, 3, hh, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if RIDE:
            lc = ((rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + msk)
            pk = rkv[li][0].expand(B_, -1, -1, -1); pv = rkv[li][1].expand(B_, -1, -1, -1)
            krd = rope_pos(pk, RIDE_POS)
            att = torch.cat([(q @ krd.transpose(-2, -1)) / math.sqrt(d), lc], dim=-1).softmax(-1)
            OFF = Pn; vv = torch.cat([pv, v], dim=2)
        elif SCALAR:
            lc = ((rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + msk)
            pad = ssc[li].view(1, -1, 1, 1).expand(B_, -1, Tm, 1)
            af = torch.cat([lc, pad], dim=-1).softmax(-1)
            att = af[..., :Tm]; scq = af[..., -1]
            OFF = 0; vv = v
        elif PREF:
            pos = torch.arange(Pn, Pn + Tm, device=dev).float()
            qr = rope_pos(q, pos); kr = rope_pos(k, pos)
            pk, pv = pref_kv(li, B_)
            pkr = rope_pos(pk, torch.arange(Pn, device=dev).float())
            att = torch.cat([(qr @ pkr.transpose(-2, -1)) / math.sqrt(d),
                             (qr @ kr.transpose(-2, -1)) / math.sqrt(d) + msk], dim=-1).softmax(-1)
            OFF = Pn; vv = torch.cat([pv, v], dim=2)
        else:
            att = ((rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + msk).softmax(-1)
            OFF = 0; vv = v
        if not SCALAR: scq = None
        # persistent-column criterion on CONTENT cols (uniform across masks): next-W queries' mean
        NC = Tm - Pm                                              # content cols in mask coords
        cmean = torch.zeros(B_, hh, NC, device=dev)
        for j in range(NC):
            r0 = Pm + j; r1 = min(r0 + W, Tm)
            cmean[:, :, j] = att[:, :, r0:r1, OFF + Pm + j].mean(-1)
        band = (cmean > a.thr / W)
        band[:, :, 0] = False                                     # p0 column is its own class
        Ad = att[:, :, qdeep, :]                                  # [B,h,Q,cols]
        p0 = Ad[:, :, :, OFF + Pm + 0].mean(-1)                            # [B,h]
        reg = (scq[:, :, qdeep].mean(-1) if SCALAR else
               (Ad[:, :, :, :OFF + Pm].sum(-1).mean(-1) if (OFF + Pm) else torch.zeros_like(p0)))
        bm = (Ad[:, :, :, OFF + Pm:] * band[:, :, None, :].float()).sum(-1).mean(-1)
        rows.append(torch.stack([p0, reg, bm, band.float().sum(-1)]))
        if a.profile:                                             # per-query profile over CONTENT rows
            Aq = att[:, :, Pm:, :]                                # [B,h,Tc,cols]
            p0q = Aq[:, :, :, OFF + Pm]                           # [B,h,Tc]
            regq = (scq[:, :, Pm:] if SCALAR else
                    (Aq[:, :, :, :OFF + Pm].sum(-1) if (OFF + Pm) else torch.zeros_like(p0q)))
            bndq = (Aq[:, :, :, OFF + Pm:] * band[:, :, None, :].float()).sum(-1)
            cur = torch.stack([p0q, regq, bndq]).mean(dim=(1, 2))  # [3, Tc] mean over B,h
            prof = cur if prof is None else prof + cur
        o = (att @ vv).transpose(1, 2).reshape(B_, T_, Cd)
        xx = xx + blk.proj(o); xx = xx + blk.mlp(blk.ln2(xx))
    Rm = torch.stack(rows)                                        # [L, 4, B, h]
    p0m, regm, bandm, ncols = Rm[:, 0], Rm[:, 1], Rm[:, 2], Rm[:, 3]
    restm = 1.0 - p0m - regm - bandm
    def s(t): return float(t.mean())
    print("DECOMP mask=%s scope=allheads p0=%.4f reg=%.4f band=%.4f rest=%.4f nbandcols=%.1f"
          % (mask, s(p0m), s(regm), s(bandm), s(restm), s(ncols)), flush=True)
    strength = (p0m + regm + bandm).mean(1)                       # [L,h] total parked mass per head
    li, hi = divmod(int(strength.argmax()), strength.shape[1])
    print("DECOMP mask=%s scope=sinkhead layer=%d head=%d p0=%.4f reg=%.4f band=%.4f rest=%.4f"
          % (mask, li, hi, float(p0m[li, :, hi].mean()), float(regm[li, :, hi].mean()),
             float(bandm[li, :, hi].mean()), float(restm[li, :, hi].mean())), flush=True)
    if a.profile and prof is not None:
        P = (prof / len(model.blocks)).cpu().numpy()              # [3, Tc]: p0, reg, band per query pos
        np.savez("/scratch1/zizhaoh/sinkprof_%s.npz" % mask, p0=P[0], reg=P[1], band=P[2], Lm=a.Lm)
        print("PROFSAVE mask=%s Tc=%d" % (mask, P.shape[1]), flush=True)

for m in a.masks.split(","):
    tag = m if m in "abcdjklmnopq" else "%s_B" % m
    path = "%s/bpe8_%s.pt" % (a.ckpt_dir, tag)
    try:
        ck = torch.load(path, map_location=dev, weights_only=False)
    except Exception as e:
        print("SKIP mask=%s (%r)" % (m, e), flush=True); continue
    decompose(m, ck)
print("DECOMP_DONE", flush=True)
