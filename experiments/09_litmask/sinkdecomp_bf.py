"""Persistent-column decomposition for b (naive SWA) vs OURS (constant-context sliding),
split by query regime -- the piece the deep-query-only measurement cannot see.

Classes (disjoint, sum to 1), using the SAME criterion as sinkdecomp_bpe9.py:
  p0    = attention on column 0 (the pos-0 chunk start)          <- the concentrated sink
  band  = persistent content columns EXCLUDING p0              <- the distributed / separator sink
          (a column is persistent iff the next-W queries give it > thr/W, thr=5 => 5x uniform)
  rest  = ordinary content (remainder to 1)

Reported for COLD queries (q < W: pos-0 is inside the window) and DEEP queries (q >= W: pos-0 is
evicted, so p0 is structurally 0). Naive scores all queries -> W of them see pos-0; ours scores
only full-window queries -> 1 does.

Run: python sinkdecomp_bf.py --tags b_w128,b_w128_fw --C 256 --W 128
"""
import torch, torch.nn as nn, math, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="a:causal,b_w128:window,b_w128_fw:window")
ap.add_argument("--C", type=int, default=256)
ap.add_argument("--W", type=int, default=128)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--thr", type=float, default=5.0)
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
    ang = torch.outer(pos.float(), freq); cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
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
def decomp(tag, kind='window'):
    ck = torch.load("/scratch1/zizhaoh/bpe8_%s.pt" % tag, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = GPT(V, cfg["n_embd"], cfg["n_head"], cfg["n_layer"]).to(dev); model.load_state_dict(ck["sd"]); model.eval()
    C, W = a.C, a.W
    g = torch.Generator().manual_seed(1234)
    ix = torch.randint(0, test_ids.numel() - C - 2, (a.n,), generator=g)
    sp = torch.stack([test_ids[i:i + C] for i in ix]).to(dev)
    ids = torch.cat([torch.full((a.n, 1), BOS_ID, device=dev, dtype=torch.long), sp[:, :C - 1]], 1)

    qi = torch.arange(C, device=dev)[:, None]; ci = torch.arange(C, device=dev)[None, :]
    ok = (ci <= qi) if kind == "causal" else ((ci <= qi) & (ci > qi - W))
    m = torch.zeros(C, C, device=dev); m.masked_fill_(~ok, float("-inf")); m = m[None, None]
    nxt = ((qi > ci) & (qi <= ci + W)).float()            # next-W queries after each column
    cnt = nxt.sum(0).clamp(min=1)
    qc = torch.arange(1, W, device=dev)                   # cold queries (pos-0 in window)
    qd = torch.arange(W, C, device=dev)                   # deep queries (pos-0 evicted)

    acc = {k: [] for k in ("p0c", "bdc", "p0d", "bdd", "ncol")}
    x = model.emb(ids)
    for blk in model.blocks:
        B, T, Cd = x.shape; h = blk.H; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = ((rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + m).softmax(-1)   # [B,h,C,C]
        cmean = (att * nxt[None, None]).sum(2) / cnt[None, None]                        # [B,h,C]
        band = cmean > (a.thr / W); band[:, :, 0] = False                               # p0 is its own class
        bf = band[:, :, None, :].float()
        acc["p0c"].append(att[:, :, qc, 0].mean(-1))
        acc["bdc"].append((att[:, :, qc, :] * bf).sum(-1).mean(-1))
        acc["p0d"].append(att[:, :, qd, 0].mean(-1))
        acc["bdd"].append((att[:, :, qd, :] * bf).sum(-1).mean(-1))
        acc["ncol"].append(band.float().sum(-1))
        o = (att @ v).transpose(1, 2).reshape(B, T, Cd)
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
    r = {k: torch.stack(v).mean().item() for k, v in acc.items()}
    return r

print("persistent-column decomposition  (thr=%.0fx uniform, W=%d, C=%d, n=%d)" % (a.thr, a.W, a.C, a.n))
print("%-12s | %-28s | %s" % ("tag", "COLD q<W  (pos-0 in window)", "DEEP q>=W (pos-0 evicted)"))
print("%-12s | %-8s %-8s %-9s | %-8s %-8s %s" % ("", "p0", "sep", "content", "p0", "sep", "content"))
out = {}
for spec in a.tags.split(","):
    t, _, kind = spec.partition(":"); kind = kind or "window"
    try:
        r = decomp(t, kind)
    except FileNotFoundError:
        print("%-12s | (ckpt missing)" % t); continue
    cc = 1.0 - r["p0c"] - r["bdc"]; cd = 1.0 - r["p0d"] - r["bdd"]
    out[t + ":" + kind] = (r["p0c"], r["bdc"], cc, r["p0d"], r["bdd"], cd)
    print("%-12s | %-8.4f %-8.4f %-9.4f | %-8.4f %-8.4f %.4f"
          % (t + "/" + kind[:4], r["p0c"], r["bdc"], cc, r["p0d"], r["bdd"], cd))
    print("   nbandcols=%.1f" % r["ncol"])
if len(out) >= 2:
    a_, b_ = list(out.values())[-2], list(out.values())[-1]
    print("DELTA cold: p0 %+.4f  sep %+.4f  content %+.4f" % (b_[0]-a_[0], b_[1]-a_[1], b_[2]-a_[2]))
    print("  -> where the freed p0 mass went (cold): sep %+.4f, content %+.4f" % (b_[1]-a_[1], b_[2]-a_[2]))
print("DECOMP_BF_DONE", flush=True)
