"""Per-query-position attention to column 0 (the pos-0 chunk-start), for b (naive SWA: score ALL
queries) vs OURS (full-window loss: score only q >= W-1).

Supports Finding 1 directly: both models share the SAME mask, W, data and budget -- the only
difference is which queries receive loss, hence how many SCORED queries have position 0 in
their receptive field (naive: W of them; ours: 1). If col-0 sink pressure comes from scored
receptive-field exposure, naive grows a pos-0 sink and ours does not.

This is invisible to the usual deep-query (>W) measurement, which structurally cannot see col 0.
Run: python sinkprofile_bf.py --tags b_w128,b_w128_fw --C 256 --W 128
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="b_w128,b_w128_fw")
ap.add_argument("--C", type=int, default=256)
ap.add_argument("--W", type=int, default=128)
ap.add_argument("--n", type=int, default=64)      # eval chunks
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
def profile(tag):
    ck = torch.load("/scratch1/zizhaoh/bpe8_%s.pt" % tag, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = GPT(V, cfg["n_embd"], cfg["n_head"], cfg["n_layer"]).to(dev)
    model.load_state_dict(ck["sd"]); model.eval()
    C, W = a.C, a.W
    g = torch.Generator().manual_seed(1234)
    ix = torch.randint(0, test_ids.numel() - C - 2, (a.n,), generator=g)
    sp = torch.stack([test_ids[i:i + C] for i in ix]).to(dev)
    # SAME input format as training: pos-0 at position 0, then content
    ids = torch.cat([torch.full((a.n, 1), BOS_ID, device=dev, dtype=torch.long), sp[:, :C - 1]], 1)
    qi = torch.arange(C, device=dev)[:, None]; ki = torch.arange(C, device=dev)[None, :]
    ok = (ki <= qi) & (ki > qi - W)                       # the training mask (sliding window)
    m = torch.zeros(C, C, device=dev); m.masked_fill_(~ok, float("-inf")); m = m[None, None]
    x = model.emb(ids); per_layer = []
    for blk in model.blocks:
        B, T, Cd = x.shape; h = blk.H; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = ((rope(q) @ rope(k).transpose(-2, -1)) / math.sqrt(d) + m).softmax(-1)
        per_layer.append(att[:, :, :, 0].mean(0))         # [h, C] attention to col 0 per query
        o = (att @ v).transpose(1, 2).reshape(B, T, Cd)
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
    A = torch.stack(per_layer)                            # [L, h, C]
    prof = A.mean(dim=(0, 1))                             # all-heads mean, per query
    peak = A.reshape(-1, C).max(0).values                 # strongest (layer,head) per query
    # uniform baseline for query q is 1/min(q+1, W); ratio > 1 == over-attended
    wsz = torch.minimum(torch.arange(1, C + 1, device=dev), torch.tensor(W, device=dev)).float()
    ratio = prof * wsz
    return prof.cpu().numpy(), peak.cpu().numpy(), ratio.cpu().numpy()

def band(v, lo, hi): return float(np.mean(v[lo:hi]))
print("query-position profile of attention to col 0 (pos-0). W=%d C=%d n=%d" % (a.W, a.C, a.n))
print("%-14s | %-28s | %-22s | %s" % ("tag", "mean attn->col0 (all-heads)", "peak head", "x uniform"))
for t in a.tags.split(","):
    try:
        p, pk, r = profile(t)
    except FileNotFoundError:
        print("%-14s | (checkpoint missing)" % t); continue
    print("%-14s | q<W/4 %.4f  q<W %.4f      | q<W %.4f            | q<W %.1fx"
          % (t, band(p, 0, a.W // 4), band(p, 0, a.W), band(pk, 0, a.W), band(r, 0, a.W)))
    np.save("/scratch1/zizhaoh/sinkprof_%s.npy" % t, np.stack([p, pk, r]))
    print("   PROFILE %s = %s" % (t, ", ".join("%.4f" % x for x in p[:16])))
print("SINKPROFILE_DONE", flush=True)
