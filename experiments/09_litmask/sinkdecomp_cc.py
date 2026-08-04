"""Persistent-column decomposition, split by query regime, for the constant-context (cc)
sliding family INCLUDING the trainable-sink designs.

Classes (disjoint, sum to 1) -- same criterion as sinkdecomp_bpe9.py (thr=5 => 5x uniform):
  p0    = attention on content column 0 (the pos-0 chunk start)
  reg   = the TRAINABLE sink (token: prepended prompt cols; prefix: per-layer K/V;
          scalar: a per-head softmax-denominator logit with NO column)
  band  = persistent content columns excluding p0 (the distributed / separator sink)
  rest  = ordinary content (remainder to 1)

COLD (q < W: pos-0 inside the window) vs DEEP (q >= W: pos-0 evicted).
Geometry mirrors toy_all8_bpe.py: token/prefix sinks are prepended and always visible.

Run: python sinkdecomp_cc.py --tags a:causal:,b_w128:window:,b_w128_fw:window:,k_w128_fw:window:token,m_w128_fw:window:prefix,l_w128_fw:window:scalar
"""
import torch, torch.nn as nn, math, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="a:causal:,b_w128:window:,b_w128_fw:window:,"
                                  "k_w128_fw:window:token,m_w128_fw:window:prefix,l_w128_fw:window:scalar")
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
def decomp(tag, kind, sink):
    ck = torch.load("/scratch1/zizhaoh/bpe8_%s.pt" % tag, map_location=dev, weights_only=False)
    cfg = ck["cfg"]; Ln, H, Cd_ = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
    Pn = cfg.get("Pn", 0) or 0
    model = GPT(V, Cd_, H, Ln).to(dev); model.load_state_dict(ck["sd"]); model.eval()
    prompt = ck.get("prompt"); prompt = prompt.to(dev) if prompt is not None else None
    pkv = ck.get("pkv"); pkv = pkv.to(dev) if pkv is not None else None
    ssc = ck.get("sink_sc"); ssc = ssc.to(dev) if ssc is not None else None
    C, W = a.C, a.W
    g = torch.Generator().manual_seed(1234)
    ix = torch.randint(0, test_ids.numel() - C - 2, (a.n,), generator=g)
    sp = torch.stack([test_ids[i:i + C] for i in ix]).to(dev)
    ids = torch.cat([torch.full((a.n, 1), BOS_ID, device=dev, dtype=torch.long), sp[:, :C - 1]], 1)

    x = model.emb(ids); Pm = 0
    if sink == "token":                              # prompt tokens prepended, always visible
        x = torch.cat([prompt[None].expand(a.n, -1, -1).to(x.dtype), x], dim=1); Pm = Pn
    T = x.shape[1]
    qi = torch.arange(T, device=dev)[:, None]; ci = torch.arange(T, device=dev)[None, :]
    rq, rc = qi - Pm, ci - Pm
    if kind == "causal": okc = (rc <= rq)
    else:                okc = (rc <= rq) & (rc > rq - W)
    ok = (ci < Pm) | (okc & (ci >= Pm))              # prompt cols always allowed
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~ok, float("-inf")); m = m[None, None]
    OFF = Pm                                         # content columns start here
    nxt = ((rq > rc) & (rq <= rc + W) & (ci >= Pm) & (qi >= Pm)).float()
    cnt = nxt.sum(0).clamp(min=1)
    qc = torch.arange(Pm + 1, Pm + W, device=dev)    # cold content queries
    qd = torch.arange(Pm + W, T, device=dev)         # deep content queries

    acc = {k: [] for k in ("p0c", "rgc", "bdc", "p0d", "rgd", "bdd")}
    for li, blk in enumerate(model.blocks):
        B, Tt, Cd = x.shape; h = blk.H; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, Tt, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        qr, kr = rope(q), rope(k)
        if sink == "prefix":                         # per-layer learned pre-RoPE K/V
            pk = pkv[li, 0].view(Pn, h, -1).transpose(0, 1)[None].expand(B, -1, -1, -1)
            pv = pkv[li, 1].view(Pn, h, -1).transpose(0, 1)[None].expand(B, -1, -1, -1)
            pos = torch.arange(Pn, Pn + Tt, device=dev).float()
            qr2, kr2 = rope_pos(q, pos), rope_pos(k, pos)
            pkr = rope_pos(pk, torch.arange(Pn, device=dev).float())
            lc = (qr2 @ kr2.transpose(-2, -1)) / math.sqrt(d) + m
            lp = (qr2 @ pkr.transpose(-2, -1)) / math.sqrt(d)
            att = torch.cat([lp, lc], dim=-1).softmax(-1)
            reg = att[:, :, :, :Pn].sum(-1); att = att[:, :, :, Pn:]
            o = (att @ v).transpose(1, 2).reshape(B, Tt, Cd)
            o = o + (torch.cat([lp, lc], -1).softmax(-1)[:, :, :, :Pn] @ pv).transpose(1, 2).reshape(B, Tt, Cd)
        elif sink == "scalar":                       # denominator logit, no column, no value
            lc = (qr @ kr.transpose(-2, -1)) / math.sqrt(d) + m
            pad = ssc[li].view(1, -1, 1, 1).expand(B, -1, Tt, 1)
            af = torch.cat([lc, pad], dim=-1).softmax(-1)
            reg = af[:, :, :, -1]; att = af[:, :, :, :Tt]
            o = (att @ v).transpose(1, 2).reshape(B, Tt, Cd)
        else:                                        # none / token (prompt already in-sequence)
            att = ((qr @ kr.transpose(-2, -1)) / math.sqrt(d) + m).softmax(-1)
            reg = att[:, :, :, :Pm].sum(-1) if Pm else torch.zeros_like(att[:, :, :, 0])
            o = (att @ v).transpose(1, 2).reshape(B, Tt, Cd)
        cm = (att * nxt[None, None]).sum(2) / cnt[None, None]
        band = cm > (a.thr / W); band[:, :, :OFF + 1] = False        # p0 + prompt are own classes
        bf = band[:, :, None, :].float()
        acc["p0c"].append(att[:, :, qc, OFF].mean(-1)); acc["p0d"].append(att[:, :, qd, OFF].mean(-1))
        acc["rgc"].append(reg[:, :, qc].mean(-1));      acc["rgd"].append(reg[:, :, qd].mean(-1))
        acc["bdc"].append((att[:, :, qc, :] * bf).sum(-1).mean(-1))
        acc["bdd"].append((att[:, :, qd, :] * bf).sum(-1).mean(-1))
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
    return {k: torch.stack(v).mean().item() for k, v in acc.items()}

print("cc-family decomposition (thr=%.0fx uniform, W=%d, C=%d, n=%d)" % (a.thr, a.W, a.C, a.n))
print("%-16s | %-34s | %s" % ("tag", "COLD q<W: p0 / reg / sep / content", "DEEP q>=W: p0 / reg / sep / content"))
for spec in a.tags.split(","):
    parts = (spec.split(":") + ["", ""])[:3]
    t, kind, sink = parts[0], (parts[1] or "window"), (parts[2] or "none")
    try:
        r = decomp(t, kind, sink)
    except FileNotFoundError:
        print("%-16s | (ckpt missing)" % t); continue
    cc = 1 - r["p0c"] - r["rgc"] - r["bdc"]; cd = 1 - r["p0d"] - r["rgd"] - r["bdd"]
    print("%-16s | %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f"
          % (t + "/" + sink[:3], r["p0c"], r["rgc"], r["bdc"], cc, r["p0d"], r["rgd"], r["bdd"], cd))
print("DECOMP_CC_DONE", flush=True)
