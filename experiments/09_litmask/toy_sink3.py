"""From-scratch toy version of cpt_sink3: train a char-RoPE GPT per mask, then measure both sink types
under a chosen deploy.  p0 = deep-query [W,Lm) attention on col-0 (mean over heads); dist = deep-query
attention on NON-pos-0 persistent low-info bands (local visible-query mean c[j]=mean_{q in [j,j+W)}>5/W),
mean over heads.  n eval sequences -> SEM.   a=full-causal, b=sliding."""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math, numpy as np
from datasets import load_dataset
ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True, choices=["a", "b"])
ap.add_argument("--window", type=int, default=64); ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--n_layer", type=int, default=4); ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--n_embd", type=int, default=256); ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--deploy", default="full", choices=["full", "native"])
ap.add_argument("--Lm", type=int, default=1024); ap.add_argument("--nseq", type=int, default=16); ap.add_argument("--S", type=int, default=4)
ap.add_argument("--bos", action="store_true")     # prepend a fixed BOS token at position 0 (stable anchor test)
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); W = a.window; L = a.ctx
ds = load_dataset("wikitext", "wikitext-103-raw-v1")
def get_text(split, nchar):
    s = []; n = 0
    for r in ds[split]:
        s.append(r["text"]); n += len(r["text"])
        if n >= nchar: break
    return "".join(s)[:nchar]
train_txt = get_text("train", 8_000_000); test_txt = get_text("test", 1_200_000)
chars = sorted(set(train_txt + test_txt)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
train_ids = enc(train_txt); test_ids = enc(test_txt).to(dev)
BOS_ID = V; V = V + 1 if a.bos else V                                            # dedicated BOS id (stable position-0 anchor)
def rope_pos(x, pos, base=10000.0):
    B, H, T, D = x.shape; half = D // 2
    freq = base ** (-torch.arange(0, half, device=x.device).float() / half)
    ang = torch.outer(pos.float(), freq); cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
def rope(x, base=10000.0): return rope_pos(x, torch.arange(x.shape[2], device=x.device), base)
class Block(nn.Module):
    def __init__(self, Cd, H):
        super().__init__(); self.H = H; self.ln1 = nn.LayerNorm(Cd); self.ln2 = nn.LayerNorm(Cd)
        self.qkv = nn.Linear(Cd, 3 * Cd); self.proj = nn.Linear(Cd, Cd)
        self.mlp = nn.Sequential(nn.Linear(Cd, 4 * Cd), nn.GELU(), nn.Linear(4 * Cd, Cd))
    def forward(self, x, mask):
        B, T, Cd = x.shape; h = self.H; d = Cd // h
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]; q = rope(q); k = rope(k)
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(d) + mask).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, Cd)
        x = x + self.proj(o); x = x + self.mlp(self.ln2(x)); return x
class GPT(nn.Module):
    def __init__(self, V, Cd, H, Ln):
        super().__init__(); self.emb = nn.Embedding(V, Cd)
        self.blocks = nn.ModuleList([Block(Cd, H) for _ in range(Ln)])
        self.lnf = nn.LayerNorm(Cd); self.head = nn.Linear(Cd, V, bias=False)
    def forward(self, ids, mask):
        x = self.emb(ids)
        for b in self.blocks: x = b(x, mask)
        return self.head(self.lnf(x))
    @torch.no_grad()
    def attn_stack(self, ids, mask):
        x = self.emb(ids); atts = []
        for blk in self.blocks:
            B, T, Cd = x.shape; h = blk.H; d = Cd // h
            qkv = blk.qkv(blk.ln1(x)).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]; q = rope(q); k = rope(k)
            att = ((q @ k.transpose(-2, -1)) / math.sqrt(d) + mask).softmax(-1); atts.append(att)
            o = (att @ v).transpose(1, 2).reshape(B, T, Cd); x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
        return torch.stack(atts, 0)                                              # [L,B,h,T,T]
model = GPT(V, a.n_embd, a.n_head, a.n_layer).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
def mk(allowed, T): m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]
qi = torch.arange(L, device=dev)[:, None]; ki = torch.arange(L, device=dev)[None, :]
tm = mk(ki <= qi, L) if a.mask == "a" else mk((ki <= qi) & (ki > qi - W), L)
def get_batch(bs, T):
    ix = torch.randint(0, len(train_ids) - T - 1, (bs,))
    if a.bos:                                                            # x = [BOS, s0..s_{T-2}], y = s0..s_{T-1}
        spans = torch.stack([train_ids[i:i + T] for i in ix])
        x = torch.cat([torch.full((bs, 1), BOS_ID, dtype=torch.long), spans[:, :T - 1]], 1); y = spans
        return x.to(dev), y.to(dev)
    x = torch.stack([train_ids[i:i + T] for i in ix]); y = torch.stack([train_ids[i + 1:i + T + 1] for i in ix])
    return x.to(dev), y.to(dev)
model.train()
for step in range(a.steps):
    x, y = get_batch(a.batch, L)
    lg = model(x, tm); loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 1000 == 0: print("  mask=%s step %d loss %.3f" % (a.mask, step, loss.item()), flush=True)
model.eval()
Lm = a.Lm; qm = torch.arange(Lm, device=dev)[:, None]; km = torch.arange(Lm, device=dev)[None, :]
if a.deploy == "full": dmask = mk(km <= qm, Lm)
else: dmask = mk(km <= qm, Lm) if a.mask == "a" else mk((km <= qm) & (km > qm - W), Lm)
qdeep = torch.arange(max(W, Lm - W), Lm, device=dev)                              # last-W settled deep queries (sink most grown here)
g = torch.Generator().manual_seed(a.seed + 999)
ix = torch.randint(0, test_ids.numel() - Lm - 2, (a.nseq,), generator=g)
oldest = (qdeep - W + 1).clamp(min=0)                                            # oldest in-window position per deep query
p0m = []; p0x = []; bdm = []; bdx = []
bos_t = torch.tensor([BOS_ID], device=dev, dtype=torch.long)
for i in ix.tolist():
    ids = (torch.cat([bos_t, test_ids[i:i + Lm - 1]]) if a.bos else test_ids[i:i + Lm])[None]
    A = model.attn_stack(ids, dmask)[:, 0].reshape(-1, Lm, Lm)                    # [LH,Lm,Lm]
    p0h = A[:, qdeep, :a.S].sum(-1).mean(-1)                                      # [LH] deep-query first-S cols (pos-0 sink, paper protocol)
    p0m.append(p0h.mean().item()); p0x.append(p0h.max().item())
    bdh = A[:, qdeep, oldest].mean(-1)                                            # [LH] att on oldest in-window (boundary sink)
    bdm.append(bdh.mean().item()); bdx.append(bdh.max().item())
f = lambda v: (np.mean(v), np.std(v) / math.sqrt(len(v)))
(p0mm, p0ms), (p0xm, p0xs), (bdmm, bdms), (bdxm, bdxs) = f(p0m), f(p0x), f(bdm), f(bdx)
print("TOYSINK mask=%s deploy=%s W=%d Lm=%d n=%d p0mean=%.4f p0meansem=%.4f p0max=%.4f p0maxsem=%.4f bdmean=%.4f bdmeansem=%.4f bdmax=%.4f bdmaxsem=%.4f" % (
    a.mask, a.deploy, W, Lm, len(p0m), p0mm, p0ms, p0xm, p0xs, bdmm, bdms, bdxm, bdxs), flush=True)
