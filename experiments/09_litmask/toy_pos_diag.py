"""Position diagnostic: train chunk=W IID 'a' (full-causal, cold-start W-windows), then deploy FULL
(growing cache [0..P]) and SLIDING@W, measuring ppl as a function of QUERY POSITION P on a fine grid.
Answers: where does full-deploy break relative to the W=64 training window?

  full_ppl(P)    = exp(mean NLL) of the token at query position P with the full growing cache [0..P]
                   (absolute RoPE). One length-(Pmax+1) full-causal forward gives every position at once.
  sliding_ppl(P) = same token, but with last-W cache + windowed RoPE (positions 0..W-1) -> position-invariant.
Both averaged over N random test-stream segments, SAME offsets/targets for a paired contrast.
"""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math
from statistics import pstdev
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=64)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--n_embd", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)       # budget = steps*batch*ctx; IID n_steps = budget//(batch*W)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pmax", type=int, default=8192)
ap.add_argument("--nseg", type=int, default=64)
ap.add_argument("--mb", type=int, default=4)             # minibatch for the big full-causal forward
ap.add_argument("--p30k", action="store_true")           # add a ~30000 point (separate longer forward)
ap.add_argument("--nseg30k", type=int, default=16)
ap.add_argument("--test_chars", type=int, default=1_200_000)
a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed)
W, L = a.window, a.ctx

ds = load_dataset("wikitext", "wikitext-103-raw-v1")
def get_text(split, nchar):
    s = []; n = 0
    for r in ds[split]:
        s.append(r["text"]); n += len(r["text"])
        if n >= nchar: break
    return "".join(s)[:nchar]
train_txt = get_text("train", 8_000_000); test_txt = get_text("test", a.test_chars)
chars = sorted(set(train_txt + test_txt)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
train_ids = enc(train_txt); test_ids = enc(test_txt).to(dev)

def rope_pos(x, pos, base=10000.0):
    B, H, T, D = x.shape; half = D // 2
    freq = base ** (-torch.arange(0, half, device=x.device).float() / half)
    ang = torch.outer(pos.float(), freq); cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
def rope(x, base=10000.0): return rope_pos(x, torch.arange(x.shape[2], device=x.device), base)

class Block(nn.Module):
    def __init__(self, Cd, H):
        super().__init__(); self.H = H
        self.ln1 = nn.LayerNorm(Cd); self.ln2 = nn.LayerNorm(Cd)
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

def causal_mask(T, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~(k <= q), float("-inf")); return m[None, None]

model = GPT(V, a.n_embd, a.n_head, a.n_layer).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

# ---- train chunk=W IID 'a' (full-causal, cold-start W-windows) ----
budget = a.steps * a.batch * a.ctx
n_steps = budget // (a.batch * W)
tm = causal_mask(W, dev)
print("TRAIN posdiag a chunk=W=%d V=%d n_steps=%d" % (W, V, n_steps), flush=True)
model.train()
for step in range(n_steps):
    ix = torch.randint(0, len(train_ids) - W - 1, (a.batch,))
    x = torch.stack([train_ids[i:i + W] for i in ix]).to(dev)
    y = torch.stack([train_ids[i + 1:i + W + 1] for i in ix]).to(dev)
    lg = model(x, tm)
    loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 2000 == 0: print("  step %d/%d loss %.3f" % (step, n_steps, loss.item()), flush=True)
model.eval()

grid = [2, 4, 8, 16, 24, 32, 48, 56, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024,
        1536, 2048, 3072, 4096, 6144, 8192]
grid = [P for P in grid if P <= a.pmax]

# random offsets so o..o+pmax+1 in range
gseed = torch.Generator().manual_seed(a.seed + 777)
maxoff = test_ids.numel() - a.pmax - 2
offs = torch.randint(0, maxoff, (a.nseg,), generator=gseed).tolist()

@torch.no_grad()
def full_pos_nll(offsets, pmax, mb):
    # per-position NLL via one length-(pmax+1) full-causal forward; returns dict P-> list of per-seg NLL
    acc = {P: [] for P in grid if P <= pmax}
    mask = causal_mask(pmax + 1, dev)
    for i in range(0, len(offsets), mb):
        sel = offsets[i:i + mb]
        inp = torch.stack([test_ids[o:o + pmax + 1] for o in sel])          # positions 0..pmax
        tgt = torch.stack([test_ids[o + 1:o + pmax + 2] for o in sel])      # next token at each position
        lg = model(inp, mask)
        nll = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="none").view(len(sel), pmax + 1)
        for P in acc: acc[P] += nll[:, P].tolist()
    return acc

@torch.no_grad()
def sliding_pos_nll(offsets):
    # query at P: last-min(W,P+1) cache ending at o+P, windowed RoPE 0..winlen-1, causal; predict token o+P+1.
    # For P<W the cache is the available [0..P] (== full there), so full==sliding for P<W by construction.
    acc = {P: [] for P in grid}
    for P in grid:
        winlen = min(W, P + 1); cm = causal_mask(winlen, dev); nlls = []
        for i in range(0, len(offsets), 64):
            sel = offsets[i:i + 64]
            inp = torch.stack([test_ids[o + P + 1 - winlen:o + P + 1] for o in sel])  # winlen tokens, positions 0..winlen-1
            tgt = torch.tensor([test_ids[o + P + 1] for o in sel], device=dev)        # token after position P
            lg = model(inp, cm)[:, -1]                                                # logit at last (query) position
            nlls += F.cross_entropy(lg, tgt, reduction="none").tolist()
        acc[P] = nlls
    return acc

def ppl_stats(acc):
    # ppl = exp(mean NLL); SEM via delta method on NLL (exp(mean) * std(NLL)/sqrt(N)) -- robust to outlier tokens
    ppl = {}; sem = {}
    for P, v in acc.items():
        m = sum(v) / len(v); ppl[P] = math.exp(m)
        sd = pstdev(v) if len(v) > 1 else 0.0
        sem[P] = math.exp(m) * sd / math.sqrt(len(v))
    return ppl, sem

fa = full_pos_nll(offs, a.pmax, a.mb)
sa = sliding_pos_nll(offs)
fp, fs = ppl_stats(fa); sp, ss = ppl_stats(sa)

# ---- print the MAIN grid first (so it survives even if the optional 30k point OOMs) ----
def arr(d): return "[" + ", ".join("%.4f" % d[P] for P in grid) + "]"
print("POSDIAG grid = %s" % ("[" + ", ".join(str(P) for P in grid) + "]"), flush=True)
print("POSDIAG full_ppl = %s" % arr(fp), flush=True)
print("POSDIAG full_sem = %s" % arr(fs), flush=True)
print("POSDIAG sliding_ppl = %s" % arr(sp), flush=True)
print("POSDIAG sliding_sem = %s" % arr(ss), flush=True)

if a.p30k:                                                # optional deep point (full is O(n^2) -> may OOM)
    P30 = 30000
    maxoff30 = test_ids.numel() - P30 - 2
    offs30 = torch.randint(0, maxoff30, (a.nseg30k,), generator=gseed).tolist()
    sl30 = []                                             # sliding@30k is cheap (W-window) -> always report
    for o in offs30:
        inp = test_ids[o + P30 - W + 1:o + P30 + 1][None]; tgt = test_ids[o + P30 + 1][None]
        sl30.append(F.cross_entropy(model(inp, causal_mask(W, dev))[:, -1], tgt, reduction="none").item())
    sp30 = math.exp(sum(sl30) / len(sl30))
    del fa, sa; torch.cuda.empty_cache()
    try:
        nll30 = []
        for o in offs30:
            inp = test_ids[o:o + P30 + 1][None]; tgt = test_ids[o + 1:o + P30 + 2][None]
            lg = model(inp, causal_mask(P30 + 1, dev))
            nll30.append(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="none").view(P30 + 1)[P30].item())
            del lg; torch.cuda.empty_cache()
        fp30 = math.exp(sum(nll30) / len(nll30))
        print("POSDIAG p30000 full_ppl=%.4f sliding_ppl=%.4f" % (fp30, sp30), flush=True)
    except torch.cuda.OutOfMemoryError:
        print("POSDIAG p30000 full_ppl=OOM sliding_ppl=%.4f" % sp30, flush=True)
print("POSDIAG_DONE W=%d nseg=%d n_steps=%d" % (W, a.nseg, n_steps), flush=True)
