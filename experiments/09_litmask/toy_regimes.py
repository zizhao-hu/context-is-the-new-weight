"""Our regimes e/f/g for the from-scratch toy GPT (char-level WikiText, RoPE), through the SAME eval as
toy_masks.py. Mirrors cpt_regimes.py semantics.
  sliding_history (e): sliding window W, trained on (W-1 real history + L content) windows, loss on content; eval plain.
  warmup (f):          P=W-1 trainable prompt embeddings fill the cold start; eval prepends them (subject to mask).
  persistent (g):      nP trainable prompt embeddings, ALWAYS attended; eval prepends them, always-kept.
"""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["sliding_history", "warmup", "persistent"])
ap.add_argument("--window", type=int, default=64)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--n_embd", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--nP", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed)
S, W, L = a.sink, a.window, a.ctx

ds = load_dataset("wikitext", "wikitext-103-raw-v1")
def get_text(split, nchar):
    s = []; n = 0
    for r in ds[split]:
        s.append(r["text"]); n += len(r["text"])
        if n >= nchar: break
    return "".join(s)[:nchar]
train_txt = get_text("train", 8_000_000); test_txt = get_text("test", 600_000)
chars = sorted(set(train_txt + test_txt)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
train_ids = enc(train_txt); test_ids = enc(test_txt)


def rope(x, base=10000.0):
    B, Hh, T, D = x.shape; half = D // 2
    freq = base ** (-torch.arange(0, half, device=x.device).float() / half)
    ang = torch.outer(torch.arange(T, device=x.device).float(), freq)
    cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, C, Hh):
        super().__init__(); self.H = Hh
        self.ln1 = nn.LayerNorm(C); self.ln2 = nn.LayerNorm(C)
        self.qkv = nn.Linear(C, 3 * C); self.proj = nn.Linear(C, C)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))
    def forward(self, x, mask):
        B, T, C = x.shape; h = self.H; d = C // h
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(q); k = rope(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d) + mask
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(o); x = x + self.mlp(self.ln2(x)); return x


class GPT(nn.Module):
    def __init__(self, V, C, Hh, Ln):
        super().__init__(); self.emb = nn.Embedding(V, C)
        self.blocks = nn.ModuleList([Block(C, Hh) for _ in range(Ln)])
        self.lnf = nn.LayerNorm(C); self.head = nn.Linear(C, V, bias=False)
    def forward(self, ids, mask, prompt=None):
        x = self.emb(ids)
        if prompt is not None:
            x = torch.cat([prompt[None].expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        for b in self.blocks: x = b(x, mask)
        return self.head(self.lnf(x))


def wmask(T, W, dev, nP=0):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    allowed = (k <= q) & (k > q - W)
    if nP: allowed = allowed | (k < nP)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]


def emask(kind, T, W, S, dev, nP=0):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    if kind == "full":        allowed = causal
    elif kind == "sliding":   allowed = causal & (k > q - W)
    elif kind == "streaming": allowed = causal & ((k < S) | (k > q - W))
    if nP: allowed = allowed | (k < nP)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]


model = GPT(V, a.n_embd, a.n_head, a.n_layer).to(dev)
prompt = None
if a.scheme == "warmup":     P = max(1, W - 1)
elif a.scheme == "persistent": P = a.nP
else:                        P = 0
if P:
    prompt = nn.Parameter(torch.randn(P, a.n_embd, device=dev) * 0.02)
nPk = a.nP if a.scheme == "persistent" else 0
params = list(model.parameters()) + ([prompt] if prompt is not None else [])
opt = torch.optim.AdamW(params, lr=a.lr)

Hist = max(1, W - 1) if a.scheme == "sliding_history" else 0
def get_batch(ids, bs, T):
    ix = torch.randint(0, len(ids) - T - 1, (bs,))
    x = torch.stack([ids[i:i + T] for i in ix]); y = torch.stack([ids[i + 1:i + T + 1] for i in ix])
    return x.to(dev), y.to(dev)

model.train()
if a.scheme == "sliding_history":
    tm = wmask(Hist + L, W, dev)
else:
    tm = wmask(P + L, W, dev, nP=nPk)
for step in range(a.steps):
    if a.scheme == "sliding_history":
        x, y = get_batch(train_ids, a.batch, Hist + L)
        lg = model(x, tm)
        loss = F.cross_entropy(lg[:, Hist:].reshape(-1, V), y[:, Hist:].reshape(-1))
    else:
        x, y = get_batch(train_ids, a.batch, L)
        lg = model(x, tm, prompt=prompt)
        loss = F.cross_entropy(lg[:, P:].reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if step % 500 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)

model.eval()
sl = L // 2
def ppl(kind):
    nll = 0.0; ntok = 0
    with torch.no_grad():
        for _ in range(30):
            x, y = get_batch(test_ids, a.batch, L)
            if a.scheme == "sliding_history":
                lg = model(x, emask(kind, L, W, S, dev))
                cl = lg[:, -sl:]
            else:
                lg = model(x, emask(kind, P + L, W, S, dev, nP=nPk), prompt=prompt)
                cl = lg[:, P:][:, -sl:]
            nll += F.cross_entropy(cl.reshape(-1, V), y[:, -sl:].reshape(-1), reduction="sum").item()
            ntok += a.batch * sl
    return math.exp(nll / ntok)
print("RESULT toy scheme=%s W=%d V=%d | full %.3f | sliding@%d %.3f | streaming(s%d+%d) %.3f"
      % (a.scheme, W, V, ppl("full"), W, ppl("sliding"), S, W, ppl("streaming")), flush=True)
print("TOYREG_DONE", flush=True)
