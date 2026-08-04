"""Tiny from-scratch GPT (char-level WikiText-103, RoPE) trained under a literature attention mask, then
eval char-level ppl under full / sliding / StreamingLLM. Produces the Fig-2 b/c/d TOY rows (mask alone
determines behaviour, no pretrained confound).
  train_mask: full | windowed (b) | swaa (c) | longformer (d, attends future -> degenerate, quantified)
"""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--train_mask", required=True, choices=["full", "windowed", "swaa", "longformer"])
ap.add_argument("--window", type=int, default=64)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--n_embd", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed)

ds = load_dataset("wikitext", "wikitext-103-raw-v1")
def get_text(split, nchar):
    s = []
    n = 0
    for r in ds[split]:
        s.append(r["text"]); n += len(r["text"])
        if n >= nchar: break
    return "".join(s)[:nchar]
train_txt = get_text("train", 8_000_000); test_txt = get_text("test", 600_000)
chars = sorted(set(train_txt + test_txt)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
train_ids = enc(train_txt); test_ids = enc(test_txt)


def rope(x, base=10000.0):
    B, H, T, D = x.shape; half = D // 2
    freq = base ** (-torch.arange(0, half, device=x.device).float() / half)
    ang = torch.outer(torch.arange(T, device=x.device).float(), freq)
    cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, C, H):
        super().__init__(); self.H = H
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
    def __init__(self, V, C, H, L):
        super().__init__(); self.emb = nn.Embedding(V, C)
        self.blocks = nn.ModuleList([Block(C, H) for _ in range(L)])
        self.lnf = nn.LayerNorm(C); self.head = nn.Linear(C, V, bias=False)
    def forward(self, ids, mask):
        x = self.emb(ids)
        for b in self.blocks: x = b(x, mask)
        return self.head(self.lnf(x))


def tmask(kind, L, W, S, dev):
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]
    causal = k <= q
    if kind == "full":         allowed = causal
    elif kind == "windowed":   allowed = causal & (k > q - W)
    elif kind == "swaa":       allowed = causal & ((k < S) | (k > q - W))
    elif kind == "longformer": allowed = (torch.abs(k - q) <= W // 2)
    m = torch.zeros(L, L, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]


def emask(kind, L, W, S, dev):
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]
    causal = k <= q
    if kind == "full":        allowed = causal
    elif kind == "sliding":   allowed = causal & (k > q - W)
    elif kind == "streaming": allowed = causal & ((k < S) | (k > q - W))
    m = torch.zeros(L, L, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]


model = GPT(V, a.n_embd, a.n_head, a.n_layer).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
L = a.ctx
def get_batch(ids, bs):
    ix = torch.randint(0, len(ids) - L - 1, (bs,))
    x = torch.stack([ids[i:i + L] for i in ix]); y = torch.stack([ids[i + 1:i + L + 1] for i in ix])
    return x.to(dev), y.to(dev)
tm = tmask(a.train_mask, L, a.window, a.sink, dev)
model.train()
for step in range(a.steps):
    x, y = get_batch(train_ids, a.batch)
    lg = model(x, tm)
    loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 500 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
model.eval()
def ppl(kind):
    em = emask(kind, L, a.window, a.sink, dev); nll = 0.0; ntok = 0; sl = L // 2
    with torch.no_grad():
        for _ in range(30):
            x, y = get_batch(test_ids, a.batch)
            lg = model(x, em)
            nll += F.cross_entropy(lg[:, -sl:].reshape(-1, V), y[:, -sl:].reshape(-1), reduction="sum").item()
            ntok += a.batch * sl
    return math.exp(nll / ntok)
print("RESULT toy train=%s W=%d sink=%d V=%d | full %.3f | sliding@%d %.3f | streaming(s%d+%d) %.3f"
      % (a.train_mask, a.window, a.sink, V, ppl("full"), a.window, ppl("sliding"), a.sink, a.window, ppl("streaming")),
      flush=True)
print("TOY_DONE", flush=True)
