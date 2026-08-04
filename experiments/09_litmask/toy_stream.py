"""Streaming-perplexity eval for the toy from-scratch GPT, mask family a-g (Fig-4 toy streaming).

Reuses the toy GPT + per-mask training of toy_masks.py (a=full/triangle, b=windowed/Gemma-SWAT,
c=swaa, d=longformer) and toy_regimes.py (e=sliding_history, f=warmup, g=persistent). Same toy
config: 4-layer RoPE GPT, char-level WikiText-103, ctx 256, W=64, ~3000 steps.

After training each model we stream PAST the 256-token training length out to ~30k tokens:

  WINDOWED curve (EVERY mask a-g): 30 bins x 1000 tokens. Per bin feed (W=64 history + 1000 bin) as ONE
    forward with RoPE positions reset to 0..1063 and a sliding-window mask (each query attends to its last
    W keys); mean NLL over the 1000 bin tokens -> ppl. Each bin is AVERAGED over N=48 INDEPENDENT segments
    at evenly-spaced offsets through the 1.2M-char test stream (report mean ppl + SEM=pstdev/sqrt(N)); this
    washes out the content-dependent bin-to-bin wiggle, leaving the position-invariant ~flat line.

  FULL curve (ONLY mask a / triangle): 30 bins x 100 of the first 3000 test tokens. Per bin feed the
    full growing prefix [0, t+100] with ABSOLUTE RoPE positions and a full causal mask; mean NLL over
    the 100 bin tokens. Past pos 256 -> diverges (RoPE OOD).

  StreamingLLM curve (ONLY mask a): like the windowed curve (also N-segment averaged) but additionally
    keep the first S=4 sink tokens of the stream. Position-remapped (sink at 0..S-1, query at slot S+W-1)
    so every attended relative distance stays <= S+W-1 (in-distribution) -> rescued to flat-ish.

Prints e.g.  STREAM a windowed = [...]  STREAM a windowed_sem = [...]  STREAM a full = [...]
             STREAM a streamingllm = [...]  STREAM a streamingllm_sem = [...]
"""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math
from statistics import pstdev
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True, choices=["a", "b", "c", "d", "e", "f", "g"])
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
ap.add_argument("--bins", type=int, default=30)          # windowed/streamingllm bins
ap.add_argument("--binsize", type=int, default=1000)     # tokens per windowed/streamingllm bin
ap.add_argument("--nseg", type=int, default=48)          # independent test-stream segments to average each bin over
ap.add_argument("--full_bins", type=int, default=30)     # full-curve bins (mask a)
ap.add_argument("--full_binsize", type=int, default=100) # tokens per full-curve bin
a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed)
S, W, L = a.sink, a.window, a.ctx

# mask a-d -> literature train mask (toy_masks.py); e-g -> our regime (toy_regimes.py)
MASK2TRAINMASK = {"a": "full", "b": "windowed", "c": "swaa", "d": "longformer"}
MASK2SCHEME = {"e": "sliding_history", "f": "warmup", "g": "persistent"}
is_regime = a.mask in MASK2SCHEME

# ------------------------------------------------------------------ data
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

# ------------------------------------------------------------------ RoPE (explicit positions)
def rope_pos(x, pos, base=10000.0):
    # x: [B,H,T,D], pos: [T] (or broadcastable) float positions
    B, H, T, D = x.shape; half = D // 2
    freq = base ** (-torch.arange(0, half, device=x.device).float() / half)
    ang = torch.outer(pos.float(), freq)               # [T, half]
    cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
def rope(x, base=10000.0):                              # default absolute positions 0..T-1
    return rope_pos(x, torch.arange(x.shape[2], device=x.device), base)

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
    def __init__(self, V, C, H, Ln):
        super().__init__(); self.emb = nn.Embedding(V, C)
        self.blocks = nn.ModuleList([Block(C, H) for _ in range(Ln)])
        self.lnf = nn.LayerNorm(C); self.head = nn.Linear(C, V, bias=False)
    def forward(self, ids, mask, prompt=None):
        x = self.emb(ids)
        if prompt is not None:
            x = torch.cat([prompt[None].expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        for b in self.blocks: x = b(x, mask)
        return self.head(self.lnf(x))

# ------------------------------------------------------------------ training masks
def tmask(kind, T, W, S, dev):                          # literature masks (a-d)
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    if kind == "full":         allowed = causal
    elif kind == "windowed":   allowed = causal & (k > q - W)
    elif kind == "swaa":       allowed = causal & ((k < S) | (k > q - W))
    elif kind == "longformer": allowed = (torch.abs(k - q) <= W // 2)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]
def wmask(T, W, dev, nP=0):                             # sliding (+optional persistent prefix) (e-g)
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    allowed = (k <= q) & (k > q - W)
    if nP: allowed = allowed | (k < nP)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]

model = GPT(V, a.n_embd, a.n_head, a.n_layer).to(dev)

# ------------------------------------------------------------------ train
if not is_regime:
    train_mask = MASK2TRAINMASK[a.mask]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    def get_batch(ids, bs, T):
        ix = torch.randint(0, len(ids) - T - 1, (bs,))
        x = torch.stack([ids[i:i + T] for i in ix]); y = torch.stack([ids[i + 1:i + T + 1] for i in ix])
        return x.to(dev), y.to(dev)
    tm = tmask(train_mask, L, W, S, dev)
    model.train()
    for step in range(a.steps):
        x, y = get_batch(train_ids, a.batch, L)
        lg = model(x, tm)
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
else:
    scheme = MASK2SCHEME[a.mask]
    prompt = None
    if scheme == "warmup":       P = max(1, W - 1)
    elif scheme == "persistent": P = a.nP
    else:                        P = 0
    if P:
        prompt = nn.Parameter(torch.randn(P, a.n_embd, device=dev) * 0.02)
    nPk = a.nP if scheme == "persistent" else 0
    params = list(model.parameters()) + ([prompt] if prompt is not None else [])
    opt = torch.optim.AdamW(params, lr=a.lr)
    Hist = max(1, W - 1) if scheme == "sliding_history" else 0
    def get_batch(ids, bs, T):
        ix = torch.randint(0, len(ids) - T - 1, (bs,))
        x = torch.stack([ids[i:i + T] for i in ix]); y = torch.stack([ids[i + 1:i + T + 1] for i in ix])
        return x.to(dev), y.to(dev)
    if scheme == "sliding_history":
        tm = wmask(Hist + L, W, dev)
    else:
        tm = wmask(P + L, W, dev, nP=nPk)
    model.train()
    for step in range(a.steps):
        if scheme == "sliding_history":
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

# ------------------------------------------------------------------ eval masks
def causal_mask(T, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~(k <= q), float("-inf")); return m[None, None]
def slide_mask(T, W, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    allowed = (k <= q) & (k > q - W)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]

# ------------------------------------------------------------------ segment starts (N independent streams)
def seg_starts(bins, binsize, N):
    # N evenly-spaced offsets so each is the start of a full `bins`-bin stream through the test text
    Twin = W + binsize
    span = (bins - 1) * binsize + Twin + 1              # length consumed by one full stream
    maxstart = test_ids.numel() - span
    assert maxstart >= 0, "test stream too short (%d<%d)" % (test_ids.numel(), span)
    if N <= 1: return [0]
    return [round(k * maxstart / (N - 1)) for k in range(N)]

# ------------------------------------------------------------------ WINDOWED streaming curve (every mask)
# Each bin is AVERAGED over N independent test-stream segments (evenly spaced through the test text) so
# the content-dependent bin-to-bin wiggle washes out; position-invariant masks -> ~flat curve + SEM band.
@torch.no_grad()
def windowed_curve(bins, binsize, N):
    means, sems = [], []
    Twin = W + binsize                                  # 1064
    mask = slide_mask(Twin, W, dev)
    starts = seg_starts(bins, binsize, N)
    for i in range(bins):
        offs = [st + i * binsize for st in starts]
        inp = torch.stack([test_ids[o:o + Twin] for o in offs])          # [N, Twin]; positions 0..Twin-1
        tgt = torch.stack([test_ids[o + 1:o + Twin + 1] for o in offs])  # [N, Twin]
        lg = model(inp, mask)                            # [N, Twin, V]
        nll = F.cross_entropy(lg[:, -binsize:].reshape(-1, V),
                              tgt[:, -binsize:].reshape(-1),
                              reduction="none").view(len(offs), binsize).mean(1)   # [N] mean-NLL/segment
        ppls = torch.exp(nll).tolist()
        means.append(sum(ppls) / len(ppls))
        sems.append(pstdev(ppls) / math.sqrt(len(ppls)) if len(ppls) > 1 else 0.0)
    return means, sems

# ------------------------------------------------------------------ FULL growing-prefix curve (mask a)
@torch.no_grad()
def full_curve(bins, binsize):
    out = []
    need = bins * binsize + 1
    assert test_ids.numel() >= need
    for i in range(bins):
        plen = (i + 1) * binsize                        # growing prefix length
        inp = test_ids[:plen][None]                     # absolute positions 0..plen-1
        tgt = test_ids[1:plen + 1]
        lg = model(inp, causal_mask(plen, dev))[0]
        nll = F.cross_entropy(lg[i * binsize:plen], tgt[i * binsize:plen], reduction="mean").item()
        out.append(math.exp(nll))
    return out

# ------------------------------------------------------------------ StreamingLLM curve (mask a)
@torch.no_grad()
def compute_sink_kv(sink_ids):
    # run the S sink tokens (causal among themselves, positions 0..S-1); capture per-block pre-RoPE k, v
    x = model.emb(sink_ids)[None]; T = sink_ids.numel()
    cm = causal_mask(T, dev); kv = []
    for blk in model.blocks:
        B, _, C = x.shape; h = blk.H; d = C // h
        qkv = blk.qkv(blk.ln1(x)).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        kv.append((k, v))
        qr = rope(q); kr = rope(k)
        att = ((qr @ kr.transpose(-2, -1)) / math.sqrt(d) + cm).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
    return kv

@torch.no_grad()
def streamingllm_curve(bins, binsize, N):
    sink_kv = compute_sink_kv(test_ids[:S])             # global sink = first S stream tokens (shared)
    Twin = W + binsize
    pos = torch.arange(Twin, device=dev)
    qpos_sink = torch.full((Twin,), float(S + W - 1), device=dev)
    spos = torch.arange(S, device=dev)
    win_mask = slide_mask(Twin, W, dev)[0, 0]           # [Twin, Twin]
    means, sems = [], []
    starts = seg_starts(bins, binsize, N)
    for i in range(bins):
        offs = [st + i * binsize for st in starts]
        ids = torch.stack([test_ids[o:o + Twin] for o in offs])          # [N, Twin]
        tgt = torch.stack([test_ids[o + 1:o + Twin + 1] for o in offs])  # [N, Twin]
        x = model.emb(ids)                               # [N, Twin, C]
        for b, blk in enumerate(model.blocks):
            B, _, C = x.shape; h = blk.H; d = C // h
            qkv = blk.qkv(blk.ln1(x)).view(B, Twin, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            sk, sv = sink_kv[b]                          # [1,H,S,d] (broadcasts over batch N)
            # window logits: natural relative positions (q@pos, k@pos)
            lw = (rope_pos(q, pos) @ rope_pos(k, pos).transpose(-2, -1)) / math.sqrt(d) + win_mask
            # sink logits: query remapped to slot S+W-1, sink keys at 0..S-1 -> rel dist <= S+W-1
            ls = (rope_pos(q, qpos_sink) @ rope_pos(sk, spos).transpose(-2, -1)) / math.sqrt(d)
            p = torch.cat([ls, lw], dim=-1).softmax(-1)  # [N,H,Twin, S+Twin]
            ps, pw = p[..., :S], p[..., S:]
            o = (ps @ sv + pw @ v).transpose(1, 2).reshape(B, Twin, C)
            x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
        lg = model.head(model.lnf(x))                    # [N, Twin, V]
        nll = F.cross_entropy(lg[:, -binsize:].reshape(-1, V),
                              tgt[:, -binsize:].reshape(-1),
                              reduction="none").view(len(offs), binsize).mean(1)   # [N]
        ppls = torch.exp(nll).tolist()
        means.append(sum(ppls) / len(ppls))
        sems.append(pstdev(ppls) / math.sqrt(len(ppls)) if len(ppls) > 1 else 0.0)
    return means, sems

# ------------------------------------------------------------------ run + report
def fmt(v): return "[" + ", ".join("%.3f" % x for x in v) + "]"
win_m, win_s = windowed_curve(a.bins, a.binsize, a.nseg)
print("STREAM %s windowed = %s" % (a.mask, fmt(win_m)), flush=True)
print("STREAM %s windowed_sem = %s" % (a.mask, fmt(win_s)), flush=True)
if a.mask == "a":
    full = full_curve(a.full_bins, a.full_binsize)      # single stream (mask a only, not plotted)
    sllm_m, sllm_s = streamingllm_curve(a.bins, a.binsize, a.nseg)
    print("STREAM a full = %s" % fmt(full), flush=True)
    print("STREAM a streamingllm = %s" % fmt(sllm_m), flush=True)
    print("STREAM a streamingllm_sem = %s" % fmt(sllm_s), flush=True)
print("STREAM_DONE mask=%s V=%d W=%d S=%d ctx=%d steps=%d bins=%d binsize=%d nseg=%d"
      % (a.mask, V, W, S, L, a.steps, a.bins, a.binsize, a.nseg), flush=True)
