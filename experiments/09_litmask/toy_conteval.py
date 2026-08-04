"""CONTINUOUS-STREAMING eval (sliding KV-cache that keeps DROPPING old tokens) for the IID-trained toy
GPTs -- the corrected Figure-4 curves.

PROBLEM with toy_stream.py windowed_curve: each 1000-tok bin is a FRESH forward with a cold W warmup and
reset positions -- not one continuous stream. FIX (here): stream the trained model through ~30k tokens
maintaining a per-layer KV-cache; for each new chunk append its (pre-RoPE) K/V and EVICT the oldest so the
cache stays at the last W (constant memory, sliding window that keeps dropping). Score per-token NLL
continuously against the real, fully-settled rolling cache. RoPE positions are windowed/relative (cached
keys 0..W-1, query at W) so it's position-invariant. Bin per 1000 tokens, average 48 independent stream
segments -> 30-bin mean+SEM (same Figure-4 format) + deep-position ppl (last bin).

Training is IID, EXACTLY as the current Figure-4 (toy_masks.py / toy_regimes.py); only the eval changes.
Masks (Fig-4 labels):
  a  full / triangle (causal). Two eval curves: 'a_full' = cache GROWS, never evicts -> absolute positions
     OOD past ctx -> diverges (the BREAKS curve); 'a_sllm' = StreamingLLM deploy (keep S=4 sink K/V + last-W
     sliding, positions remapped in-distribution -> rescued).
  b  Gemma/SWAT  (plain sliding window).      continuous windowed cache.
  c  SWAA        (sink + window).             continuous windowed cache + first-S sink K/V kept.
  f  sliding-history.                         continuous windowed cache.
  g  warmup (trainable soft-prompt cold-start filler).   continuous windowed; prompt at the stream start only.
  h  persistent (nP always-attended registers).          continuous windowed + registers always attended.
"""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math
from statistics import pstdev
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True, choices=["a", "b", "c", "f", "g", "h"])
ap.add_argument("--window", type=int, default=64)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--nP", type=int, default=8)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--n_embd", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bins", type=int, default=30)
ap.add_argument("--binsize", type=int, default=1000)
ap.add_argument("--nseg", type=int, default=48)          # windowed (constant-cache) curves
ap.add_argument("--nseg_grow", type=int, default=12)     # growing-cache (a_full) curve (expensive)
ap.add_argument("--mb_grow", type=int, default=4)        # minibatch for the growing-cache stream
ap.add_argument("--test_chars", type=int, default=1_200_000)
a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed)
S, W, L = a.sink, a.window, a.ctx
C = W                                                    # streaming chunk length == window

# ------------------------------------------------------------------ data
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

# ------------------------------------------------------------------ model (RoPE GPT, explicit positions)
def rope_pos(x, pos, base=10000.0):
    B, H, T, D = x.shape; half = D // 2
    freq = base ** (-torch.arange(0, half, device=x.device).float() / half)
    ang = torch.outer(pos.float(), freq)
    cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
def rope(x, base=10000.0):
    return rope_pos(x, torch.arange(x.shape[2], device=x.device), base)

class Block(nn.Module):
    def __init__(self, Cd, H):
        super().__init__(); self.H = H
        self.ln1 = nn.LayerNorm(Cd); self.ln2 = nn.LayerNorm(Cd)
        self.qkv = nn.Linear(Cd, 3 * Cd); self.proj = nn.Linear(Cd, Cd)
        self.mlp = nn.Sequential(nn.Linear(Cd, 4 * Cd), nn.GELU(), nn.Linear(4 * Cd, Cd))
    def forward(self, x, mask):
        B, T, Cd = x.shape; h = self.H; d = Cd // h
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(q); k = rope(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d) + mask
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, Cd)
        x = x + self.proj(o); x = x + self.mlp(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self, V, Cd, H, Ln):
        super().__init__(); self.emb = nn.Embedding(V, Cd)
        self.blocks = nn.ModuleList([Block(Cd, H) for _ in range(Ln)])
        self.lnf = nn.LayerNorm(Cd); self.head = nn.Linear(Cd, V, bias=False)
    def forward(self, ids, mask, prompt=None):
        x = self.emb(ids)
        if prompt is not None:
            x = torch.cat([prompt[None].expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        for b in self.blocks: x = b(x, mask)
        return self.head(self.lnf(x))

model = GPT(V, a.n_embd, a.n_head, a.n_layer).to(dev)

# ------------------------------------------------------------------ training masks (IID; toy_masks/toy_regimes-faithful)
def tmask(kind, T, W, S, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    if kind == "full":       allowed = causal
    elif kind == "windowed": allowed = causal & (k > q - W)
    elif kind == "swaa":     allowed = causal & ((k < S) | (k > q - W))
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]
def wmask(T, W, dev, nP=0):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    allowed = (k <= q) & (k > q - W)
    if nP: allowed = allowed | (k < nP)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]

# Fig-4 label -> IID training scheme
M2TRAIN = {"a": "full", "b": "windowed", "c": "swaa", "f": "sliding_history", "g": "warmup", "h": "persistent"}
scheme = M2TRAIN[a.mask]
prompt = None
if scheme == "warmup":       Pn = max(1, W - 1)
elif scheme == "persistent": Pn = a.nP
else:                        Pn = 0
if Pn:
    prompt = nn.Parameter(torch.randn(Pn, a.n_embd, device=dev) * 0.02)
nPk = a.nP if scheme == "persistent" else 0
Hist = max(1, W - 1) if scheme == "sliding_history" else 0
params = list(model.parameters()) + ([prompt] if prompt is not None else [])
opt = torch.optim.AdamW(params, lr=a.lr)

def get_batch(ids, bs, T):
    ix = torch.randint(0, len(ids) - T - 1, (bs,))
    x = torch.stack([ids[i:i + T] for i in ix]); y = torch.stack([ids[i + 1:i + T + 1] for i in ix])
    return x.to(dev), y.to(dev)

print("TRAIN mask=%s scheme=%s V=%d W=%d Pn=%d steps=%d" % (a.mask, scheme, V, W, Pn, a.steps), flush=True)
model.train()
if scheme in ("full", "windowed", "swaa"):
    tm = tmask(scheme, L, W, S, dev)
    for step in range(a.steps):
        x, y = get_batch(train_ids, a.batch, L)
        lg = model(x, tm)
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 500 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
elif scheme == "sliding_history":
    tm = wmask(Hist + L, W, dev)
    for step in range(a.steps):
        x, y = get_batch(train_ids, a.batch, Hist + L)
        lg = model(x, tm)
        loss = F.cross_entropy(lg[:, Hist:].reshape(-1, V), y[:, Hist:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 500 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
else:  # warmup / persistent
    tm = wmask(Pn + L, W, dev, nP=nPk)
    for step in range(a.steps):
        x, y = get_batch(train_ids, a.batch, L)
        lg = model(x, tm, prompt=prompt)
        loss = F.cross_entropy(lg[:, Pn:].reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 500 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
model.eval()

# ------------------------------------------------------------------ KV-cache forward (reused from streaming-training stream_step)
def compute_kv_from_emb(emb_in, detach):
    x = emb_in; P = x.shape[1]
    q_ = torch.arange(P, device=dev)[:, None]; k_ = torch.arange(P, device=dev)[None, :]
    cm = torch.zeros(P, P, device=dev); cm.masked_fill_(~(k_ <= q_), float("-inf")); cm = cm[None, None]
    kv = []
    for blk in model.blocks:
        B, _, Cd = x.shape; h = blk.H; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, P, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        kv.append((k.detach(), v.detach()) if detach else (k, v))
        qr = rope(q); kr = rope(k)
        att = ((qr @ kr.transpose(-2, -1)) / math.sqrt(d) + cm).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, P, Cd)
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
    return kv

@torch.no_grad()
def stream_step(ids, rolling, prefix_kv, realmask, growing):
    x = model.emb(ids); B, Cl = ids.shape; new_rolling = []
    for li, blk in enumerate(model.blocks):
        h = blk.H; Cd = x.shape[2]; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, Cl, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        rk = rv = None; Wc = 0
        if rolling is not None:
            rk, rv = rolling[li]; Wc = rk.shape[2]
        pk = pv = None; Pfx = 0
        if prefix_kv is not None:
            pk, pv = prefix_kv[li]
            if pk.shape[0] == 1 and B > 1: pk = pk.expand(B, -1, -1, -1); pv = pv.expand(B, -1, -1, -1)
            Pfx = pk.shape[2]
        klist = ([pk] if pk is not None else []) + ([rk] if rk is not None else []) + [k]
        vlist = ([pv] if pv is not None else []) + ([rv] if rv is not None else []) + [v]
        kf = torch.cat(klist, dim=2); vf = torch.cat(vlist, dim=2)
        Lk = Pfx + Wc + Cl
        kpos = torch.arange(Lk, device=ids.device).float()
        qpos = torch.arange(Pfx + Wc, Pfx + Wc + Cl, device=ids.device).float()
        qr = rope_pos(q, qpos); kr = rope_pos(kf, kpos)
        att = (qr @ kr.transpose(-2, -1)) / math.sqrt(d)
        col = torch.arange(Lk, device=ids.device)[None, :]
        t = torch.arange(Cl, device=ids.device)[:, None]
        is_prefix = col < Pfx
        jp = col - Pfx; rpos = Wc + t
        real_ok = (jp <= rpos) & (jp > rpos - W) if realmask == "window" else (jp <= rpos)
        allowed = is_prefix | (~is_prefix & real_ok)
        att = att.masked_fill(~allowed[None, None], float("-inf")).softmax(-1)
        o = (att @ vf).transpose(1, 2).reshape(B, Cl, Cd)
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
        if rk is not None:
            allk = torch.cat([rk, k], dim=2); allv = torch.cat([rv, v], dim=2)
        else:
            allk = k; allv = v
        if not growing:
            allk = allk[:, :, -W:, :]; allv = allv[:, :, -W:, :]
        new_rolling.append((allk.detach(), allv.detach()))
    return model.head(model.lnf(x)), new_rolling

# ------------------------------------------------------------------ CONTINUOUS-STREAMING curve
def seg_starts(N, need):
    maxstart = test_ids.numel() - need
    assert maxstart >= 0, "test stream too short (%d<%d)" % (test_ids.numel(), need)
    if N <= 1: return [0]
    return [round(k * maxstart / (N - 1)) for k in range(N)]

@torch.no_grad()
def continuous_curve(prefix_mode, realmask, growing, N, mb, fixed_prefix=None):
    # prefix_mode: 'none' | 'sink' (capture first S of stream) | 'fixed_always' | 'fixed_cold'
    bins = a.bins; binsize = a.binsize; warm = W; Total = bins * binsize
    starts = seg_starts(N, warm + Total + 1)
    bin_ppls = [[] for _ in range(bins)]
    for mb0 in range(0, len(starts), mb):
        sel = starts[mb0:mb0 + mb]; B = len(sel)
        seq = torch.stack([test_ids[s:s + warm + Total + 1] for s in sel])  # [B, warm+Total+1]
        rolling = None; sink_kv = None
        pertok = torch.empty(B, warm + Total, device=dev)
        ninp = warm + Total
        for c0 in range(0, ninp, C):
            end = min(c0 + C, ninp); clen = end - c0
            cx = seq[:, c0:end]; cy = seq[:, c0 + 1:end + 1]
            pkv = None
            if prefix_mode == "sink":
                if rolling is None:
                    sink_kv = compute_kv_from_emb(model.emb(cx[:, :S]), True); pkv = None
                else: pkv = sink_kv
            elif prefix_mode == "fixed_always":
                pkv = fixed_prefix
            elif prefix_mode == "fixed_cold":
                pkv = fixed_prefix if rolling is None else None
            lg, rolling = stream_step(cx, rolling, pkv, realmask, growing)
            nll = F.cross_entropy(lg.reshape(-1, V), cy.reshape(-1), reduction="none").view(B, clen)
            pertok[:, c0:c0 + clen] = nll
        scored = pertok[:, warm:warm + Total].view(B, bins, binsize).mean(-1)  # [B, bins] mean-NLL
        ppl = torch.exp(scored)
        for i in range(bins): bin_ppls[i].extend(ppl[:, i].tolist())
    means = [sum(x) / len(x) for x in bin_ppls]
    sems = [pstdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0 for x in bin_ppls]
    return means, sems

# ------------------------------------------------------------------ run + report
def fmt(v): return "[" + ", ".join("%.4f" % x for x in v) + "]"
def emit(name, m, s):
    print("CURVE %s means = %s" % (name, fmt(m)), flush=True)
    print("CURVE %s sems = %s" % (name, fmt(s)), flush=True)
    print("DEEP %s deep_ppl=%.4f sem=%.4f stream_mean=%.4f min=%.4f max=%.4f"
          % (name, m[-1], s[-1], sum(m) / len(m), min(m), max(m)), flush=True)

if a.mask == "a":
    m, s = continuous_curve("none", "causal", True, a.nseg_grow, a.mb_grow)      # full, grows -> breaks
    emit("a_full", m, s)
    m, s = continuous_curve("sink", "window", False, a.nseg, a.nseg)             # StreamingLLM rescue
    emit("a_sllm", m, s)
elif a.mask == "c":
    m, s = continuous_curve("sink", "window", False, a.nseg, a.nseg)             # SWAA: sink + window
    emit("c", m, s)
elif a.mask == "g":
    pkv = compute_kv_from_emb(prompt[None], True)                                # warmup prompt at cold start
    m, s = continuous_curve("fixed_cold", "window", False, a.nseg, a.nseg, fixed_prefix=pkv)
    emit("g", m, s)
elif a.mask == "h":
    pkv = compute_kv_from_emb(prompt[None], True)                                # persistent registers always
    m, s = continuous_curve("fixed_always", "window", False, a.nseg, a.nseg, fixed_prefix=pkv)
    emit("h", m, s)
else:  # b, f : plain continuous windowed
    m, s = continuous_curve("none", "window", False, a.nseg, a.nseg)
    emit(a.mask, m, s)
print("CONTEVAL_DONE mask=%s" % a.mask, flush=True)
