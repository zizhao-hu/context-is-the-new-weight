"""FIXED all-8 toy experiment (W=64): each mask trained its FAITHFUL way, then the continuous
sliding-KV-cache eval (random N=128 offsets) + 3 deploy columns. Streaming Table-1 + Figure-4.

TRAINING
  LITERATURE (IID random chunks, normal -- they LEARN the cold start; loss over ALL positions incl. the
  first-W partial windows):
    a Triangle (full causal)        : IID, full mask.        (unbounded baseline)
    b Gemma/SWAT (plain sliding)    : IID, sliding-W mask, loss over all (learns to predict early tokens).
    c SWAA (sink + window)          : IID, swaa mask, loss over all.
    d Longformer (attends future)   : IID, longformer mask.  (degenerate baseline)
  OURS + TXL (truncated-BPTT streaming, carried DETACHED cache, B=32 instances, chunk C=W=64), BOTH
  A=carry-cache-across-instances and B=reset-per-instance:
    e Transformer-XL : full-causal WITHIN chunk + carried previous-segment cache (segment recurrence).
    f sliding-history: PURE FULL-WINDOW -- score ONLY positions that already have a full real W-window;
                       SKIP the first-W cold start of each instance ("wait to accumulate"). No prefix.
    g sliding+warmup : prepend a trainable warmup soft-prompt that FILLS the cold start; DO score the early
                       positions (the prompt supplies the missing context).
    h sliding+persist: nP=8 always-attended trainable registers.
  All match the IID per-run loss-token budget (~24.6M).

EVAL (continuous sliding-KV-cache: append each chunk's pre-RoPE K/V, EVICT oldest -> last-W cache; per-token
NLL vs the fully-settled rolling history; windowed/relative RoPE). RANDOM seeded segment offsets, N=128.
  (1) 30-bin streaming curve (mean + SEM)   [= deploy-sliding]
  (2) deep-position ppl = last (30th) bin
  (3) deploy columns (deep-position ppl): full / sliding@W / streaming(S4+W).
Deploy columns use PLAIN inference masks (no train-time prompt/registers); h also reports streaming-with-its-registers.
"""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math
from statistics import pstdev
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True, choices=["a", "b", "c", "d", "e", "f", "g", "h"])
ap.add_argument("--boundary", default="B", choices=["A", "B"])   # only e/f/g/h
ap.add_argument("--window", type=int, default=64)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--nP", type=int, default=8)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--n_embd", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)       # IID steps; defines the token budget
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--inst_chunks", type=int, default=64)
ap.add_argument("--stream_steps", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bins", type=int, default=30)
ap.add_argument("--binsize", type=int, default=1000)
ap.add_argument("--nseg", type=int, default=128)         # random eval segments
ap.add_argument("--nseg_full", type=int, default=16)     # segments for the expensive growing-cache full deploy
ap.add_argument("--mb_grow", type=int, default=4)        # minibatch for the growing-cache stream
ap.add_argument("--test_chars", type=int, default=1_200_000)
ap.add_argument("--iid_w", action="store_true")          # fairness v1: IID masks train on W-token windows (chunk=W, cold-start)
ap.add_argument("--iid_2w", action="store_true")         # fairness v2: IID masks train on 2W windows, score ONLY last W (full W-window of context each, matching sliding cache+chunk)
ap.add_argument("--log_every", type=int, default=0)      # >0: emit TRAINPT (step, running-mean loss) every N steps for convergence curves
a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed)
S, W, L = a.sink, a.window, a.ctx
C = W
STREAM = a.mask in ("e", "f", "g", "h")
tag = ("%s_%s" % (a.mask, a.boundary)) if STREAM else a.mask

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
    def forward(self, ids, mask):
        x = self.emb(ids)
        for b in self.blocks: x = b(x, mask)
        return self.head(self.lnf(x))

model = GPT(V, a.n_embd, a.n_head, a.n_layer).to(dev)
prompt = None; Pn = 0
if a.mask == "g":   Pn = max(1, W - 1)
elif a.mask == "h": Pn = a.nP
if Pn: prompt = nn.Parameter(torch.randn(Pn, a.n_embd, device=dev) * 0.02)
params = list(model.parameters()) + ([prompt] if prompt is not None else [])
opt = torch.optim.AdamW(params, lr=a.lr)

# ------------------------------------------------------------------ masks
def causal_mask(T, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~(k <= q), float("-inf")); return m[None, None]
def slide_mask(T, W, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    a_ = (k <= q) & (k > q - W)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~a_, float("-inf")); return m[None, None]
def tmask(kind, T, W, S, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    if kind == "full":         allowed = causal
    elif kind == "windowed":   allowed = causal & (k > q - W)
    elif kind == "swaa":       allowed = causal & ((k < S) | (k > q - W))
    elif kind == "longformer": allowed = (torch.abs(k - q) <= W // 2)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]

def compute_kv_from_emb(emb_in, detach):
    x = emb_in; P = x.shape[1]; cm = causal_mask(P, dev); kv = []
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

def stream_step(ids, rolling, prefix_kv, realmask, growing):
    """Per-layer pre-RoPE KV-cache forward (truncated-BPTT in training via cache .detach())."""
    x = model.emb(ids); B, Cl = ids.shape; new_rolling = []
    for li, blk in enumerate(model.blocks):
        h = blk.H; Cd = x.shape[2]; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, Cl, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        rk = rv = None; Wc = 0
        if rolling is not None: rk, rv = rolling[li]; Wc = rk.shape[2]
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
        allk = torch.cat([rk, k], dim=2) if rk is not None else k
        allv = torch.cat([rv, v], dim=2) if rv is not None else v
        if not growing: allk = allk[:, :, -W:, :]; allv = allv[:, :, -W:, :]
        new_rolling.append((allk.detach(), allv.detach()))
    return model.head(model.lnf(x)), new_rolling

def draw_instances(ids, B, Linst):
    span = Linst + 1
    ix = torch.randint(0, len(ids) - span, (B,))
    blk = torch.stack([ids[i:i + span] for i in ix]).to(dev)
    return blk[:, :Linst], blk[:, 1:Linst + 1]

# ------------------------------------------------------------------ TRAIN
budget_tokens = a.steps * a.batch * a.ctx
print("TRAIN mask=%s boundary=%s tag=%s V=%d W=%d Pn=%d budget=%d" % (a.mask, a.boundary, tag, V, W, Pn, budget_tokens), flush=True)
_lacc = [0.0, 0]                                          # [sum_loss, count] for running-mean convergence logging
def log_pt(stp, lossval):
    if a.log_every <= 0: return
    _lacc[0] += lossval; _lacc[1] += 1
    if stp % a.log_every == 0:
        print("TRAINPT tag=%s step=%d loss=%.5f" % (tag, stp, _lacc[0] / _lacc[1]), flush=True)
        _lacc[0] = 0.0; _lacc[1] = 0
model.train()
if not STREAM:                                            # LITERATURE: IID random chunks
    LIT = {"a": "full", "b": "windowed", "c": "swaa", "d": "longformer"}
    if a.iid_2w:                                           # fairness v2: 2W window, score last W (full W-context each)
        Lc = 2 * W; score_from = W; n_steps = budget_tokens // (a.batch * W)
    elif a.iid_w:                                          # fairness v1: W window, cold-start, score all
        Lc = W; score_from = 0; n_steps = budget_tokens // (a.batch * W)
    else:                                                  # original: ctx-256 window, score all
        Lc = L; score_from = 0; n_steps = a.steps
    print("IID_CHUNK mask=%s iid_w=%s iid_2w=%s Lc=%d score_from=%d n_steps=%d"
          % (a.mask, a.iid_w, a.iid_2w, Lc, score_from, n_steps), flush=True)
    tm = tmask(LIT[a.mask], Lc, W, S, dev)
    def get_batch(ids, bs, T):
        ix = torch.randint(0, len(ids) - T - 1, (bs,))
        x = torch.stack([ids[i:i + T] for i in ix]); y = torch.stack([ids[i + 1:i + T + 1] for i in ix])
        return x.to(dev), y.to(dev)
    for step in range(n_steps):
        x, y = get_batch(train_ids, a.batch, Lc)
        lg = model(x, tm)
        loss = F.cross_entropy(lg[:, score_from:].reshape(-1, V), y[:, score_from:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        log_pt(step + 1, loss.item())
        if step % 2000 == 0: print("  step %d/%d loss %.3f" % (step, n_steps, loss.item()), flush=True)
    used_tokens = n_steps * a.batch * (Lc - score_from)
else:                                                     # OURS + TXL: truncated-BPTT streaming
    realmask = "causal" if a.mask == "e" else "window"
    prefix_kind = "cold" if a.mask == "g" else ("always" if a.mask == "h" else "none")
    skip_cold = (a.mask == "f")
    reset_boundary = (a.boundary == "B")
    Linst = a.inst_chunks * C
    target = a.stream_steps * a.batch * C if a.stream_steps else budget_tokens   # match IID loss-token budget
    rolling = None; xi = yi = None; ptr = a.inst_chunks; scored = 0; step = 0
    while scored < target:                                 # budget-driven (so f's cold-skip still matches budget)
        if ptr >= a.inst_chunks:
            xi, yi = draw_instances(train_ids, a.batch, Linst); ptr = 0
            if reset_boundary: rolling = None
        Wc = 0 if rolling is None else rolling[0][0].shape[2]
        prefix_kv = None
        if prefix_kind == "always":  prefix_kv = compute_kv_from_emb(prompt[None], detach=False)
        elif prefix_kind == "cold" and rolling is None: prefix_kv = compute_kv_from_emb(prompt[None], detach=False)
        cx = xi[:, ptr * C:(ptr + 1) * C]; cy = yi[:, ptr * C:(ptr + 1) * C]
        lg, rolling = stream_step(cx, rolling, prefix_kv, realmask, growing=False)
        if skip_cold:
            t0 = max(0, W - 1 - Wc)                        # first position with a full real W-window
            if t0 >= C: ptr += 1; continue                 # whole chunk still cold -> skip (never hits at C=W)
            loss = F.cross_entropy(lg[:, t0:].reshape(-1, V), cy[:, t0:].reshape(-1)); scored += a.batch * (C - t0)
        else:
            loss = F.cross_entropy(lg.reshape(-1, V), cy.reshape(-1)); scored += a.batch * C
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        ptr += 1; step += 1; log_pt(step, loss.item())
        if step % 2000 == 0: print("  step %d scored %d/%d loss %.3f" % (step, scored, target, loss.item()), flush=True)
    used_tokens = scored
    print("STREAM_STEPS mask=%s boundary=%s steps=%d Linst=%d realmask=%s prefix=%s skip_cold=%s scored=%d"
          % (a.mask, a.boundary, step, Linst, realmask, prefix_kind, skip_cold, scored), flush=True)
model.eval()

# ------------------------------------------------------------------ EVAL: continuous sliding-KV-cache, RANDOM offsets
bins, binsize, warm = a.bins, a.binsize, W
Total = bins * binsize
need = warm + Total + 1
maxstart = test_ids.numel() - need
assert maxstart > 0, "test stream too short"
g = torch.Generator().manual_seed(a.seed + 12345)
starts = torch.randint(0, maxstart, (a.nseg,), generator=g).tolist()

@torch.no_grad()
def continuous_curve(prefix_mode, realmask, growing, seg, mb, fixed_prefix=None):
    # Returns (bin_means[30], bin_sems[30], avg, avg_sem, deep, deep_sem). avg = mean over 30 bins
    # (whole-stream), deep = last bin (~30k). Each bin = exp(mean-NLL). Minibatched over `seg` segments.
    ninp = warm + Total
    rows = []                                                  # per-segment ppl row [30]
    for m0 in range(0, len(seg), mb):
        sel = seg[m0:m0 + mb]; Bn = len(sel)
        s_seq = torch.stack([test_ids[s:s + need] for s in sel])
        rolling = None; sink_kv = None
        pertok = torch.empty(Bn, ninp, device=dev)
        for c0 in range(0, ninp, C):
            end = min(c0 + C, ninp); clen = end - c0
            cx = s_seq[:, c0:end]; cy = s_seq[:, c0 + 1:end + 1]
            pkv = None
            if prefix_mode == "sink":
                if rolling is None: sink_kv = compute_kv_from_emb(model.emb(cx[:, :S]), True); pkv = None
                else: pkv = sink_kv
            elif prefix_mode == "fixed_always": pkv = fixed_prefix
            elif prefix_mode == "fixed_cold":   pkv = fixed_prefix if rolling is None else None
            lg, rolling = stream_step(cx, rolling, pkv, realmask, growing)
            pertok[:, c0:end] = F.cross_entropy(lg.reshape(-1, V), cy.reshape(-1), reduction="none").view(Bn, clen)
        binnll = pertok[:, warm:warm + Total].view(Bn, bins, binsize).mean(-1)     # [Bn, bins] mean-NLL
        rows += torch.exp(binnll).tolist()                     # per-seg per-bin ppl
    ppl = torch.tensor(rows)                                   # [Nseg, bins]
    N = ppl.shape[0]
    bin_means = ppl.mean(0).tolist(); bin_sems = (ppl.std(0, unbiased=False) / math.sqrt(N)).tolist()
    seg_avg = ppl.mean(1)                                      # [Nseg] whole-stream avg per seg
    avg = seg_avg.mean().item(); avg_sem = (seg_avg.std(unbiased=False) / math.sqrt(N)).item()
    deep = bin_means[-1]; deep_sem = bin_sems[-1]
    return bin_means, bin_sems, avg, avg_sem, deep, deep_sem

def fmt(v): return "[" + ", ".join("%.4f" % x for x in v) + "]"
seg_full = starts[:a.nseg_full]
sl = continuous_curve("none", "window", False, starts, a.nseg)            # sliding@W
st = continuous_curve("sink", "window", False, starts, a.nseg)            # StreamingLLM(S4+W)
fu = continuous_curve("none", "causal", True,  seg_full, a.mb_grow)       # full (growing, breaks)
print("CURVE %s means = %s" % (tag, fmt(sl[0])), flush=True)              # sliding 30-bin curve (Fig-4)
print("CURVE %s sems = %s" % (tag, fmt(sl[1])), flush=True)
print("DEPLOY mask=%s boundary=%s tag=%s V=%d W=%d used_tokens=%d Nseg=%d Nfull=%d | "
      "full_avg %.4f (sem %.4f) full_deep %.4f (sem %.4f) | "
      "sliding_avg %.4f (sem %.4f) sliding_deep %.4f (sem %.4f) | "
      "streaming_avg %.4f (sem %.4f) streaming_deep %.4f (sem %.4f)"
      % (a.mask, a.boundary, tag, V, W, used_tokens, a.nseg, a.nseg_full,
         fu[2], fu[3], fu[4], fu[5], sl[2], sl[3], sl[4], sl[5], st[2], st[3], st[4], st[5]), flush=True)
if a.mask == "h":
    reg_kv = compute_kv_from_emb(prompt[None], True)
    hr = continuous_curve("fixed_always", "window", False, starts, a.nseg, fixed_prefix=reg_kv)
    print("DEPLOY_EXTRA mask=h boundary=%s | streamingregs_avg %.4f (sem %.4f) streamingregs_deep %.4f (sem %.4f)"
          % (a.boundary, hr[2], hr[3], hr[4], hr[5]), flush=True)
@torch.no_grad()
def measure_sink():
    """Read the trained model's sink pattern with a single settled forward under each mask's
    train/deploy config (h prepends its always-on registers). p0 = attn mass on key col 0;
    bd = attn mass on the oldest in-window key (relative boundary). mean/max over (layers,heads)."""
    model.eval()
    Bm, Lm = 4, max(L, 1024)
    gms = torch.Generator().manual_seed(a.seed + 777)
    ixm = torch.randint(0, test_ids.numel() - Lm - 2, (Bm,), generator=gms)
    seq = torch.stack([test_ids[i:i + Lm + 1].cpu() for i in ixm]).to(dev)
    ids_m = seq[:, :Lm]; tgt_m = seq[:, 1:Lm + 1]
    x = model.emb(ids_m); Pm = 0
    if a.mask == "h":
        x = torch.cat([prompt[None].expand(Bm, -1, -1).to(x.dtype), x], dim=1); Pm = Pn
    Tm = x.shape[1]
    qi = torch.arange(Tm, device=dev)[:, None]; ki = torch.arange(Tm, device=dev)[None, :]
    if a.mask in ("a", "e"):   allowed = ki <= qi
    elif a.mask == "c":        allowed = (ki <= qi) & ((ki < S) | (ki > qi - W))
    elif a.mask == "d":        allowed = torch.abs(ki - qi) <= W // 2
    elif a.mask == "h":
        rq = qi - Pm; rk = ki - Pm
        allowed = (ki < Pm) | ((rk <= rq) & (rk > rq - W))
    else:                      allowed = (ki <= qi) & (ki > qi - W)
    msk = torch.zeros(Tm, Tm, device=dev); msk.masked_fill_(~allowed, float("-inf")); msk = msk[None, None]
    qts = torch.arange(max(Pm + W, Tm - W), Tm, device=dev)      # DEEP settled queries (last W: full's sink is most grown here)
    nsink = Pm if a.mask == "h" else S                            # sink region = nP registers (h) or first-S cols
    oldest = (qts - W + 1).clamp(min=Pm)
    a0s = []; rbs = []; attL = []; xx = x
    for blk in model.blocks:
        B_, T_, Cd = xx.shape; hh = blk.H; d = Cd // hh
        qkv = blk.qkv(blk.ln1(xx)).view(B_, T_, 3, hh, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]; q = rope(q); k = rope(k)
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(d) + msk).softmax(-1)
        attL.append(att.detach())
        a0s.append(att[:, :, qts, :nsink].sum(-1).mean(dim=(0, 2)))
        rbs.append(att[:, :, qts, oldest].mean(dim=(0, 2)))
        o = (att @ v).transpose(1, 2).reshape(B_, T_, Cd); xx = xx + blk.proj(o); xx = xx + blk.mlp(blk.ln2(xx))
    A0 = torch.stack(a0s); RB = torch.stack(rbs)
    try:                                                          # band-head map + bands-per-window count
        import numpy as _np
        _bs = torch.stack([at[0, :, qts, :].mean(1).max(-1).values for at in attL])   # [layers,h] strongest persistent column
        _li, _hi = divmod(int(_bs.argmax().item()), _bs.shape[1])
        _ch = [chars[int(t)] for t in ids_m[0].tolist()]
        _Mh = attL[_li][0, _hi].float().cpu().numpy()
        _colp = _Mh[W:].mean(0)                                    # per-key mean attn over deep queries
        for _thr in (2.0, 3.0, 5.0):
            _nb = int((_colp > _thr / W).sum())
            print('BANDCOUNT mask=%s W=%d Lm=%d thr=%gx nbands=%d bands_per_window=%.2f'
                  % (a.mask, W, Lm, _thr, _nb, _nb / (Lm / W)), flush=True)
        _np.savez('/scratch1/zizhaoh/toymap_%s_W%d.npz' % (a.mask, W), M=_Mh,
                  chars=_np.array(_ch, dtype=object), Pm=int(Pm), W=int(W))
        print('MAPSAVE mask=%s W=%d layer=%d head=%d bandstrength=%.3f' % (a.mask, W, _li, _hi, float(_bs.max())), flush=True)
    except Exception as _e:
        print('MAPSAVE_FAIL', repr(_e), flush=True)
    lg = model.head(model.lnf(xx))[:, Pm:, :]
    vl = F.cross_entropy(lg[:, W:].reshape(-1, V), tgt_m[:, W:].reshape(-1))
    return A0.mean().item(), A0.max().item(), RB.mean().item(), RB.max().item(), math.exp(vl.item())

_p0m, _p0x, _bdm, _bdx, _vppl = measure_sink()
print("SINK mask=%s boundary=%s p0_mean=%.4f p0_max=%.4f bd_mean=%.4f bd_max=%.4f valppl=%.4f"
      % (a.mask, a.boundary, _p0m, _p0x, _bdm, _bdx, _vppl), flush=True)
@torch.no_grad()
def band_analysis():
    """Persistent-band sink-mass over N instances -> mean +/- SEM. Per-column visible-query mean;
    sink cols = visible-mean > 5x uniform(1/W); sink-mass = frac of deep-query attention on them."""
    model.eval()
    import numpy as _np
    Bm, Lm = 16, 512
    gms = torch.Generator().manual_seed(a.seed + 999)
    ixm = torch.randint(0, test_ids.numel() - Lm - 2, (Bm,), generator=gms)
    seq = torch.stack([test_ids[i:i + Lm + 1].cpu() for i in ixm]).to(dev)
    ids_b = seq[:, :Lm]
    x = model.emb(ids_b); Pm = 0
    if a.mask == "h":
        x = torch.cat([prompt[None].expand(Bm, -1, -1).to(x.dtype), x], dim=1); Pm = Pn
    Tm = x.shape[1]
    qi = torch.arange(Tm, device=dev)[:, None]; ki = torch.arange(Tm, device=dev)[None, :]
    if a.mask in ("a", "e"):   allowed = ki <= qi
    elif a.mask == "c":        allowed = (ki <= qi) & ((ki < S) | (ki > qi - W))
    elif a.mask == "h":
        rq = qi - Pm; rk = ki - Pm; allowed = (ki < Pm) | ((rk <= rq) & (rk > rq - W))
    else:                      allowed = (ki <= qi) & (ki > qi - W)
    msk = torch.zeros(Tm, Tm, device=dev); msk.masked_fill_(~allowed, float("-inf")); msk = msk[None, None]
    qdeep = torch.arange(Pm + W, Tm, device=dev)
    qearly = torch.arange(1, min(Pm + W + 1, Tm), device=dev)                       # queries that can still see col-0
    bsL = []; smL = []; p0L = []; xx = x
    for blk in model.blocks:
        B_, T_, Cd = xx.shape; hh = blk.H; d = Cd // hh
        qkv = blk.qkv(blk.ln1(xx)).view(B_, T_, 3, hh, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]; q = rope(q); k = rope(k)
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(d) + msk).softmax(-1)
        c = torch.zeros(B_, hh, Tm, device=dev)
        for j in range(Tm):
            r1 = min(j + W, Tm); c[:, :, j] = att[:, :, j:r1, j].mean(-1)
        band = (c > 5.0 / W).float()
        sm = (att[:, :, qdeep, :] * band[:, :, None, :]).sum(-1).mean(-1)         # [B,h] distributed sink-mass (deep q)
        p0 = att[:, :, qearly, 0].mean(-1)                                        # [B,h] pos-0 mass (early q that see it)
        bsL.append(c.max(-1).values); smL.append(sm); p0L.append(p0)
        o = (att @ v).transpose(1, 2).reshape(B_, T_, Cd); xx = xx + blk.proj(o); xx = xx + blk.mlp(blk.ln2(xx))
    BS = torch.stack(bsL); SM = torch.stack(smL); P0 = torch.stack(p0L)            # [L,B,h]
    _li, _hi = divmod(int(BS.mean(1).argmax().item()), BS.shape[2])                # global sink head (distributed)
    _pli, _phi = divmod(int(P0.mean(1).argmax().item()), P0.shape[2])             # global pos-0 head
    dm = SM[_li, :, _hi].cpu().numpy(); p0a = P0[_pli, :, _phi].cpu().numpy(); unif = 1.0 / W
    print("SWEEP mask=%s W=%d n=%d distmass=%.4f distsem=%.4f distratio=%.1f p0mass=%.4f p0sem=%.4f p0ratio=%.1f" % (
        a.mask, W, len(dm), float(dm.mean()), float(dm.std() / _np.sqrt(len(dm))), float(dm.mean()) / unif,
        float(p0a.mean()), float(p0a.std() / _np.sqrt(len(p0a))), float(p0a.mean()) / unif), flush=True)

band_analysis()
print("ALL8_DONE mask=%s boundary=%s" % (a.mask, a.boundary), flush=True)
