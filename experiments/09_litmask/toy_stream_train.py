"""STREAMING (truncated-BPTT) training for the toy GPT, all masks a-h (+ IID windowed baseline w),
streaming version of Table 1 + Figure 4.

Reuses the toy GPT (4-layer RoPE, char-level WikiText-103, W=64) and the averaged streaming eval
(windowed_curve, 48 segments, 30 bins to 30k). REPLACES training with a real truncated-BPTT stream;
each mask carries its OWN cache/mask logic DURING streaming (B=32 instances, chunk C=W=64, cache
DETACHED between chunks). Two instance-boundary variants per windowed mask: A=carry, B=reset.

  w  IID windowed baseline (random fixed-256 windows, sliding mask). boundary n/a. ONE run.
  a  Triangle / full causal: unbounded cache within an instance (full-causal over all seen tokens),
     RESET per instance. A/B not meaningful -> ONE run. Unbounded baseline.
  b  Gemma/SWAT (plain sliding window): windowed last-W cache. A + B.
  c  SWAA (sink + window): cache keeps the first S=sink tokens of the stream PLUS the last W. A + B.
  d  Longformer (attends future): NOT streamable causally -> trained the IID way (degenerate). ONE run.
  e  Transformer-XL: full-causal WITHIN [prev-segment cache (W) ; chunk] (segment recurrence). A + B.
     Asymmetric foil (full-causal over 2 segments, not a per-query sliding window).
  f  sliding-history: windowed last-W cache (== b mechanically). A + B.
  g  sliding + warmup: windowed cache + trainable warmup soft-prompt (P=W-1) filling the cold start
     (used at every instance start under B; moot under A, cold-starts only at step 0). A + B.
  h  sliding + persistent: windowed cache + nP persistent ALWAYS-attended trainable registers. A + B.

All runs match the IID per-run token budget (~24.6M loss tokens). EVAL at W=64 per model:
  (1) deep_ppl       : held-out deep-position ppl = last 1000-tok bin of windowed_curve (48-seg mean).
  (2) deploy columns : deploy the trained weights under full / sliding@W / streaming(S4+W); deep-position
                       ppl for each (litmask Table-1 style; sliding@W deep == deep_ppl).
  (3) Fig-4 curve    : windowed_curve 30-bin mean + SEM.
  (+) h only also reports deploy_streaming with its persistent REGISTERS as the StreamingLLM prefix.
Deploy columns use PLAIN inference masks (no train-time prompt/registers fed) for a uniform comparison
of the trained weights; the h-with-registers number shows the registers' deploy contribution.
"""
import torch, torch.nn as nn, torch.nn.functional as F, argparse, math
from statistics import pstdev
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True, choices=["w", "a", "b", "c", "d", "e", "f", "g", "h"])
ap.add_argument("--boundary", default="B", choices=["A", "B"])   # A=carry, B=reset (ignored for w/a/d)
ap.add_argument("--window", type=int, default=64)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--nP", type=int, default=8)
ap.add_argument("--ctx", type=int, default=256)          # IID training window length (w/d)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--n_embd", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)       # IID steps; defines the token budget
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--inst_chunks", type=int, default=64)   # chunks per streaming instance (inst len = inst_chunks*W)
ap.add_argument("--stream_steps", type=int, default=0)   # 0 -> auto: match IID token budget
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bins", type=int, default=30)
ap.add_argument("--binsize", type=int, default=1000)
ap.add_argument("--nseg", type=int, default=48)
ap.add_argument("--test_chars", type=int, default=1_200_000)
a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed)
S, W, L = a.sink, a.window, a.ctx
C = W                                                    # streaming chunk length == window
tag = "%s_%s" % (a.mask, a.boundary)

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

# ------------------------------------------------------------------ RoPE (explicit positions)
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

# trainable prompt/registers for g (warmup) / h (persistent)
prompt = None; Pn = 0
if a.mask == "g":   Pn = max(1, W - 1)
elif a.mask == "h": Pn = a.nP
if Pn:
    prompt = nn.Parameter(torch.randn(Pn, a.n_embd, device=dev) * 0.02)
params = list(model.parameters()) + ([prompt] if prompt is not None else [])
opt = torch.optim.AdamW(params, lr=a.lr)

# ------------------------------------------------------------------ helpers
def causal_mask(T, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~(k <= q), float("-inf")); return m[None, None]
def slide_mask(T, W, dev):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    allowed = (k <= q) & (k > q - W)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]
def tmask(kind, T, W, S, dev):                          # IID literature masks (w=windowed, d=longformer)
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    if kind == "windowed":     allowed = causal & (k > q - W)
    elif kind == "longformer": allowed = (torch.abs(k - q) <= W // 2)
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~allowed, float("-inf")); return m[None, None]

def compute_kv_from_emb(emb_in, detach):
    # emb_in: [B,P,Cd] -> run through stack (causal self-attention), return per-layer pre-RoPE (k,v).
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

# ------------------------------------------------------------------ STREAMING forward (truncated-BPTT, per-layer pre-RoPE KV cache)
def stream_step(ids, rolling, prefix_kv, realmask, growing):
    """ids:[B,C] one chunk. rolling: per-layer (k,v) pre-RoPE detached or None (sliding/full cache).
       prefix_kv: per-layer (k,v) ALWAYS-allowed prefix (sink/registers) or None.
       realmask: 'window' or 'causal' (over the [rolling;chunk] real block). growing: unbounded cache (a).
       Returns logits [B,C,V] and NEW rolling cache (detached)."""
    x = model.emb(ids); B, Cl = ids.shape; new_rolling = []
    for li, blk in enumerate(model.blocks):
        h = blk.H; Cd = x.shape[2]; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, Cl, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # [B,h,Cl,d] pre-RoPE
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
        jp = col - Pfx                                   # real-block index (valid where col>=Pfx)
        rpos = Wc + t                                    # query's real-timeline index
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

def draw_instances(ids, B, Linst):
    span = Linst + 1
    ix = torch.randint(0, len(ids) - span, (B,))
    blk = torch.stack([ids[i:i + span] for i in ix]).to(dev)
    return blk[:, :Linst], blk[:, 1:Linst + 1]

# ------------------------------------------------------------------ TRAIN
budget_tokens = a.steps * a.batch * a.ctx
print("BUDGET mask=%s boundary=%s tokens=%d V=%d W=%d Pn=%d" % (a.mask, a.boundary, budget_tokens, V, W, Pn), flush=True)
model.train()
if a.mask in ("w", "d"):                                  # IID baselines (windowed / longformer)
    kind = "windowed" if a.mask == "w" else "longformer"
    tm = tmask(kind, L, W, S, dev)
    def get_batch(ids, bs, T):
        ix = torch.randint(0, len(ids) - T - 1, (bs,))
        x = torch.stack([ids[i:i + T] for i in ix]); y = torch.stack([ids[i + 1:i + T + 1] for i in ix])
        return x.to(dev), y.to(dev)
    for step in range(a.steps):
        x, y = get_batch(train_ids, a.batch, L)
        lg = model(x, tm)
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 500 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)
    used_tokens = a.steps * a.batch * L
else:                                                     # streaming masks a,b,c,e,f,g,h
    growing = (a.mask == "a")
    realmask = "causal" if a.mask in ("a", "e") else "window"
    prefix_kind = "sink" if a.mask == "c" else ("prompt" if a.mask in ("g", "h") else "none")
    prompt_when = "always" if a.mask == "h" else "cold"   # g: cold only
    reset_boundary = (a.boundary == "B") or (a.mask == "a")
    Linst = a.inst_chunks * C
    s_steps = a.stream_steps or (budget_tokens // (a.batch * C))
    rolling = None; sink_kv = None; xi = yi = None; ptr = a.inst_chunks
    for step in range(s_steps):
        if ptr >= a.inst_chunks:                          # instance boundary
            xi, yi = draw_instances(train_ids, a.batch, Linst); ptr = 0
            if reset_boundary:
                rolling = None; sink_kv = None            # B / a: cold-start each instance
            # A: keep rolling (carry); keep sink_kv (global sink)
        cx = xi[:, ptr * C:(ptr + 1) * C]; cy = yi[:, ptr * C:(ptr + 1) * C]
        prefix_kv = None
        if prefix_kind == "sink":
            if rolling is None:                           # cold start: capture sink from first S real tokens
                sink_kv = compute_kv_from_emb(model.emb(cx[:, :S]), detach=True)
                prefix_kv = None                          # chunk0 sees the sink as part of its own window
            else:
                prefix_kv = sink_kv
        elif prefix_kind == "prompt":
            if prompt_when == "always" or rolling is None:
                prefix_kv = compute_kv_from_emb(prompt[None], detach=False)
        lg, rolling = stream_step(cx, rolling, prefix_kv, realmask, growing)
        loss = F.cross_entropy(lg.reshape(-1, V), cy.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        ptr += 1
        if step % 1000 == 0: print("  step %d/%d loss %.3f" % (step, s_steps, loss.item()), flush=True)
    used_tokens = s_steps * a.batch * C
    print("STREAM_STEPS mask=%s boundary=%s s_steps=%d Linst=%d realmask=%s prefix=%s growing=%s"
          % (a.mask, a.boundary, s_steps, Linst, realmask, prefix_kind, growing), flush=True)
model.eval()

# ------------------------------------------------------------------ EVAL (averaged deep-position streaming, plain deploy masks)
def seg_starts(bins, binsize, N):
    Twin = W + binsize
    span = (bins - 1) * binsize + Twin + 1
    maxstart = test_ids.numel() - span
    assert maxstart >= 0, "test stream too short (%d<%d)" % (test_ids.numel(), span)
    if N <= 1: return [0]
    return [round(k * maxstart / (N - 1)) for k in range(N)]

@torch.no_grad()
def windowed_curve(bins, binsize, N):                    # sliding@W deploy; full 30-bin Fig-4 curve
    means, sems = [], []
    Twin = W + binsize; mask = slide_mask(Twin, W, dev); starts = seg_starts(bins, binsize, N)
    for i in range(bins):
        offs = [st + i * binsize for st in starts]
        inp = torch.stack([test_ids[o:o + Twin] for o in offs])
        tgt = torch.stack([test_ids[o + 1:o + Twin + 1] for o in offs])
        lg = model(inp, mask)
        nll = F.cross_entropy(lg[:, -binsize:].reshape(-1, V), tgt[:, -binsize:].reshape(-1),
                              reduction="none").view(len(offs), binsize).mean(1)
        ppls = torch.exp(nll).tolist()
        means.append(sum(ppls) / len(ppls))
        sems.append(pstdev(ppls) / math.sqrt(len(ppls)) if len(ppls) > 1 else 0.0)
    return means, sems

@torch.no_grad()
def deep_full(bins, binsize, N):                         # full-causal deploy at the deepest bin
    Twin = W + binsize; mask = causal_mask(Twin, dev); starts = seg_starts(bins, binsize, N)
    offs = [st + (bins - 1) * binsize for st in starts]
    inp = torch.stack([test_ids[o:o + Twin] for o in offs])
    tgt = torch.stack([test_ids[o + 1:o + Twin + 1] for o in offs])
    lg = model(inp, mask)
    nll = F.cross_entropy(lg[:, -binsize:].reshape(-1, V), tgt[:, -binsize:].reshape(-1),
                          reduction="none").view(len(offs), binsize).mean(1)
    ppls = torch.exp(nll).tolist()
    return sum(ppls) / len(ppls), (pstdev(ppls) / math.sqrt(len(ppls)) if len(ppls) > 1 else 0.0)

@torch.no_grad()
def deep_streaming(bins, binsize, N, prefix_kv=None, Sp=None):
    # StreamingLLM deploy at the deepest bin: prefix (sink/registers) at 0..Pfx-1 (always attended,
    # position-remapped) + last-W sliding window. prefix_kv None -> capture S real sink tokens.
    Twin = W + binsize; starts = seg_starts(bins, binsize, N)
    if prefix_kv is None:
        prefix_kv = compute_kv_from_emb(model.emb(test_ids[:S][None]), detach=True); Sp = S
    pos = torch.arange(Twin, device=dev)
    qpos_pref = torch.full((Twin,), float(Sp + W - 1), device=dev)
    ppos = torch.arange(Sp, device=dev)
    win_mask = slide_mask(Twin, W, dev)[0, 0]
    offs = [st + (bins - 1) * binsize for st in starts]
    ids = torch.stack([test_ids[o:o + Twin] for o in offs])
    tgt = torch.stack([test_ids[o + 1:o + Twin + 1] for o in offs])
    x = model.emb(ids)
    for b, blk in enumerate(model.blocks):
        B, _, Cd = x.shape; h = blk.H; d = Cd // h
        qkv = blk.qkv(blk.ln1(x)).view(B, Twin, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        sk, sv = prefix_kv[b]
        if sk.shape[0] == 1 and B > 1: sk = sk.expand(B, -1, -1, -1); sv = sv.expand(B, -1, -1, -1)
        lw = (rope_pos(q, pos) @ rope_pos(k, pos).transpose(-2, -1)) / math.sqrt(d) + win_mask
        ls = (rope_pos(q, qpos_pref) @ rope_pos(sk, ppos).transpose(-2, -1)) / math.sqrt(d)
        p = torch.cat([ls, lw], dim=-1).softmax(-1)
        ps, pw = p[..., :Sp], p[..., Sp:]
        o = (ps @ sv + pw @ v).transpose(1, 2).reshape(B, Twin, Cd)
        x = x + blk.proj(o); x = x + blk.mlp(blk.ln2(x))
    lg = model.head(model.lnf(x))
    nll = F.cross_entropy(lg[:, -binsize:].reshape(-1, V), tgt[:, -binsize:].reshape(-1),
                          reduction="none").view(len(offs), binsize).mean(1)
    ppls = torch.exp(nll).tolist()
    return sum(ppls) / len(ppls), (pstdev(ppls) / math.sqrt(len(ppls)) if len(ppls) > 1 else 0.0)

def fmt(v): return "[" + ", ".join("%.4f" % x for x in v) + "]"
win_m, win_s = windowed_curve(a.bins, a.binsize, a.nseg)
deep_ppl = win_m[-1]; deep_sem = win_s[-1]
dfull, dfull_s = deep_full(a.bins, a.binsize, a.nseg)
dstream, dstream_s = deep_streaming(a.bins, a.binsize, a.nseg)
print("STREAM %s windowed = %s" % (tag, fmt(win_m)), flush=True)
print("STREAM %s windowed_sem = %s" % (tag, fmt(win_s)), flush=True)
print("RESULT mask=%s boundary=%s V=%d W=%d used_tokens=%d | deep_ppl %.4f (sem %.4f) | deploy_full %.4f | deploy_sliding %.4f | deploy_streaming %.4f | stream_mean %.4f min %.4f max %.4f"
      % (a.mask, a.boundary, V, W, used_tokens, deep_ppl, deep_sem, dfull, deep_ppl, dstream,
         sum(win_m) / len(win_m), min(win_m), max(win_m)), flush=True)
if a.mask == "h":                                         # extra: deploy with persistent registers as the StreamingLLM prefix
    reg_kv = compute_kv_from_emb(prompt[None], detach=True)
    dreg, dreg_s = deep_streaming(a.bins, a.binsize, a.nseg, prefix_kv=reg_kv, Sp=Pn)
    print("RESULT_EXTRA mask=h boundary=%s | deploy_streaming_regs %.4f (sem %.4f)" % (a.boundary, dreg, dreg_s), flush=True)
print("STREAMTRAIN_DONE mask=%s boundary=%s" % (a.mask, a.boundary), flush=True)
