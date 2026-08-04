"""CPT a pretrained full-causal LLM (Llama-3.2-3B-Instruct) under each attention mask, then deploy under
full / sliding@W / StreamingLLM(S4+W) (continuous stream) and report avg + deep ppl. Real-LLM replication
of the toy methodology (experiments/09_litmask/toy_all8.py). Full fine-tune, bf16 AdamW, grad checkpointing.

MASKS (7, no Longformer):
  IID (4D additive mask over L=2048, one fwd/step, loss on all positions):
    a full causal     b Gemma/SWAT sliding-W     c SWAA (sink S + sliding-W)
  STREAMING (carried DETACHED KV-cache truncated-BPTT, chunk C=W; instance = inst_chunks chunks = L tokens;
  A=carry cache across instances, B=reset per instance). Position_ids grow absolutely -> RoPE is RELATIVE so
  windowed attention (rel dist <= W) is in-distribution regardless of absolute magnitude:
    e Transformer-XL   : cache=prev chunk (W), full-causal over [cache ; chunk] (block, 1-seg memory).
    f sliding-history  : cache=last-W, sliding-window over [cache ; chunk].
    g sliding+warmup   : f + a trainable warmup prompt prepended ONLY at the cold-start chunk (cache empty).
    h sliding+persistent: f + nP trainable registers prepended EVERY chunk (always attended).

DEPLOY EVAL (continuous, single fwd over eval_len = warm + bins*binsize + 1, capped for the O(n^2) full):
per deploy mask, per-token NLL, skip first `warm` query positions, bin -> ppl/bin; avg=mean bins, deep=last.
"""
import torch, argparse, math, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True, choices=["a", "b", "c", "e", "f", "g", "h", "g_iid", "h_iid", "j_iid", "a_scal", "b_scal", "n_iid", "o_iid", "x2a", "x2b", "f_scal", "f_rtok", "f_rpref", "f_x2"])   # f_* = t-BPTT sliding-history + each sink design   # x2a/x2b = a/b + TWO per-layer K/V sinks with --split geometry   # n_iid/o_iid = b + RIDING sink token / prefix (constant offset W..W+nP-1 behind every query)   # j_iid = a + sink token (IID full-causal + always-on prefix); a_scal/b_scal = sink SCALAR (per-head softmax-denominator logit)
ap.add_argument("--boundary", default="B", choices=["A", "B"])   # streaming only
ap.add_argument("--window", type=int, default=512)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--nP", type=int, default=8)
ap.add_argument("--ctx", type=int, default=2048)         # instance length (IID seq len; streaming inst = ctx tokens)
ap.add_argument("--steps", type=int, default=120)        # optimizer steps (1 instance/step)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--n_seq", type=int, default=400)
ap.add_argument("--n_eval", type=int, default=8)
ap.add_argument("--warm", type=int, default=512)
ap.add_argument("--bins", type=int, default=4)
ap.add_argument("--binsize", type=int, default=512)
ap.add_argument("--pos_cap", type=int, default=120000)   # A: soft-reset cache+pos when abs RoPE pos exceeds this
ap.add_argument("--base", action="store_true")           # skip CPT, analyze the base model
ap.add_argument("--sink_npz", default="")                # if set: attention-sink analysis (full attn) + save npz, skip deploy eval
ap.add_argument("--sink_len", type=int, default=1536)    # fixed-sample length for sink analysis (>W so token-0 evicts under sliding)
ap.add_argument("--save_ckpt", default="")               # dir to save the CPT'd model (+prompt) for reuse
ap.add_argument("--load_ckpt", default="")               # dir to load a CPT'd model (+prompt) from, skip CPT
ap.add_argument("--deploy_stream", action="store_true")  # CONTINUOUS streaming deploy eval (rolling cache to ~30k)
ap.add_argument("--eval_bins", type=int, default=30)
ap.add_argument("--eval_binsize", type=int, default=1000)
ap.add_argument("--eval_warm", type=int, default=512)
ap.add_argument("--full_bins", type=int, default=15)     # full deploy capped (O(n^2)): warm + full_bins*full_binsize
ap.add_argument("--full_binsize", type=int, default=512)
ap.add_argument("--split", default="1p1r", choices=["2p0", "2ride", "1p1r"])   # x2 masks: sink-slot geometry split
ap.add_argument("--hotpot", action="store_true")          # zero-shot HotpotQA eval (F1/EM/ans-ppl) under full + streaming deploys
ap.add_argument("--hp_n", type=int, default=150)
ap.add_argument("--hp_maxnew", type=int, default=12)
ap.add_argument("--hp_ctx", type=int, default=2048)
ap.add_argument("--own_only", action="store_true")       # deploy_stream: ONLY the full-context-with-own-sink eval
ap.add_argument("--perlayer_reg", action="store_true")   # h/h_iid: ADD per-layer trainable raw K/V prefix (prefix-tuning) on top of the input prompt
ap.add_argument("--tiny", action="store_true")           # tiny LlamaConfig for fast mechanism testing
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
S, W, L = a.sink, a.window, a.ctx
C = W
STREAM = a.mask in ("e", "f", "g", "h", "f_scal", "f_rtok", "f_rpref", "f_x2")
PREFIX_IID = a.mask in ("g_iid", "h_iid", "j_iid")
SCAL = a.mask in ("a_scal", "b_scal", "f_scal")
RIDING = a.mask in ("n_iid", "o_iid", "f_rtok", "f_rpref")
RIDE2 = a.mask in ("x2a", "x2b", "f_x2")               # IID single-forward with a prepended trainable prefix (no carried cache)
INST_CHUNKS = L // C
tag = ("%s_%s" % (a.mask, a.boundary)) if STREAM else (("%s_%s" % (a.mask, a.split)) if a.mask in ("x2a", "x2b") else a.mask)
NINF = float("-inf")

# ------------------------------------------------------------------ model + data
print("loading %s (tiny=%s) ..." % (MODEL, a.tiny), flush=True)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
if a.tiny:
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=8192,
                      attn_implementation="eager")
    model = LlamaForCausalLM(cfg).to(dev).to(bf16)
elif a.load_ckpt:
    model = AutoModelForCausalLM.from_pretrained(a.load_ckpt, dtype=bf16, attn_implementation="eager").to(dev); a.base = True
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)
H = model.config.hidden_size
def pack(split, n, Lp):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= Lp:
            out.append(buf[:Lp]); buf = buf[Lp:]
            if len(out) >= n: return out
    return out

# ------------------------------------------------------------------ masks
def add4d(allowed, dev, dt):
    Lq, Lk = allowed.shape; m = torch.zeros(Lq, Lk, device=dev, dtype=dt); m.masked_fill_(~allowed, NINF); return m[None, None]
def iid_mask(kind, L, W, S, dev, dt):
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]; causal = k <= q
    if kind == "a":   al = causal
    elif kind == "b": al = causal & (k > q - W)
    elif kind == "c": al = causal & ((k < S) | (k > q - W))
    return add4d(al, dev, dt)
def iid_prefix_mask(kind, T, Pn, W, dev, dt):
    # combined [prefix(Pn) ; real(L)] = T. h_iid: real queries attend prefix(always) + sliding-W over real.
    # g_iid: plain sliding-W over the combined (warmup prefix falls inside early positions' windows). loss on real only.
    r = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    is_pref_row = r < Pn
    if kind == "h_iid":   real_ok = (k < Pn) | ((k >= Pn) & (k <= r) & (k > r - W))
    elif kind == "j_iid": real_ok = (k < Pn) | ((k >= Pn) & (k <= r))
    else:                 real_ok = (k <= r) & (k > r - W)
    pref_ok = (k <= r) & (k < Pn)
    al = torch.where(is_pref_row, pref_ok, real_ok)
    return add4d(al, dev, dt)
def deploy_mask(kind, L, W, S, dev, dt):
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]; causal = k <= q
    if kind == "full":        al = causal
    elif kind == "sliding":   al = causal & (k > q - W)
    elif kind == "streaming": al = causal & ((k < S) | (k > q - W))
    return add4d(al, dev, dt)
def stream_mask(T, Wc, Pn, C, realmask, persistent, dev, dt):
    # rows = T = Pn(prefix)+C(chunk) queries; cols = Wc(cache)+Pn(prefix)+C(chunk) keys
    Lk = Wc + T; r = torch.arange(T, device=dev)[:, None]; c = torch.arange(Lk, device=dev)[None, :]
    is_pref_row = r < Pn
    is_pref_col = (c >= Wc) & (c < Wc + Pn)
    real_idx = torch.where(c < Wc, c, c - Pn)            # cache 0..Wc-1 ; chunk Wc..Wc+C-1
    rq = Wc + (r - Pn)                                   # chunk query real index
    if realmask == "sliding": real_ok = (real_idx <= rq) & (real_idx > rq - W)
    else:                     real_ok = (real_idx <= rq)
    chunk_row_ok = torch.where(is_pref_col, torch.full_like(real_ok, persistent or Pn > 0), real_ok)
    pref_row_ok = is_pref_col & ((c - Wc) <= r)          # prefix rows: causal among prefix
    allowed = torch.where(is_pref_row, pref_row_ok, chunk_row_ok)
    return add4d(allowed, dev, dt)

def build_real_cache(pkv, Wc, Pn, W, grow=False):
    data = []
    for Lr in pkv.layers:
        k, v = Lr.keys, Lr.values
        if Pn:
            k = torch.cat([k[:, :, :Wc, :], k[:, :, Wc + Pn:, :]], dim=2)
            v = torch.cat([v[:, :, :Wc, :], v[:, :, Wc + Pn:, :]], dim=2)
        if grow: data.append((k.detach(), v.detach()))                      # keep ALL history (growing full cache)
        else:    data.append((k[:, :, -W:, :].detach(), v[:, :, -W:, :].detach()))
    return DynamicCache(ddp_cache_data=data)

def stream_chunk(chunk_ids, cache, pos, realmask, prefix_emb, persistent, grow=False):
    emb = model.get_input_embeddings()
    Wc = 0 if cache is None else cache.layers[0].keys.shape[2]
    cemb = emb(chunk_ids); Pn = 0; inp = cemb
    if prefix_emb is not None:
        Pn = prefix_emb.shape[0]; inp = torch.cat([prefix_emb[None].to(cemb.dtype), cemb], dim=1)
    T = inp.shape[1]
    position_ids = torch.arange(pos, pos + T, device=dev)[None]
    m4 = stream_mask(T, Wc, Pn, C, realmask, persistent, dev, inp.dtype)
    out = model(inputs_embeds=inp, past_key_values=cache, position_ids=position_ids, attention_mask=m4, use_cache=True)
    logits = out.logits[0, Pn:Pn + C]
    newc = build_real_cache(out.past_key_values, Wc, Pn, W, grow=grow)
    return logits, newc, pos + T

# ------------------------------------------------------------------ per-layer K/V prefix (prefix-tuning) for h / h_iid
def build_cache_pl(rolling, reg_k, reg_v):
    # cache = [per-layer raw trainable K/V prefix (nPL, NOT detached, NO RoPE) ; rolling-W (detached)]
    data = []
    for l in range(reg_k.shape[0]):
        pk = reg_k[l].permute(1, 0, 2)[None].to(bf16)   # [nP,nkv,hd] -> [1,nkv,nP,hd]
        pv = reg_v[l].permute(1, 0, 2)[None].to(bf16)
        if rolling is not None:
            k = torch.cat([pk, rolling.layers[l].keys], 2); v = torch.cat([pv, rolling.layers[l].values], 2)
        else: k, v = pk, pv
        data.append((k, v))
    return DynamicCache(ddp_cache_data=data)
def stream_mask_pl(T, nPL, Wc, Pn, C, realmask, dev, dt):
    # cols: [perlayer(nPL) ; rolling(Wc) ; input_prompt(Pn) ; chunk(C)] ; rows: [input_prompt(Pn) ; chunk(C)]
    Lk = nPL + Wc + T; r = torch.arange(T, device=dev)[:, None]; c = torch.arange(Lk, device=dev)[None, :]
    is_pl = c < nPL; c2 = c - nPL
    is_prompt_col = (c2 >= Wc) & (c2 < Wc + Pn)
    real_idx = torch.where(c2 < Wc, c2, c2 - Pn)        # rolling 0..Wc-1 ; chunk Wc..Wc+C-1
    is_pref_row = r < Pn; rq = Wc + (r - Pn)
    real_ok = (real_idx <= rq) & (real_idx > rq - W) if realmask == "sliding" else (real_idx <= rq)
    chunk_allowed = is_pl | is_prompt_col | (~is_pl & ~is_prompt_col & real_ok)
    prompt_allowed = is_pl | (is_prompt_col & ((c2 - Wc) <= r))
    allowed = torch.where(is_pref_row, prompt_allowed, chunk_allowed)
    return add4d(allowed, dev, dt)
def build_real_cache_pl(pkv, nPL, Wc, Pn, W, grow=False):
    # post-fwd layer keys = [perlayer(nPL) ; rolling(Wc) ; input_prompt(Pn) ; chunk(C)] -> keep last-W (or all if grow)
    data = []
    for Lr in pkv.layers:
        k, v = Lr.keys, Lr.values
        kr = torch.cat([k[:, :, nPL:nPL + Wc, :], k[:, :, nPL + Wc + Pn:, :]], 2)
        vr = torch.cat([v[:, :, nPL:nPL + Wc, :], v[:, :, nPL + Wc + Pn:, :]], 2)
        if grow: data.append((kr.detach(), vr.detach()))
        else:    data.append((kr[:, :, -W:, :].detach(), vr[:, :, -W:, :].detach()))
    return DynamicCache(ddp_cache_data=data)
def stream_chunk_pl(chunk_ids, rolling, pos, prompt_emb, reg_k, reg_v, realmask, chunk_len=None, ret_cache=True, grow=False):
    emb = model.get_input_embeddings()
    Wc = 0 if rolling is None else rolling.layers[0].keys.shape[2]
    nPL = reg_k.shape[1]; Pn = prompt_emb.shape[0]; Cl = chunk_len or C
    inp = torch.cat([prompt_emb[None].to(bf16), emb(chunk_ids)], 1)        # [1, Pn+Cl, H]
    T = inp.shape[1]
    cache = build_cache_pl(rolling, reg_k, reg_v)
    position_ids = torch.arange(pos, pos + T, device=dev)[None]
    cache_position = torch.arange(nPL + Wc, nPL + Wc + T, device=dev)
    m4 = stream_mask_pl(T, nPL, Wc, Pn, Cl, realmask, dev, inp.dtype)
    out = model(inputs_embeds=inp, past_key_values=cache, position_ids=position_ids,
                cache_position=cache_position, attention_mask=m4, use_cache=ret_cache)
    logits = out.logits[0, Pn:Pn + Cl]
    newc = build_real_cache_pl(out.past_key_values, nPL, Wc, Pn, W, grow=grow) if ret_cache else None
    return logits, newc, pos + T

# ------------------------------------------------------------------ CPT
RIDE_ON = {"on": False}
RIDE_KV = {}
if RIDING:
    import transformers.models.llama.modeling_llama as _ml2
    from transformers.models.llama.modeling_llama import repeat_kv as _repkv2, rotate_half as _rh
    nLr = model.config.num_hidden_layers; nkvr = model.config.num_key_value_heads
    hdr_ = model.config.hidden_size // model.config.num_attention_heads
    RIDE_D = (W + torch.arange(a.nP, device=dev).flip(0)).float()      # reg j at offset W + (nP-1-j)
    _dmy = torch.zeros(1, a.nP, hdr_, device=dev, dtype=bf16)
    with torch.no_grad():
        COS_D, SIN_D = model.model.rotary_emb(_dmy, RIDE_D[None].long())   # [1, nP, hd] at the offset angles
    def set_ride_kv_o():
        for li in range(nLr):
            RIDE_KV[li] = (reg_k[li].permute(1, 0, 2)[None], reg_v[li].permute(1, 0, 2)[None])   # [1,nkv,nP,hd] pre-RoPE
    def set_ride_kv_n(grad):
        RIDE_ON["on"] = False
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            pos = torch.arange(a.nP, device=dev)
            out = model(inputs_embeds=prompt[None], position_ids=pos[None], use_cache=True,
                        past_key_values=DynamicCache())
            cosP, sinP = model.model.rotary_emb(_dmy, pos[None])
            for li, Lr in enumerate(out.past_key_values.layers):
                kc = Lr.keys                                            # [1,nkv,nP,hd] POST-RoPE at 0..nP-1
                kp = kc * cosP.unsqueeze(1) - _rh(kc) * sinP.unsqueeze(1)   # un-rotate -> pre-RoPE
                RIDE_KV[li] = (kp, Lr.values)
        RIDE_ON["on"] = True
    _orig_attn_fwd = _ml2.LlamaAttention.forward
    def _ride_fwd(self, hidden_states, position_embeddings, attention_mask=None, past_key_value=None, cache_position=None, **kw):
        if not RIDE_ON["on"]:
            return _orig_attn_fwd(self, hidden_states, position_embeddings, attention_mask=attention_mask,
                                  past_key_value=past_key_value, cache_position=cache_position, **kw)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _arp
        Bq, Tq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(Bq, Tq, -1, hdr_).transpose(1, 2)
        k = self.k_proj(hidden_states).view(Bq, Tq, -1, hdr_).transpose(1, 2)
        v = self.v_proj(hidden_states).view(Bq, Tq, -1, hdr_).transpose(1, 2)
        cos, sin = position_embeddings
        qr, kr = _arp(q, k, cos, sin)
        _pkv = past_key_value if past_key_value is not None else kw.get("past_key_values", None)
        if _pkv is not None:
            kr, v = _pkv.update(kr, v, self.layer_idx, {"cache_position": cache_position})
        ks = _repkv2(kr, self.num_key_value_groups); vs = _repkv2(v, self.num_key_value_groups)
        sc_ = getattr(self, "scaling", hdr_ ** -0.5)
        aw = torch.matmul(qr, ks.transpose(2, 3)) * sc_
        if attention_mask is not None: aw = aw + attention_mask[:, :, :, :ks.shape[-2]]
        qp = qr * cos.unsqueeze(1) - _rh(qr) * sin.unsqueeze(1)          # un-rotated queries
        pk, pv = RIDE_KV[self.layer_idx]
        krd = pk * COS_D.unsqueeze(1) - _rh(pk) * SIN_D.unsqueeze(1)     # reg keys at R(-D)
        krg = _repkv2(krd.expand(Bq, -1, -1, -1), self.num_key_value_groups)
        lr = torch.matmul(qp, krg.transpose(2, 3)) * sc_
        aw = torch.cat([lr, aw], dim=-1)
        aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(qr.dtype)
        nPr = pk.shape[2]
        vrg = _repkv2(pv.expand(Bq, -1, -1, -1), self.num_key_value_groups)
        out = torch.matmul(aw[..., :nPr], vrg) + torch.matmul(aw[..., nPr:], vs)
        out = out.transpose(1, 2).reshape(Bq, Tq, -1)
        return self.o_proj(out), None
    _ml2.LlamaAttention.forward = _ride_fwd
    RIDE_ON["on"] = True
    print("RIDING_PATCHED mask=%s offsets=%s..%s" % (a.mask, int(RIDE_D.min()), int(RIDE_D.max())), flush=True)
if RIDE2:
    import transformers.models.llama.modeling_llama as _ml3
    from transformers.models.llama.modeling_llama import repeat_kv as _repkv3, rotate_half as _rh3, apply_rotary_pos_emb as _arp3
    nLr = model.config.num_hidden_layers; nkvr = model.config.num_key_value_heads
    hdr_ = model.config.hidden_size // model.config.num_attention_heads
    SPLIT = {"2p0": (2, 0), "2ride": (0, 2), "1p1r": (1, 1)}[a.split]
    NF, NR = SPLIT
    _dm2 = torch.zeros(1, max(NF, NR, 1), hdr_, device=dev, dtype=bf16)
    with torch.no_grad():
        if NF: COS_F, SIN_F = model.model.rotary_emb(_dm2[:, :NF], torch.arange(NF, device=dev)[None])          # front slots at abs 0..NF-1
        if NR: COS_R, SIN_R = model.model.rotary_emb(_dm2[:, :NR], (W + torch.arange(NR, device=dev))[None])    # riding slots at offsets W..W+NR-1
    def set_x2_kv():
        for li in range(nLr):
            RIDE_KV[li] = (reg_k[li].permute(1, 0, 2)[None], reg_v[li].permute(1, 0, 2)[None])   # [1,nkv,2,hd] pre-RoPE
    _orig_attn_fwd2 = _ml3.LlamaAttention.forward
    def _x2_fwd(self, hidden_states, position_embeddings, attention_mask=None, past_key_value=None, cache_position=None, **kw):
        if not RIDE_ON["on"]:
            return _orig_attn_fwd2(self, hidden_states, position_embeddings, attention_mask=attention_mask,
                                   past_key_value=past_key_value, cache_position=cache_position, **kw)
        Bq, Tq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(Bq, Tq, -1, hdr_).transpose(1, 2)
        k = self.k_proj(hidden_states).view(Bq, Tq, -1, hdr_).transpose(1, 2)
        v = self.v_proj(hidden_states).view(Bq, Tq, -1, hdr_).transpose(1, 2)
        cos, sin = position_embeddings
        qr, kr = _arp3(q, k, cos, sin)
        _pkv = past_key_value if past_key_value is not None else kw.get("past_key_values", None)
        if _pkv is not None:
            kr, v = _pkv.update(kr, v, self.layer_idx, {"cache_position": cache_position})
        ks = _repkv3(kr, self.num_key_value_groups); vs = _repkv3(v, self.num_key_value_groups)
        sc_ = getattr(self, "scaling", hdr_ ** -0.5)
        aw = torch.matmul(qr, ks.transpose(2, 3)) * sc_
        if attention_mask is not None: aw = aw + attention_mask[:, :, :, :ks.shape[-2]]
        pk, pv = RIDE_KV[self.layer_idx]                                     # [1,nkv,2,hd]: [front | ride] slots
        blocks = []
        if NF:
            kf = pk[:, :, :NF] * COS_F.unsqueeze(1) + _rh3(pk[:, :, :NF]) * SIN_F.unsqueeze(1)     # R(+pos): true pos-0 geometry
            blocks.append(torch.matmul(qr, _repkv3(kf.expand(Bq, -1, -1, -1), self.num_key_value_groups).transpose(2, 3)) * sc_)
        if NR:
            qp = qr * cos.unsqueeze(1) - _rh3(qr) * sin.unsqueeze(1)                                # un-rotated queries
            krd = pk[:, :, NF:] * COS_R.unsqueeze(1) - _rh3(pk[:, :, NF:]) * SIN_R.unsqueeze(1)     # R(-D): riding geometry
            blocks.append(torch.matmul(qp, _repkv3(krd.expand(Bq, -1, -1, -1), self.num_key_value_groups).transpose(2, 3)) * sc_)
        aw = torch.cat(blocks + [aw], dim=-1)
        aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(qr.dtype)
        vrg = _repkv3(pv.expand(Bq, -1, -1, -1), self.num_key_value_groups)
        out = torch.matmul(aw[..., :2], vrg) + torch.matmul(aw[..., 2:], vs)
        out = out.transpose(1, 2).reshape(Bq, Tq, -1)
        return self.o_proj(out), None
    _ml3.LlamaAttention.forward = _x2_fwd
    RIDE_ON["on"] = True
    print("X2_PATCHED mask=%s split=%s (front=%d ride=%d)" % (a.mask, a.split, NF, NR), flush=True)
USE_CACHE = STREAM or (a.perlayer_reg and a.mask in ("h", "h_iid", "j_iid")) or RIDING or RIDE2   # cache injection needs use_cache=True (HF ignores input cache under grad-ckpt)
if USE_CACHE:
    model.config.use_cache = True                        # h_iid per-layer: single 2k-token fwd, ~48GB (fits A6000); h streaming chunks ~32GB
else:
    model.gradient_checkpointing_enable(); model.config.use_cache = False
prompt = None
if a.mask in ("g", "g_iid"): prompt = torch.nn.Parameter(torch.randn(max(1, W - 1), H, device=dev, dtype=bf16) * 0.02)
elif a.mask in ("h", "h_iid", "j_iid"): prompt = torch.nn.Parameter(torch.randn(a.nP, H, device=dev, dtype=bf16) * 0.02)
SINK_SC = None
if SCAL:                                                   # per-(layer,head) learned softmax-denominator logit; no K/V/value
    from transformers.models.llama import modeling_llama as _ml
    from transformers.models.llama.modeling_llama import repeat_kv as _repkv
    SINK_SC = torch.nn.Parameter(torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads, device=dev, dtype=bf16))
    _orig_eager = _ml.eager_attention_forward
    def _eager_sink(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        sc = getattr(module, "sink_scalar", None)
        if sc is None: return _orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
        ks = _repkv(key, module.num_key_value_groups); vs = _repkv(value, module.num_key_value_groups)
        aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None: aw = aw + attention_mask[:, :, :, :ks.shape[-2]]
        Bq, Hq, Tq, Kq = aw.shape
        pad = sc.view(1, -1, 1, 1).expand(Bq, -1, Tq, 1).to(aw.dtype)
        aw = torch.cat([aw, pad], dim=-1)
        aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)[..., :Kq]
        aw = torch.nn.functional.dropout(aw, p=dropout, training=module.training)
        out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
        return out, aw
    _ml.eager_attention_forward = _eager_sink
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = _eager_sink
    except Exception as _e:
        print("ATTN_REGISTRY_PATCH_SKIP", repr(_e), flush=True)
    for _li, _lyr in enumerate(model.model.layers):
        _lyr.self_attn.sink_scalar = SINK_SC[_li]
    print("SINK_SCALAR_PATCHED layers=%d heads=%d" % tuple(SINK_SC.shape), flush=True)
reg_k = reg_v = None
if a.perlayer_reg and a.mask in ("h", "h_iid", "j_iid"):           # per-layer raw trainable K/V prefix (prefix-tuning)
    nL = model.config.num_hidden_layers; nkv = model.config.num_key_value_heads; hd = model.config.hidden_size // model.config.num_attention_heads
    reg_k = torch.nn.Parameter(torch.randn(nL, a.nP, nkv, hd, device=dev, dtype=bf16) * 0.02)
    reg_v = torch.nn.Parameter(torch.randn(nL, a.nP, nkv, hd, device=dev, dtype=bf16) * 0.02)
if a.mask in ("o_iid", "f_rpref"):                                 # riding sink prefix: free per-layer pre-RoPE K/V
    reg_k = torch.nn.Parameter(torch.randn(nLr, a.nP, nkvr, hdr_, device=dev, dtype=bf16) * 0.02)
    reg_v = torch.nn.Parameter(torch.randn(nLr, a.nP, nkvr, hdr_, device=dev, dtype=bf16) * 0.02)
elif RIDE2:                                                        # x2: two per-layer K/V sink slots (geometry per --split)
    reg_k = torch.nn.Parameter(torch.randn(nLr, 2, nkvr, hdr_, device=dev, dtype=bf16) * 0.02)
    reg_v = torch.nn.Parameter(torch.randn(nLr, 2, nkvr, hdr_, device=dev, dtype=bf16) * 0.02)
elif a.mask in ("n_iid", "f_rtok"):                                # riding sink token: trainable prompt embedding
    prompt = torch.nn.Parameter(torch.randn(a.nP, H, device=dev, dtype=bf16) * 0.02)
if a.load_ckpt:
    import os
    if prompt is not None and os.path.exists(os.path.join(a.load_ckpt, "prompt.pt")):
        prompt = torch.nn.Parameter(torch.load(os.path.join(a.load_ckpt, "prompt.pt"), map_location=dev).to(bf16))
    if reg_k is not None and os.path.exists(os.path.join(a.load_ckpt, "reg_k.pt")):
        reg_k = torch.nn.Parameter(torch.load(os.path.join(a.load_ckpt, "reg_k.pt"), map_location=dev).to(bf16))
        reg_v = torch.nn.Parameter(torch.load(os.path.join(a.load_ckpt, "reg_v.pt"), map_location=dev).to(bf16))
    if RIDING:
        if a.mask in ("o_iid", "f_rpref"): set_ride_kv_o()
        else: set_ride_kv_n(grad=False)
    if RIDE2: set_x2_kv()
    if SCAL and os.path.exists(os.path.join(a.load_ckpt, "sink_sc.pt")):
        SINK_SC = torch.nn.Parameter(torch.load(os.path.join(a.load_ckpt, "sink_sc.pt"), map_location=dev).to(bf16))
        for _li, _lyr in enumerate(model.model.layers): _lyr.self_attn.sink_scalar = SINK_SC[_li]
if not a.base:
    model.train()
    params = list(model.parameters()) + ([prompt] if prompt is not None else []) + ([reg_k, reg_v] if reg_k is not None else []) + ([SINK_SC] if SCAL else [])
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95))
    seqs = pack("train", a.n_seq, L)
    print("packed %d train seqs. CPT mask=%s boundary=%s steps=%d inst_chunks=%d ..." % (len(seqs), a.mask, a.boundary, a.steps, INST_CHUNKS), flush=True)
    t0 = time.time()
    if PREFIX_IID:                                        # IID single-forward with a prepended trainable prefix (no carried cache)
        Pn = prompt.shape[0]; emb = model.get_input_embeddings()
        tm = iid_prefix_mask(a.mask, Pn + L, Pn, W, dev, bf16) if reg_k is None else None
        for step in range(a.steps):
            ids = torch.tensor(seqs[step % len(seqs)], device=dev)
            if reg_k is not None:                                                   # + per-layer K/V prefix (single fwd, no carry)
                lg, _, _ = stream_chunk_pl(ids[None], None, 0, prompt, reg_k, reg_v, "causal" if a.mask == "j_iid" else "sliding", chunk_len=L)
                loss = F.cross_entropy(lg[:L - 1].float(), ids[1:])
            else:
                inp = torch.cat([prompt[None], emb(ids[None])], dim=1)              # [1, Pn+L, H]
                lg = model(inputs_embeds=inp, attention_mask=tm).logits[0]
                loss = F.cross_entropy(lg[Pn:Pn + L - 1].float(), ids[1:])          # loss on real positions only
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            if step % 25 == 0 or step == a.steps - 1:
                print("  step %d/%d loss %.4f (%.1fs peakGB %.1f)" % (step, a.steps, loss.item(), time.time() - t0, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    elif not STREAM:
        _mk = {"a_scal": "a", "b_scal": "b", "n_iid": "b", "o_iid": "b", "x2a": "a", "x2b": "b"}.get(a.mask, a.mask)
        tm = iid_mask(_mk, L, W, S, dev, bf16)
        for step in range(a.steps):
            ids = torch.tensor([seqs[step % len(seqs)]], device=dev)
            if a.mask == "n_iid": set_ride_kv_n(grad=True)
            elif a.mask == "o_iid": set_ride_kv_o()          # rebuild views each step (fresh autograd graph)
            elif RIDE2: set_x2_kv()
            lg = model(ids, attention_mask=tm).logits[0]
            loss = F.cross_entropy(lg[:-1].float(), ids[0, 1:])
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            if step % 25 == 0 or step == a.steps - 1:
                print("  step %d/%d loss %.4f (%.1fs peakGB %.1f)" % (step, a.steps, loss.item(), time.time() - t0, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    else:
        realmask = "causal" if a.mask == "e" else "sliding"
        persistent = (a.mask == "h")
        cache = None; pos = 0
        for step in range(a.steps):
            ids = torch.tensor(seqs[step % len(seqs)], device=dev)
            if a.mask == "f_rpref": set_ride_kv_o()             # refresh riding-prefix KV views (param views, per-step ok)
            elif a.mask == "f_x2": set_x2_kv()                   # refresh 2-slot KV views
            if a.boundary == "B" or pos > a.pos_cap: cache = None; pos = 0   # B resets per instance; A soft-resets to bound RoPE pos
            opt.zero_grad(); lastloss = 0.0
            for ci in range(INST_CHUNKS):
                if a.mask == "f_rtok": set_ride_kv_n(grad=True)  # per-chunk fresh graph (prompt-derived KV; avoids double-backward)
                ch = ids[ci * C:(ci + 1) * C][None]; tg = ids[ci * C + 1:(ci + 1) * C + 1]
                if a.mask == "h" and reg_k is not None:                       # per-layer K/V prefix + input prompt
                    lg, cache, pos = stream_chunk_pl(ch, cache, pos, prompt, reg_k, reg_v, "sliding")
                else:
                    prefix_emb = None
                    if a.mask == "h": prefix_emb = prompt
                    elif a.mask == "g" and cache is None: prefix_emb = prompt
                    lg, cache, pos = stream_chunk(ch, cache, pos, realmask, prefix_emb, persistent)
                loss = F.cross_entropy(lg[:tg.shape[0]].float(), tg) / INST_CHUNKS
                loss.backward(); lastloss += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % 25 == 0 or step == a.steps - 1:
                print("  step %d/%d loss %.4f (%.1fs peakGB %.1f)" % (step, a.steps, lastloss, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9), flush=True)
model.eval()

# ------------------------------------------------------------------ save checkpoint (for reuse)
if a.save_ckpt and not a.base:
    import os
    os.makedirs(a.save_ckpt, exist_ok=True)
    model.save_pretrained(a.save_ckpt); tok.save_pretrained(a.save_ckpt)
    if prompt is not None: torch.save(prompt.detach().cpu(), os.path.join(a.save_ckpt, "prompt.pt"))
    if reg_k is not None:
        torch.save(reg_k.detach().cpu(), os.path.join(a.save_ckpt, "reg_k.pt"))
        torch.save(reg_v.detach().cpu(), os.path.join(a.save_ckpt, "reg_v.pt"))
    if SCAL: torch.save(SINK_SC.detach().cpu(), os.path.join(a.save_ckpt, "sink_sc.pt"))
    print("SAVED_CKPT %s" % a.save_ckpt, flush=True)

# ------------------------------------------------------------------ ZERO-SHOT HOTPOTQA (own-sink, full + streaming deploys)
if a.hotpot:
    import re as _re, string as _st, collections as _co, glob as _gl, os as _os
    import pyarrow as _pa
    model.gradient_checkpointing_disable(); model.config.use_cache = False
    def _norm(x):
        x = "".join(ch for ch in x.lower() if ch not in _st.punctuation)
        return " ".join(_re.sub(r"\b(a|an|the)\b", " ", x).split())
    def _f1(pred, gold):
        pr, gd = _norm(pred).split(), _norm(gold).split()
        com = sum((_co.Counter(pr) & _co.Counter(gd)).values())
        if com == 0 or not pr or not gd: return 0.0
        p_, r_ = com / len(pr), com / len(gd); return 2 * p_ * r_ / (p_ + r_)
    _cand = (_gl.glob(_os.path.expandvars("$HF_HOME/datasets/hotpotqa___hotpot_qa/distractor/*/*/hotpot_qa-validation.arrow"))
             or _gl.glob("/scratch1/zizhaoh/hf_cache/datasets/hotpotqa___hotpot_qa/distractor/*/*/hotpot_qa-validation.arrow"))
    _vf = _cand[0]
    try: _rd = _pa.ipc.open_stream(_pa.memory_map(_vf, "r"))
    except Exception: _rd = _pa.ipc.open_file(_pa.memory_map(_vf, "r"))
    hp = _rd.read_all().to_pylist()[:a.hp_n]
    EOS = tok.eos_token_id; NLid = tok.convert_tokens_to_ids("\u010a") if False else tok("\n", add_special_tokens=False).input_ids[-1]
    TOKMODEL = (prompt is not None) and not RIDING                       # prompt-prepend family (g/h/j token models)
    PLMODEL = (reg_k is not None) and not RIDING and not RIDE2           # per-layer prefix family
    Pn_ = prompt.shape[0] if TOKMODEL else 0
    HPC = a.hp_ctx
    def _build(ex):
        ctx = tok("\n".join("".join(pp) for pp in ex["context"]["sentences"]), add_special_tokens=False).input_ids
        qq = tok("\n\nAnswer the question with a short span.\nQuestion: " + ex["question"] + "\nAnswer:", add_special_tokens=False).input_ids
        ans = tok(" " + ex["answer"], add_special_tokens=False).input_ids[:a.hp_maxnew]
        return ctx, qq, ans
    @torch.no_grad()
    def _logits(ids_list, kind, W_):
        L2 = len(ids_list); x = torch.tensor([ids_list], device=dev)
        if PLMODEL:
            lg, _, _ = stream_chunk_pl(x, None, 0, prompt, reg_k, reg_v, "causal" if kind == "full" else "sliding", chunk_len=L2, ret_cache=False)
            return lg
        if TOKMODEL:
            mk = iid_prefix_mask("j_iid" if kind == "full" else "h_iid", Pn_ + L2, Pn_, W_, dev, bf16)
            inp = torch.cat([prompt[None].to(bf16), model.get_input_embeddings()(x)], 1)
            return model(inputs_embeds=inp, attention_mask=mk, use_cache=False).logits[0, Pn_:]
        dm = deploy_mask({"windowed": "sliding"}.get(kind, kind), L2, W_, S, dev, bf16)
        return model(x, attention_mask=dm, use_cache=False).logits[0]
    @torch.no_grad()
    def _hp_eval(kind, W_):
        tf1 = tem = tacc = tnll = 0.0; tk = 0; N = 0
        for ex in hp:
            ctx, qq, ans = _build(ex)
            if not ans: continue
            N += 1
            base_ids = (ctx + qq)[-(HPC - Pn_):]
            full_ids = (base_ids + ans)[-(HPC - Pn_):]; kp = len(full_ids) - len(ans)
            lg = _logits(full_ids, kind, W_)
            tgt = torch.tensor(full_ids[kp:], device=dev)
            tnll += F.cross_entropy(lg[kp - 1:len(full_ids) - 1].float(), tgt, reduction="sum").item(); tk += len(ans)
            cur = list(base_ids)
            for _ in range(a.hp_maxnew):
                body = cur[-(HPC - Pn_):]
                nl = _logits(body, kind, W_)[-1]
                nt = int(nl.argmax())
                if nt == EOS or nt == NLid: break
                cur.append(nt)
            pred = tok.decode(cur[len(base_ids):]).strip()
            gold_n = _norm(ex["answer"]); pred_n = _norm(pred)
            tf1 += _f1(pred, ex["answer"]); tem += float(pred_n == gold_n)
            tacc += float(bool(gold_n) and gold_n in pred_n)          # containment accuracy (verbosity-robust)
        print("HOTPOT model=%s deploy=%s W=%d | F1 %.3f | EM %.3f | Acc %.3f | ans-ppl %.2f (N=%d)"
              % (tag if not (a.base and not a.load_ckpt) else "base", kind, W_, tf1 / N, tem / N, tacc / N, math.exp(tnll / tk), N), flush=True)
    _hp_eval("full", HPC)
    _skind = "streaming" if (a.mask in ("a", "c") and not (TOKMODEL or PLMODEL or SCAL or RIDING or RIDE2)) or (a.base and not a.load_ckpt) else "windowed"
    _hp_eval(_skind, W)
    print("HOTPOT_DONE", flush=True)
    import sys; sys.exit(0)

# ------------------------------------------------------------------ CONTINUOUS STREAMING DEPLOY EVAL (rolling cache to ~30k)
if a.deploy_stream:
    model.gradient_checkpointing_disable(); model.config.use_cache = True       # rolling-cache eval needs returned caches
    @torch.no_grad()
    def chunked(streams, realmask, use_sink):
        bins, bs, warm = a.eval_bins, a.eval_binsize, a.eval_warm; bin_nll = [0.0] * bins; ntok = bs * len(streams)
        for s in streams:
            sid = torch.tensor(s, device=dev); ninp = len(s) - 1
            sink_emb = model.get_input_embeddings()(sid[:S][None])[0] if use_sink else None
            cache = None; pos = 0; nll_all = torch.empty(ninp, device=dev); ci0 = 0
            while ci0 < ninp:
                end = min(ci0 + C, ninp); ch = sid[ci0:end][None]; tg = sid[ci0 + 1:end + 1]
                prefix = sink_emb if use_sink else None
                lg, cache, pos = stream_chunk(ch, cache, pos, realmask, prefix, use_sink)
                nll_all[ci0:end] = F.cross_entropy(lg[:end - ci0].float(), tg, reduction="none"); ci0 = end
            sc = nll_all[warm:warm + bins * bs].view(bins, bs)
            for b in range(bins): bin_nll[b] += sc[b].sum().item()
        bp = [math.exp(x / ntok) for x in bin_nll]; return math.exp(sum(bin_nll) / (ntok * bins)), bp[-1], bp
    @torch.no_grad()
    def hreg_stream(streams):
        bins, bs, warm = a.eval_bins, a.eval_binsize, a.eval_warm; bin_nll = [0.0] * bins; ntok = bs * len(streams)
        for s in streams:
            sid = torch.tensor(s, device=dev); ninp = len(s) - 1; cache = None; pos = 0; nll_all = torch.empty(ninp, device=dev); ci0 = 0
            while ci0 < ninp:
                end = min(ci0 + C, ninp); ch = sid[ci0:end][None]; tg = sid[ci0 + 1:end + 1]
                if reg_k is not None:                                        # per-layer K/V + input prompt
                    lg, cache, pos = stream_chunk_pl(ch, cache, pos, prompt, reg_k, reg_v, "sliding", chunk_len=end - ci0)
                else:
                    lg, cache, pos = stream_chunk(ch, cache, pos, "sliding", prompt, True)
                nll_all[ci0:end] = F.cross_entropy(lg[:end - ci0].float(), tg, reduction="none"); ci0 = end
            sc = nll_all[warm:warm + bins * bs].view(bins, bs)
            for b in range(bins): bin_nll[b] += sc[b].sum().item()
        bp = [math.exp(x / ntok) for x in bin_nll]; return math.exp(sum(bin_nll) / (ntok * bins)), bp[-1], bp
    @torch.no_grad()
    def full_single(streams, cap):
        bins, bs, warm = a.full_bins, a.full_binsize, a.eval_warm; bin_nll = [0.0] * bins; ntok = bs * len(streams)
        dm = deploy_mask("full", cap, W, S, dev, bf16)
        for s in streams:
            ids = torch.tensor([s[:cap]], device=dev)
            lg = model(ids, attention_mask=dm, use_cache=False).logits[0]
            nll = F.cross_entropy(lg[:-1].float(), ids[0, 1:], reduction="none")
            sc = nll[warm:warm + bins * bs].view(bins, bs)
            for b in range(bins): bin_nll[b] += sc[b].sum().item()
        bp = [math.exp(x / ntok) for x in bin_nll]; return math.exp(sum(bin_nll) / (ntok * bins)), bp[-1], bp
    @torch.no_grad()
    def full_own(streams, cap):
        # full-context deploy WITH the model's own trained sink attached (prompt and/or per-layer K/V prefix)
        bins, bs, warm = a.full_bins, a.full_binsize, a.eval_warm; bin_nll = [0.0] * bins; ntok = bs * len(streams)
        emb = model.get_input_embeddings(); Pn = prompt.shape[0] if prompt is not None else 0
        pm = iid_prefix_mask("j_iid", Pn + cap, Pn, W, dev, bf16) if (reg_k is None and Pn) else None
        for s_ in streams:
            sid = torch.tensor(s_[:cap], device=dev)
            if reg_k is not None:
                lg, _, _ = stream_chunk_pl(sid[None], None, 0, prompt, reg_k, reg_v, "causal", chunk_len=cap, ret_cache=False)
            else:
                inp = torch.cat([prompt[None].to(bf16), emb(sid[None])], 1)
                lg = model(inputs_embeds=inp, attention_mask=pm, use_cache=False).logits[0, Pn:]
            nll = F.cross_entropy(lg[:-1].float(), sid[1:], reduction="none")
            sc = nll[warm:warm + bins * bs].view(bins, bs)
            for b in range(bins): bin_nll[b] += sc[b].sum().item()
        bp = [math.exp(x / ntok) for x in bin_nll]
        return math.exp(sum(bin_nll) / (ntok * bins)), bp[-1]
    @torch.no_grad()
    def full_grow(streams):
        # FULL-context deploy streamed to 30k via a GROWING (non-evicting) KV cache + causal attention,
        # with the model's own sink attached (token: prepend once; prefix: per-layer regs; scalar/riding: patch).
        bins, bs, warm = a.eval_bins, a.eval_binsize, a.eval_warm; bin_nll = [0.0] * bins; ntok = bs * len(streams)
        TOK = (prompt is not None) and (reg_k is None) and not RIDING and not RIDE2
        PL = (reg_k is not None) and not RIDING and not RIDE2
        for sseq in streams:
            sid = torch.tensor(sseq, device=dev); ninp = len(sseq) - 1
            cache = None; pos = 0; nll_all = torch.empty(ninp, device=dev); ci0 = 0
            while ci0 < ninp:
                end = min(ci0 + C, ninp); ch = sid[ci0:end][None]; tg = sid[ci0 + 1:end + 1]; cl = end - ci0
                if PL:
                    lg, cache, pos = stream_chunk_pl(ch, cache, pos, prompt, reg_k, reg_v, "causal", chunk_len=cl, grow=True)
                elif TOK:
                    pe = prompt if cache is None else None                 # prepend sink once; it stays in the growing cache
                    lg, cache, pos = stream_chunk(ch, cache, pos, "causal", pe, False, grow=True)
                else:
                    lg, cache, pos = stream_chunk(ch, cache, pos, "causal", None, False, grow=True)   # plain/scalar/riding/2slot (patch active)
                nll_all[ci0:end] = F.cross_entropy(lg[:cl].float(), tg, reduction="none"); ci0 = end
            sc = nll_all[warm:warm + bins * bs].view(bins, bs)
            for b in range(bins): bin_nll[b] += sc[b].sum().item()
        bp = [math.exp(x / ntok) for x in bin_nll]; return math.exp(sum(bin_nll) / (ntok * bins)), bp[-1]
    mlabel = "base" if (a.base and not a.load_ckpt) else tag
    full_cap = a.eval_warm + a.full_bins * a.full_binsize + 1
    slen = a.eval_warm + a.eval_bins * a.eval_binsize + 1
    streams = pack("test", a.n_eval, slen)
    print("deploy_stream model=%s n_eval=%d slen=%d full_cap=%d ..." % (mlabel, len(streams), slen, full_cap), flush=True)
    if a.own_only:
        if RIDING or RIDE2 or (prompt is None and reg_k is None):
            oa, od, _ = full_single(streams, full_cap)          # sink-less, scalar, or riding: sink lives in the patch
        else:
            oa, od = full_own(streams, full_cap)
        print("DSTREAM_FULLOWN model=%s | full_own(avg %.4f deep %.4f @%d)" % (mlabel, oa, od, full_cap - 1), flush=True)
        print("DSTREAM_DONE model=%s" % mlabel, flush=True)
        import sys; sys.exit(0)
    fa, fd = full_grow(streams)          # full-context streamed to 30k (growing cache)
    sa, sd, _ = chunked(streams, "sliding", False)
    ta, td, _ = chunked(streams, "sliding", True)
    _fdepth = a.eval_bins * a.eval_binsize
    print("DSTREAM model=%s | full(avg %.4f deep %.4f @%dk) | sliding(avg %.4f deep %.4f @30k) | streamingLLM(avg %.4f deep %.4f @30k)"
          % (mlabel, fa, fd, _fdepth // 1000, sa, sd, ta, td), flush=True)
    is_cpt = not (a.base and not a.load_ckpt)
    if a.mask == "e" and is_cpt:
        ea, ed, _ = chunked(streams, "causal", False)
        print("DSTREAM_E model=%s | e@2W_causal(avg %.4f deep %.4f @30k)" % (mlabel, ea, ed), flush=True)
    if a.mask in ("h", "h_iid", "j_iid") and is_cpt and prompt is not None:
        ha, hd, _ = hreg_stream(streams)
        print("DSTREAM_H model=%s | sliding_own_regs(avg %.4f deep %.4f @30k)" % (mlabel, ha, hd), flush=True)
    print("DSTREAM_DONE model=%s" % mlabel, flush=True)
    import sys; sys.exit(0)

# ------------------------------------------------------------------ ATTENTION-SINK GRID (per deploy mask)
if a.sink_npz:
    import numpy as np
    T = a.sink_len; samp = pack("test", 1, T)[0]; ids = torch.tensor([samp], device=dev)
    toks = [tok.decode([t]) for t in samp]
    is_sep_t = torch.tensor([(s.strip() in {".", ",", ";", ":", "!", "?"}) or (s.strip() == "") for s in toks], device=dev)
    mlabel = "base" if (a.base and not a.load_ckpt) else tag
    qi = torch.arange(T, device=dev)[:, None]; ki = torch.arange(T, device=dev)[None, :]
    qfull = (torch.arange(T, device=dev) >= W)                   # queries whose window has evicted token-0 (sliding)
    def get_attn(deploy):
        dm = deploy_mask(deploy, T, W, S, dev, bf16)
        with torch.no_grad():
            out = model(ids, attention_mask=dm, use_cache=False, output_attentions=True)
        return torch.stack(out.attentions, 0)[:, 0].float()      # [Lyr,Hd,T,T]
    npz = {}
    # FULL: innate pos-0 sink
    Af = get_attn("full"); Lyr, Hd = Af.shape[0], Af.shape[1]
    ph0 = Af[:, :, 1:, 0].mean(-1); full_pos0 = ph0.mean().item(); full_pct = (ph0 > 0.3).float().mean().item()
    per_layer = ph0.mean(1).tolist(); hf = int(torch.topk(ph0.flatten(), 1).indices.item())
    npz["full_map"] = Af[hf // Hd, hf % Hd].cpu().numpy(); npz["full_head"] = np.array([hf // Hd, hf % Hd])
    del Af; torch.cuda.empty_cache()
    # STREAMING(S4+W): mass on the pinned first-S sink tokens
    As = get_attn("streaming"); phs = As[:, :, W:, :S].sum(-1).mean(-1)
    str_sink = phs.mean().item(); str_pct = (phs > 0.3).float().mean().item()
    hs = int(torch.topk(phs.flatten(), 1).indices.item())
    npz["stream_map"] = As[hs // Hd, hs % Hd].cpu().numpy(); npz["stream_head"] = np.array([hs // Hd, hs % Hd])
    del As; torch.cuda.empty_cache()
    # SLIDING@W: token-0 evicted -> 4-way relocation over queries q>=W
    Asl = get_attn("sliding")
    inwin = (ki > qi - W) & (ki <= qi)
    wstart = inwin & (ki <= qi - W + 4)                          # oldest 4 in-window (relocated window-start sink)
    local = inwin & (ki >= qi - 2) & ~wstart                     # recent 3 (diagonal)
    sepm = inwin & is_sep_t[None, :] & ~wstart & ~local          # in-window separators
    diffuse = inwin & ~wstart & ~local & ~sepm
    def frac(M): return (Asl * M[None, None].float()).sum(-1)[:, :, qfull].mean().item()
    fr_ws, fr_lo, fr_sp, fr_df = frac(wstart), frac(local), frac(sepm), frac(diffuse)
    ws_ph = (Asl * wstart[None, None].float()).sum(-1)[:, :, qfull].mean(-1)
    hsl = int(torch.topk(ws_ph.flatten(), 1).indices.item())
    npz["slide_map"] = Asl[hsl // Hd, hsl % Hd].cpu().numpy(); npz["slide_head"] = np.array([hsl // Hd, hsl % Hd])
    cats = {"wstart": fr_ws, "local": fr_lo, "sep": fr_sp, "diffuse": fr_df}; dom = max(cats, key=cats.get)
    np.savez_compressed(a.sink_npz, tokens=np.array(toks, dtype=object), per_layer_full_pos0=np.array(per_layer),
                        full_pos0=full_pos0, full_pct=full_pct, stream_sink=str_sink, stream_pct=str_pct,
                        slide_wstart=fr_ws, slide_local=fr_lo, slide_sep=fr_sp, slide_diffuse=fr_df, **npz)
    print("SINKGRID model=%s | FULL pos0 %.3f (sinkheads %.2f) | STREAM firstS %.3f (sinkheads %.2f) | SLIDING wstart %.3f local %.3f sep %.3f diffuse %.3f -> dominant %s"
          % (mlabel, full_pos0, full_pct, str_sink, str_pct, fr_ws, fr_lo, fr_sp, fr_df, dom), flush=True)
    print("SINK per_layer_full_pos0 = %s" % (["%.3f" % x for x in per_layer]), flush=True)
    print("SINK_DONE model=%s npz=%s" % (mlabel, a.sink_npz), flush=True)
    import sys; sys.exit(0)

# ------------------------------------------------------------------ DEPLOY EVAL
eval_len = a.warm + a.bins * a.binsize + 1
ev = pack("test", a.n_eval, eval_len)
print("deploy eval (eval_len %d, n_eval %d) ..." % (eval_len, len(ev)), flush=True)
@torch.no_grad()
def deploy_eval(kind):
    dm = deploy_mask(kind, eval_len, W, S, dev, bf16); bin_nll = [0.0] * a.bins; ntok = a.binsize * len(ev)
    for s in ev:
        ids = torch.tensor([s], device=dev)
        lg = model(ids, attention_mask=dm, use_cache=False).logits[0]
        nll = F.cross_entropy(lg[:-1].float(), ids[0, 1:], reduction="none")
        scored = nll[a.warm:a.warm + a.bins * a.binsize].view(a.bins, a.binsize)
        for b in range(a.bins): bin_nll[b] += scored[b].sum().item()
    binppl = [math.exp(x / ntok) for x in bin_nll]
    return math.exp(sum(bin_nll) / (ntok * a.bins)), binppl[-1], binppl

@torch.no_grad()
def deploy_eval_hreg():
    # h own-registers sliding deploy: chunked continuous stream, registers prepended EVERY chunk (bounded
    # positions, matching training), sliding cache. Score query positions >= warm, binned.
    bin_nll = [0.0] * a.bins; ntok = a.binsize * len(ev); ninp = eval_len - 1
    for s in ev:
        ids = torch.tensor(s, device=dev); cache = None; pos = 0
        nll_all = torch.empty(ninp, device=dev); ci0 = 0
        while ci0 < ninp:
            end = min(ci0 + C, ninp); ch = ids[ci0:end][None]; tg = ids[ci0 + 1:end + 1]
            lg, cache, pos = stream_chunk(ch, cache, pos, "sliding", prompt, True)
            nll_all[ci0:end] = F.cross_entropy(lg[:end - ci0].float(), tg, reduction="none"); ci0 = end
        scored = nll_all[a.warm:a.warm + a.bins * a.binsize].view(a.bins, a.binsize)
        for b in range(a.bins): bin_nll[b] += scored[b].sum().item()
    binppl = [math.exp(x / ntok) for x in bin_nll]
    return math.exp(sum(bin_nll) / (ntok * a.bins)), binppl[-1]

res = {}
for kind in ["full", "sliding", "streaming"]:
    avg, deep, bp = deploy_eval(kind); res[kind] = (avg, deep)
    print("DEPLOY %s: avg %.4f deep %.4f bins %s" % (kind, avg, deep, ["%.2f" % x for x in bp]), flush=True)
if a.mask == "h":
    ra, rd = deploy_eval_hreg(); print("DEPLOY sliding_regs: avg %.4f deep %.4f" % (ra, rd), flush=True)
    print("RESULT_EXTRA mask=h boundary=%s | sliding_regs(avg %.4f deep %.4f)" % (a.boundary, ra, rd), flush=True)
print("RESULT cpt mask=%s boundary=%s tag=%s W=%d S=%d ctx=%d steps=%d | full(avg %.4f deep %.4f) | sliding(avg %.4f deep %.4f) | streaming(avg %.4f deep %.4f) | peakGB %.1f"
      % (a.mask, a.boundary, tag, W, S, L, a.steps, res["full"][0], res["full"][1], res["sliding"][0], res["sliding"][1],
         res["streaming"][0], res["streaming"][1], torch.cuda.max_memory_allocated() / 1e9), flush=True)
print("CPT_DONE tag=%s" % tag, flush=True)
