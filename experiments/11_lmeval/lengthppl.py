"""Length-swept perplexity (SWAT Table-2 style) on PG-19 test books.

For each eval length L in {128, 1024, 4096, 16384}: score the first L tokens of each of
--ndocs PG-19 test books under the given deploy; report mean ppl +/- SEM over books.
Deploy geometry + register handling mirrors lmeval_sswa.py / cpt_masks.py (validated paths).

Run: python lengthppl.py --ckpt /scratch1/zizhaoh/cceq_ckpts/prefix --deploy sliding
"""
import argparse, json, math, os
import torch

def _install_riding(model, mode, W, prompt, reg_k, reg_v, dev, bf16):
    """Riding registers at constant offset (port of cpt_masks.py RIDING patch)."""
    import transformers.models.llama.modeling_llama as _ml2
    from transformers.models.llama.modeling_llama import repeat_kv as _repkv2, rotate_half as _rh
    from transformers.cache_utils import DynamicCache
    nLr = model.config.num_hidden_layers
    hdr_ = model.config.hidden_size // model.config.num_attention_heads
    nP = (reg_k.shape[1] if mode == "prefix" else prompt.shape[0])
    RIDE_KV = {}
    RIDE_D = (W + torch.arange(nP, device=dev).flip(0)).float()
    _dmy = torch.zeros(1, nP, hdr_, device=dev, dtype=bf16)
    with torch.no_grad():
        COS_D, SIN_D = model.model.rotary_emb(_dmy, RIDE_D[None].long())
    if mode == "prefix":
        for li in range(nLr):
            RIDE_KV[li] = (reg_k[li].permute(1, 0, 2)[None], reg_v[li].permute(1, 0, 2)[None])
    else:
        with torch.no_grad():
            pos = torch.arange(nP, device=dev)
            out = model(inputs_embeds=prompt[None], position_ids=pos[None], use_cache=True,
                        past_key_values=DynamicCache())
            cosP, sinP = model.model.rotary_emb(_dmy, pos[None])
            for li, Lr in enumerate(out.past_key_values.layers):
                kc = Lr.keys
                kp = kc * cosP.unsqueeze(1) - _rh(kc) * sinP.unsqueeze(1)
                RIDE_KV[li] = (kp, Lr.values)
    _orig = _ml2.LlamaAttention.forward
    def _ride_fwd(self, hidden_states, position_embeddings, attention_mask=None,
                  past_key_value=None, cache_position=None, **kw):
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
        qp = qr * cos.unsqueeze(1) - _rh(qr) * sin.unsqueeze(1)
        pk, pv = RIDE_KV[self.layer_idx]
        krd = pk * COS_D.unsqueeze(1) - _rh(pk) * SIN_D.unsqueeze(1)
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
    print("RIDING_PATCHED mode=%s nP=%d W=%d" % (mode, nP, W), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--deploy", default="full", choices=["full", "sliding", "streaming"])
ap.add_argument("--window", type=int, default=1024)
ap.add_argument("--sink_keep", type=int, default=4)
ap.add_argument("--riding", default="none", choices=["none", "token", "prefix"])   # riding registers (constant offset); ckpt files alone cannot distinguish
ap.add_argument("--lengths", default="128,1024,4096,16384")
ap.add_argument("--ndocs", type=int, default=40)
a = ap.parse_args()
dev = "cuda"; bf16 = torch.bfloat16
os.environ.setdefault("HF_HOME", "/scratch1/zizhaoh/.cache/huggingface")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_is_scalar = os.path.isdir(a.ckpt) and os.path.exists(os.path.join(a.ckpt, "sink_sc.pt"))
ATTN = "eager" if _is_scalar else "sdpa"     # eager fp32 scores OOM at 16k on 3B; sdpa handles 4D masks
model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=bf16, attn_implementation=ATTN).to(dev).eval()
tok = AutoTokenizer.from_pretrained(a.ckpt)
prompt = reg_k = reg_v = None; sink = "none"
if os.path.isdir(a.ckpt):
    pp, rk, rv, sc = [os.path.join(a.ckpt, f) for f in ("prompt.pt", "reg_k.pt", "reg_v.pt", "sink_sc.pt")]
    if os.path.exists(rk):
        sink = "prefix"; reg_k = torch.load(rk, map_location=dev).to(bf16); reg_v = torch.load(rv, map_location=dev).to(bf16)
        if os.path.exists(pp): prompt = torch.load(pp, map_location=dev).to(bf16)
    elif os.path.exists(pp):
        sink = "token"; prompt = torch.load(pp, map_location=dev).to(bf16)
    elif os.path.exists(sc):
        sink = "scalar"
        SINK_SC = torch.load(sc, map_location=dev).to(bf16)
        import transformers.models.llama.modeling_llama as _ml
        from transformers.models.llama.modeling_llama import repeat_kv as _repkv
        def _eager_sink(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            s2 = getattr(module, "sink_scalar", None)
            ks = _repkv(key, module.num_key_value_groups); vs = _repkv(value, module.num_key_value_groups)
            aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
            if attention_mask is not None: aw = aw + attention_mask[:, :, :, :ks.shape[-2]]
            Bq, Hq, Tq, Kq = aw.shape
            pad = s2.view(1, -1, 1, 1).expand(Bq, -1, Tq, 1).to(aw.dtype)
            aw = torch.cat([aw, pad], dim=-1)
            aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)[..., :Kq]
            return torch.matmul(aw, vs).transpose(1, 2).contiguous(), aw
        _ml.eager_attention_forward = _eager_sink
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = _eager_sink
        for li, lyr in enumerate(model.model.layers): lyr.self_attn.sink_scalar = SINK_SC[li]
if a.riding != "none":
    _install_riding(model, a.riding, a.window, prompt, reg_k, reg_v, dev, bf16)
    sink = "riding"; prompt = reg_k = reg_v = None
print("LENGTHPPL ckpt=%s deploy=%s sink=%s" % (a.ckpt, a.deploy, sink), flush=True)

def add4d(al, dt):
    m = torch.zeros(al.shape, device=dev, dtype=dt)
    m.masked_fill_(~al, torch.finfo(dt).min)
    return m[None, None]

def mask_plain(T, dt):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    if a.deploy == "full":      al = causal
    elif a.deploy == "sliding": al = causal & (k > q - a.window)
    else:                       al = causal & ((k < a.sink_keep) | (k > q - a.window))
    return add4d(al, dt)

def mask_token(T, Pn, dt):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    rq, rk_ = q - Pn, k - Pn
    causal = rk_ <= rq
    okc = causal if a.deploy == "full" else causal & (rk_ > rq - a.window)
    al = (k < Pn) | ((q >= Pn) & (k >= Pn) & okc) | ((q < Pn) & (k <= q))
    return add4d(al, dt)

def mask_prefix(T2, nPL, Pn, dt):
    Lk = nPL + T2
    r = torch.arange(T2, device=dev)[:, None]; c = torch.arange(Lk, device=dev)[None, :]
    is_pl = c < nPL; c2 = c - nPL
    is_prompt = (c2 >= 0) & (c2 < Pn)
    ri = c2 - Pn; rq = r - Pn
    ok = (ri <= rq) & ((a.deploy == "full") | (ri > rq - a.window))
    al = torch.where(r < Pn, is_pl | (is_prompt & (c2 <= r)), is_pl | is_prompt | ((~is_pl) & (~is_prompt) & ok))
    return add4d(al, dt)

@torch.no_grad()
def nll_doc(ids):
    x = ids[:-1][None].to(dev); y = ids[1:].to(dev); T = x.shape[1]
    if sink == "prefix":
        from transformers.cache_utils import DynamicCache
        emb = model.get_input_embeddings()
        Pn = prompt.shape[0] if prompt is not None else 0
        nPL = reg_k.shape[1]
        pieces = [emb(x)]
        if Pn: pieces.insert(0, prompt[None])
        inp = torch.cat(pieces, 1); T2 = inp.shape[1]
        cache = DynamicCache(ddp_cache_data=[(reg_k[l].permute(1, 0, 2)[None], reg_v[l].permute(1, 0, 2)[None])
                                             for l in range(reg_k.shape[0])])
        out = model(inputs_embeds=inp, past_key_values=cache,
                    position_ids=torch.arange(T2, device=dev)[None],
                    cache_position=torch.arange(nPL, nPL + T2, device=dev),
                    attention_mask=mask_prefix(T2, nPL, Pn, bf16), use_cache=False)
        lg = out.logits[0, Pn:]
    elif sink == "token":
        emb = model.get_input_embeddings()
        Pn = prompt.shape[0]
        inp = torch.cat([prompt[None], emb(x)], 1)
        lg = model(inputs_embeds=inp, attention_mask=mask_token(inp.shape[1], Pn, bf16)).logits[0, Pn:]
    else:
        lg = model(x, attention_mask=mask_plain(T, bf16)).logits[0]
    tot = 0.0; cnt = y.numel()
    for i in range(0, cnt, 2048):
        tot += torch.nn.functional.cross_entropy(lg[i:i + 2048].float(), y[i:i + 2048],
                                                 reduction="sum").item()
    return tot / cnt

docs = load_dataset("emozilla/pg19-test", split="test")
name = os.path.basename(a.ckpt.rstrip("/"))
for L in [int(x) for x in a.lengths.split(",")]:
    if _is_scalar and L > 4096:
        print("LENPPL ckpt=%s deploy=%s L=%d SKIPPED (scalar sink requires eager attention)"
              % (os.path.basename(a.ckpt.rstrip("/")), a.deploy, L), flush=True)
        continue
    nlls = []
    for d in docs.select(range(a.ndocs)):
        ids = tok(d["text"], return_tensors="pt", add_special_tokens=True).input_ids[0][:L + 1]
        if len(ids) < L + 1: continue
        nlls.append(nll_doc(ids))
    n = len(nlls)
    mean = sum(nlls) / n
    ppls = [math.exp(v) for v in nlls]
    mp = sum(ppls) / n
    sem = (sum((p - mp) ** 2 for p in ppls) / n) ** 0.5 / n ** 0.5
    print("LENPPL ckpt=%s deploy=%s L=%d n=%d ppl=%.3f sem=%.3f (tokmean %.3f)"
          % (name, a.deploy, L, n, mp, sem, math.exp(mean)), flush=True)
print("LENGTHPPL_DONE", flush=True)
