"""Length-swept perplexity (SWAT Table-2 style) on PG-19 test books.

For each eval length L in {128, 1024, 4096, 16384}: score the first L tokens of each of
--ndocs PG-19 test books under the given deploy; report mean ppl +/- SEM over books.
Deploy geometry + register handling mirrors lmeval_sswa.py / cpt_masks.py (validated paths).

Run: python lengthppl.py --ckpt /scratch1/zizhaoh/cceq_ckpts/prefix --deploy sliding
"""
import argparse, json, math, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--deploy", default="full", choices=["full", "sliding", "streaming"])
ap.add_argument("--window", type=int, default=1024)
ap.add_argument("--sink_keep", type=int, default=4)
ap.add_argument("--lengths", default="128,1024,4096,16384")
ap.add_argument("--ndocs", type=int, default=40)
a = ap.parse_args()
dev = "cuda"; bf16 = torch.bfloat16
os.environ.setdefault("HF_HOME", "/scratch1/zizhaoh/.cache/huggingface")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=bf16, attn_implementation="eager").to(dev).eval()
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
    return torch.nn.functional.cross_entropy(lg.float(), y).item()

docs = load_dataset("emozilla/pg19-test", split="test")
name = os.path.basename(a.ckpt.rstrip("/"))
for L in [int(x) for x in a.lengths.split(",")]:
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
