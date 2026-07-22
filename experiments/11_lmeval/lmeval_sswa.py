"""lm-evaluation-harness wrapper for the CPT checkpoints with S-SWA deploy geometries.

Deploys (mirrors experiments/10_cpt/cpt_masks.py, the source of the validated deploy_eval_reg):
  full      : full causal (4D additive mask)
  sliding   : banded causal window W
  streaming : StreamingLLM mask (first S tokens + window W)
Sinks (auto-detected from checkpoint dir):
  token  : prompt.pt  -- prepended always-visible embeddings (logits sliced past them)
  prefix : reg_k.pt/reg_v.pt (+prompt.pt) -- per-layer raw K/V injected via DynamicCache,
           content positions offset by nP (exact stream_chunk_pl geometry, rolling=None)
  scalar : sink_sc.pt -- per-(layer,head) softmax-denominator logit via eager-attention patch
Riding sinks are NOT supported here (constant-offset attention patch); skip those checkpoints.

batch_size is forced to 1: custom 4D masks/prepends do not compose with padded batches.
max_length capped at the CPT training context (2048) so wikitext rolling windows match training.

Run: python lmeval_sswa.py --ckpt /scratch1/zizhaoh/cpt_ckpts/a --deploy full \
       --tasks boolq,piqa --out /scratch1/zizhaoh/lmeval_out/a_full.json
"""
import argparse, json, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)            # HF dir, or plain model name for the base row
ap.add_argument("--deploy", default="full", choices=["full", "sliding", "streaming"])
ap.add_argument("--window", type=int, default=1024)
ap.add_argument("--sink_keep", type=int, default=4)  # S for streaming
ap.add_argument("--tasks", default="lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,siqa_pq,boolq")
ap.add_argument("--include_path", default="/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/experiments/11_lmeval/tasks")
ap.add_argument("--limit", type=int, default=0)      # >0: debug subset
ap.add_argument("--max_length", type=int, default=2048)
ap.add_argument("--vanilla", action="store_true")    # validation: bypass wrapper entirely (plain HFLM)
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda"
bf16 = torch.bfloat16

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def add4d(allowed, dt):
    m = torch.zeros(allowed.shape, device=dev, dtype=dt)
    m.masked_fill_(~allowed, torch.finfo(dt).min)
    return m[None, None]


class SSWALM(HFLM):
    def __init__(self, ckpt, deploy, W, S, **kw):
        super().__init__(pretrained=ckpt, backend="causal", dtype=bf16, device=dev,
                         batch_size=1, max_length=a.max_length,
                         attn_implementation="eager", **kw)
        self.deploy, self.W, self.S = deploy, W, S
        self.prompt = self.reg_k = self.reg_v = None
        self.sink = "none"
        if os.path.isdir(ckpt):
            pp = os.path.join(ckpt, "prompt.pt")
            rk = os.path.join(ckpt, "reg_k.pt"); rv = os.path.join(ckpt, "reg_v.pt")
            sc = os.path.join(ckpt, "sink_sc.pt")
            if os.path.exists(rk):
                self.sink = "prefix"
                self.reg_k = torch.load(rk, map_location=dev).to(bf16)
                self.reg_v = torch.load(rv, map_location=dev).to(bf16)
                if os.path.exists(pp): self.prompt = torch.load(pp, map_location=dev).to(bf16)
            elif os.path.exists(pp):
                self.sink = "token"
                self.prompt = torch.load(pp, map_location=dev).to(bf16)
            elif os.path.exists(sc):
                self.sink = "scalar"
                self._patch_scalar(torch.load(sc, map_location=dev).to(bf16))
        print("SSWALM ckpt=%s deploy=%s W=%d sink=%s" % (ckpt, deploy, W, self.sink), flush=True)

    # ---- scalar sink: per-head softmax-denominator logit (copied from cpt_masks.py) ----
    def _patch_scalar(self, SINK_SC):
        import transformers.models.llama.modeling_llama as _ml
        from transformers.models.llama.modeling_llama import repeat_kv as _repkv
        _orig = _ml.eager_attention_forward
        def _eager_sink(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            sc = getattr(module, "sink_scalar", None)
            if sc is None:
                return _orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
            ks = _repkv(key, module.num_key_value_groups); vs = _repkv(value, module.num_key_value_groups)
            aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
            if attention_mask is not None: aw = aw + attention_mask[:, :, :, :ks.shape[-2]]
            Bq, Hq, Tq, Kq = aw.shape
            pad = sc.view(1, -1, 1, 1).expand(Bq, -1, Tq, 1).to(aw.dtype)
            aw = torch.cat([aw, pad], dim=-1)
            aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)[..., :Kq]
            out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
            return out, aw
        _ml.eager_attention_forward = _eager_sink
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS["eager"] = _eager_sink
        except Exception as e:
            print("ATTN_REGISTRY_PATCH_SKIP", repr(e), flush=True)
        for li, lyr in enumerate(self.model.model.layers):
            lyr.self_attn.sink_scalar = SINK_SC[li]

    # ---- masks ----
    def _mask_plain(self, T, dt):
        q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
        causal = k <= q
        if self.deploy == "full":      al = causal
        elif self.deploy == "sliding": al = causal & (k > q - self.W)
        else:                          al = causal & ((k < self.S) | (k > q - self.W))
        return add4d(al, dt)

    def _mask_token(self, T, Pn, dt):
        # cols/rows: [prompt(Pn); content(T-Pn)]; prompt cols always visible; content windowed
        q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
        rq, rk = q - Pn, k - Pn
        causal = rk <= rq
        if self.deploy == "full": okc = causal
        else:                     okc = causal & (rk > rq - self.W)
        al = (k < Pn) | ((q >= Pn) & (k >= Pn) & okc)
        al = al | ((q < Pn) & (k <= q))                       # prompt rows: causal among themselves
        return add4d(al, dt)

    def _mask_prefix(self, T2, nPL, Pn, dt):
        # cols: [perlayer(nPL); prompt(Pn); chunk]; rows: [prompt(Pn); chunk]  (stream_mask_pl, Wc=0)
        Lk = nPL + T2
        r = torch.arange(T2, device=dev)[:, None]; c = torch.arange(Lk, device=dev)[None, :]
        is_pl = c < nPL; c2 = c - nPL
        is_prompt = (c2 >= 0) & (c2 < Pn)
        real_idx = c2 - Pn; rqq = r - Pn
        real_ok = (real_idx <= rqq) & ((self.deploy == "full") | (real_idx > rqq - self.W))
        chunk_allowed = is_pl | is_prompt | ((~is_pl) & (~is_prompt) & real_ok)
        prompt_allowed = is_pl | (is_prompt & (c2 <= r))
        al = torch.where(r < Pn, prompt_allowed, chunk_allowed)
        return add4d(al, dt)

    # ---- forward ----
    def _model_call(self, inps, attn_mask=None, labels=None):
        assert inps.shape[0] == 1, "batch_size must be 1 for custom deploy masks"
        T = inps.shape[1]
        if self.sink == "prefix":
            from transformers.cache_utils import DynamicCache
            emb = self.model.get_input_embeddings()
            Pn = self.prompt.shape[0] if self.prompt is not None else 0
            nPL = self.reg_k.shape[1]
            pieces = [emb(inps)]
            if Pn: pieces.insert(0, self.prompt[None])
            inp = torch.cat(pieces, 1); T2 = inp.shape[1]
            data = [(self.reg_k[l].permute(1, 0, 2)[None], self.reg_v[l].permute(1, 0, 2)[None])
                    for l in range(self.reg_k.shape[0])]
            cache = DynamicCache(ddp_cache_data=data)
            position_ids = torch.arange(0, T2, device=dev)[None]
            cache_position = torch.arange(nPL, nPL + T2, device=dev)
            m4 = self._mask_prefix(T2, nPL, Pn, inp.dtype)
            out = self.model(inputs_embeds=inp, past_key_values=cache, position_ids=position_ids,
                             cache_position=cache_position, attention_mask=m4, use_cache=False)
            return out.logits[:, Pn:]
        if self.sink == "token":
            emb = self.model.get_input_embeddings()
            Pn = self.prompt.shape[0]
            inp = torch.cat([self.prompt[None], emb(inps)], 1)
            m4 = self._mask_token(inp.shape[1], Pn, inp.dtype)
            return self.model(inputs_embeds=inp, attention_mask=m4).logits[:, Pn:]
        m4 = self._mask_plain(T, self.model.dtype if hasattr(self.model, "dtype") else bf16)
        return self.model(inps, attention_mask=m4).logits

    def _model_generate(self, *args, **kw):
        raise NotImplementedError("generation tasks not supported by the S-SWA wrapper")


tasks = a.tasks.split(",")
if a.vanilla:
    lm = HFLM(pretrained=a.ckpt, backend="causal", dtype=bf16, device=dev,
              batch_size=1, max_length=a.max_length, attn_implementation="eager")
else:
    lm = SSWALM(a.ckpt, a.deploy, a.window, a.sink_keep)

from lm_eval.tasks import TaskManager
tm = TaskManager(include_path=a.include_path) if a.include_path else None
res = simple_evaluate(model=lm, tasks=tasks, limit=(a.limit or None), task_manager=tm)
name = os.path.basename(a.ckpt.rstrip("/"))
for t, r in sorted(res["results"].items()):
    keys = [k for k in r if not k.endswith("_stderr") and isinstance(r[k], (int, float))]
    kv = " ".join("%s=%.4f" % (k, r[k]) for k in sorted(keys))
    err = " ".join("%s=%.4f" % (k, r[k]) for k in sorted(r) if k.endswith("_stderr") and isinstance(r[k], (int, float)))
    print("LMEVAL ckpt=%s deploy=%s task=%s | %s | %s" % (name, a.deploy, t, kv, err), flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({k: v for k, v in res["results"].items()}, open(a.out, "w"), indent=1, default=str)
    print("WROTE", a.out, flush=True)
print("LMEVAL_DONE", flush=True)
