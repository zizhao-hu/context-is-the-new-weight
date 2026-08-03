#!/usr/bin/env python3
"""Provenance-clean WikiText check for the Table 2 stream column: one harness,
matched deploys, within-length and streaming, base against CPT checkpoints.

Per checkpoint and policy prints:
  VERIFY ckpt=<name> policy=<full|slide|sllm> within=<ppl>+-<sem> \
         stream_all=<ppl>+-<sem> stream_far=<ppl>+-<sem>
within: full attention on L-token chunks, scored q >= 64 (the ppl<Cmax analogue).
stream_all: TOTAL-token streams at the W KV budget (chunked prefill c=256 with
eviction to W-c retained entries), scored q >= W. stream_far: same pass scored
q >= 4096, strictly beyond the CPT length. slide keeps the last W-c entries;
sllm (StreamingLLM) keeps the first 4 real tokens plus the last W-c-4; full is
the unbounded reference.
"""
import argparse, math, os

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", default=None)
ap.add_argument("--W", type=int, default=1024)
ap.add_argument("--L", type=int, default=1024)
ap.add_argument("--total", type=int, default=16384)
ap.add_argument("--nseq", type=int, default=4)
ap.add_argument("--nchunk", type=int, default=24)
ap.add_argument("--policies", default="full,slide,sllm")
a = ap.parse_args()
dev, bf16 = "cuda", torch.bfloat16
W, L = a.W, a.L
os.environ.setdefault("HF_HOME", "/scratch1/zizhaoh/.cache/huggingface")

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=bf16,
                                             attn_implementation="sdpa").to(dev).eval()
tok = AutoTokenizer.from_pretrained(a.ckpt)
name = a.name or os.path.basename(a.ckpt.rstrip("/"))

wt = load_dataset("wikitext", "wikitext-103-raw-v1")["validation"]
ids = tok("\n\n".join(r["text"] for r in wt if r["text"].strip()),
          add_special_tokens=False, return_tensors="pt")["input_ids"][0]
print("TOKENS", len(ids), flush=True)

def kv_layers(pkv):
    if hasattr(pkv, "layers"):
        return [(l.keys, l.values) for l in pkv.layers]
    if hasattr(pkv, "key_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return list(pkv)

def nll_of(logits, y, lo):
    """per-position NLL from lo onward, gathered in slices to bound fp32 memory"""
    out = []
    for j in range(lo, y.size(1), 2048):
        sl = slice(j, min(j + 2048, y.size(1)))
        lp = F.log_softmax(logits[:, sl].float(), dim=-1)
        out.append(-lp.gather(-1, y[:, sl, None]).squeeze(-1))
    return torch.cat(out, dim=1)

def agg(chunk_means):
    n = len(chunk_means)
    mu = sum(chunk_means) / n
    var = sum((c - mu) ** 2 for c in chunk_means) / max(n - 1, 1)
    ppl = math.exp(mu)
    return ppl, ppl * math.sqrt(var / n)

@torch.no_grad()
def within():
    ms = []
    for i in range(a.nchunk):
        o = i * L
        if o + L + 1 > len(ids):
            break
        x = ids[o:o + L][None].to(dev); y = ids[o + 1:o + L + 1][None].to(dev)
        ms.append(nll_of(model(input_ids=x).logits, y, 64).mean().item())
    return agg(ms)

@torch.no_grad()
def stream(policy):
    c, keep = 256, W - 256
    alls, fars = [], []
    for i in range(a.nseq):
        o = i * a.total
        if o + a.total + 1 > len(ids):
            break
        x = ids[o:o + a.total][None].to(dev); y = ids[o + 1:o + a.total + 1][None].to(dev)
        if policy == "full":
            logits = model(input_ids=x).logits
        else:
            cache, outs = DynamicCache(), []
            for j in range(0, a.total, c):
                seg = x[:, j:j + c]
                pos = torch.arange(j, j + seg.size(1), device=dev)[None]
                o_ = model(input_ids=seg, past_key_values=cache,
                           position_ids=pos, use_cache=True)
                outs.append(o_.logits)
                cache = DynamicCache()
                for i_, (k, v) in enumerate(kv_layers(o_.past_key_values)):
                    if k.size(2) > keep:
                        if policy == "slide":
                            k, v = k[:, :, -keep:], v[:, :, -keep:]
                        else:                                     # sllm
                            k = torch.cat([k[:, :, :4], k[:, :, -(keep - 4):]], 2)
                            v = torch.cat([v[:, :, :4], v[:, :, -(keep - 4):]], 2)
                    cache.update(k, v, i_)
            logits = torch.cat(outs, dim=1)
        nll = nll_of(logits, y, W)
        alls.append(nll.mean().item())
        fars.append(nll[:, 4096 - W:].mean().item())
        del logits
        torch.cuda.empty_cache()
    return agg(alls), agg(fars)

wi = within()
for pol in a.policies.split(","):
    (sa, ses), (sf, sfs) = stream(pol)
    print("VERIFY ckpt=%s policy=%s within=%.3f+-%.3f stream_all=%.3f+-%.3f "
          "stream_far=%.3f+-%.3f" % (name, pol, wi[0], wi[1], sa, ses, sf, sfs),
          flush=True)
print("VDONE", flush=True)
