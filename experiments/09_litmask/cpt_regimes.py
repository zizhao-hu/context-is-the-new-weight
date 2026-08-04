"""Our training regimes e/f/g on Qwen2.5-0.5B, through the SAME eval harness as cpt_masks.py so the ppl
is directly comparable for Table 1 (tab:litmask). Each job trains one scheme at window W, then evals
held-out ppl over the last `score_last` positions under full / sliding / StreamingLLM.

  sliding_history (e): plain causal sliding window W; trained on chunks carrying W-1 REAL preceding tokens
                       as history (loss on content only). EVAL: plain, no prompt (identical to cpt_masks).
  warmup (f):          P=W-1 TRAINABLE soft-prompts fill the cold start; sliding mask; loss on content.
                       EVAL: prepend the trained prompts; content scored under full/sliding/streaming
                       (prompt is subject to the deployment mask, so it only aids near the start).
  persistent (g):      nP TRAINABLE soft-prompts, ALWAYS attended by every content query (registers);
                       sliding among content; loss on content. EVAL: prepend prompts; the nP prompt
                       columns are ALWAYS kept in the deployment mask (allowed |= k<nP).
"""
import torch, argparse, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "Qwen/Qwen2.5-0.5B"
ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["sliding_history", "warmup", "persistent"])
ap.add_argument("--window", type=int, default=256)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--n_seq", type=int, default=2000)
ap.add_argument("--score_last", type=int, default=1024)
ap.add_argument("--n_eval", type=int, default=40)
ap.add_argument("--nP", type=int, default=64)               # persistent registers (g)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
S, W, L, sl = a.sink, a.window, a.ctx, a.score_last


def wmask(T, W, dev, dt, nP=0):                             # sliding TRAIN mask (+persistent cols)
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    allowed = (k <= q) & (k > q - W)
    if nP: allowed = allowed | (k < nP)
    m = torch.zeros(T, T, device=dev, dtype=dt); m.masked_fill_(~allowed, float("-inf")); return m[None, None]


def emask(kind, T, W, S, dev, dt, nP=0):                    # EVAL masks (+always-kept persistent cols)
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    causal = k <= q
    if kind == "full":        allowed = causal
    elif kind == "sliding":   allowed = causal & (k > q - W)
    elif kind == "streaming": allowed = causal & ((k < S) | (k > q - W))
    if nP: allowed = allowed | (k < nP)
    m = torch.zeros(T, T, device=dev, dtype=dt); m.masked_fill_(~allowed, float("-inf")); return m[None, None]


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)
H = model.get_input_embeddings().weight.shape[1]
emb = model.get_input_embeddings()


def pack(split, n, T):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= T:
            out.append(buf[:T]); buf = buf[T:]
            if len(out) >= n: return out
    return out


def pack_hist(split, n, T, Hist):                          # (history[Hist], content[T]) contiguous-stream pairs
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= Hist + T:
            out.append((buf[:Hist], buf[Hist:Hist + T])); buf = buf[T:]
            if len(out) >= n: return out
    return out


# ---- trainable prompt for f / g ----
prompt = None
if a.scheme == "warmup":
    P = max(1, W - 1)
    prompt = torch.nn.Parameter(torch.randn(P, H, device=dev, dtype=torch.float32) * 0.02)
elif a.scheme == "persistent":
    P = a.nP
    prompt = torch.nn.Parameter(torch.randn(P, H, device=dev, dtype=torch.float32) * 0.02)
nPk = a.nP if a.scheme == "persistent" else 0              # eval always-keep count

model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
params = list(model.parameters()) + ([prompt] if prompt is not None else [])
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95))

if a.scheme == "sliding_history":
    seqs = pack_hist("train", a.n_seq, L, max(1, W - 1))
else:
    seqs = pack("train", a.n_seq, L)
print("[%s W=%d] %d train seqs" % (a.scheme, W, len(seqs)), flush=True)

for step in range(a.steps):
    item = seqs[step % len(seqs)]
    if a.scheme == "sliding_history":
        hist, content = item; nh = len(hist)
        ids = torch.tensor([hist + content], device=dev)
        lg = model(input_ids=ids, attention_mask=wmask(nh + L, W, dev, bf16)).logits[0][nh:]
    else:
        content = item
        ids = torch.tensor([content], device=dev)
        inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], dim=1)      # (1, P+L, H)
        lg = model(inputs_embeds=inp, attention_mask=wmask(P + L, W, dev, bf16, nP=nPk)).logits[0][P:]
    tgt = torch.tensor(content[1:], device=dev)
    loss = F.cross_entropy(lg[:-1].float(), tgt)
    loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
    if step % 100 == 0: print("  step %d/%d loss %.3f" % (step, a.steps, loss.item()), flush=True)

model.eval()
ev = pack("test", a.n_eval, L)
def ppl(kind):
    nll = 0.0; ntok = 0
    for s in ev:
        ids = torch.tensor([s], device=dev)
        with torch.no_grad():
            if a.scheme == "sliding_history":
                lg = model(input_ids=ids, attention_mask=emask(kind, L, W, S, dev, bf16)).logits[0]; off = 0
            else:
                inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], dim=1)
                lg = model(inputs_embeds=inp, attention_mask=emask(kind, P + L, W, S, dev, bf16, nP=nPk)).logits[0]
                off = P
        tgt = ids[0, L - sl:L]; pred = lg[off + L - sl - 1:off + L - 1].float()
        nll += F.cross_entropy(pred, tgt, reduction="sum").item(); ntok += sl
    return math.exp(nll / ntok)
print("RESULT cpt scheme=%s W=%d | full %.3f | sliding@%d %.3f | streaming(s%d+%d) %.3f"
      % (a.scheme, W, ppl("full"), W, ppl("sliding"), S, W, ppl("streaming")), flush=True)
print("CPTREG_DONE", flush=True)
