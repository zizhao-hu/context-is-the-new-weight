"""Continual-learning benchmark for the attention-sink paper: full-causal vs SWA vs T-SWA
continued pretraining over a 4-task sequence, against simple continual-learning baselines.

Tasks (all offline HF caches on CARC):
  T1 wikitext   wikitext-103-raw-v1 train        (encyclopedic prose)
  T2 gsm8k      openai/gsm8k main train          (math word problems, Q/A text)
  T3 tofu       locuslab/tofu forget10           (fictitious-author facts, Q/A text; 360/40 split)
  T4 arc        allenai/ai2_arc ARC-Challenge    (science QA text)
  R  fineweb    one parquet of fineweb sample    (never trained: general-language retention probe)

One job = one (mask, method) config trained sequentially T1->T2->T3->T4 at a matched step
budget per stage, with per-task held-out perplexity measured after every stage.

Masks (Qwen2.5-0.5B, additive 4D attention mask, L=1024 chunks):
  a  full causal
  b  SWA, sliding window W=256
  t  T-SWA: same sliding mask as b, loss only on queries with a full window (q >= W)

Methods (replay = ER, ewc = online EWC, lwf = LwF, the traditional continual-learning trio):
  naive   sequential AdamW, nothing else
  replay  each micro-batch drawn from a uniformly sampled PREVIOUS task with prob --replay_p
  l2      L2-SP to the stage-start weights: loss += lam/2 * ||theta - theta_stage_start||^2
  ewc     online EWC: diagonal Fisher estimated on each finished task (mean-normalized,
          accumulated), penalty lam/2 * sum F * (theta - anchor)^2, anchor = last stage end
  lwf     learning without forgetting: KL to the frozen previous-stage model on the
          current batch, loss += alpha * KL(teacher || student)

Deploy for eval is matched to the design (a: full causal; b,t: sliding W), and NLL is scored
only on positions >= W under every deploy so each scored position has at least W context.
Machine-readable output: CLEVAL stage=<0..4> task=<name> ppl=<x>; CLDONE at the end.
"""
import argparse, math, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True, choices=["a", "b", "t"])
ap.add_argument("--method", required=True, choices=["naive", "replay", "l2", "ewc", "lwf"])
ap.add_argument("--W", type=int, default=256)
ap.add_argument("--L", type=int, default=1024)
ap.add_argument("--steps", type=int, default=250)          # optimizer steps per stage
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--replay_p", type=float, default=0.2)
ap.add_argument("--l2_lam", type=float, default=0.01)
ap.add_argument("--ewc_lam", type=float, default=1.0)      # on mean-normalized Fisher
ap.add_argument("--ewc_batches", type=int, default=24)     # Fisher estimation batches
ap.add_argument("--lwf_alpha", type=float, default=1.0)    # KD weight
ap.add_argument("--n_eval", type=int, default=24)          # eval chunks per task
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fineweb", default="/scratch1/zizhaoh/fineweb/sample/100BT/001_00005.parquet")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--hybrid", action="store_true")           # freeze all but full-attention layers; 8-bit AdamW + grad ckpt
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
dev, bf16 = "cuda", torch.bfloat16
W, L = a.W, a.L

MODEL = a.model
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=bf16, attn_implementation="sdpa",
                                             trust_remote_code=True).to(dev)
if a.hybrid:
    # train only the full-(softmax-)attention decoder layers of the hybrid; freeze the rest
    lt = getattr(model.config, "layer_types", None)
    if lt:
        full_idx = [i for i, t in enumerate(lt) if "full" in t]
    else:
        full_idx = [i for i, l in enumerate(model.model.layers)
                    if "linear" not in type(getattr(l, "self_attn", l)).__name__.lower()
                    and not hasattr(l, "linear_attn")]
    print("HYBRID full-attention layers:", full_idx, flush=True)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for i in full_idx:
        for p_ in model.model.layers[i].parameters():
            p_.requires_grad_(True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
else:
    model.gradient_checkpointing_disable()
TRAINABLE = [p_ for p_ in model.parameters() if p_.requires_grad]
print("trainable params: %.2fB / %.2fB" % (sum(p_.numel() for p_ in TRAINABLE) / 1e9,
      sum(p_.numel() for p_ in model.parameters()) / 1e9), flush=True)

# ---------------- data: one flat token stream per task ----------------
def stream(texts, cap_tokens=6_000_000):
    ids = []
    for t in texts:
        if not t or not t.strip():
            continue
        ids.extend(tok(t + "\n\n", add_special_tokens=False)["input_ids"])
        if len(ids) >= cap_tokens:
            break
    return torch.tensor(ids[:cap_tokens], dtype=torch.long)

def build_tasks():
    wt = load_dataset("wikitext", "wikitext-103-raw-v1")
    t1_tr = stream([r["text"] for r in wt["train"]][:60000])
    t1_va = stream([r["text"] for r in wt["validation"]], 200_000)

    gs = load_dataset("openai/gsm8k", "main")
    ser_g = lambda r: "Q: %s\nA: %s" % (r["question"], r["answer"])
    t2_tr = stream([ser_g(r) for r in gs["train"]])
    t2_va = stream([ser_g(r) for r in gs["test"]], 200_000)

    tf = load_dataset("locuslab/tofu", "forget10")["train"]
    ser_t = lambda r: "Q: %s\nA: %s" % (r["question"], r["answer"])
    rows = [ser_t(r) for r in tf]
    t3_tr = stream(rows[:360], 2_000_000)
    t3_va = stream(rows[360:], 200_000)

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    def ser_a(r):
        ch = " ".join("%s) %s" % (l, x) for l, x in zip(r["choices"]["label"], r["choices"]["text"]))
        return "Question: %s %s Answer: %s" % (r["question"], ch, r["answerKey"])
    t4_tr = stream([ser_a(r) for r in arc["train"]])
    t4_va = stream([ser_a(r) for r in arc["validation"]], 200_000)

    import pyarrow.parquet as pq                     # direct read: no datasets cache, no FileLock
    fw_texts = pq.read_table(a.fineweb, columns=["text"])["text"].to_pylist()[:1500]
    r_va = stream(fw_texts, 200_000)

    tasks = [("wikitext", t1_tr, t1_va), ("gsm8k", t2_tr, t2_va),
             ("tofu", t3_tr, t3_va), ("arc", t4_tr, t4_va)]
    for n, tr, va in tasks:
        print("TASK %s train_toks=%d val_toks=%d" % (n, len(tr), len(va)), flush=True)
    print("TASK fineweb val_toks=%d" % len(r_va), flush=True)
    return tasks, ("fineweb", r_va)

tasks, probe = build_tasks()

# ---------------- masks ----------------
def make_mask(kind):
    q = torch.arange(L)[:, None]; k = torch.arange(L)[None, :]
    if kind == "full":
        keep = k <= q
    else:
        keep = (k <= q) & (k > q - W)
    m = torch.zeros(L, L, dtype=bf16)
    m[~keep] = float("-inf")
    return m[None, None].to(dev)

TRAIN_MASK = make_mask("full" if a.mask == "a" else "slide")
DEPLOY_MASK = TRAIN_MASK           # matched deploy: a -> full, b/t -> sliding
loss_from = W if a.mask == "t" else 0

def batch_from(tr, bs):
    off = torch.randint(0, len(tr) - L - 1, (bs,))
    x = torch.stack([tr[o:o + L] for o in off])
    y = torch.stack([tr[o + 1:o + L + 1] for o in off])
    return x.to(dev), y.to(dev)

def lm_loss(x, y, mask, from_pos):
    out = model(input_ids=x, attention_mask=mask.expand(x.size(0), -1, -1, -1)).logits
    lp = F.log_softmax(out.float(), dim=-1)
    nll = -lp.gather(-1, y[..., None]).squeeze(-1)
    return nll[:, from_pos:].mean()

@torch.no_grad()
def eval_ppl(va):
    model.eval()
    tot, n = 0.0, 0
    for i in range(a.n_eval):
        o = i * L
        if o + L + 1 > len(va):
            break
        x = va[o:o + L][None].to(dev); y = va[o + 1:o + L + 1][None].to(dev)
        out = model(input_ids=x, attention_mask=DEPLOY_MASK).logits
        lp = F.log_softmax(out.float(), dim=-1)
        nll = -lp.gather(-1, y[..., None]).squeeze(-1)[:, W:]      # >= W context under every deploy
        tot += nll.sum().item(); n += nll.numel()
    model.train()
    return math.exp(tot / max(n, 1))

def eval_all(stage):
    for name, _, va in tasks:
        print("CLEVAL stage=%d task=%s ppl=%.4f" % (stage, name, eval_ppl(va)), flush=True)
    print("CLEVAL stage=%d task=%s ppl=%.4f" % (stage, probe[0], eval_ppl(probe[1])), flush=True)

# ---------------- sequential training ----------------
def fisher_diag(tr):
    """diagonal Fisher of the LM loss on task tr, mean-normalized"""
    F = [torch.zeros_like(p, dtype=torch.float32) for p in TRAINABLE]
    model.train()
    for _ in range(a.ewc_batches):
        model.zero_grad(set_to_none=True)
        x, y = batch_from(tr, a.bs)
        lm_loss(x, y, TRAIN_MASK, loss_from).backward()
        for f, p in zip(F, TRAINABLE):
            if p.grad is not None:
                f += p.grad.float() ** 2
    model.zero_grad(set_to_none=True)
    tot = sum(f.sum() for f in F); n = sum(f.numel() for f in F)
    scale = (tot / n).clamp_min(1e-12)
    return [f / scale for f in F]                             # mean 1

if a.hybrid:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(TRAINABLE, lr=a.lr, weight_decay=0.0)
else:
    opt = torch.optim.AdamW(TRAINABLE, lr=a.lr, weight_decay=0.0)
eval_all(0)                                                   # base model row
ewc_F, ewc_anchor, teacher = None, None, None
for s, (name, tr, _) in enumerate(tasks, start=1):
    anchor = None
    if a.method == "l2":
        anchor = [p.detach().clone() for p in TRAINABLE]
    if a.method == "lwf" and s > 1:
        import copy
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if a.hybrid:
            # frozen layers never train: alias them to the student's storage to halve memory
            sp = dict(model.named_parameters())
            for n_, tp in teacher.named_parameters():
                if not sp[n_].requires_grad:
                    tp.data = sp[n_].data
    model.train()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            src = tr
            if a.method == "replay" and s > 1 and random.random() < a.replay_p:
                src = tasks[random.randrange(s - 1)][1]
            x, y = batch_from(src, a.bs)
            if a.method == "lwf" and teacher is not None:
                out = model(input_ids=x, attention_mask=TRAIN_MASK.expand(x.size(0), -1, -1, -1)).logits
                lp = F.log_softmax(out.float(), dim=-1)
                nll = -lp.gather(-1, y[..., None]).squeeze(-1)[:, loss_from:].mean()
                with torch.no_grad():
                    tl = teacher(input_ids=x, attention_mask=TRAIN_MASK.expand(x.size(0), -1, -1, -1)).logits
                kd = F.kl_div(lp[:, loss_from:], F.log_softmax(tl.float(), dim=-1)[:, loss_from:],
                              log_target=True, reduction="batchmean") / (L - loss_from)
                loss = nll + a.lwf_alpha * kd
            else:
                loss = lm_loss(x, y, TRAIN_MASK, loss_from)
            if a.method == "l2":
                reg = sum(((p - q0) ** 2).sum() for p, q0 in zip(TRAINABLE, anchor))
                loss = loss + a.l2_lam * 0.5 * reg / 1e6      # scaled per million params
            if a.method == "ewc" and ewc_F is not None:
                reg = sum((f * (p - q0) ** 2).sum()
                          for f, p, q0 in zip(ewc_F, TRAINABLE, ewc_anchor))
                loss = loss + a.ewc_lam * 0.5 * reg
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(TRAINABLE, 1.0)
        opt.step()
        if step % 50 == 0:
            print("STAGE %d (%s) step %d loss %.4f" % (s, name, step, loss.item()), flush=True)
    if a.method == "ewc":
        Fn = fisher_diag(tr)
        ewc_F = Fn if ewc_F is None else [f0 + f1 for f0, f1 in zip(ewc_F, Fn)]
        ewc_anchor = [p.detach().clone() for p in TRAINABLE]
    eval_all(s)
print("CLDONE mask=%s method=%s" % (a.mask, a.method), flush=True)
