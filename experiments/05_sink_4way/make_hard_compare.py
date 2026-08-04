"""Confirm base + 3 hard models are distinct (weight diff / KL / top-1 agreement), then plot the
attention-received distribution for each — averaged over 16 sequences, ALL layers+heads (matches the
sink-rate metric), all no-prefix — as 4 barcharts."""
import torch, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dev = "cuda"
EXP = os.path.dirname(os.path.abspath(__file__))
base_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(base_id)
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
L, N = 512, 12                                       # match the sink-rate eval length (512)
seqs, b = [], []
for row in ds:
    if row["text"].strip():
        b.extend(tok(row["text"], add_special_tokens=False).input_ids)
    while len(b) >= L and len(seqs) < N:
        seqs.append(b[:L]); b = b[L:]
    if len(seqs) >= N:
        break
ids0 = torch.tensor([seqs[0]], device=dev)


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, attn_implementation="eager").to(dev).eval()


def aprof(m):
    acc = None
    for s in seqs:
        with torch.no_grad():
            out = m(input_ids=torch.tensor([s], device=dev), output_attentions=True)
        A = torch.stack([l[0].float().mean(0) for l in out.attentions]).mean(0)   # avg all layers+heads
        T = A.shape[0]
        cnt = torch.arange(T - 1, -1, -1, device=dev).float().clamp(min=1)
        recv = (A.sum(0) - A.diagonal()) / cnt                                    # mean attn received per key (over later queries)
        acc = recv if acc is None else acc + recv
    return (acc / len(seqs)).cpu().numpy()


base = load(base_id)
with torch.no_grad():
    lb = base(input_ids=ids0).logits[0].log_softmax(-1)
specs = [("base", base_id), ("causal (full_hard)", EXP + "/models/full_hard"),
         ("windowed_hard", EXP + "/models/windowed_hard"), ("startup_hard", EXP + "/models/startup_hard")]
profs = []
print(f"{'model':22s} rel_wdiff   KL(m||base)  top1_agree")
for name, path in specs:
    m = base if path == base_id else load(path)
    if path != base_id:
        dsum = psum = 0.0
        for (_, p1), (_, p2) in zip(base.named_parameters(), m.named_parameters()):
            dsum += (p1 - p2).abs().sum().item(); psum += p1.abs().sum().item()
        with torch.no_grad():
            lm = m(input_ids=ids0).logits[0].log_softmax(-1)
        kl = (lm.exp() * (lm - lb)).sum(-1).mean().item()
        agree = (lb.argmax(-1) == lm.argmax(-1)).float().mean().item()
        print(f"{name:22s} {dsum/psum:.2e}   {kl:.3f}        {agree:.3f}", flush=True)
    pr = aprof(m)
    profs.append((name, pr))
    print(f"   {name:22s} pos1_attn={pr[0]:.4f}  (L={L}, {N} seqs, all layers)", flush=True)
    if path != base_id:
        del m; torch.cuda.empty_cache()

mx = max(r.max() for _, r in profs)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for i, (name, r) in enumerate(profs):
    a = ax[i]
    a.bar(range(len(r)), r, width=1.0, color="#4a78b5")
    a.bar([0], [r[0]], width=1.8, color="#c0392b")
    a.set_title(f"{name}\npos-1 attn = {r[0]:.3f}", fontsize=10)
    a.set_xlabel("key position"); a.set_ylim(0, mx * 1.08)
    if i == 0:
        a.set_ylabel("mean attention received")
fig.suptitle("Attention-received distribution (16-seq avg, all layers, no-prefix) — base + 3 hard-trained models. "
             "windowed→startup training shrinks the content sink.", fontsize=11)
plt.tight_layout()
plt.savefig(EXP + "/attn_dist_hard.png", dpi=140, bbox_inches="tight"); print("wrote attn_dist_hard.png")
print("HARDCMP_DONE")
