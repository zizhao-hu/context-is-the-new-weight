"""base vs trained, three schemes, each pair evaluated under ITS OWN deployment:
  causal   : full attention            (base vs full_hard)
  windowed : windowed mask W=256       (base vs windowed_hard)
  startup  : windowed mask + prompts   (base+prompts vs startup_hard, same trained prompts)
Two metrics on the first token (abs pos 0): mean attention, and sink rate (% heads mean>0.3). 16 held-out seqs, all layers, L=512."""
import torch, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dev = "cuda"
EXP = os.path.dirname(os.path.abspath(__file__))
base_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(base_id)
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
L, W, THRESH, NEVAL = 512, 256, 0.3, 16
buf = []
for row in ds.select(range(50000, 52000)):
    if row["text"].strip():
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
    if len(buf) >= L * NEVAL:
        break
seqs = [buf[i * L:(i + 1) * L] for i in range(NEVAL)]
SP = os.path.join(EXP, "models/startup_hard/startup_prompt.pt")


def wmask(T):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~((k <= q) & (k > q - W)), float("-inf")); return m[None, None]


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()


def measure(path, mode):
    m = load(path); heads = []
    pr = torch.load(SP).to(dev) if mode == "startup" else None
    emb = m.get_input_embeddings() if mode == "startup" else None
    for seq in seqs:
        ids = torch.tensor([seq], device=dev)
        with torch.no_grad():
            if mode == "startup":
                inp = torch.cat([pr.to(torch.bfloat16).unsqueeze(0), emb(ids)], dim=1)
                out = m(inputs_embeds=inp, attention_mask=wmask(pr.shape[0] + L), output_attentions=True, use_cache=False)
            elif mode == "windowed":
                out = m(input_ids=ids, attention_mask=wmask(L), output_attentions=True, use_cache=False)
            else:
                out = m(input_ids=ids, output_attentions=True, use_cache=False)
        for layer in out.attentions:
            heads.append(layer[0].float()[:, 1:, 0].mean(1).cpu())      # attention to abs pos 0
    del m; torch.cuda.empty_cache()
    allh = torch.cat(heads)
    return allh.mean().item(), (allh > THRESH).float().mean().item() * 100


groups = [
    ("causal",   measure(base_id, "full"),     measure(os.path.join(EXP, "models/full_hard"), "full")),
    ("windowed", measure(base_id, "windowed"), measure(os.path.join(EXP, "models/windowed_hard"), "windowed")),
    ("startup",  measure(base_id, "startup"),  measure(os.path.join(EXP, "models/startup_hard"), "startup")),
]
for g, b, t in groups:
    print(f"{g:9s}  base: mean={b[0]:.3f} sink={b[1]:.1f}%   trained: mean={t[0]:.3f} sink={t[1]:.1f}%", flush=True)

x = np.arange(3); wb = 0.36
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
for i, (idx, title, fmt) in enumerate([(0, "mean attention to first token", "%.2f"), (1, "sink rate  (% heads, mean attn > 0.3)", "%.0f%%")]):
    bvals = [g[1][idx] for g in groups]; tvals = [g[2][idx] for g in groups]
    ax[i].bar(x - wb / 2, bvals, wb, label="base", color="#9ecae1", edgecolor="#444", lw=.4)
    ax[i].bar(x + wb / 2, tvals, wb, label="trained", color="#08519c", edgecolor="#444", lw=.4)
    for xi, (bv, tv) in enumerate(zip(bvals, tvals)):
        ax[i].text(xi - wb / 2, bv, fmt % bv, ha="center", va="bottom", fontsize=8.5)
        ax[i].text(xi + wb / 2, tv, fmt % tv, ha="center", va="bottom", fontsize=8.5)
    ax[i].set_xticks(x); ax[i].set_xticklabels([g[0] for g in groups]); ax[i].set_title(title, fontsize=11); ax[i].legend(fontsize=9)
fig.suptitle("base vs trained, each under its own deployment  ·  causal = full attention · windowed = windowed mask · startup = windowed + prompts  ·  sink at first token (abs pos 0)", fontsize=9.5)
plt.tight_layout()
plt.savefig(EXP + "/pairs_bars.png", dpi=145, bbox_inches="tight"); print("wrote pairs_bars.png"); print("PAIRS_DONE")
