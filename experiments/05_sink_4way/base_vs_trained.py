"""Same wikitext example, same sliding-with-history setup (63-token real history prefix + 64 content, windowed
W=64). Compare the UNTRAINED base model vs the trained sliding_history_64 — and their difference — to isolate
what the windowed+history training changed in the attention (beyond the structural prefix effect). Rows =
content-query attention only; 0 = first content token; non-attended white."""
import torch, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dev = "cuda"
EXP = os.path.dirname(os.path.abspath(__file__))
base_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(base_id)
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
buf = []
for row in ds:
    if row["text"].strip():
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
    if len(buf) >= 63 + 64:
        break
HIST, CL, W = 63, 64, 64
hist, content = buf[:HIST], buf[HIST:HIST + CL]


def wmask(T):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~((k <= q) & (k > q - W)), float("-inf")); return m[None, None]


def amap(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    with torch.no_grad():
        out = m(input_ids=torch.tensor([hist + content], device=dev), attention_mask=wmask(HIST + CL), output_attentions=True, use_cache=False)
    A = torch.stack([l[0].float().mean(0) for l in out.attentions]).mean(0).cpu().numpy()[HIST:, :]   # content queries
    del m, out; torch.cuda.empty_cache()
    return A


Ab, At = amap(base_id), amap(EXP + "/models/sliding_history_64")
nrow, ncol = Ab.shape
ext = [-HIST - .5, ncol - HIST - .5, nrow - .5, -.5]
cmap = plt.cm.magma_r.copy(); cmap.set_bad("white")
vmax = max(np.nanmax(np.where(Ab > 1e-6, Ab, np.nan)), np.nanmax(np.where(At > 1e-6, At, np.nan)))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for a, (name, A) in zip(ax[:2], [("BEFORE — untrained base", Ab), ("AFTER — windowed+history trained", At)]):
    a.imshow(np.where(A > 1e-6, A, np.nan), cmap=cmap, norm=PowerNorm(0.5, vmin=0, vmax=vmax), aspect="equal", interpolation="nearest", extent=ext)
    a.axvline(-.5, color="#1b7a3d", lw=1.4); a.text(1, 3, "0 = first content token", color="#1b7a3d", fontsize=8)
    a.set_title(f"sliding-with-history setup\n{name}", fontsize=10)
    a.set_xlabel("attends to → (0 = first content token)", fontsize=8); a.set_ylabel("content token ↓", fontsize=8)
D = np.where((np.tril(np.ones_like(At), k=HIST) > 0), At - Ab, np.nan)                # difference where attended
dmax = np.nanpercentile(np.abs(D), 99)
ax[2].imshow(D, cmap="RdBu_r", norm=TwoSlopeNorm(0, -dmax, dmax), aspect="equal", interpolation="nearest", extent=ext)
ax[2].axvline(-.5, color="#1b7a3d", lw=1.4)
ax[2].set_title("difference (trained − base)\nred = trained attends MORE, blue = LESS", fontsize=10)
ax[2].set_xlabel("attends to → (0 = first content token)", fontsize=8); ax[2].set_ylabel("content token ↓", fontsize=8)
plt.tight_layout()
plt.savefig(EXP + "/base_vs_trained.png", dpi=140, bbox_inches="tight"); print("wrote base_vs_trained.png"); print("BVT_DONE")
