"""Base-model attention sink (real attention of untrained Qwen2.5-0.5B on a wikitext sample): a 2D map (left)
+ attention-received-per-position (right). For §1."""
import torch, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
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
    if len(buf) >= 96:
        break
ids = torch.tensor([buf[:96]], device=dev)
m = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
with torch.no_grad():
    out = m(input_ids=ids, output_attentions=True, use_cache=False)
A = torch.stack([l[0].float().mean(0) for l in out.attentions]).mean(0).cpu().numpy()
T = A.shape[0]
recv = np.array([A[k + 1:, k].mean() if k + 1 < T else 0.0 for k in range(T)])

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7))
Am = np.where(A > 1e-6, A, np.nan)
cmap = plt.cm.magma_r.copy(); cmap.set_bad("white")
ax[0].imshow(Am, cmap=cmap, norm=PowerNorm(0.45, vmin=0, vmax=float(np.nanpercentile(A, 99.5))), aspect="equal", interpolation="nearest")
ax[0].axvline(0, color="#c0392b", lw=1.3)
ax[0].set_title("base-model attention map (avg heads + layers)", fontsize=11)
ax[0].set_xlabel("attends to (key) →"); ax[0].set_ylabel("predicting token ↓")
ax[1].bar(range(T), recv, width=1.0, color="#4a78b5")
ax[1].bar([0], [recv[0]], width=1.8, color="#c0392b")
ax[1].text(3, recv[0] * 0.9, f"sink: token 1 receives {recv[0]:.2f}", color="#c0392b", fontsize=9)
ax[1].set_title("attention received per position", fontsize=11)
ax[1].set_xlabel("key position"); ax[1].set_ylabel("mean attention received")
plt.tight_layout()
plt.savefig(EXP + "/base_sink.png", dpi=145, bbox_inches="tight"); print("wrote base_sink.png"); print("BASESINK_DONE")
