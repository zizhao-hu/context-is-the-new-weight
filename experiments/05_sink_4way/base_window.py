"""CORRECT sliding window: each window is its OWN fresh length-W sequence (positions 0..W-1) — there is no
global absolute token 0, it does not exist in the computation. So the window's first token IS position 0 (the
sequence start), and we check whether the base model sinks onto it (the fair, sliding register). Left: the
fresh-window attention map (sink = bright column at position 0 = window-first). Right: attention received per
in-window position. Averaged over 32 fresh windows, all heads, all layers."""
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
W, NWIN = 64, 32
buf = []
for row in ds:
    if row["text"].strip():
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
    if len(buf) >= W * NWIN:
        break
windows = [buf[i * W:(i + 1) * W] for i in range(NWIN)]          # each a SEPARATE fresh sequence
ids = torch.tensor(windows, device=dev)                          # (NWIN, W); each row gets positions 0..W-1
m = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
with torch.no_grad():
    out = m(input_ids=ids, output_attentions=True, use_cache=False)   # ordinary causal attention, per fresh window
A = torch.stack([l.float().mean(1).mean(0) for l in out.attentions]).mean(0).cpu().numpy()   # avg heads, windows, layers -> (W,W)
recv = np.array([A[k + 1:, k].mean() if k + 1 < W else 0.0 for k in range(W)])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
Am = np.where(A > 1e-6, A, np.nan)
cmap = plt.cm.magma_r.copy(); cmap.set_bad("white")
ax[0].imshow(Am, cmap=cmap, norm=PowerNorm(0.45, vmin=0, vmax=float(np.nanpercentile(A, 99.5))), aspect="equal", interpolation="nearest")
ax[0].axvline(0, color="#c0392b", lw=1.3)
ax[0].set_title("base · FRESH window W=64\n(each window its own sequence, positions 0..63; no global token 0)", fontsize=10.5)
ax[0].set_xlabel("attends to (key);  0 = window's first →"); ax[0].set_ylabel("predicting token in window ↓")
ax[1].bar(range(W), recv, width=1.0, color="#4a78b5")
ax[1].bar([0], [recv[0]], width=1.8, color="#c0392b")
ax[1].text(2, recv[0] * 0.9, f"sink on window's first: {recv[0]:.2f}", color="#c0392b", fontsize=9)
ax[1].set_title("attention received per in-window position", fontsize=10.5)
ax[1].set_xlabel("position in window (0 = window's first / oldest)"); ax[1].set_ylabel("mean attention received")
plt.tight_layout()
plt.savefig(EXP + "/base_window.png", dpi=145, bbox_inches="tight"); print("wrote base_window.png"); print("BASEWIN_DONE")
