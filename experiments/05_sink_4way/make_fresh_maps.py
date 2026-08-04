"""§5 attention maps, CORRECT (fresh-window) form. Base model. LEFT: ordinary full-causal attention — the sink
is pinned to the fixed first token (column 0), and context grows. RIGHT: a fresh sliding window W=64 — each
predicting token's window is computed as its OWN length-W sequence (positions 0..W-1, no global token 0), and
its row is placed back at absolute columns [q-W+1 .. q]; the sink rides the window's first/oldest token (the
band's left edge), sliding forward — a constant-width, position-fair register. Avg heads + all layers."""
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
    if len(buf) >= 200:
        break
L, W = 128, 64
m = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()

# LEFT — full causal over L tokens
with torch.no_grad():
    of = m(input_ids=torch.tensor([buf[:L]], device=dev), output_attentions=True, use_cache=False)
Af = torch.stack([l[0].float().mean(0) for l in of.attentions]).mean(0).cpu().numpy()      # (L,L)

# RIGHT — each predicting token q gets a FRESH length-W window [q-W+1 .. q] (positions 0..W-1)
wins = [buf[q - W + 1:q + 1] for q in range(W - 1, L)]
ids = torch.tensor(wins, device=dev)                                                        # (nwin, W)
with torch.no_grad():
    ow = m(input_ids=ids, output_attentions=True, use_cache=False)
Aw = torch.stack([l[:, :, -1, :].float().mean(1) for l in ow.attentions]).mean(0).cpu().numpy()   # (nwin, W) last-token attn
P = np.full((L, L), np.nan)
for i, q in enumerate(range(W - 1, L)):
    P[q, q - W + 1:q + 1] = Aw[i]                                                            # place row back at absolute columns

cmap = plt.cm.magma_r.copy(); cmap.set_bad("white")
vmax = float(np.nanpercentile(np.concatenate([Af[Af > 1e-6], P[~np.isnan(P)]]), 99.5))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
ax[0].imshow(np.where(Af > 1e-6, Af, np.nan), cmap=cmap, norm=PowerNorm(0.45, vmin=0, vmax=vmax), aspect="equal", interpolation="nearest")
ax[0].axvline(0, color="#c0392b", lw=1.2)
ax[0].set_title("causal (full attention)\nsink pinned to the FIXED first token (column 0); context grows", fontsize=10.5)
ax[0].set_xlabel("attends to (key) →"); ax[0].set_ylabel("predicting token ↓")
ax[1].imshow(P, cmap=cmap, norm=PowerNorm(0.45, vmin=0, vmax=vmax), aspect="equal", interpolation="nearest")
ax[1].set_title("fresh sliding window W=64\nsink RIDES the window's first token (band's left edge); constant width", fontsize=10.5)
ax[1].set_xlabel("attends to (key) →"); ax[1].set_ylabel("predicting token ↓")
fig.suptitle("Base model. Each window computed fresh (positions 0..W-1, no global token 0) — the sink slides with the window's oldest token instead of pinning to a fixed first token.", fontsize=10)
plt.tight_layout()
plt.savefig(EXP + "/fresh_maps.png", dpi=145, bbox_inches="tight"); print("wrote fresh_maps.png"); print("FRESH_DONE")
