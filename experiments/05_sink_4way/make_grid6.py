"""2x3 attention grid. Rows = content (predicted) tokens; x: context/prompts at NEGATIVE positions, 0 = first
content token. Cols: causal (no context) / history FULL / startup-soft-prompt FULL. On the two right panels a
BLACK parallelogram outlines the sliding window W=64 (whole triangle = full attention, inside box = windowed
deployment). Green box = predicted content tokens. TOP = base, BOTTOM = trained. The startup column uses the
trained soft-prompts (startup_64) as the prefix; base+prompts vs startup-trained."""
import torch, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle, Polygon
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dev = "cuda"
EXP = os.path.dirname(os.path.abspath(__file__))
base_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(base_id)
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
HIST, CL, W = 63, 128, 64
buf = []
for row in ds:
    if row["text"].strip():
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
    if len(buf) >= HIST + CL:
        break
hist, content = buf[:HIST], buf[HIST:HIST + CL]
SP = torch.load(os.path.join(EXP, "models/startup_64/startup_prompt.pt"), map_location="cpu")[:HIST]   # (63, H) trained soft-prompts


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()


def amap(m, mode):
    emb = m.get_input_embeddings()
    with torch.no_grad():
        if mode == "causal":
            out = m(input_ids=torch.tensor([content], device=dev), output_attentions=True, use_cache=False); pfx = 0
        elif mode == "hist":
            out = m(input_ids=torch.tensor([hist + content], device=dev), output_attentions=True, use_cache=False); pfx = HIST
        else:  # startup soft-prompts as the prefix
            inp = torch.cat([SP.to(dev).to(torch.bfloat16).unsqueeze(0), emb(torch.tensor([content], device=dev))], dim=1)
            out = m(inputs_embeds=inp, output_attentions=True, use_cache=False); pfx = HIST
    A = torch.stack([l[0].float().mean(0) for l in out.attentions]).mean(0).cpu().numpy()[pfx:, :]
    return np.where(A > 1e-6, A, np.nan), pfx


P = [[None] * 3 for _ in range(2)]
mb = load(base_id)
P[0][0] = ("base · causal (no context)", *amap(mb, "causal"))
P[0][1] = ("base + history · full", *amap(mb, "hist"))
P[0][2] = ("base + startup-prompts · full", *amap(mb, "startup"))
del mb; torch.cuda.empty_cache()
mf = load(os.path.join(EXP, "models/full_64")); P[1][0] = ("causal · trained", *amap(mf, "causal")); del mf; torch.cuda.empty_cache()
ms = load(os.path.join(EXP, "models/sliding_history_64")); P[1][1] = ("sliding-history · trained · full", *amap(ms, "hist")); del ms; torch.cuda.empty_cache()
mt = load(os.path.join(EXP, "models/startup_64")); P[1][2] = ("startup · trained · full", *amap(mt, "startup")); del mt; torch.cuda.empty_cache()

vmax = max(np.nanpercentile(P[r][c][1], 99.5) for r in range(2) for c in range(3))
cmap = plt.cm.magma_r.copy(); cmap.set_bad("white")
fig, ax = plt.subplots(2, 3, figsize=(16, 8.2), gridspec_kw={"width_ratios": [CL, HIST + CL, HIST + CL]})
for r in range(2):
    for c in range(3):
        name, A, pfx = P[r][c]; a = ax[r][c]
        a.imshow(A, cmap=cmap, norm=PowerNorm(0.45, vmin=0, vmax=vmax), aspect="equal", interpolation="nearest",
                 extent=[-pfx - .5, CL - .5, CL - .5, -.5])
        a.add_patch(Rectangle((-.5, -.5), CL, CL, fill=False, edgecolor="#1b7a3d", lw=1.3))
        if pfx > 0:
            a.axvline(-.5, color="#1b7a3d", lw=0.8, ls="--")
            a.add_patch(Polygon([(-W + .5, -.5), (.5, -.5), (CL - .5, CL - .5), (CL - W - .5, CL - .5)],
                                closed=True, fill=False, edgecolor="black", lw=1.7))
        a.set_title(name, fontsize=9)
        a.set_xlabel("attends to  (prefix < 0,  0 = first content) →", fontsize=7); a.set_ylabel("predicted token ↓", fontsize=7)
fig.suptitle("TOP = base · BOTTOM = trained.  Cols: causal · history (real prior text) · startup (trainable soft-prompts).  Whole triangle = full attention; inside black box = sliding window W=64.", fontsize=8.6)
plt.tight_layout()
plt.savefig(EXP + "/grid6.png", dpi=140, bbox_inches="tight"); print("wrote grid6.png"); print("GRID6_DONE")
