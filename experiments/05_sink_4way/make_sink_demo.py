"""Demonstrate the sink-rate bars with the actual attention. LEFT: the two causal attention maps (base, trained)
— the bright first column IS the sink. RIGHT: the sink-rate bars across schemes, colored by the SAME magma scale
as the maps (dark = high sink), so the bars match the maps' style and show what they quantify."""
import torch, os, numpy as np, json
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
CL = 80
buf = []
for row in ds:
    if row["text"].strip():
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
    if len(buf) >= CL:
        break
content = buf[:CL]


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()


def cmap_attn(m):
    with torch.no_grad():
        out = m(input_ids=torch.tensor([content], device=dev), output_attentions=True, use_cache=False)
    A = torch.stack([l[0].float().mean(0) for l in out.attentions]).mean(0).cpu().numpy()
    return np.where(A > 1e-6, A, np.nan)


mb = load(base_id); Ab = cmap_attn(mb); del mb; torch.cuda.empty_cache()
mf = load(os.path.join(EXP, "models/full_hard")); Af = cmap_attn(mf); del mf; torch.cuda.empty_cache()


def sr(name):
    return json.load(open(os.path.join(EXP, "sink", name + ".json")))["sink_rate_pos1"] * 100


rates = [("base", sr("base")), ("causal", sr("full_hard")), ("windowed", sr("windowed_hard")), ("startup", sr("startup_hard"))]
print("rates:", rates, flush=True)

mcmap = plt.cm.magma_r.copy(); mcmap.set_bad("white")
vmax = max(np.nanpercentile(Ab, 99.5), np.nanpercentile(Af, 99.5))
fig = plt.figure(figsize=(12, 4.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.5], hspace=0.4, wspace=0.18)
for r, (A, lab) in enumerate([(Ab, "base · causal attention"), (Af, "trained · causal attention")]):
    ax = fig.add_subplot(gs[r, 0])
    ax.imshow(A, cmap=mcmap, norm=PowerNorm(0.45, vmin=0, vmax=vmax), aspect="equal", interpolation="nearest")
    ax.axvline(0, color="#1b7a3d", lw=1.3)
    ax.set_title(lab, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel("query ↓", fontsize=7); ax.set_xlabel("attends to →", fontsize=7)
axb = fig.add_subplot(gs[:, 1])
names = [n for n, _ in rates]; vals = [v for _, v in rates]
cols = [plt.cm.magma_r(0.22 + 0.72 * min(v, 60) / 60) for v in vals]      # value -> same magma scale as the maps
bars = axb.bar(names, vals, color=cols, edgecolor="#333", lw=0.5, width=0.6)
for b, v in zip(bars, vals):
    axb.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
axb.set_ylabel("attention-sink rate\n(% of heads, mean attention to token 1 > 0.3)", fontsize=9)
axb.set_ylim(0, max(vals) * 1.2); axb.spines[["top", "right"]].set_visible(False); axb.tick_params(labelsize=10)
fig.suptitle("The bright first column in the attention map (left) IS the sink; the bars (right) quantify it across schemes — colored by the same scale.", fontsize=8.8, y=1.0)
plt.tight_layout()
plt.savefig(EXP + "/sink_demo.png", dpi=150, bbox_inches="tight"); print("wrote sink_demo.png"); print("DEMO_DONE")
