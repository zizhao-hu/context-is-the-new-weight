"""Matched-scale attention maps (real attention, avg heads+layers) for base + the 3 W=64/L=64 models, §3-style:
non-attended cells left WHITE. base/causal = full attention on 64-token content; sliding_history_64 and
startup_64 = windowed (W=64) over [63-token prefix ⊕ 64-token content], so the history/prompt prefix is visible
and you can see the sink relocate onto it. Content is 64 in every panel so the panels line up equally."""
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
    if len(buf) >= 63 + 64:
        break
HIST, CL, W = 63, 64, 64
hist, content = buf[:HIST], buf[HIST:HIST + CL]


def wmask(T):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev)
    m.masked_fill_(~((k <= q) & (k > q - W)), float("-inf"))
    return m[None, None]


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()


def amap(path, mode):
    m = load(path); emb = m.get_input_embeddings()
    with torch.no_grad():
        if mode == "startup":
            pr = torch.load(os.path.join(path, "startup_prompt.pt")).to(dev)[:HIST]
            inp = torch.cat([pr.to(torch.bfloat16).unsqueeze(0), emb(torch.tensor([content], device=dev))], dim=1)
            out = m(inputs_embeds=inp, attention_mask=wmask(HIST + CL), output_attentions=True, use_cache=False); pfx = HIST
        elif mode == "history":
            out = m(input_ids=torch.tensor([hist + content], device=dev), attention_mask=wmask(HIST + CL), output_attentions=True, use_cache=False); pfx = HIST
        else:
            out = m(input_ids=torch.tensor([content], device=dev), output_attentions=True, use_cache=False); pfx = 0
    A = torch.stack([l[0].float().mean(0) for l in out.attentions]).mean(0).cpu().numpy()
    T = A.shape[0]
    Am = np.where(A > 1e-6, A, np.nan)                                # non-attended (masked / ~0) -> white
    del m, out; torch.cuda.empty_cache()
    return Am, pfx


specs = [("base", base_id, "plain"), ("causal (full_64)", EXP + "/models/full_64", "plain"),
         ("sliding with history", EXP + "/models/sliding_history_64", "history"),
         ("startup (trainable prompt)", EXP + "/models/startup_64", "startup")]
maps = [(n,) + amap(p, mo) for n, p, mo in specs]
vmax = max(np.nanmax(A) for _, A, _ in maps)
cmap = plt.cm.magma_r.copy(); cmap.set_bad("white")
widths = [m[1].shape[0] for m in maps]
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2), gridspec_kw={"width_ratios": widths})
for a, (name, A, pfx) in zip(ax, maps):
    Acon = A[pfx:, :]                                                     # only CONTENT-query rows (drop the prefix's own attention)
    nrow, ncol = Acon.shape                                              # nrow = 64 content tokens; ncol = prefix+content
    a.imshow(Acon, cmap=cmap, norm=PowerNorm(0.5, vmin=0, vmax=vmax), aspect="equal", interpolation="nearest",
             extent=[-pfx - .5, ncol - pfx - .5, nrow - .5, -.5])         # x: 0 = first content token (prefix at <0); y: content queries
    if pfx > 0:
        a.axvline(-.5, color="#1b7a3d", lw=1.4)                           # prefix | content boundary at x=0
        a.text(1, 3, "0 = first content token", color="#1b7a3d", fontsize=7.5, va="top")
    a.set_title(f"{name}\n(content tokens; prefix={pfx} at x<0)", fontsize=9.5)
    a.set_xlabel("attends to →  (0 = first content token)", fontsize=8); a.set_ylabel("content token ↓", fontsize=8)
fig.suptitle("Real attention (avg heads+layers); non-attended = white. Content=64 in all panels; sliding/startup show the 63-token prefix.", fontsize=10)
plt.tight_layout()
out = EXP + "/attn_maps64.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)
