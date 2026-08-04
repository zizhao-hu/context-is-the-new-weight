"""Visualize the REAL attention (avg over heads, late layers) of the SAME wikitext chunk on the 4 models:
base, causal, sliding-with-history, sliding-with-trainable-tokens. Each model is fed its deployment input
(base/causal: content only; history: 255 real history + content; startup: 255 trainable prompts + content).
The cyan line marks the first CONTENT token — the sink sits on the first token of the *input*, so for base/causal
that's the content's first token, but for the filled schemes it's relocated onto the prefix."""
import torch, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    if len(buf) >= 255 + 64:
        break
HIST, CL = 255, 64
hist, content = buf[:HIST], buf[HIST:HIST + CL]


def amap(path, mode):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    emb = m.get_input_embeddings()
    with torch.no_grad():
        if mode == "startup":
            prompt = torch.load(os.path.join(path, "startup_prompt.pt")).to(dev)
            inp = torch.cat([prompt.to(torch.bfloat16).unsqueeze(0), emb(torch.tensor([content], device=dev))], dim=1)
            out = m(inputs_embeds=inp, output_attentions=True, use_cache=False); pfx = prompt.shape[0]
        elif mode == "history":
            out = m(input_ids=torch.tensor([hist + content], device=dev), output_attentions=True, use_cache=False); pfx = HIST
        else:
            out = m(input_ids=torch.tensor([content], device=dev), output_attentions=True, use_cache=False); pfx = 0
    sel = out.attentions[16:]                                # late layers (sink is strongest there)
    A = torch.stack([l[0].float().mean(0) for l in sel]).mean(0).cpu().numpy()   # avg heads + late layers
    del m, out; torch.cuda.empty_cache()
    return A, pfx


specs = [("base", base_id, "plain"), ("causal (full)", EXP + "/models/full", "plain"),
         ("sliding with history", EXP + "/models/sliding_history", "history"),
         ("sliding with trainable tokens", EXP + "/models/startup", "startup")]
fig, ax = plt.subplots(2, 2, figsize=(12, 11))
for i, (name, path, mode) in enumerate(specs):
    A, pfx = amap(path, mode)
    a = ax[i // 2][i % 2]
    a.imshow(A, cmap="magma", norm=PowerNorm(0.45, vmin=0, vmax=0.35), aspect="equal", interpolation="nearest")
    if pfx > 0:
        a.axvline(pfx - 0.5, color="cyan", lw=1.2); a.axhline(pfx - 0.5, color="cyan", lw=1.2)
        a.text(pfx + 2, 6, "first content\ntoken", color="cyan", fontsize=8, va="top")
    else:
        a.text(1.5, 6, "first token\n(= content)", color="cyan", fontsize=8, va="top")
    a.set_title(f"{name}   (T={A.shape[0]}, prefix={pfx})", fontsize=11)
    a.set_xlabel("attends to (key) →", fontsize=9); a.set_ylabel("query ↓", fontsize=9)
plt.tight_layout()
out = EXP + "/attn_maps.png"
plt.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
