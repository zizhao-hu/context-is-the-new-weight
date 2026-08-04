"""For each model, the attention-RECEIVED distribution across key positions (mean over later queries, heads,
late layers) on the same wikitext chunk. Shows WHERE attention concentrates: base/causal spike on the content's
first token (pos 0); the filled schemes spike on the prefix (pos 0 of the prompt/history block) and stay flat
at the first content token (red dashed line)."""
import torch, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def profile(path, mode):
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
    A = torch.stack([l[0].float().mean(0) for l in out.attentions[16:]]).mean(0).cpu().numpy()  # avg heads+late layers
    T = A.shape[0]
    recv = np.array([A[k + 1:, k].mean() if k + 1 < T else 0.0 for k in range(T)])   # mean attn received per key
    del m, out; torch.cuda.empty_cache()
    return recv, pfx


specs = [("base", base_id, "plain"), ("causal (full)", EXP + "/models/full", "plain"),
         ("sliding with history", EXP + "/models/sliding_history", "history"),
         ("sliding with trainable tokens", EXP + "/models/startup", "startup")]
profs = [(n,) + profile(p, mo) for n, p, mo in specs]
mx = max(r.max() for _, r, _ in profs)
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for i, (name, recv, pfx) in enumerate(profs):
    a = ax[i // 2][i % 2]
    a.bar(range(len(recv)), recv, width=1.0, color="#4a78b5")
    a.bar([0], [recv[0]], width=1.6, color="#c0392b")                     # the input's first token (the sink)
    if pfx > 0:
        a.axvline(pfx, color="#1b7a3d", ls="--", lw=1.6)
        a.text(pfx + 4, mx * 0.82, "first content token\n(no sink)", color="#1b7a3d", fontsize=8.5)
        a.text(0, recv[0] + mx * 0.02, "sink on prefix", color="#c0392b", fontsize=8, ha="left")
    else:
        a.text(0, recv[0] + mx * 0.02, "sink on first content token", color="#c0392b", fontsize=8, ha="left")
    a.set_title(f"{name}   (T={len(recv)})", fontsize=11)
    a.set_xlabel("key position"); a.set_ylabel("mean attention received")
    a.set_ylim(0, mx * 1.08)
plt.tight_layout()
out = EXP + "/attn_dist.png"
plt.savefig(out, dpi=135, bbox_inches="tight"); print("wrote", out)

# ---- overlay: all four aligned at the content start (x=0 = first content token; x<0 = prefix) ----
cols = {"base": "#888", "causal (full)": "#c0392b", "sliding with history": "#16a085",
        "sliding with trainable tokens": "#2e6da4"}
fig2, a2 = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
for name, recv, pfx in profs:
    xs = np.arange(len(recv)) - pfx
    a2[0].plot(xs, recv, color=cols[name], lw=1.4, label=name)
    a2[1].plot(xs, recv, color=cols[name], lw=1.4, label=name)
a2[0].set_xlim(-258, -248); a2[0].set_title("zoom: prefix start (input's first token)", fontsize=10)
a2[0].axvline(-255, color="#aaa", ls=":", lw=1)
a2[1].set_xlim(-12, 64); a2[1].axvline(0, color="#1b7a3d", ls="--", lw=1.6)
a2[1].text(1.5, mx * 0.7, "first content token", color="#1b7a3d", fontsize=9)
a2[1].set_title("zoom: content region (from the first content token)", fontsize=10)
for a in a2:
    a.set_xlabel("key position relative to first content token"); a.set_ylabel("mean attention received"); a.set_ylim(0, mx * 1.08)
a2[1].legend(fontsize=8.5, loc="upper right")
fig2.suptitle("Attention-received distribution, aligned at content start — one spike, just relocated off the content", fontsize=12)
plt.tight_layout()
out2 = EXP + "/attn_dist_overlay.png"
plt.savefig(out2, dpi=135, bbox_inches="tight"); print("wrote", out2)
