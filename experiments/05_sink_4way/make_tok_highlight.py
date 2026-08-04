"""Token-level attention, SAME example as the heatmaps (63 history + 128 content, W=64). Compute attention
RECEIVED per token from the content (predicting) queries within the window, for BASE and TRAINED. Dump tokens +
values as JSON; the page renders them as flowing colored text (highlighter), not boxes."""
import torch, os, numpy as np, json
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
seq = buf[:HIST + CL]; T = HIST + CL


def wmask(n):
    q = torch.arange(n, device=dev)[:, None]; k = torch.arange(n, device=dev)[None, :]
    m = torch.zeros(n, n, device=dev); m.masked_fill_(~((k <= q) & (k > q - W)), float("-inf")); return m[None, None]


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()


def recv_dist(path):
    m = load(path)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([seq], device=dev), attention_mask=wmask(T), output_attentions=True, use_cache=False)
    A = torch.stack([l[0].float().mean(0) for l in out.attentions]).mean(0).cpu().numpy()
    rec = np.array([np.mean([A[q, k] for q in range(max(k + 1, HIST), min(k + W, T))])
                    if min(k + W, T) > max(k + 1, HIST) else 0.0 for k in range(T)])
    del m; torch.cuda.empty_cache()
    return rec


rb = recv_dist(base_id)
rt = recv_dist(os.path.join(EXP, "models/sliding_history_64"))
toks = [tok.decode([t]) for t in seq]
json.dump({"tokens": toks, "base": rb.tolist(), "trained": rt.tolist(), "HIST": HIST,
           "vmax": float(max(rb.max(), rt.max()))}, open(os.path.join(EXP, "tok_highlight.json"), "w"))
print("wrote tok_highlight.json"); print("TOKHL_DONE")
