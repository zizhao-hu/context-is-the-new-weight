"""With-context comparison. Setup: 63-token real history prefix + 256 content, W=64. For base+history vs the
trained sliding_history_64, under FULL attention vs the WINDOWED mask, measure the sink at TWO locations:
- abs pos 0 = the history's first token (where a prefix relocates the sink), and
- pos 63 = the content's first token (the original content sink).
Metric = % of heads (all layers) whose mean attention to that token (over later queries) exceeds 0.3."""
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dev = "cuda"
EXP = os.path.dirname(os.path.abspath(__file__))
base_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(base_id)
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
HIST, CL, W, THRESH, NEVAL = 63, 256, 64, 0.3, 8
buf = []
for row in ds.select(range(50000, 52000)):
    if row["text"].strip():
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
    if len(buf) >= (HIST + CL) * NEVAL:
        break
seqs = [buf[i * (HIST + CL):(i + 1) * (HIST + CL)] for i in range(NEVAL)]


def wmask(T):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~((k <= q) & (k > q - W)), float("-inf")); return m[None, None]


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()


def measure(path, windowed):
    m = load(path); a0, ac = [], []
    for seq in seqs:
        ids = torch.tensor([seq], device=dev)
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=(wmask(HIST + CL) if windowed else None), output_attentions=True, use_cache=False)
        for layer in out.attentions:
            A = layer[0].float()
            a0.append(A[:, 1:, 0].mean(1).cpu())                 # sink at history token 0 (abs pos 0)
            ac.append(A[:, HIST + 1:, HIST].mean(1).cpu())       # sink at content token 0 (pos 63)
    del m; torch.cuda.empty_cache()
    return (torch.cat(a0) > THRESH).float().mean().item() * 100, (torch.cat(ac) > THRESH).float().mean().item() * 100


print("setup: 63 history + 256 content, W=64")
print(f"{'model':22s} {'attn':9s} {'sink@history-tok0':>18s} {'sink@content-tok0':>18s}")
for name, sub in [("base + history", None), ("sliding_history_64", "models/sliding_history_64")]:
    path = base_id if sub is None else os.path.join(EXP, sub)
    for windowed in [False, True]:
        h0, c0 = measure(path, windowed)
        print(f"{name:22s} {'windowed' if windowed else 'full':9s} {h0:16.1f}% {c0:17.1f}%", flush=True)
print("SINKCTX_DONE")
