"""Abs-position-0 sink rate under FULL attention vs under the WINDOWED mask (the real sliding-window deployment).
Under the window, position 0 is masked for every query past W, so it cannot be attended — the deployment sink
rate should collapse. Metric = fraction of heads (all layers) whose mean attention to abs pos 0 (over queries
1..L-1) exceeds 0.3."""
import torch, os, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dev = "cuda"
EXP = os.path.dirname(os.path.abspath(__file__))
base_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(base_id)
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
buf = []
for row in ds.select(range(50000, 52000)):
    if row["text"].strip():
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
    if len(buf) >= 512 * 16:
        break
L, W, THRESH, NEVAL = 512, 256, 0.3, 16
seqs = [buf[i * L:(i + 1) * L] for i in range(NEVAL)]


def wmask(T):
    q = torch.arange(T, device=dev)[:, None]; k = torch.arange(T, device=dev)[None, :]
    m = torch.zeros(T, T, device=dev); m.masked_fill_(~((k <= q) & (k > q - W)), float("-inf")); return m[None, None]


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()


def sink_rate(path, windowed, startup=False):
    m = load(path); heads = []
    pr = None
    if startup:
        pr = torch.load(os.path.join(path, "startup_prompt.pt")).to(dev); emb = m.get_input_embeddings()
    for seq in seqs:
        ids = torch.tensor([seq], device=dev)
        with torch.no_grad():
            if startup:
                inp = torch.cat([pr.to(torch.bfloat16).unsqueeze(0), emb(ids)], dim=1)
                out = m(inputs_embeds=inp, attention_mask=wmask(pr.shape[0] + L), output_attentions=True, use_cache=False)
            else:
                out = m(input_ids=ids, attention_mask=(wmask(L) if windowed else None), output_attentions=True, use_cache=False)
        for layer in out.attentions:
            A = layer[0].float()                              # (H, T, T)
            heads.append(A[:, 1:, 0].mean(1).cpu())           # mean attention to ABS pos 0 over later queries
    del m; torch.cuda.empty_cache()
    return (torch.cat(heads) > THRESH).float().mean().item() * 100


print(f"{'model':16s} {'full-attn':>10s} {'windowed mask (deploy)':>24s}")
for name, sub, st in [("base", None, False), ("windowed_hard", "models/windowed_hard", False), ("startup_hard", "models/startup_hard", True)]:
    path = base_id if sub is None else os.path.join(EXP, sub)
    full = "  n/a" if st else f"{sink_rate(path, False, False):6.1f}%"
    win = f"{sink_rate(path, True, st):6.1f}%"
    print(f"{name:16s} {full:>10s} {win:>24s}", flush=True)
print("SINKWIN_DONE")
