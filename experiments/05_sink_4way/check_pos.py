"""Is the model's output invariant to absolute position (RoPE relativity)? Feed the SAME 64-token window with
position_ids shifted by different offsets; compare the last-token logits. If identical -> natural positions ==
per-window reset (the window's first token IS the effective 0). If they drift -> the reset matters."""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dev = "cuda"
base_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(base_id)
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
buf = []
for row in ds:
    if row["text"].strip():
        buf.extend(tok(row["text"], add_special_tokens=False).input_ids)
    if len(buf) >= 64:
        break
ids = torch.tensor([buf[:64]], device=dev)
m = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32, attn_implementation="eager").to(dev).eval()


def last_logits(offset):
    pos = torch.arange(offset, offset + 64, device=dev).unsqueeze(0)
    with torch.no_grad():
        return m(input_ids=ids, position_ids=pos).logits[0, -1]


l0 = last_logits(0)
print("offset   max|logit-diff vs offset0|   top1_same   KL(p_off||p_0)")
for off in [0, 64, 256, 512, 960]:
    l = last_logits(off)
    d = (l - l0).abs().max().item()
    kl = F.kl_div(l.log_softmax(-1), l0.softmax(-1), reduction="sum").item()
    print(f"{off:6d}   {d:>10.5f}                {str((l.argmax()==l0.argmax()).item()):>5}      {kl:.6f}", flush=True)
print("CHECKPOS_DONE")
