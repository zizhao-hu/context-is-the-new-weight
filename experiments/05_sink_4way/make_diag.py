"""Diagnostic: did the fine-tunes actually change the model, or are they ≈ base? Compares each trained model
to the base on (a) relative weight change, (b) output-distribution KL on a sample, (c) top-1 next-token agreement."""
import torch, os
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
    if len(buf) >= 256:
        break
ids = torch.tensor([buf[:256]], device=dev)


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, attn_implementation="eager").to(dev).eval()


base = load(base_id)
with torch.no_grad():
    lb = base(input_ids=ids).logits[0].log_softmax(-1)
print(f"{'model':20s} rel_weight_diff   KL(m||base)   top1_agree")
for name, path in [("full (causal)", EXP + "/models/full"), ("startup", EXP + "/models/startup"),
                   ("sliding_history", EXP + "/models/sliding_history"), ("full_alpaca", EXP + "/models/full_alpaca")]:
    m = load(path)
    dsum = psum = 0.0
    for (_, p1), (_, p2) in zip(base.named_parameters(), m.named_parameters()):
        dsum += (p1 - p2).abs().sum().item(); psum += p1.abs().sum().item()
    with torch.no_grad():
        lm = m(input_ids=ids).logits[0].log_softmax(-1)
    kl = (lm.exp() * (lm - lb)).sum(-1).mean().item()
    agree = (lb.argmax(-1) == lm.argmax(-1)).float().mean().item()
    print(f"{name:20s} {dsum/psum:.3e}      {kl:.4f}      {agree:.3f}", flush=True)
    del m; torch.cuda.empty_cache()
print("DIAG_DONE")
