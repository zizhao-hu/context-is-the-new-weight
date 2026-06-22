"""Cross-architecture attention-pattern comparison (gated vs our windowed on softmax). The fla softmax Attention
is flash-attn-only (no attention weights) and gla is linear (no L x L matrix), so we compare PATTERNS via a
gradient influence map: infl[i,j] = || d h_L[i] / d x_j || (how much input token j drives the layer-L hidden at
position i). Works identically for flash-attn softmax and linear gla. We compute it at FULL attention so it
reflects what the model INTRINSICALLY uses, not the inference mask. Hypothesis: windowed-trained softmax becomes
gla-like (local, decaying, no position-0 sink), unlike the base softmax (sink at j=0)."""
import torch, argparse, numpy as np
import fla
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--scheme", default="base", choices=["base", "windowed"])  # windowed => windowed CPT first
ap.add_argument("--window", type=int, default=256)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--L", type=int, default=512)
ap.add_argument("--n_seq", type=int, default=6)
ap.add_argument("--n_pos", type=int, default=10)      # output positions sampled per seq
ap.add_argument("--n_proj", type=int, default=3)      # random projections per position
ap.add_argument("--out", required=True)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=bf16).to(dev)
attns = [m for m in model.modules() if m.__class__.__name__ == "Attention" and hasattr(m, "window_size")]
print("softmax-attn layers: %d" % len(attns), flush=True)


def packseqs(split, n, L):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n: return out
    return out


if a.scheme == "windowed" and attns:
    for m in attns: m.window_size = a.window           # sliding window via flash-attn during training
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95)); tr = packseqs("train", a.n_seq * 200, a.L)
    import torch.nn.functional as F
    for s in range(a.steps):
        ids = torch.tensor([tr[s % len(tr)]], device=dev)
        lg = model(ids).logits[0]
        loss = F.cross_entropy(lg[:-1].float(), ids[0, 1:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        if s % 150 == 0: print("  step %d loss %.3f" % (s, loss.item()), flush=True)
    model.gradient_checkpointing_disable()
# influence at FULL attention (intrinsic pattern, not the inference mask)
for m in attns: m.window_size = None
model.eval()
emb_layer = model.get_input_embeddings()
infl = np.zeros((a.L, a.L)); cnt = np.zeros((a.L, a.L))
ev = packseqs("test", a.n_seq, a.L)
torch.manual_seed(0)
positions = torch.linspace(8, a.L - 1, a.n_pos).round().long().tolist()
for s in ev:
    ids = torch.tensor([s], device=dev)
    x = emb_layer(ids).detach().clone().requires_grad_(True)
    out = model(inputs_embeds=x, output_hidden_states=True)
    h = out.hidden_states[-1][0].float()               # [L, D]
    for i in positions:
        for _ in range(a.n_proj):
            r = torch.randn(h.shape[-1], device=dev)
            if x.grad is not None: x.grad = None
            (h[i] * r).sum().backward(retain_graph=True)
            g = x.grad[0, :i + 1].float().norm(dim=-1).cpu().numpy()  # ||d/dx_j||
            infl[i, :i + 1] += g; cnt[i, :i + 1] += 1
infl = np.divide(infl, cnt, out=np.zeros_like(infl), where=cnt > 0)
# row-normalise each sampled output position to a distribution over sources
rows = sorted(set(positions))
M = infl[rows]; M = M / (M.sum(1, keepdims=True) + 1e-9)
np.savez(a.out, infl=M, rows=np.array(rows), model=a.model.split('/')[-1], scheme=a.scheme, n_attn=len(attns))
print("RESULT influence saved %s | rows=%s | row0_share@j0=%.4f mean@j0=%.4f" %
      (a.out, len(rows), M[-1, 0], M[:, 0].mean()), flush=True)
print("INFL_DONE", flush=True)
