"""windowed + startup schemes for the GATED (gla) 1.3B model, to mirror the softmax transformer. gla has no
attention window (it is an O(1)-state recurrence), so we bound its memory by RESETTING the recurrent state every
W tokens = processing the sequence in independent W-token blocks (block-local recurrence). 'windowed' = block
reset; 'startup' = block reset + a learned warm-up prompt prepended to EVERY block (initializes the cold state,
the recurrent analog of the cold-start prompt). One job = one matrix row: train at --train_window, eval block
perplexity at every deploy window. Tests whether train-small-deploy-anywhere appears in gla too."""
import torch, argparse, math
import fla
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
GLA = "fla-hub/gla-1.3B-100B"
ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["windowed", "startup"])
ap.add_argument("--train_window", type=int, default=256); ap.add_argument("--nP", type=int, default=64)
ap.add_argument("--deploy", default="128,256,512,1024,0")
ap.add_argument("--steps", type=int, default=800); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--n_seq", type=int, default=2000)
ap.add_argument("--n_eval", type=int, default=48); ap.add_argument("--eval_len", type=int, default=2048)
ap.add_argument("--tag", required=True)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(GLA)
model = AutoModelForCausalLM.from_pretrained(GLA, dtype=bf16).to(dev)
emb = model.get_input_embeddings(); EOS = tok.eos_token_id or 0
prompt = None
if a.scheme == "startup":
    prompt = torch.nn.Parameter(torch.randn(a.nP, model.config.hidden_size, device=dev, dtype=torch.float32) * 0.02)


def packseqs(split, n, L):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n: return out
    return out


def block_logits(ids, W):
    """logits for [1,L] under W-block-reset recurrence (fresh state every W tokens); prompt warms each block."""
    L = ids.shape[1]
    if W is None or W <= 0 or W >= L:                       # full recurrence (1 block)
        blocks = ids
    else:
        pad = (W - L % W) % W
        idp = F.pad(ids, (0, pad), value=EOS) if pad else ids
        blocks = idp.view(idp.shape[1] // W, W)             # [nb, W]
    if prompt is not None:
        pe = prompt.to(bf16).unsqueeze(0).expand(blocks.shape[0], -1, -1)
        lg = model(inputs_embeds=torch.cat([pe, emb(blocks)], 1)).logits[:, a.nP:]
    else:
        lg = model(blocks).logits
    return lg, blocks                                       # [nb, W, V], [nb, W]


def block_ce(ids, W, reduce_sum=False):
    lg, blocks = block_logits(ids, W)
    p = lg[:, :-1].float().reshape(-1, lg.shape[-1]); t = blocks[:, 1:].reshape(-1)
    return F.cross_entropy(p, t, reduction="sum" if reduce_sum else "mean"), t.numel()


# ---- train (block-windowed CPT at train_window) ----
model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
opt = torch.optim.AdamW(list(model.parameters()) + ([prompt] if prompt is not None else []), lr=a.lr, betas=(0.9, 0.95))
tr = packseqs("train", a.n_seq, a.ctx)
for s in range(a.steps):
    ids = torch.tensor([tr[s % len(tr)]], device=dev)
    loss, _ = block_ce(ids, a.train_window)
    loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + ([prompt] if prompt is not None else []), 1.0)
    opt.step(); opt.zero_grad()
    if s % 200 == 0: print("  step %d loss %.3f" % (s, loss.item()), flush=True)
model.gradient_checkpointing_disable(); model.eval()
ev = packseqs("test", a.n_eval, a.eval_len); out = []
for W in [int(x) for x in a.deploy.split(",")]:
    tot = 0.0; ntok = 0
    for sq in ev:
        ids = torch.tensor([sq], device=dev)
        with torch.no_grad():
            ce, nt = block_ce(ids, W, reduce_sum=True)
        tot += ce.item(); ntok += nt
    out.append("deploy%s %.2f" % ("full" if W == 0 else str(W), math.exp(tot / ntok)))
print("RESULT glamatrix %s | %s" % (a.tag, " | ".join(out)), flush=True)
print("GLASCHEME_DONE", flush=True)
