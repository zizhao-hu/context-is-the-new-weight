"""fla-hub 1.3B-100B architecture suite under a bounded memory budget (eval-only, no finetuning). Every model
shares the GPT skeleton + 100B training tokens; only the token-mixer differs (softmax / gla / gsa / delta_net /
hgrn2 / mamba / retnet). We bound memory UNIFORMLY for every architecture by processing the sequence in
independent W-token blocks (fresh state / no cross-block context = a 'chunked LM'), so the column is a fair
cross-architecture budget. full = one block (native). Held-out wikitext perplexity. One job = one row."""
import torch, argparse, math
import fla
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--budgets", default="128,256,512,1024,0")
ap.add_argument("--n_eval", type=int, default=48); ap.add_argument("--eval_len", type=int, default=2048)
ap.add_argument("--tag", required=True)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=bf16).to(dev).eval()
EOS = tok.eos_token_id or 0
nattn = len([m for m in model.modules() if m.__class__.__name__ == "Attention"])
print("model %s | softmax-attn layers %d" % (a.model.split('/')[-1], nattn), flush=True)


def packseqs(split, n, L):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split); buf = []; out = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= L:
            out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n: return out
    return out


def block_ce(ids, W):
    L = ids.shape[1]
    if W is None or W <= 0 or W >= L:
        blocks = ids
    else:
        pad = (W - L % W) % W
        idp = F.pad(ids, (0, pad), value=EOS) if pad else ids
        blocks = idp.view(idp.shape[1] // W, W)
    with torch.no_grad():
        lg = model(blocks).logits
    p = lg[:, :-1].float().reshape(-1, lg.shape[-1]); t = blocks[:, 1:].reshape(-1)
    return F.cross_entropy(p, t, reduction="sum").item(), t.numel()


ev = packseqs("test", a.n_eval, a.eval_len); out = []
for W in [int(x) for x in a.budgets.split(",")]:
    tot = 0.0; ntok = 0
    for sq in ev:
        ce, nt = block_ce(torch.tensor([sq], device=dev), W); tot += ce; ntok += nt
    out.append("budget%s %.2f" % ("full" if W == 0 else str(W), math.exp(tot / ntok)))
print("RESULT suite %s | attn=%d | %s" % (a.tag, nattn, " | ".join(out)), flush=True)
print("SUITE_DONE", flush=True)
