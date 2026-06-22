"""Qwen2.5-0.5B HotpotQA under bounded working memory (aim 3b + persist fix). Schemes: base / windowed /
persist (nP learned always-on registers + windowed content = a TRAINED StreamingLLM). Context truncated to ctx,
scored as answer-NLL under the scheme's deployment. Tests whether learned global registers (persist) close the
windowed-deployment gap to base+StreamingLLM on far-retrieval QA. Train+eval one process."""
import torch, argparse, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "Qwen/Qwen2.5-0.5B"
ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["base", "windowed", "persist"])
ap.add_argument("--task", default="hotpotqa"); ap.add_argument("--window", type=int, default=256)
ap.add_argument("--sink", type=int, default=4); ap.add_argument("--nP", type=int, default=64)
ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--n_seq", type=int, default=2000)
ap.add_argument("--n_eval", type=int, default=400)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)
emb = model.get_input_embeddings()


def cmask(kind, L, W, S, nP, dev, dt):
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]; c = k <= q
    a_ = {"full": c, "windowed": c & (k > q - W), "streaming": c & ((k < S) | (k > q - W)),
          "persist": c & ((k < nP) | (k > q - W))}[kind]
    m = torch.zeros(L, L, device=dev, dtype=dt); m.masked_fill_(~a_, float("-inf")); return m[None, None]


prompt = None
if a.scheme == "persist":
    prompt = torch.nn.Parameter(torch.randn(a.nP, model.config.hidden_size, device=dev, dtype=torch.float32) * 0.02)
if a.scheme in ("windowed", "persist"):
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    opt = torch.optim.AdamW(list(model.parameters()) + ([prompt] if prompt is not None else []), lr=a.lr, betas=(0.9, 0.95))
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"); buf = []; sq = []
    for r in ds:
        if not r["text"].strip(): continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= a.ctx:
            sq.append(buf[:a.ctx]); buf = buf[a.ctx:]
            if len(sq) >= a.n_seq: break
        if len(sq) >= a.n_seq: break
    for s in range(a.steps):
        ids = torch.tensor([sq[s % len(sq)]], device=dev)
        if prompt is not None:
            inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], 1)
            lg = model(inputs_embeds=inp, attention_mask=cmask("persist", a.nP + a.ctx, a.window, a.sink, a.nP, dev, bf16)).logits[0][a.nP:]
        else:
            lg = model(ids, attention_mask=cmask("windowed", a.ctx, a.window, a.sink, a.nP, dev, bf16)).logits[0]
        loss = F.cross_entropy(lg[:-1].float(), ids[0, 1:]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        if s % 100 == 0: print("  step %d loss %.3f" % (s, loss.item()), flush=True)
model.eval()
lb = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
def ans_nll(kind, ex):
    ctx = tok("\n".join("".join(p) for p in ex["context"]["sentences"]), add_special_tokens=False).input_ids[-(a.ctx - 256):]
    p = tok("\n\nAnswer the question based on the passages.\nQuestion: " + ex["question"] + "\nAnswer:", add_special_tokens=False).input_ids
    ans = tok(" " + ex["answer"], add_special_tokens=False).input_ids[:24]
    if not ans: return None
    body = ctx + p + ans
    if kind == "persist":
        body = body[-(a.ctx - a.nP):]; ids = torch.tensor([body], device=dev); L = len(body); kp = L - len(ans)
        inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], 1); Lf = a.nP + L
        with torch.no_grad():
            lg = model(inputs_embeds=inp, attention_mask=cmask("persist", Lf, a.window, a.sink, a.nP, dev, bf16)).logits[0][a.nP:]
    else:
        body = body[-a.ctx:]; ids = torch.tensor([body], device=dev); L = len(body); kp = L - len(ans)
        with torch.no_grad():
            lg = model(ids, attention_mask=cmask(kind, L, a.window, a.sink, a.nP, dev, bf16)).logits[0]
    return F.cross_entropy(lg[kp - 1:L - 1].float(), ids[0, kp:L], reduction="sum").item(), len(ans)
kinds = ["full", "persist"] if a.scheme == "persist" else ["full", "windowed", "streaming"]
for kind in kinds:
    tn = 0.0; tk = 0
    for ex in list(lb)[:a.n_eval]:
        r = ans_nll(kind, ex)
        if r: tn += r[0]; tk += r[1]
    print("RESULT longbench-%s %s | %s | answer-ppl %.3f (n=%d)" % (a.task, a.scheme, kind, math.exp(tn / tk), tk), flush=True)
print("LONGBENCH_DONE", flush=True)
