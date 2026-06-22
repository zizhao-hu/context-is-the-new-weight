"""HotpotQA downstream, FAIR mask-only control + STANDARD metric. Every variant is continued-pretrained (CPT) on
the SAME wikitext for the SAME steps -- only the attention mask differs (triangle=full-causal, windowed, persist)
-- so QA-ability drift is held constant and the comparison isolates the mask. base = no CPT (reference); we also
score base under a post-hoc StreamingLLM mask. Metrics: SQuAD-style generation EM/F1 (the real downstream metric)
AND teacher-forced answer-NLL (sensitive proxy). One process: train (if CPT) then eval."""
import torch, argparse, math, re, string, collections
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "Qwen/Qwen2.5-0.5B"
ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["base", "triangle", "windowed", "persist"])
ap.add_argument("--window", type=int, default=256); ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--nP", type=int, default=64); ap.add_argument("--ctx", type=int, default=1536)
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--n_seq", type=int, default=2000); ap.add_argument("--n_eval", type=int, default=200)
ap.add_argument("--maxnew", type=int, default=12)
a = ap.parse_args(); dev = "cuda"; bf16 = torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)
emb = model.get_input_embeddings(); EOS = tok.eos_token_id


def cmask(kind, L, nP, dev, dt):
    q = torch.arange(L, device=dev)[:, None]; k = torch.arange(L, device=dev)[None, :]; c = k <= q
    al = {"full": c, "windowed": c & (k > q - a.window), "streaming": c & ((k < a.sink) | (k > q - a.window)),
          "persist": c & ((k < nP) | (k > q - a.window))}[kind]
    m = torch.zeros(L, L, device=dev, dtype=dt); m.masked_fill_(~al, float("-inf")); return m[None, None]


def norm(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    com = sum((collections.Counter(p) & collections.Counter(g)).values())
    if com == 0 or not p or not g: return 0.0
    pr, rc = com / len(p), com / len(g); return 2 * pr * rc / (pr + rc)


prompt = None
if a.scheme != "base":
    if a.scheme == "persist":
        prompt = torch.nn.Parameter(torch.randn(a.nP, model.config.hidden_size, device=dev, dtype=torch.float32) * 0.02)
    trainmask = {"triangle": "full", "windowed": "windowed", "persist": "persist"}[a.scheme]
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
        ids = torch.tensor([sq[s % len(sq)]], device=dev); P = a.nP if prompt is not None else 0
        if prompt is not None:
            inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], 1)
            lg = model(inputs_embeds=inp, attention_mask=cmask("persist", P + a.ctx, a.nP, dev, bf16)).logits[0][P:]
        else:
            lg = model(ids, attention_mask=cmask(trainmask, a.ctx, 0, dev, bf16)).logits[0]
        loss = F.cross_entropy(lg[:-1].float(), ids[0, 1:]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        if s % 150 == 0: print("  step %d loss %.3f" % (s, loss.item()), flush=True)
model.eval()
lb = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))[:a.n_eval]


def build(ex):
    ctx = tok("\n".join("".join(p) for p in ex["context"]["sentences"]), add_special_tokens=False).input_ids
    q = tok("\n\nAnswer the question with a short span.\nQuestion: " + ex["question"] + "\nAnswer:", add_special_tokens=False).input_ids
    ans = tok(" " + ex["answer"], add_special_tokens=False).input_ids[:a.maxnew]
    return ctx, q, ans


def deploy_mask(kind, L, nP):
    return cmask(kind, L, nP, dev, bf16)


@torch.no_grad()
def evaluate(kind):
    tf1 = 0.0; tem = 0.0; tnll = 0.0; tk = 0; N = 0
    for ex in lb:
        ctx, q, ans = build(ex)
        if not ans: continue
        N += 1; nP = a.nP if kind == "persist" else 0
        cap = a.ctx - nP
        base_ids = (ctx + q)[-cap:]                                   # context window for the prompt
        # ---- teacher-forced answer NLL ----
        ids = torch.tensor([(base_ids + ans)[-(a.ctx - nP):]], device=dev); L = ids.shape[1]; kp = L - len(ans)
        if prompt is not None and kind == "persist":
            inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(ids)], 1); Lf = nP + L
            lg = model(inputs_embeds=inp, attention_mask=deploy_mask("persist", Lf, nP)).logits[0][nP:]
        else:
            lg = model(ids, attention_mask=deploy_mask(kind, L, 0)).logits[0]
        tnll += F.cross_entropy(lg[kp - 1:L - 1].float(), ids[0, kp:L], reduction="sum").item(); tk += len(ans)
        # ---- greedy generation EM/F1 ----
        cur = base_ids[:]
        for _ in range(a.maxnew):
            body = cur[-(a.ctx - nP):]; x = torch.tensor([body], device=dev); L2 = len(body)
            if prompt is not None and kind == "persist":
                inp = torch.cat([prompt.to(bf16).unsqueeze(0), emb(x)], 1)
                nl = model(inputs_embeds=inp, attention_mask=deploy_mask("persist", nP + L2, nP)).logits[0, -1]
            else:
                nl = model(x, attention_mask=deploy_mask(kind, L2, 0)).logits[0, -1]
            nt = int(nl.argmax())
            if nt == EOS or nt == tok.convert_tokens_to_ids("\n"): break
            cur.append(nt)
        pred = tok.decode(cur[len(base_ids):]).strip()
        gold = ex["answer"]; tf1 += f1(pred, gold); tem += float(norm(pred) == norm(gold))
    print("RESULT hotpotfair %s | deploy=%-9s | F1 %.3f | EM %.3f | ans-ppl %.2f (N=%d)" %
          (a.scheme, kind, tf1 / N, tem / N, math.exp(tnll / tk), N), flush=True)


kinds = {"base": ["full", "windowed", "streaming"], "triangle": ["full", "windowed", "streaming"],
         "windowed": ["windowed"], "persist": ["persist"]}[a.scheme]
for kd in kinds:
    evaluate(kd)
print("HOTPOTFAIR_DONE", flush=True)
