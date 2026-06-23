"""HotpotQA downstream on LLaMA-3.2-1B (all-softmax, vs Qwen3.5-9B hybrid). Continued-pretrain on wikitext under a
mask, then SQuAD-style greedy-generation F1/EM + teacher-forced answer-ppl, deployed under a constant W-window.
Schemes: base / triangle(full-causal) / windowed(sliding-history) / startup(trainable cold-start prompt, windowed
so it slides out) / uniwin(uniform context: fixed-length-W predict-last). base+triangle eval at W in {128,256,512}."""
import torch, argparse, math, re, string, collections, random
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "unsloth/Llama-3.2-1B"
ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["base","triangle","windowed","startup","uniwin"])
ap.add_argument("--window", type=int, default=256); ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--nP", type=int, default=64); ap.add_argument("--ctx", type=int, default=1536)
ap.add_argument("--steps", type=int, default=400); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--n_seq", type=int, default=2000); ap.add_argument("--n_eval", type=int, default=150)
ap.add_argument("--maxnew", type=int, default=12)
a = ap.parse_args(); dev="cuda"; bf16=torch.bfloat16; random.seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=bf16, attn_implementation="eager").to(dev)
emb = model.get_input_embeddings(); EOS = tok.eos_token_id; H = model.config.hidden_size; NL = tok.convert_tokens_to_ids("\n")

def cmask(kind, L, nP, W, dt):
    q=torch.arange(L,device=dev)[:,None]; k=torch.arange(L,device=dev)[None,:]; c=k<=q
    al={"full":c,"windowed":c&(k>q-W),"streaming":c&((k<a.sink)|(k>q-W)),"persist":c&((k<nP)|(k>q-W))}[kind]
    m=torch.zeros(L,L,device=dev,dtype=dt); m.masked_fill_(~al,float("-inf")); return m[None,None]
def norm(s):
    s="".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b"," ",s).split())
def f1(pred,gold):
    p,g=norm(pred).split(),norm(gold).split(); com=sum((collections.Counter(p)&collections.Counter(g)).values())
    if com==0 or not p or not g: return 0.0
    pr,rc=com/len(p),com/len(g); return 2*pr*rc/(pr+rc)
def pack(n,L):
    ds=load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train"); buf=[]; out=[]
    for r in ds:
        if not r["text"].strip(): continue
        buf+=tok(r["text"],add_special_tokens=False).input_ids
        while len(buf)>=L:
            out.append(buf[:L]); buf=buf[L:]
            if len(out)>=n: return out
    return out

prompt=None
if a.scheme!="base":
    model.gradient_checkpointing_enable(); model.config.use_cache=False; model.train(); params=list(model.parameters())
    if a.scheme=="startup":
        prompt=torch.nn.Parameter(torch.randn(a.nP,H,device=dev,dtype=torch.float32)*0.02); params+=[prompt]
    opt=torch.optim.AdamW(params,lr=a.lr,betas=(0.9,0.95))
    if a.scheme=="uniwin":
        stream=[t for sq in pack(a.n_seq,a.ctx) for t in sq]; W=a.window
        for s in range(a.steps):
            st=[random.randrange(0,len(stream)-W-1) for _ in range(16)]
            wins=torch.tensor([stream[j:j+W+1] for j in st],device=dev)
            loss=F.cross_entropy(model(wins[:,:-1]).logits[:,-1].float(),wins[:,-1])
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()
            if s%100==0: print("  step %d loss %.3f"%(s,loss.item()),flush=True)
    else:
        sqs=pack(a.n_seq,a.ctx); tm={"triangle":"full","windowed":"windowed","startup":"windowed"}[a.scheme]
        for s in range(a.steps):
            ids=torch.tensor([sqs[s%len(sqs)]],device=dev)
            if a.scheme=="startup":
                inp=torch.cat([prompt.to(bf16).unsqueeze(0),emb(ids)],1)
                lg=model(inputs_embeds=inp,attention_mask=cmask("windowed",a.nP+a.ctx,0,a.window,bf16)).logits[0][a.nP:]
            else:
                lg=model(ids,attention_mask=cmask(tm,a.ctx,0,a.window,bf16)).logits[0]
            loss=F.cross_entropy(lg[:-1].float(),ids[0,1:])
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad()
            if s%100==0: print("  step %d loss %.3f"%(s,loss.item()),flush=True)
model.eval(); model.config.use_cache=False
lb=list(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation"))[:a.n_eval]
def build(ex):
    ctx=tok("\n".join("".join(p) for p in ex["context"]["sentences"]),add_special_tokens=False).input_ids
    q=tok("\n\nAnswer the question with a short span.\nQuestion: "+ex["question"]+"\nAnswer:",add_special_tokens=False).input_ids
    ans=tok(" "+ex["answer"],add_special_tokens=False).input_ids[:a.maxnew]; return ctx,q,ans
@torch.no_grad()
def evaluate(kind,W):
    hasP=(kind=="startup") and prompt is not None; mk="windowed" if kind=="startup" else kind
    tf1=tem=tnll=0.0; tk=0; N=0
    for ex in lb:
        ctx,q,ans=build(ex)
        if not ans: continue
        N+=1; nP=a.nP if hasP else 0; base_ids=(ctx+q)[-(a.ctx-nP):]
        ids=torch.tensor([(base_ids+ans)[-(a.ctx-nP):]],device=dev); L=ids.shape[1]; kp=L-len(ans)
        if hasP:
            inp=torch.cat([prompt.to(bf16).unsqueeze(0),emb(ids)],1)
            lg=model(inputs_embeds=inp,attention_mask=cmask("windowed",nP+L,nP,W,bf16)).logits[0][nP:]
        else:
            lg=model(ids,attention_mask=cmask(mk,L,0,W,bf16)).logits[0]
        tnll+=F.cross_entropy(lg[kp-1:L-1].float(),ids[0,kp:L],reduction="sum").item(); tk+=len(ans)
        cur=base_ids[:]
        for _ in range(a.maxnew):
            body=cur[-(a.ctx-nP):]; x=torch.tensor([body],device=dev); L2=len(body)
            if hasP:
                inp=torch.cat([prompt.to(bf16).unsqueeze(0),emb(x)],1)
                nl=model(inputs_embeds=inp,attention_mask=cmask("windowed",nP+L2,nP,W,bf16)).logits[0,-1]
            else:
                nl=model(x,attention_mask=cmask(mk,L2,0,W,bf16)).logits[0,-1]
            nt=int(nl.argmax())
            if nt==EOS or nt==NL: break
            cur.append(nt)
        pred=tok.decode(cur[len(base_ids):]).strip()
        tf1+=f1(pred,ex["answer"]); tem+=float(norm(pred)==norm(ex["answer"]))
    print("RESULT llamahotpot %-8s W=%-4d | F1 %.3f | EM %.3f | ans-ppl %.2f (N=%d)"%(a.scheme,W,tf1/N,tem/N,math.exp(tnll/tk),N),flush=True)
if a.scheme in ("base","triangle"):
    for W in [128,256,512]: evaluate("windowed",W)
elif a.scheme=="startup":
    evaluate("startup",a.window)
else:
    evaluate("windowed",a.window)
print("LLAMAHOTPOT_DONE",flush=True)
