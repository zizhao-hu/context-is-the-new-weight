"""Finding-5 follow-up: during passkey retrieval, does the separator sink act as a
location-coupled landmark (top-attended separator near the passkey) or stay
recency-bound (near the query) while retrieval runs through content attention?

Per trial: build the passkey context exactly as passkey_eval.py, prefill, then decode
the answer greedily step by step with output_attentions=True (eager attention), taking
each step's 1 x past attention row. Deep layers (last third), mean over heads, averaged
over the first 5 answer steps. Metrics per trial:
  top_sep_dist   token distance from the argmax-attended separator to the passkey center
  near_ratio     per-separator attention mass within +-64 tok of the passkey / elsewhere
  digit_mass     attention mass on the passkey digit tokens
  argmax_is_pk   1 if the globally argmax-attended context position lies in the passkey span
Machine-readable: SEPATT (per trial) and SEPAGG (per depth); SEPDONE at the end."""
import torch, argparse, random
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--label", default="m")
ap.add_argument("--length", type=int, default=4096)
ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
ap.add_argument("--ntrials", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--near", type=int, default=64); ap.add_argument("--steps", type=int, default=5)
a = ap.parse_args(); dev = "cuda"; random.seed(a.seed); torch.manual_seed(a.seed)

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                             attn_implementation="eager").to(dev).eval()
print("SEP_MODEL %s -> %s" % (a.label, a.model), flush=True)

FILL = tok(("The grass is green. The sky is blue. The sun is yellow. Here we go. "
            "There and back again. " * 20).strip(), add_special_tokens=False).input_ids
def filler(n):
    out = []
    while len(out) < n: out += FILL
    return out[:n]
PRE = tok("There is important info hidden inside a lot of irrelevant text. Find it and memorize it.\n\n",
          add_special_tokens=False).input_ids
Q = tok("\n\nWhat is the pass key? The pass key is ", add_special_tokens=False).input_ids
SEP_STR = {".", ",", ";", ":", "!", "?"}

@torch.no_grad()
def run_trial(depth):
    pk = "".join(random.choice("0123456789") for _ in range(5))
    key = tok("The pass key is %s. Remember it. " % pk, add_special_tokens=False).input_ids
    body = a.length - len(PRE) - len(Q) - len(key)
    at = int(depth * body)
    ctx = PRE + filler(at) + key + filler(body - at) + Q
    pk_start, pk_end = len(PRE) + at, len(PRE) + at + len(key)          # passkey span
    toks = tok.convert_ids_to_tokens(ctx)
    sep_pos = [i for i, t in enumerate(toks)
               if t.replace("Ġ", "").replace("Ċ", "\n").strip() in SEP_STR]
    sep_pos_t = torch.tensor(sep_pos, device=dev)
    near = ((sep_pos_t >= pk_start - a.near) & (sep_pos_t < pk_end + a.near))
    ids = torch.tensor([ctx], device=dev)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nl = model.config.num_hidden_layers
    deep = list(range(2 * nl // 3, nl))
    nxt = out.logits[:, -1:].argmax(-1)
    rows = torch.zeros(len(ctx), device=dev, dtype=torch.float32)
    gen = []
    for step in range(a.steps):
        o = model(input_ids=nxt, past_key_values=past, use_cache=True, output_attentions=True)
        past = o.past_key_values
        att = torch.stack([o.attentions[l][0, :, 0, :len(ctx)].float().mean(0) for l in deep]).mean(0)
        rows += att
        gen.append(nxt.item())
        nxt = o.logits[:, -1:].argmax(-1)
    rows /= a.steps
    sep_att = rows[sep_pos_t]
    top_sep = sep_pos[int(sep_att.argmax())]
    pk_center = (pk_start + pk_end) // 2
    m_near = sep_att[near].sum().item(); n_near = int(near.sum())
    m_far = sep_att[~near].sum().item(); n_far = int((~near).sum())
    ratio = (m_near / max(n_near, 1)) / max(m_far / max(n_far, 1), 1e-9)
    digit_mass = rows[pk_start:pk_end].sum().item()
    argmax_pos = int(rows.argmax())
    hit = pk in tok.decode(gen)
    return dict(top_sep_dist=top_sep - pk_center, near_ratio=ratio, digit_mass=digit_mass,
                argmax_is_pk=int(pk_start <= argmax_pos < pk_end), correct=int(hit),
                top_sep=top_sep, pk_center=pk_center)

for D in [float(x) for x in a.depths.split(",")]:
    rs = []
    for t in range(a.ntrials):
        r = run_trial(D)
        rs.append(r)
        print("SEPATT label=%s depth=%.1f trial=%d top_sep=%d pk=%d dist=%d ratio=%.2f "
              "digit=%.4f argmaxpk=%d correct=%d" % (a.label, D, t, r["top_sep"], r["pk_center"],
              r["top_sep_dist"], r["near_ratio"], r["digit_mass"], r["argmax_is_pk"], r["correct"]),
              flush=True)
    n = len(rs)
    frac_near = sum(abs(r["top_sep_dist"]) <= a.near for r in rs) / n
    print("SEPAGG label=%s depth=%.1f n=%d top_near_frac=%.2f mean_ratio=%.2f "
          "mean_digit=%.4f argmaxpk_frac=%.2f acc=%.2f" % (a.label, D, n, frac_near,
          sum(r["near_ratio"] for r in rs) / n, sum(r["digit_mass"] for r in rs) / n,
          sum(r["argmax_is_pk"] for r in rs) / n, sum(r["correct"] for r in rs) / n), flush=True)
print("SEPDONE label=%s" % a.label, flush=True)
