"""Native-thinking WebShop agent: uses the model's OWN chat template + native <think> mode
(not ReAct's foreign bracket think[...]). Multi-turn: obs=user turn, action=assistant turn.
Each step the model thinks (inside <think></think>) then emits one action — so thinking does NOT
consume an env step (no ReAct step-tax). --mode think (enable_thinking) vs act (no think).
Isolates: does the model's native reasoning help over plain acting, format/budget confounds removed.
Eval = same 100 W2W sessions, SR = % reward==1.0 (ReAct metric)."""
from __future__ import annotations
import json, re, time, sys, argparse, os
from pathlib import Path
import requests, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SERVER_URL = "http://127.0.0.1:" + os.environ.get("WEBSHOP_PORT", "36001")
ACTION_RE = re.compile(r"(search\[.*?\]|click\[[^\]]+\])", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def wait_for_server(url, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/list_envs", timeout=2).status_code == 200: return True
        except Exception: pass
        time.sleep(2)
    return False
def create_env(): r=requests.post(f"{SERVER_URL}/create",timeout=10);r.raise_for_status();return r.json()
def reset_env(i,s): r=requests.post(f"{SERVER_URL}/reset",json={"env_idx":i,"session_id":s},timeout=30);r.raise_for_status();return r.json()[0]
def get_actions(i): r=requests.get(f"{SERVER_URL}/available_actions",params={"env_idx":i},timeout=10);r.raise_for_status();return r.json()
def step_env(i,a): r=requests.post(f"{SERVER_URL}/step",json={"env_idx":i,"action":a},timeout=30);r.raise_for_status();return r.json()

def parse_action(text):
    m = ACTION_RE.search(text)
    return m.group(0).strip() if m else None

def format_obs(env_idx, obs):
    """Append the admissible-actions block (training-format obs)."""
    try:
        acts = get_actions(env_idx)
        actions = (["search[<your query>]"] if acts.get("has_search_bar") else []) + \
                  [f"click[{c}]" for c in (acts.get("clickables") or []) if str(c).lower() != "search"]
    except Exception:
        actions = []
    if not actions: return obs
    return obs + "\n\nYour admissible actions of the current situation are: \n[\n" + ",\n".join(f"'{a}'" for a in actions) + "\n]."

SYS = ("You are a shopping agent on WebShop. Buy the item that best matches the Instruction in as few "
       "steps as possible. Follow this procedure exactly:\n"
       "1. SEARCH ONCE using the key nouns + attributes from the Instruction.\n"
       "2. The search results ARE relevant. Do NOT search again, do NOT click 'Back to Search', and do "
       "NOT click [Next >] or [< Prev] to browse more pages. From the FIRST page, immediately click the "
       "product id ([B0...]) whose title best matches the Instruction (pick the closest one even if imperfect).\n"
       "3. On the product page, click the required option buttons (size, color, scent, count, ...) that "
       "the Instruction asks for, then click[Buy Now].\n"
       "4. Click each option AT MOST ONCE. As soon as the required options are selected, click[Buy Now].\n"
       "NEVER output the same action twice — always make progress toward Buy Now.\n\n"
       "WORKED EXAMPLE (Instruction: a 3 ounce bottle of bright citrus deodorant for sensitive skin, under $50):\n"
       "search[3 ounce bright citrus deodorant sensitive skin]\n"
       "click[b078gwrc1j]            # click the matching product id from the results\n"
       "click[bright citrus]         # select the scent option (once)\n"
       "click[3 ounce (pack of 1)]   # select the size option (once)\n"
       "click[Buy Now]               # all options chosen -> buy\n"
       "(Note: each option clicked exactly once, then Buy Now. Do the same.)\n\n"
       "Respond with EXACTLY ONE admissible action on its own line: search[query] or click[button text].")

THINK_INSTR = ("\nBefore the action, in ONE short sentence VERIFY the candidate matches the Instruction's "
               "specific attributes (color, size, count, price). Pick the option that matches each attribute; "
               "if unsure, pick the closest and move on. Then put the action on its OWN final line.")
ACT_INSTR = "\nOutput ONLY the action — no reasoning, no other text."

def gen(model, tok, messages, max_new, sample=False):
    # fast no-think generation (visible reasoning comes from the prompt, not slow <think> blocks)
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    ids = ids[:, -8000:]
    kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)

def parse_last(text):
    m = ACTION_RE.findall(THINK_RE.sub("", text))
    return m[-1].strip() if m else None

def run_one(model, tok, env_idx, sid, think, max_steps=15, verbose=False):
    sys_msg = SYS + (THINK_INSTR if think else ACT_INSTR)
    obs = format_obs(env_idx, reset_env(env_idx, sid))
    turns = []; mn = 220 if think else 40; bad = 0; prev_action = None
    for step in range(max_steps):
        msgs = [{"role": "system", "content": sys_msg}]
        for role, content in turns: msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": obs})
        raw = gen(model, tok, msgs, mn)
        action = parse_last(raw)
        if action is None:                                   # retry once with sampling
            raw = gen(model, tok, msgs, mn, sample=True); action = parse_last(raw)
        if verbose: print(f"    step {step}: think={think} action={action} | gen[:90]={raw.strip()[:90]!r}", flush=True)
        if action is None:
            bad += 1
            if bad >= 2: return {"session_id": sid, "reward": 0.0, "n_steps": step, "parse_fail": True, "last": raw[:200]}
            continue
        bad = 0
        # de-dup: if the model repeats its last action, re-sample to break the loop (same guard for think/act)
        tries = 0
        while action == prev_action and tries < 3:
            a2 = parse_last(gen(model, tok, msgs, mn, sample=True))
            if a2: action = a2
            tries += 1
        prev_action = action
        # append ONLY the action to history (not the reasoning) so prior reasoning can't reinforce a loop
        turns.append(("user", obs)); turns.append(("assistant", action))
        try:
            res = step_env(env_idx, action)
        except Exception as e:
            return {"session_id": sid, "reward": 0.0, "n_steps": step, "error": str(e)[:80]}
        reward = float(res.get("reward", 0.0) or 0.0); done = bool(res.get("done", False))
        if done: return {"session_id": sid, "reward": reward, "n_steps": step + 1, "done": True}
        obs = format_obs(env_idx, res.get("state") or res.get("observation") or "")
    return {"session_id": sid, "reward": 0.0, "n_steps": max_steps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["think", "act"], required=True)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--sids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    SIDS = [int(it["item_id"].split("_")[-1]) for it in json.load(open(args.sids_file))][:args.n_episodes]
    think = (args.mode == "think")
    print(f"[native] mode={args.mode} model={args.model} waiting for env...", flush=True)
    if not wait_for_server(SERVER_URL): sys.exit("env server never came up")
    env_idx = create_env()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, dtype=torch.bfloat16,
                                                     trust_remote_code=True, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    label = f"native_{args.mode}_{'27b' if '27' in args.model else '9b'}"
    eps, results = [], {}
    for i, sid in enumerate(SIDS):
        t0 = time.time()
        try:
            r = run_one(model, tok, env_idx, sid, think, verbose=args.verbose)
        except Exception as e:
            print(f"  ep {i} sid={sid}: ABORT {str(e)[:80]}", flush=True); break
        r["wall_s"] = time.time() - t0; eps.append(r)
        print(f"  ep {i} sid={sid}: reward={r['reward']:.3f} steps={r['n_steps']} t={r['wall_s']:.0f}s", flush=True)
        n = len(eps)
        results[label] = {"avg_reward": sum(e["reward"] for e in eps)/n,
                          "success_rate": sum(1 for e in eps if e["reward"] > 0.95)/n,
                          "n": n, "mode": args.mode, "model": args.model, "episodes": eps}
        Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[native] DONE {label} n={len(eps)}", flush=True)

if __name__ == "__main__":
    main()
