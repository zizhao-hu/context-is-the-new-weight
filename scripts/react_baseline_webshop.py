"""ReAct-paper replication on WebShop (Table 4 rows: Act, ReAct) with Qwen3.5-9B
as the substitute LM (text-davinci-002 / PaLM-540B are retired).

Faithful to ysymyth/ReAct WebShop.ipynb:
  - exact few-shot prompts (prompt1 = ReAct with think[...], prompt1_actonly = Act)
  - loop: Action: -> env step -> Observation: ... ; think[...] gets observation "OK."
  - greedy decoding, stop at newline, max 15 steps, context char-truncated to 6400
Deviations (logged in the output): substitute LM; our 100 held-out W2W sessions instead
of their fixed_0..499 (so the row is directly comparable to our adapter cells); our
agentenv server's [SEP] observations converted to ReAct's bracketed-button format
using the server's clickables list.
"""
from __future__ import annotations
import json, re, time, sys, argparse, os
from pathlib import Path
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SERVER_URL = "http://127.0.0.1:" + os.environ.get("WEBSHOP_PORT", "36001")

# ---------------- ReAct's exact few-shot prompts (WebShop.ipynb cells 4-5) ----------------
PROMPT_REACT = """Webshop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B078GTKVXY]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B08KBVJ4XN]
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)
$15.95

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation:
[Back to Search]
[< Prev]
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
Price: $10.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]
"""

PROMPT_ACT = """Webshop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B078GTKVXY]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B08KBVJ4XN]
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)
$15.95

Action: click[B078GWRC1J]
Observation:
[Back to Search]
[< Prev]
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
Price: $10.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]
"""

# ---------------- env client (same HTTP API as rollout_webshop.py) ----------------
def wait_for_server(url, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/list_envs", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def create_env():
    r = requests.post(f"{SERVER_URL}/create", timeout=10); r.raise_for_status(); return r.json()

def reset_env(env_idx, session_id):
    r = requests.post(f"{SERVER_URL}/reset", json={"env_idx": env_idx, "session_id": session_id}, timeout=30)
    r.raise_for_status(); return r.json()[0]

def get_actions(env_idx):
    r = requests.get(f"{SERVER_URL}/available_actions", params={"env_idx": env_idx}, timeout=10)
    r.raise_for_status(); return r.json()

def step_env(env_idx, action):
    r = requests.post(f"{SERVER_URL}/step", json={"env_idx": env_idx, "action": action}, timeout=30)
    r.raise_for_status(); return r.json()

# ---------------- [SEP] obs -> ReAct's bracketed format ----------------
def to_react_format(obs: str, env_idx) -> str:
    """The agentenv server emits 'A [SEP] B [SEP] C'; ReAct's prompt shows buttons in
    [brackets], one segment per line. Bracket exactly the segments that are clickable."""
    try:
        acts = get_actions(env_idx)
        clickables = {str(c).strip().lower() for c in (acts.get("clickables") or [])}
    except Exception:
        clickables = set()
    lines = []
    for seg in (s.strip() for s in obs.split("[SEP]")):
        if not seg:
            continue
        if seg.lower() in clickables:
            lines.append(f"[{seg}]")
        else:
            lines.append(seg)
    return "\n".join(lines)

ACTION_RE = re.compile(r"^(think\[.*?\]|search\[.*?\]|click\[[^\]]+\])", re.DOTALL)

# ---------------- the ReAct loop (webshop_run, faithful) ----------------
def webshop_run(model, tok, env_idx, session_id, init_prompt, max_steps=15, verbose=False):
    obs = to_react_format(reset_env(env_idx, session_id), env_idx)
    prompt = ""
    n_env_steps = 0
    for i in range(max_steps):
        if i:
            prompt += f" {ACTION}\nObservation: {OBS}\n\nAction:"
        else:
            prompt += f"{obs}\n\nAction:"
        ctx = init_prompt + prompt[-(6400 - len(init_prompt)):]
        ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=100, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
        action = gen.split("\n")[0].strip().lstrip()        # stop=['\n'] equivalent
        m = ACTION_RE.match(action)
        action = m.group(0) if m else action
        if verbose:
            print(f"    step {i}: Action: {action[:120]}", flush=True)
        if action.startswith("think["):
            OBS = "OK."
            ACTION = action
            continue
        try:
            res = step_env(env_idx, action)
            raw = res.get("state") or res.get("observation") or ""
            reward = float(res.get("reward", 0.0) or 0.0)
            done = bool(res.get("done", False))
            n_env_steps += 1
            if done:
                return reward, i + 1, None
            OBS = to_react_format(raw, env_idx)
        except Exception:
            OBS = "Invalid action!"
        ACTION = action
    return 0.0, max_steps, "max_steps"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["react", "act"], required=True)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--sids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    SIDS = [int(it["item_id"].split("_")[-1]) for it in json.load(open(args.sids_file))][:args.n_episodes]
    init_prompt = PROMPT_REACT if args.mode == "react" else PROMPT_ACT

    print(f"[react-base] mode={args.mode} waiting for env server at {SERVER_URL} ...", flush=True)
    if not wait_for_server(SERVER_URL):
        sys.exit("env server never came up")
    env_idx = create_env()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, dtype=torch.bfloat16,
                                                     trust_remote_code=True, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     trust_remote_code=True).to("cuda")
    model.eval()

    label = f"react_paper_{args.mode}_qwen"
    episodes = []
    results = {}
    for ep, sid in enumerate(SIDS):
        t0 = time.time()
        try:
            reward, nsteps, err = webshop_run(model, tok, env_idx, sid, init_prompt, verbose=args.verbose)
        except Exception as e:
            print(f"  ep {ep} sid={sid}: ABORTED ({str(e)[:80]})", flush=True)
            break
        episodes.append({"session_id": sid, "reward": reward, "n_steps": nsteps, "err": err,
                         "wall_s": time.time() - t0})
        print(f"  ep {ep} sid={sid}: reward={reward:.3f} steps={nsteps} time={episodes[-1]['wall_s']:.1f}s", flush=True)
        n = len(episodes)
        results[label] = {
            "avg_reward": sum(e["reward"] for e in episodes) / n,
            "success_rate": sum(1 for e in episodes if e["reward"] > 0.95) / n,
            "n": n, "protocol": "ReAct WebShop.ipynb: greedy, 15 steps, think->OK., 6400-char ctx; "
                                "substitute LM Qwen3.5-9B; same 100 W2W sessions as our cells",
            "episodes": episodes,
        }
        Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[react-base] DONE {label}: n={len(episodes)}", flush=True)

if __name__ == "__main__":
    main()
