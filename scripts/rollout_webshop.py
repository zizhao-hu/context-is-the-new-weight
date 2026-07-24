"""A1: Live Webshop env rollouts for each adapter. Measures task reward.

Architecture: webshop FastAPI server runs in background (webshop conda env).
This script loads the adapter (cinw conda env) and steps the env via HTTP.
"""
from __future__ import annotations
import json, re, time, sys, argparse, gc
from pathlib import Path
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, "/project2/jessetho_1732/zizhaoh/context-is-the-new-weight")

STREAM_ROOT = Path("/scratch1/zizhaoh/context-is-the-new-weight/outputs/webshop_streaming")
FOURWAY_ROOT = Path("/scratch1/zizhaoh/context-is-the-new-weight/outputs/fourway")
SERVER_URL = "http://127.0.0.1:" + __import__("os").environ.get("WEBSHOP_PORT", "36001")

# Agentic task-solving comparison: the model acts as the WebShop agent (emits search/click,
# env returns reward). Compares the policy baseline, the iid joint SFT, and experience tuning.
CELLS = [
    ("base (no training)",                  "base",  None),
    # "Instruction tuning (o->a)" dropped — already evaluated at N=200 (ourgoals_instruction_200.json:
    # 22.5% success / 34.3% score). Skipping keeps this job short so it schedules/backfills easily; merge at report.
    ("World-action model (o,a->o',a')",     "peft",  FOURWAY_ROOT / "hist_world_action"),
    ("Experience tuning (streaming)",       "peft",  STREAM_ROOT / "alltrajs_lora_ce_nwm8_nev2_r256"),
    ("Experience tuning (+posrec)",         "peft",  STREAM_ROOT / "alltrajs_lora_ce_nwm8_nev2_r256_recall_prope_posrec"),
]

ACTION_RE = re.compile(r"(search\[.*?\]|click\[[^\]]+\])", re.DOTALL)
# Qwen is bilingual; the all-token-trained adapters sometimes emit the action VERB in Chinese
# (搜索=search, 点击=click) while the bracket content is correct. Normalize to English so we
# measure task competence, not tokenizer language drift.
CN_VERB = [("搜索", "search"), ("檢索", "search"), ("查找", "search"), ("搜尋", "search"),
           ("点击", "click"), ("點擊", "click"), ("单击", "click"), ("選擇", "click")]
PLACEHOLDERS = {"search[<your query>]", "click[<your query>]", "search[query]"}


def wait_for_server(url, timeout=900):
    """Poll /list_envs until the server responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/list_envs", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def create_env():
    r = requests.post(f"{SERVER_URL}/create", timeout=10)
    r.raise_for_status()
    return r.json()  # env_idx


def reset_env(env_idx, session_id):
    r = requests.post(f"{SERVER_URL}/reset", json={"env_idx": env_idx, "session_id": session_id}, timeout=30)
    r.raise_for_status()
    out = r.json()
    return out[0]  # initial observation text


def get_actions(env_idx):
    r = requests.get(f"{SERVER_URL}/available_actions", params={"env_idx": env_idx}, timeout=10)
    r.raise_for_status()
    return r.json()


def step_env(env_idx, action):
    r = requests.post(f"{SERVER_URL}/step", json={"env_idx": env_idx, "action": action}, timeout=30)
    r.raise_for_status()
    return r.json()  # StepResponse


def parse_action(text: str):
    """Extract the first valid Webshop action from the model's generated text.
    Normalizes Chinese action verbs to English and rejects the literal admissible-action
    placeholder (which the model occasionally copies verbatim)."""
    for cn, en in CN_VERB:
        text = text.replace(cn + "[", en + "[")
    m = ACTION_RE.search(text)
    if not m:
        return None
    act = m.group(0).strip()
    if act in PLACEHOLDERS:
        return None
    return act


def format_obs_with_actions(env_idx, obs):
    """Append the admissible-actions block to an observation, matching the training format.

    Training obs_0 (system msg) looked like:
        WebShop [SEP] Instruction: [SEP] <instr> [SEP] Search

        Your admissible actions of the current situation are:
        [
        'search[<your query>]',
        'click[search]',
        ].
    The env's /reset and /step return only the first line, so we reconstruct the action block.
    """
    try:
        av = get_actions(env_idx)
    except Exception:
        return obs
    actions = []
    if av.get("has_search_bar"):
        actions.append("search[<your query>]")
    for c in av.get("clickables", []):
        if c == "search":
            continue
        actions.append(f"click[{c}]")
    if not actions:
        return obs
    block = "\n\nYour admissible actions of the current situation are: \n[\n" + ",\n".join(f"'{a}'" for a in actions) + "\n]."
    return obs + block


def generate_action(model, tok, prompt: str, max_new_tokens=64, sample=False) -> str:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)


def rollout_one(model, tok, env_idx, session_id, max_steps=20, react=False, obs0_label="obs_0", think_mode="bracket", verbose=False, no_history=False, fewshot_prefix=""):
    obs = format_obs_with_actions(env_idx, reset_env(env_idx, session_id))
    history = [(obs0_label, obs)]
    transcript = []; thoughts = []
    for step in range(max_steps):
        if no_history:
            # memoryless (single-step instruction-tuning eval): context = current obs only
            ctx = next((f"[{lbl}] {text}" for lbl, text in reversed(history) if lbl.startswith("obs")), "")
        else:
            ctx = "\n".join(f"[{lbl}] {text}" for lbl, text in history)
        ctx = fewshot_prefix + ctx          # fixed few-shot exemplars (finetuned x few-shot at test), if any
        if react:
            # ReAct-WM: the agent first THINKS (predict + reason), then acts.
            ctx += "\n<think>\n" if think_mode == "native" else f"\n[think_{step}] "
            gen = generate_action(model, tok, ctx, max_new_tokens=220)
            action = parse_action(gen)
            if action is None:
                gen = generate_action(model, tok, ctx, max_new_tokens=220, sample=True)
                action = parse_action(gen)
            if think_mode == "native":
                think = gen.split("</think>")[0].strip()[:400]
            else:
                cut = gen.find("[act"); think = (gen[:cut] if cut > 0 else gen.split("search[")[0].split("click[")[0]).strip()[:400]
        else:
            ctx += f"\n[act_{step}] "
            gen = generate_action(model, tok, ctx)
            action = parse_action(gen)
            if action is None:
                gen = generate_action(model, tok, ctx, sample=True)
                action = parse_action(gen)
            think = None
        if verbose:
            last_obs = next((t for l, t in reversed(history) if l.startswith("obs")), "")
            print(f"\n===== STEP {step} (sid {session_id}) =====", flush=True)
            print(f"  [last obs ~800c]: {last_obs[:800]}", flush=True)
            print(f"  [gen ~350c]: {gen[:350]!r}", flush=True)
            print(f"  [parsed action]: {action}", flush=True)
        if action is None:
            return {"session_id": session_id, "reward": 0.0, "done": False, "n_steps": step,
                    "parse_failure": True, "last_gen": gen[:300], "transcript": transcript, "thoughts": thoughts}
        transcript.append(action)
        if think is not None:
            thoughts.append(think)
            history.append((f"think_{step}", think))
        try:
            res = step_env(env_idx, action)
        except Exception as e:
            return {"session_id": session_id, "reward": 0.0, "done": False, "n_steps": step,
                    "error": str(e)[:100], "transcript": transcript, "thoughts": thoughts}
        raw_obs = res.get("state") or res.get("observation") or ""
        reward = float(res.get("reward", 0.0) or 0.0)
        done = bool(res.get("done", False))
        history.append((f"act_{step}", action))
        if done:
            return {"session_id": session_id, "reward": reward, "done": True, "n_steps": step + 1,
                    "transcript": transcript, "thoughts": thoughts}
        new_obs = format_obs_with_actions(env_idx, raw_obs)
        history.append((f"obs_{step+1}", new_obs))
    return {"session_id": session_id, "reward": 0.0, "done": False, "n_steps": max_steps,
            "transcript": transcript, "thoughts": thoughts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--session-id-start", type=int, default=500, help="Sessions 0..499 used in training-style trajectories; use 500+ to avoid overlap.")
    ap.add_argument("--cells", default="all", help="comma-separated cell labels or 'all'")
    ap.add_argument("--adapter", default=None, help="eval a single arbitrary adapter dir (overrides CELLS)")
    ap.add_argument("--label", default=None, help="label for --adapter cell")
    ap.add_argument("--react", action="store_true", help="ReAct-WM: agent generates [think] then [act] each step")
    ap.add_argument("--obs0-label", default="obs_0", help="first-obs event label; ET streaming-training format uses 'obs_0(=sys)'")
    ap.add_argument("--think-mode", default="bracket", choices=["bracket", "native"], help="react think slot: '[think_t]' bracket vs native '<think>'")
    ap.add_argument("--verbose", action="store_true", help="print full obs+gen+action each step (diagnostic)")
    ap.add_argument("--no-history", action="store_true", help="memoryless: context = current obs only (for single-step instruction-tuned models)")
    ap.add_argument("--fewshot-file", default=None, help="prepend k exemplar trajectories (training [label] format) as a fixed few-shot prefix (finetuned x static-few-shot at test)")
    ap.add_argument("--fewshot-k", type=int, default=1)
    ap.add_argument("--sids-file", default=None, help="Word2World webshop_test.json: use its exact goal session-ids (item_id suffix)")
    ap.add_argument("--out", default="/scratch1/zizhaoh/context-is-the-new-weight/outputs/rollout_webshop.json")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()
    SIDS = None
    if args.sids_file:
        SIDS = [int(it["item_id"].split("_")[-1]) for it in json.load(open(args.sids_file))][:args.n_episodes]
        print(f"using {len(SIDS)} Word2World goal session-ids from {args.sids_file}")
    fewshot_prefix = ""
    if args.fewshot_file:
        ex = json.load(open(args.fewshot_file))[:args.fewshot_k]
        parts = ["\n".join(f"[{e['label']}] {e['content']}" for e in tr["events"]) for tr in ex]
        fewshot_prefix = "Here are example shopping episodes:\n" + "\n\n".join(parts) + "\n\nNow your task:\n"
        print(f"few-shot prefix: {args.fewshot_k} exemplar traj(s), {len(fewshot_prefix)} chars")

    print("Waiting for Webshop server at", SERVER_URL, "...")
    if not wait_for_server(SERVER_URL):
        print("ERROR: Webshop server not responding")
        sys.exit(1)
    print("server up")

    if args.adapter:
        # single arbitrary adapter (its dir contains an 'adapter/' subdir)
        cells = [(args.label or "adapter", "peft", Path(args.adapter))]
    else:
        requested = set(args.cells.split(",")) if args.cells != "all" else None
        cells = [c for c in CELLS if requested is None or c[0] in requested]
    print(f"Running {len(cells)} cells × {args.n_episodes} episodes")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16
    print(f"Loading base {args.model} (4bit={args.load_4bit})...")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, dtype=dtype,
                                                    trust_remote_code=True, device_map={"": 0})
    else:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to("cuda")
    base.eval()

    env_idx = create_env()
    print(f"created env_idx={env_idx}")

    all_results = {}
    for label, kind, cell_dir in cells:
        print(f"\n=== {label} ===")
        try:
            if kind == "base":
                model = base
            elif kind == "peft":
                ad = cell_dir / "adapter"
                if not ad.exists():
                    print(f"  adapter missing: {ad}")
                    all_results[label] = {"error": "adapter missing"}
                    continue
                model = PeftModel.from_pretrained(base, str(ad))
                model.eval()
            episodes = []
            for ep in range(args.n_episodes):
                sid = SIDS[ep] if SIDS is not None else args.session_id_start + ep
                t0 = time.time()
                try:
                    res = rollout_one(model, tok, env_idx, sid, react=args.react, obs0_label=args.obs0_label,
                                      think_mode=args.think_mode, verbose=args.verbose, no_history=args.no_history,
                                      fewshot_prefix=fewshot_prefix)
                except Exception as e:
                    # server died / transient — keep episodes so far, stop this cell
                    print(f"  ep {ep} sid={sid}: ABORTED ({str(e)[:80]}) — keeping {len(episodes)} episodes")
                    break
                res["wall_s"] = time.time() - t0
                print(f"  ep {ep} sid={sid}: reward={res['reward']:.3f} done={res.get('done')} steps={res['n_steps']} time={res['wall_s']:.1f}s", flush=True)
                episodes.append(res)
                # incremental save so a later crash never wipes completed episodes
                avg_reward = sum(e["reward"] for e in episodes) / max(1, len(episodes))
                success = sum(1 for e in episodes if e["reward"] > 0.95) / max(1, len(episodes))
                all_results[label] = {"avg_reward": avg_reward, "success_rate": success, "n": len(episodes), "episodes": episodes}
                Path(args.out).write_text(json.dumps(all_results, indent=2))
            avg_reward = sum(e["reward"] for e in episodes) / max(1, len(episodes))
            success = sum(1 for e in episodes if e["reward"] > 0.95) / max(1, len(episodes))
            all_results[label] = {"avg_reward": avg_reward, "success_rate": success, "n": len(episodes), "episodes": episodes}
            print(f"  avg_reward = {avg_reward:.3f}  success_rate = {success:.3f}  (n={len(episodes)})")
            if kind == "peft":
                try: base = model.unload()
                except Exception: pass
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results[label] = {"error": str(e)[:200]}

    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
