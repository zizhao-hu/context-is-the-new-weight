"""ReAct-paper replication on ALFWorld (Table 3) with Qwen3.5-9B as substitute LM.

Faithful port of ysymyth/ReAct alfworld.ipynb:
  - base_config.yaml, split=eval_out_of_distribution, 134 games, batch_size=1
  - per task type: 2-shot prompt (react_{v}_1 + react_{v}_0) from prompts/alfworld_3prompts.json
  - loop max 50 steps; action 'think: ...' -> observation 'OK.'; greedy, stop at newline
  - SR per task type (Pick/Clean/Heat/Cool/Look/Pick2) + overall, as in Table 3
Substitute LM = frozen Qwen3.5-9B (text-davinci-002/PaLM-540B retired).
"""
from __future__ import annotations
import json, os, sys, argparse, yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PREFIXES = {
    "pick_and_place": "put", "pick_clean_then_place": "clean", "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool", "look_at_obj": "examine", "pick_two_obj": "puttwo",
}
COLS = ["put", "clean", "heat", "cool", "examine", "puttwo"]  # Table 3 order: Pick/Clean/Heat/Cool/Look/Pick2

def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-games", type=int, default=134)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from alfworld.agents.environment import get_environment
    with open(args.config) as r:
        config = yaml.safe_load(r)
    split = "eval_out_of_distribution"
    env = get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    d = json.load(open(args.prompts))

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16,
                                                 trust_remote_code=True).to("cuda")
    model.eval()

    def llm(prompt, max_new=100):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        ids = ids[:, -7000:]  # keep context bounded
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
        return text.split("\n")[0].strip()  # stop=['\n']

    def alfworld_run(prompt, ob=""):
        init_prompt = prompt + ob + "\n>"
        run = ""
        for i in range(1, 50):
            action = llm(init_prompt + run).strip()
            # the model sometimes echoes "> action"; strip a leading '>'
            action = action.lstrip("> ").strip()
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info["won"][0], done[0]
            if action.startswith("think:"):
                observation = "OK."
            if args.verbose:
                print(f"    {i} Act: {action[:100]} | Obs: {observation[:80]}", flush=True)
            run += f" {action}\n{observation}\n>"
            if done:
                return reward
        return 0

    cnts = {k: 0 for k in COLS}
    rs = {k: 0 for k in COLS}
    log = []
    for g in range(args.n_games):
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
        matched = None
        for k, v in PREFIXES.items():
            if name.startswith(k):
                matched = v
                prompt = ("Interact with a household to solve a task. Here are two examples.\n"
                          + d[f"react_{v}_1"] + d[f"react_{v}_0"] + "\nHere is the task.\n")
                r = alfworld_run(prompt, ob=ob)
                rs[v] += r; cnts[v] += 1
                break
        tot_r = sum(rs.values()); tot_c = sum(cnts.values())
        log.append({"game": g, "type": matched, "reward": r if matched else None})
        print(f"[alfworld] {g+1}/{args.n_games} type={matched} r={r if matched else '-'} "
              f"running SR={100*tot_r/max(1,tot_c):.1f} ({tot_r}/{tot_c})", flush=True)
        # incremental save
        res = {"by_type": {k: {"sr": 100*rs[k]/cnts[k] if cnts[k] else None, "n": cnts[k]} for k in COLS},
               "all": {"sr": 100*tot_r/max(1, tot_c), "n": tot_c},
               "protocol": "ReAct alfworld.ipynb: 2-shot, 50-step, think->OK., greedy; substitute Qwen3.5-9B",
               "log": log}
        json.dump(res, open(args.out, "w"), indent=2)
    print(f"[alfworld] DONE  by-type SR: " +
          "  ".join(f"{k}={100*rs[k]/cnts[k]:.0f}" if cnts[k] else f"{k}=NA" for k in COLS) +
          f"  ALL={100*sum(rs.values())/max(1,sum(cnts.values())):.1f}", flush=True)

if __name__ == "__main__":
    main()
