"""Lifetime Experience Tuning — agentic deployment as one continuous, ever-growing stream.

A global clock G never resets; the persistent memory ΔW keeps integrating; a recall buffer is
indexed by G. The ReAct-WM agent is deployed on a sequence of held-out goals; after each episode
its experience is (optionally) integrated into ΔW at the current G (persistent-RoPE positions),
and G advances. Task reward is recorded PREQUENTIALLY (before integrating that episode), so the
reward-vs-episode curve shows whether lifetime accumulation makes the agent better at the task.

--integration variants (the thing we are comparing):
  frozen        no integration, no G growth — current ReAct-WM (in-context only). Baseline.
  everything    integrate all episode events (loss on obs+think+act). Pure continual accumulation.
  reward_gated  always integrate obs (world model); reinforce think+act only if reward>=thresh.
  hybrid        reward_gated + replay a past high-reward episode at its ORIGINAL G-positions (recall consolidation).
  expert        after the attempt, integrate the matching EXPERT demo for the goal (BC on expert actions).

Within-episode reasoning uses in-context accumulation; G/persistent-RoPE anchor the INTEGRATION
and recall (where the adapter learns to bind content to absolute lifetime positions).
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, "/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/scripts")
from rollout_webshop import (reset_env, step_env, format_obs_with_actions, parse_action,
                             generate_action, create_env, wait_for_server, SERVER_URL)
sys.path.insert(0, "/project2/jessetho_1732/zizhaoh/context-is-the-new-weight")

GOALS_FILE = "/project2/jessetho_1732/zizhaoh/Word2World/data/llama_factory/webshop_train_70790.json"


def set_gen_mode(m):
    try: m.gradient_checkpointing_disable()
    except Exception: pass
    m.config.use_cache = True; m.eval()


def set_train_mode(m):
    try:
        m.gradient_checkpointing_enable(); m.enable_input_require_grads()
    except Exception: pass
    m.config.use_cache = False; m.train()


def rollout_react(model, tok, env_idx, sid, max_steps=15, prefix="", act_only=False, no_history=False,
                  prompt_param=None, window=0):
    """ReAct-WM rollout; returns (events[obs/(think)/act], reward, done, n_steps).
    prefix: text prepended each step (context_accum). act_only: no think step (Act). no_history:
    context = current obs only (instruction-tuning / memoryless). prompt_param/window: if set, deploy with the
    windowed generation (trainable prompt + last-W context) consistent with windowed finetuning."""
    def _gen(ctx, mnew, sample=False):
        if prompt_param is not None and window > 0:
            return generate_windowed(model, tok, ctx, prompt_param, window, max_new_tokens=mnew, sample=sample)
        return generate_action(model, tok, ctx, max_new_tokens=mnew, sample=sample)
    obs = format_obs_with_actions(env_idx, reset_env(env_idx, sid))
    events = [{"label": "obs_0", "role": "obs", "content": obs}]
    history = [("obs_0", obs)]
    reward = 0.0; done = False; step = 0
    for step in range(max_steps):
        if no_history:                                   # memoryless (instruction-tuning): only current obs
            hist_txt = next((f"[{l}] {t}" for l, t in reversed(history) if l.startswith("obs")), "")
        else:
            hist_txt = "\n".join(f"[{l}] {t}" for l, t in history)
        mnew = 64 if act_only else 220
        ctx = prefix + hist_txt + (f"\n[act_{step}] " if act_only else f"\n[think_{step}] ")
        gen = _gen(ctx, mnew)
        action = parse_action(gen)
        if action is None:
            gen = _gen(ctx, mnew, sample=True); action = parse_action(gen)
        if action is None:
            break
        if not act_only:                                 # react: capture the think too
            cut = gen.find("[act"); think = (gen[:cut] if cut > 0 else gen.split("search[")[0].split("click[")[0]).strip()[:400]
            events.append({"label": f"think_{step}", "role": "think", "content": think})
            history.append((f"think_{step}", think))
        events.append({"label": f"act_{step}", "role": "act", "content": action})
        history.append((f"act_{step}", action))
        try:
            res = step_env(env_idx, action)
        except Exception:
            break
        reward = float(res.get("reward", 0.0) or 0.0); done = bool(res.get("done", False))
        if done:
            break
        nobs = format_obs_with_actions(env_idx, res.get("state") or res.get("observation") or "")
        events.append({"label": f"obs_{step+1}", "role": "obs", "content": nobs})
        history.append((f"obs_{step+1}", nobs))
    return events, reward, done, step + 1


def expert_events(demo):
    out = []; oi = 0; ai = 0
    for i, m in enumerate(demo["messages"]):
        if i == 0: out.append({"label": "obs_0", "role": "obs", "content": m["content"]}); oi = 1
        elif m["role"] == "user": out.append({"label": f"act_{ai}", "role": "act", "content": m["content"]}); ai += 1
        else: out.append({"label": f"obs_{oi}", "role": "obs", "content": m["content"]}); oi += 1
    return out


def flat_ids_roles(events, tok):
    ids = []; roles = []
    for e in events:
        seg = tok(f"[{e['label']}] {e['content']}\n", add_special_tokens=False).input_ids
        ids.extend(seg); roles.extend([e["role"]] * len(seg))
    return ids, roles


def integrate(student, optim, trainable, ids, roles, G, loss_roles, pos_start, max_tokens=2400):
    """One gradient step on a flat event sequence at persistent-RoPE positions [pos_start, ...].
    Returns loss (or None if nothing to supervise)."""
    if len(ids) > max_tokens:
        ids = ids[-max_tokens:]; roles = roles[-max_tokens:]
    keep = [r in loss_roles for r in roles[1:]]
    if len(ids) < 2 or not any(keep):
        return None
    set_train_mode(student)
    t = torch.tensor([ids], dtype=torch.long, device="cuda")
    pos = torch.arange(pos_start, pos_start + t.shape[1], device="cuda").unsqueeze(0)
    out = student(input_ids=t, position_ids=pos)
    extra = out.logits.shape[1] - t.shape[1]
    logits = out.logits[0, extra:extra + t.shape[1] - 1, :]
    targets = t[0, 1:]
    km = torch.tensor(keep, dtype=torch.bool, device="cuda")
    loss = F.cross_entropy(logits[km], targets[km])
    loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optim.step(); optim.zero_grad()
    return loss.item()


def distill_episode(student, optim, trainable, tok, events, pos_start, temp=1.0, keep_frac=0.5,
                    mask_think=False, max_ctx=3000, mem_ids=None, mem_cap=6000, loss_type="kl"):
    """CONTEXT DISTILLATION with a MIDDLE split, over the in-context trajectory 1..n. The STUDENT
    conditions only on the recent TAIL — the last `keep_frac` of the trajectory tokens ('start from the
    middle'), and predicts the think/act tokens in that tail.
      loss_type='kl' (default): a TEACHER on the FULL trajectory (+ optional cross-episode MEMORY) gives
        soft targets; KL(teacher || student) forces the dropped early history into ΔW. Two forwards.
      loss_type='ce' (CONTROL): the SAME truncated student is trained with HARD cross-entropy on the
        actual tail tokens — no teacher. Isolates the soft-KL/self-distillation objective from the context
        truncation (identical truncation; only the loss differs from the kl variant). One forward."""
    dev = "cuda"
    full_ids, roles = [], []
    for e in events:
        s = tok(f"[{e['label']}] {e['content']}\n", add_special_tokens=False).input_ids
        full_ids.extend(s); roles.extend([e["role"]] * len(s))
    L = len(full_ids)
    if L < 4 or L > max_ctx:
        return None
    split = min(L - 2, max(1, int(L * (1.0 - keep_frac))))      # student keeps [split:L]; teacher keeps all
    want = ("act",) if mask_think else ("think", "act")
    tgt = [j for j in range(split + 1, L) if roles[j] in want]  # think/act tokens in the recent tail
    if not tgt:
        return None
    teacher_logits, moff = None, 0
    if loss_type == "kl":
        mem = (mem_ids or [])[-mem_cap:]; moff = len(mem)       # teacher: full (+memory) soft distributions
        set_gen_mode(student)
        t_in = torch.tensor([mem + full_ids], dtype=torch.long, device=dev)
        t_pos = torch.arange(pos_start, pos_start + moff + L, device=dev).unsqueeze(0)
        with torch.no_grad():
            teacher_logits = student(input_ids=t_in, position_ids=t_pos).logits[0]   # (moff+L, V)
    # student: recent tail only — must recover the dropped early half from ΔW
    set_train_mode(student)
    s_ids = full_ids[split:]
    s_in = torch.tensor([s_ids], dtype=torch.long, device=dev)
    s_pos = torch.arange(pos_start, pos_start + len(s_ids), device=dev).unsqueeze(0)
    s_logits = student(input_ids=s_in, position_ids=s_pos).logits[0]             # (L-split, V)
    if loss_type == "ce":                                      # CONTROL: hard CE on the same tail tokens
        sel = torch.stack([s_logits[(j - split) - 1] for j in tgt])
        tg = torch.tensor([full_ids[j] for j in tgt], dtype=torch.long, device=dev)
        loss = F.cross_entropy(sel, tg)
    else:
        losses = []
        for j in tgt:
            t_d = teacher_logits[moff + j - 1] / temp          # teacher predicts token j from the FULL prefix
            s_d = s_logits[(j - split) - 1] / temp             # student predicts token j from the TAIL prefix
            losses.append(F.kl_div(F.log_softmax(s_d, -1).unsqueeze(0), F.softmax(t_d, -1).unsqueeze(0),
                                   reduction="sum") * (temp * temp))
        loss = torch.stack(losses).sum() / len(losses)
    loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optim.step(); optim.zero_grad()
    return loss.item()


def window_mask(T, W, device, dtype):
    """4D additive causal BAND mask [1,1,T,T]: query q attends to the W most-recent keys (q-W < k <= q) — a
    sliding DIAGONAL window. Over a [P=W-1 prompts | trajectory] sequence, the prompts cover the 'top triangle'
    (the upper-left deficit where the band runs off the start), completing a full-width-W diagonal for EVERY
    token: the first real token's window = [all W-1 prompts ⊕ itself], later tokens slide into all-real."""
    q = torch.arange(T, device=device)[:, None]
    k = torch.arange(T, device=device)[None, :]
    allowed = (k <= q) & (k > q - W)
    m = torch.zeros(T, T, device=device, dtype=dtype)
    m.masked_fill_(~allowed, float("-inf"))
    return m[None, None]


def windowed_episode(student, optim, trainable, tok, events, pos_start, prompt_param, W,
                     temp=1.0, mask_think=False, loss_type="ce", max_ctx=3000):
    """NO-COLD-START windowed finetuning = a sliding DIAGONAL WINDOW of width W, with P=W-1 trainable prompts
    covering the 'top triangle' so EVERY token (incl. the first) is predicted from a full W-wide context: the
    prompts fill the early deficit, then the window slides into all-real tokens. Supervise think/act — CE on
    the actual tokens (default) or KL toward a full-context teacher. Only the 8 softmax layers honor the mask."""
    dev = "cuda"
    full_ids, roles = [], []
    for e in events:
        s = tok(f"[{e['label']}] {e['content']}\n", add_special_tokens=False).input_ids
        full_ids.extend(s); roles.extend([e["role"]] * len(s))
    if len(full_ids) > W:                              # keep last W so the window covers the WHOLE kept trajectory
        full_ids, roles = full_ids[-W:], roles[-W:]
    L = len(full_ids)
    if L < 2:
        return None
    want = ("act",) if mask_think else ("think", "act")
    tgt = [j for j in range(1, L) if roles[j] in want]
    if not tgt:
        return None
    P = prompt_param.shape[0]
    teacher_logits = None
    if loss_type == "kl":                                          # optional full-context teacher (no window/prompt)
        set_gen_mode(student)
        with torch.no_grad():
            t_pos = torch.arange(pos_start, pos_start + L, device=dev).unsqueeze(0)
            teacher_logits = student(input_ids=torch.tensor([full_ids], device=dev), position_ids=t_pos).logits[0]
    set_train_mode(student)
    emb = student.get_input_embeddings()
    traj_emb = emb(torch.tensor([full_ids], device=dev))           # (1, L, H)
    pe = prompt_param.to(traj_emb.dtype).unsqueeze(0)              # (1, P, H)  trainable prior-context
    inp = torch.cat([pe, traj_emb], dim=1)                          # (1, P+L, H)
    Tt = P + L
    pos = torch.arange(pos_start, pos_start + Tt, device=dev).unsqueeze(0)
    mask = window_mask(Tt, W, dev, traj_emb.dtype)   # diagonal window; the P=W-1 prompts cover the top triangle
    s_logits = student(inputs_embeds=inp, attention_mask=mask, position_ids=pos).logits[0]   # (P+L, V)
    if loss_type == "ce":
        sel = torch.stack([s_logits[P + j - 1] for j in tgt])      # predicts token j (seq pos P+j) from window
        tg = torch.tensor([full_ids[j] for j in tgt], dtype=torch.long, device=dev)
        loss = F.cross_entropy(sel, tg)
    else:
        losses = [F.kl_div(F.log_softmax(s_logits[P + j - 1] / temp, -1).unsqueeze(0),
                           F.softmax(teacher_logits[j - 1] / temp, -1).unsqueeze(0),
                           reduction="sum") * (temp * temp) for j in tgt]
        loss = torch.stack(losses).sum() / len(losses)
    loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optim.step(); optim.zero_grad()
    return loss.item()


def generate_windowed(model, tok, prompt, prompt_param, W, max_new_tokens=64, sample=False):
    """Deploy-side windowed generation, matching the training parallelogram: the next-token query sees the last
    W positions of [prompt-block ⊕ accumulated O/T/A]. The soft-prompt PADS the initial (short) context and
    SLIDES OUT as the real context accumulates, so the total context length is held at W (prompt count =
    W - real_len, dropping to 0 once real_len ≥ W)."""
    dev = "cuda"
    ids = tok(prompt, add_special_tokens=False).input_ids[-W:]      # last W real O/T/A tokens
    rlen = len(ids)
    emb = model.get_input_embeddings()
    ctx = emb(torch.tensor([ids], device=dev))
    n_p = max(0, W - rlen)                                          # prompts pad the deficit; slide out as real grows
    if n_p > 0:
        pe = prompt_param[-n_p:].to(ctx.dtype).unsqueeze(0)        # the W-real_len prompts nearest the trajectory
        inp = torch.cat([pe, ctx], dim=1)                          # total length held at W
    else:
        inp = ctx
    kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(inputs_embeds=inp, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0].tolist(), skip_special_tokens=True)    # inputs_embeds => out is only new tokens


def pstart(args, G, ep):
    """RoPE position offset for a consolidation step. persistent: never-reset clock G.
    reset: always 0. compressed: ep*stride so the persistent ordering survives but positions stay
    under the RoPE max even for long lifetimes (G itself still advances for buffer/recall bookkeeping)."""
    if args.pos_mode == "reset":
        return 0
    if args.pos_mode == "compressed":
        return ep * args.pos_stride
    return G


def _dump(args, log, G, wall, partial=False):
    """Write the running summary. Called every 10 episodes (partial=True) so a job that hits the
    walltime cap still leaves a usable reward-vs-episode curve, and once at the end (partial=False)."""
    rewards = [r["reward"] for r in log]
    third = max(1, len(rewards) // 3)
    summary = {
        "integration": args.integration, "n_episodes": len(log), "G_final": G, "wall_s": wall,
        "partial": partial,
        "reward_overall": sum(rewards) / len(rewards),
        "reward_first_third": sum(rewards[:third]) / third,
        "reward_last_third": sum(rewards[-third:]) / third,
        "success_rate": sum(1 for r in rewards if r > 0.95) / len(rewards),
        "completed": sum(1 for r in log if r["done"]),
        "log": log,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integration", required=True,
                    choices=["frozen", "everything", "reward_gated", "hybrid", "expert", "context_accum", "distill"])
    ap.add_argument("--distill-temp", type=float, default=1.0, help="distill: softmax temperature for the KL teacher/student.")
    ap.add_argument("--distill-window", type=int, default=0,
                    help="NO-COLD-START windowed finetuning: sliding diagonal window of width W (0=off). Every "
                         "token predicts from a full W-wide context; P=W-1 trainable prompts cover the top "
                         "triangle (early deficit). Windows the 8 softmax layers; trains+deploys windowed (reset).")
    ap.add_argument("--distill-n-prompts", type=int, default=0,
                    help="# trainable prompt tokens covering the top triangle; 0 => W-1 (full diagonal window).")
    ap.add_argument("--distill-loss", choices=["kl", "ce"], default="kl",
                    help="distill: 'kl' = soft KL toward a full-context teacher (real distillation); "
                         "'ce' = CONTROL, hard cross-entropy on the same truncated-student tail tokens, no "
                         "teacher — isolates the KL objective from the context truncation.")
    ap.add_argument("--distill-keep-frac", type=float, default=0.5,
                    help="distill: fraction of the trajectory the STUDENT keeps as context (the recent tail); "
                         "teacher always sees the full trajectory. 0.5 = 'start from the middle'. Smaller = more "
                         "aggressive (more history pushed into ΔW, but riskier); 1.0 = no distillation.")
    ap.add_argument("--distill-mask-think", action="store_true", help="distill: KL on act tokens only (skip think).")
    ap.add_argument("--distill-teacher-memory", action="store_true",
                    help="distill: roll out WITH a windowed cross-episode memory and give the KL teacher that memory "
                         "prefix (its think retrieves past episodes); the student distills it from the bare current obs "
                         "-> innate weight-based recall. Implies a windowed memory (bounded by --ctx-budget).")
    ap.add_argument("--ctx-budget", type=int, default=16384,
                    help="context_accum: token budget for past-episode memory carried in the prompt "
                         "(most-recent kept). Bounds the in-context rival to weight consolidation.")
    ap.add_argument("--act-mode", action="store_true", help="rollout WITHOUT a think step (Act). For CCD-act / instruction-tuning rows.")
    ap.add_argument("--no-history", action="store_true", help="memoryless rollout: context = current obs only (instruction tuning).")
    ap.add_argument("--single-step", action="store_true", help="consolidate each (obs->act) step separately, no history (instruction tuning).")
    ap.add_argument("--adapter", required=True, help="starting ReAct-WM adapter dir (contains adapter/)")
    ap.add_argument("--start-idx", type=int, default=150, help="goal slice start in train_70790 (held-out from training)")
    ap.add_argument("--n-episodes", type=int, default=80)
    ap.add_argument("--sids-file", default=None, help="Word2World webshop_test.json: use its exact goal session-ids (item_id suffix), same goals as the frozen task-success eval. 'expert' integration unavailable with this.")
    ap.add_argument("--reward-thresh", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-decay", action="store_true",
                    help="cosine-decay the consolidation LR from --lr to 0 across the lifetime (drift control: "
                         "many online steps at a fixed LR push ΔW away from the good init).")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="if >0, DEPLOY an exponential moving average of the LoRA weights (decay=this, e.g. 0.99): "
                         "train live ΔW but roll out / measure reward on the EMA — smooths the online drift. 0 = off.")
    ap.add_argument("--pos-mode", choices=["persistent", "reset", "compressed"], default="persistent",
                    help="persistent: consolidate at the never-reset clock G (component 7). "
                         "reset: standard RoPE positions (pos_start=0) every consolidation -> continual WITHOUT the time clock, "
                         "to isolate the impact of persistent time (7) from continual consolidation (9). "
                         "compressed: bucket each episode into a fixed --pos-stride (pos_start=ep*stride) so the persistent "
                         "timeline keeps its ordering but stays UNDER the RoPE max for long lifetimes — the fix for the "
                         "n>~150 decline where G exhausts the trained position range.")
    ap.add_argument("--pos-stride", type=int, default=1200,
                    help="compressed pos-mode: RoPE positions allotted per episode (pos_start=ep*stride). 1200 -> 200 eps ~240K, under 262K.")
    ap.add_argument("--mask-obs", action="store_true",
                    help="everything: supervise only {think,act} (match static experience-tuning loss), not obs. "
                         "Control for the audit finding that CCD trains on obs while static ET masks it.")
    ap.add_argument("--fewshot-file", default=None,
                    help="Prepend k exemplar trajectories (training [label] format) as a FIXED few-shot prefix each "
                         "episode (finetuned x static-few-shot at test).")
    ap.add_argument("--fewshot-k", type=int, default=1)
    ap.add_argument("--carry-context", action="store_true",
                    help="With an integration mode: ALSO carry a windowed in-context memory prefix (last ctx-budget "
                         "tokens of past episodes) while consolidating to ΔW — memory in context AND weights.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(0); torch.manual_seed(0)
    if not wait_for_server(SERVER_URL):
        print("ERROR: server down"); sys.exit(1)
    print("server up")
    if args.sids_file:
        # W2W test goals: identical session-ids to the frozen task-success eval (item_id suffix)
        goals = [{"id": int(it["item_id"].split("_")[-1])} for it in json.load(open(args.sids_file))][:args.n_episodes]
    else:
        goals = json.load(open(GOALS_FILE))[args.start_idx: args.start_idx + args.n_episodes]
    print(f"[lifetime:{args.integration}] {len(goals)} goals (sids_file={bool(args.sids_file)})")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    student = PeftModel.from_pretrained(base, str(Path(args.adapter) / "adapter"), is_trainable=True)
    trainable = [p for p in student.parameters() if p.requires_grad]
    prompt_param = None
    if args.distill_window > 0:                                    # trainable prior-context prompts (no cold-start)
        H = student.get_input_embeddings().weight.shape[1]
        P = args.distill_n_prompts or (args.distill_window - 1)    # W-1 prompts cover the top triangle
        prompt_param = torch.nn.Parameter(torch.randn(P, H, device="cuda", dtype=torch.float32) * 0.02)
        trainable = trainable + [prompt_param]
        print(f"[windowed] W={args.distill_window} P={P} trainable prompts (H={H})")
    optim = None if args.integration in ("frozen", "context_accum") else torch.optim.AdamW(trainable, lr=args.lr)
    ema = [p.detach().clone() for p in trainable] if (args.ema > 0 and optim is not None) else None
    env_idx = create_env()

    # Fixed few-shot exemplar prefix (finetuned x static-few-shot at test): render k training-format trajs once.
    fewshot_prefix = ""
    if args.fewshot_file:
        ex = json.load(open(args.fewshot_file))[:args.fewshot_k]
        parts = ["\n".join(f"[{e['label']}] {e['content']}" for e in tr["events"]) for tr in ex]
        fewshot_prefix = "Here are example shopping episodes:\n" + "\n\n".join(parts) + "\n\nNow your task:\n"

    G = 0; buffer = []   # buffer: (ids, roles, G_start) of high-reward episodes
    ctx_mem = []         # context_accum / carry_context: accumulated (label,text) events from PAST episodes
    windowed = args.integration == "context_accum" or args.carry_context or args.distill_teacher_memory
    log = []; t0 = time.time()
    for ep, demo in enumerate(goals):
        sid = int(demo["id"])
        set_gen_mode(student)
        prefix = fewshot_prefix; mem_ids = None
        if windowed and ctx_mem:
            # carry the most-recent ctx_budget tokens of past episodes in the prompt (memory-in-context)
            mids = tok(("\n".join(f"[{l}] {t}" for l, t in ctx_mem)), add_special_tokens=False).input_ids
            prefix += "Previous shopping episodes you have done (memory):\n" + \
                      tok.decode(mids[-args.ctx_budget:]) + "\n\nNow the new task:\n"
            if args.distill_teacher_memory:
                mem_ids = tok(prefix, add_special_tokens=False).input_ids   # teacher sees this; student won't
        if optim is not None and args.lr_decay:                  # cosine LR -> 0 over the lifetime (drift control)
            lr_t = 0.5 * args.lr * (1 + math.cos(math.pi * ep / max(1, len(goals))))
            for g in optim.param_groups: g["lr"] = lr_t
        if ema is not None:                                      # DEPLOY the EMA weights for this rollout...
            _live = [p.detach().clone() for p in trainable]
            for p, e in zip(trainable, ema): p.data.copy_(e)
        with torch.no_grad():
            events, reward, done, nsteps = rollout_react(student, tok, env_idx, sid, prefix=prefix,
                                                          act_only=args.act_mode, no_history=args.no_history,
                                                          prompt_param=prompt_param, window=args.distill_window)
        if ema is not None:                                      # ...then restore live weights for consolidation
            for p, l in zip(trainable, _live): p.data.copy_(l)
        log.append({"ep": ep, "sid": sid, "reward": reward, "done": done, "n_steps": nsteps, "G": G})

        if windowed:
            ctx_mem.extend((e["label"], e["content"]) for e in events)   # accumulate memory in CONTEXT
        if args.integration not in ("frozen", "context_accum"):          # ...and/or consolidate into ΔW
            if args.integration == "expert":
                ids, roles = flat_ids_roles(expert_events(demo), tok)
                integrate(student, optim, trainable, ids, roles, G, {"act"}, pstart(args, G, ep))   # BC on expert actions
                G += min(len(ids), 2400)
            elif args.integration == "distill":
                ids, _ = flat_ids_roles(events, tok)
                if args.distill_window > 0:                       # windowed uniform-context finetuning (no cold-start)
                    windowed_episode(student, optim, trainable, tok, events, pstart(args, G, ep), prompt_param,
                                     args.distill_window, temp=args.distill_temp,
                                     mask_think=args.distill_mask_think, loss_type=args.distill_loss)
                else:                                             # context distillation: teacher-full || student-tail
                    distill_episode(student, optim, trainable, tok, events, pstart(args, G, ep),
                                    temp=args.distill_temp, keep_frac=args.distill_keep_frac,
                                    mask_think=args.distill_mask_think, mem_ids=mem_ids, loss_type=args.distill_loss)
                G += min(len(ids), 2400)
            elif args.single_step:
                # continual instruction tuning: consolidate each (obs -> act) step ALONE, no history
                cur = None
                for e in events:
                    if e["role"] == "obs": cur = [e]
                    elif e["role"] == "act" and cur is not None:
                        pids, proles = flat_ids_roles(cur + [e], tok)
                        integrate(student, optim, trainable, pids, proles, G, {"act"}, pstart(args, G, ep))
                        G += min(len(pids), 2400); cur = None
            else:
                ids, roles = flat_ids_roles(events, tok)
                if args.integration == "everything":
                    lr_roles = {"think", "act"} if args.mask_obs else {"obs", "think", "act"}
                else:  # reward_gated or hybrid: always world-model obs; reinforce policy only if good
                    lr_roles = {"obs", "think", "act"} if reward >= args.reward_thresh else {"obs"}
                integrate(student, optim, trainable, ids, roles, G, lr_roles, pstart(args, G, ep))
                gstart = G; G += min(len(ids), 2400)
                if reward >= args.reward_thresh:
                    buffer.append((ids[:2400], roles[:2400], gstart, ep))
                if args.integration == "hybrid" and buffer:
                    rids, rroles, rg, rep = random.choice(buffer)      # recall-replay at the episode's original positions
                    integrate(student, optim, trainable, rids, rroles, rg, {"think", "act"}, pstart(args, rg, rep))
            if ema is not None:                                  # track the EMA of the freshly-updated live ΔW
                for e, p in zip(ema, trainable): e.mul_(args.ema).add_(p.detach(), alpha=1 - args.ema)

        if ep % 5 == 0:
            recent = [r["reward"] for r in log[-20:]]
            print(f"  ep {ep} sid={sid} reward={reward:.2f} done={done} G={G} | recent20_avg={sum(recent)/len(recent):.3f}")
        if ep % 10 == 0 and ep > 0:
            _dump(args, log, G, time.time() - t0, partial=True)   # crash/timeout-safe checkpoint

    wall = time.time() - t0
    summary = _dump(args, log, G, wall, partial=False)
    print(f"\n[{args.integration}] reward first→last third: {summary['reward_first_third']:.3f} -> {summary['reward_last_third']:.3f}"
          f"  overall={summary['reward_overall']:.3f}  Gfinal={G}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
