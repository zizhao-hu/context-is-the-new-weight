"""Train a ReAct-WM agent: the model learns to THINK (predict outcome + reason) then ACT.

Input: think-augmented trajectories [{instruction, events:[{label,role,content}]}] with roles
obs / think / act (from gen_think_annotations.py).

Rendered flat per trajectory:
    [obs_0] ...
    [think_0] Prediction: ... Reason: ...
    [act_0] search[...]
    [obs_1] ...
    [think_1] ...
    [act_1] click[...]
    ...

LOSS IS ON think + act TOKENS ONLY (the agent's outputs). Observations are context, NOT
predicted — this is what stops the all-token model from drifting into observation-spewing.
The world-model knowledge lives inside the think text (the "Prediction:" clause), so training
the think tokens teaches outcome-anticipation in language AND the reasoning that uses it.

--stream: process windows in trajectory order with a persistent accumulating memory (experience
tuning). Default: i.i.d. shuffled (SFT-style). This flag is the ET-vs-iid knob.
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/project2/jessetho_1732/zizhaoh/context-is-the-new-weight")
from src import peft_adapter

LOSS_ROLES = {"think", "act"}            # the agent's outputs
LABELS = {"obs": "obs", "think": "think", "act": "act"}


def render_with_roles(events, tok):
    ids = []; roles = []
    for e in events:
        seg = tok(f"[{e['label']}] {e['content']}\n", add_special_tokens=False).input_ids
        ids.extend(seg); roles.extend([e["role"]] * len(seg))
    return ids, roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-file", required=True)
    ap.add_argument("--lora-r", type=int, default=256)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="base model repo id")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA: load the base in 4-bit (nf4) for large models on a single GPU")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="If >0: hard cap on optimizer steps (safety ceiling for early-stopping).")
    ap.add_argument("--val-file", default=None,
                    help="Held-out think-augmented val set. Enables EARLY-STOPPING to convergence: train until "
                         "val-loss stops improving (same criterion for every cell -> fair scaling comparison).")
    ap.add_argument("--patience", type=int, default=4, help="stop after this many val evals with no improvement")
    ap.add_argument("--eval-every", type=int, default=15, help="evaluate val-loss every N optimizer steps")
    ap.add_argument("--min-steps", type=int, default=15, help="do not early-stop before this many optimizer steps")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--n-epochs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--stream", action="store_true", help="trajectory-order streaming (experience tuning) vs iid")
    ap.add_argument("--also-obs", action="store_true", help="also put loss on observation tokens (joint world model)")
    ap.add_argument("--n-wm", type=int, default=0,
                    help="If >0: split each trajectory's event list into consecutive non-overlapping "
                         "n_wm-event windows; each window is a training item. 0 = whole-trajectory (default).")
    ap.add_argument("--chunk", action="store_true",
                    help="With n_wm==0: tile each trajectory into consecutive <=max_tokens token-chunks "
                         "covering ALL events (no truncation); each chunk is a training item. Ensures the "
                         "whole trajectory is consumed regardless of length.")
    ap.add_argument("--recall", action="store_true",
                    help="After each LM step, add a recall step: feed [recall_prompt, window] and put "
                         "CE loss (think+act mask) on predicting the window's tokens. Backward accumulates.")
    ap.add_argument("--recall-prompt", default="Recall the previous experience:\n",
                    help="Text fed before the target window during recall step.")
    ap.add_argument("--persistent-rope", action="store_true",
                    help="Use a monotonically increasing global token counter as RoPE position_ids "
                         "across all training windows, so every event has a unique absolute position.")
    ap.add_argument("--position-recall", action="store_true",
                    help="Extra recall step on a RANDOM PAST window with the recall cue placed JUST BEFORE "
                         "that past window's ORIGINAL training positions. Requires --persistent-rope.")
    ap.add_argument("--win", type=int, default=0,
                    help="Design-A sliding window: if >0, tile each FULL (un-truncated) trajectory into "
                         "overlapping token windows of this size; each window is a training item tagged with "
                         "its absolute start position in the trajectory. Covers the whole trajectory.")
    ap.add_argument("--stride", type=int, default=0,
                    help="Stride for --win sliding window (default win//2 => 50%% overlap).")
    ap.add_argument("--traj-rope", action="store_true",
                    help="Design A: intra-trajectory persistent RoPE. position_ids = the window's true "
                         "absolute positions within its trajectory (grows across chunks, RESETS at each new "
                         "trajectory). Pairs with --win. Mutually exclusive with --persistent-rope (lifecycle).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-adapter", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build windows, print stats, exit before model load")
    args = ap.parse_args()
    if args.position_recall and not args.persistent_rope:
        ap.error("--position-recall requires --persistent-rope")
    if args.traj_rope and args.persistent_rope:
        ap.error("--traj-rope (intra-trajectory) and --persistent-rope (lifecycle) are mutually exclusive")
    if args.stride == 0 and args.win > 0:
        args.stride = max(1, args.win // 2)

    loss_roles = set(LOSS_ROLES) | ({"obs"} if args.also_obs else set())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(0); torch.manual_seed(0)
    data = json.load(open(args.aug_file))
    print(f"[react] {len(data)} augmented trajs | loss on {sorted(loss_roles)} | stream={args.stream}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Build training windows. n_wm==0 (default): ONE whole-trajectory window per traj, truncated to
    # max_tokens from the FRONT. n_wm>0: split each trajectory's EVENT list into consecutive
    # non-overlapping n_wm-event windows; each window is a training item.
    def build_items(src):                    # same windowing logic for train and validation
        # Each item is (ids, roles, abs_start). abs_start = the window's true start position within
        # its trajectory (0 for single-window items); --traj-rope (design A) uses it for intra-traj
        # RoPE that grows across chunks and resets per trajectory.
        out = []
        if args.win > 0:                     # Design A: sliding window over the FULL trajectory
            for traj in src:
                ids, roles = render_with_roles(traj["events"], tok)
                n = len(ids)
                starts = list(range(0, max(1, n - args.win + 1), args.stride))
                if starts and starts[-1] + args.win < n:
                    starts.append(n - args.win)        # ensure the trajectory tail is covered
                for s in starts:
                    w_ids = ids[s:s + args.win]; w_roles = roles[s:s + args.win]
                    if len(w_ids) >= 2 and any(r in loss_roles for r in w_roles[1:]):
                        out.append((w_ids, w_roles, s))
        elif args.n_wm > 0:
            for traj in src:
                evs = traj["events"]
                for s in range(0, len(evs), args.n_wm):
                    ids, roles = render_with_roles(evs[s:s + args.n_wm], tok)
                    if len(ids) > args.max_tokens:
                        ids = ids[-args.max_tokens:]; roles = roles[-args.max_tokens:]
                    if len(ids) >= 2 and any(r in loss_roles for r in roles[1:]):
                        out.append((ids, roles, 0))
        elif args.chunk:
            for traj in src:
                cur = []; cur_len = 0
                for e in traj["events"]:
                    L = len(tok(f"[{e['label']}] {e['content']}\n", add_special_tokens=False).input_ids)
                    if cur and cur_len + L > args.max_tokens:
                        ids, roles = render_with_roles(cur, tok)
                        if len(ids) >= 2 and any(r in loss_roles for r in roles[1:]):
                            out.append((ids, roles, 0))
                        cur = []; cur_len = 0
                    cur.append(e); cur_len += L
                if cur:
                    ids, roles = render_with_roles(cur, tok)
                    if len(ids) > args.max_tokens:
                        ids = ids[-args.max_tokens:]; roles = roles[-args.max_tokens:]
                    if len(ids) >= 2 and any(r in loss_roles for r in roles[1:]):
                        out.append((ids, roles, 0))
        else:
            for traj in src:
                ids, roles = render_with_roles(traj["events"], tok)
                if len(ids) > args.max_tokens:
                    ids = ids[-args.max_tokens:]; roles = roles[-args.max_tokens:]
                if len(ids) >= 2 and any(r in loss_roles for r in roles[1:]):
                    out.append((ids, roles, 0))
        return out
    items = build_items(data)
    val_items = build_items(json.load(open(args.val_file)))[:40] if args.val_file else []
    n_sup = sum(sum(1 for r in roles[1:] if r in loss_roles) for _, roles, _ in items)
    print(f"[react] {len(items)} windows (n_wm={args.n_wm}), {n_sup} supervised (think+act) target tokens")
    if args.win > 0:
        starts_all = [a for _, _, a in items]
        lens = [len(i) for i, _, _ in items]
        print(f"[react][designA] win={args.win} stride={args.stride} | windows={len(items)} "
              f"max_abs_start={max(starts_all) if starts_all else 0} max_win_len={max(lens) if lens else 0} "
              f"(intra-traj positions, reset per trajectory)")
    if args.dry_run:
        print("[react] DRY-RUN done (no model load)"); return

    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                    dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0})
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
        try:
            base.gradient_checkpointing_enable(); base.enable_input_require_grads()
        except Exception: pass
    student = peft_adapter.make_adapter(base, method="lora", num_virtual_tokens=16,
                                        lora_r=args.lora_r, lora_alpha=args.lora_r * 2)
    trainable = [p for p in student.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)
    opt_steps = max(1, (len(items) // args.accum)) * args.n_epochs
    if args.max_steps > 0: opt_steps = min(opt_steps, args.max_steps)   # cap LR schedule to the step budget
    warmup = max(10, int(0.05 * opt_steps))
    def lr_at(s):
        if s < warmup: return s / warmup
        p = (s - warmup) / max(1, opt_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    student.train()

    # Persistent-RoPE global token counter (grows monotonically) + past-window buffer for
    # position-recall. _rng is separate from the data-shuffle RNG so toggling recall doesn't
    # perturb the iid shuffle order.
    G_token = 0
    past_windows = []                        # list of (G_window_start, ids, roles)
    PAST_BUFFER_SIZE = 100
    _rng = random.Random(0)

    def _think_act_mask_full(roles_seq):
        # CE mask over ALL window tokens (recall steps predict the full window from the cue, so the
        # target is the whole window t[0,:], length T). Keep think+act (NOT obs).
        return torch.tensor([r in LOSS_ROLES for r in roles_seq], dtype=torch.bool, device="cuda")

    def val_loss():                          # avg think+act CE on held-out val items (no_grad)
        student.eval(); tot = 0.0; ntok = 0
        with torch.no_grad():
            for ids, roles, _ in val_items:
                t = torch.tensor([ids], dtype=torch.long, device="cuda"); T = t.shape[1]
                out = student(t); extra = out.logits.shape[1] - T
                logits = out.logits[0, extra:extra + T - 1, :]; targets = t[0, 1:]
                keep = torch.tensor([r in loss_roles for r in roles[1:]], dtype=torch.bool, device="cuda")
                if keep.any():
                    tot += F.cross_entropy(logits[keep], targets[keep], reduction="sum").item(); ntok += int(keep.sum())
        student.train(); return tot / max(1, ntok)
    best_val = float("inf"); bad = 0; stop = False; saved_best = False
    t0 = time.time(); step = 0; micro = 0; optim.zero_grad()
    for ep in range(args.n_epochs):
        order = list(range(len(items)))
        if not args.stream:
            random.shuffle(order)            # iid; --stream keeps trajectory order (experience tuning)
        for idx in order:
            ids, roles, abs_start = items[idx]
            t = torch.tensor([ids], dtype=torch.long, device="cuda")
            T = t.shape[1]
            G_window_start = G_token
            # RoPE base for this window: intra-trajectory abs_start (design A, resets per traj) or
            # the lifecycle global counter (design B). rope_on gates whether we override positions.
            rope_on = args.persistent_rope or args.traj_rope
            base_pos = abs_start if args.traj_rope else G_window_start
            # --- LM forward (next-token CE on think+act tokens) ---
            if rope_on:
                pos = torch.arange(base_pos, base_pos + T, device="cuda").unsqueeze(0)
                out_l = student(input_ids=t, position_ids=pos)
            else:
                out_l = student(t)
            extra = out_l.logits.shape[1] - T
            logits = out_l.logits[0, extra:extra + T - 1, :]
            targets = t[0, 1:]
            keep = torch.tensor([r in loss_roles for r in roles[1:]], dtype=torch.bool, device="cuda")
            loss = F.cross_entropy(logits[keep], targets[keep])
            (loss / args.accum).backward(); micro += 1

            # --- Recall step: predict the window's tokens given only the recall cue in context ---
            if args.recall:
                recall_ids = tok(args.recall_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                P = recall_ids.shape[1]
                full_ids = torch.cat([recall_ids, t], dim=1)        # (1, P + T)
                if rope_on:
                    cue_pos = torch.arange(base_pos + T, base_pos + T + P, device="cuda")
                    tgt_pos = torch.arange(base_pos, base_pos + T, device="cuda")
                    out_r = student(input_ids=full_ids,
                                    position_ids=torch.cat([cue_pos, tgt_pos]).unsqueeze(0))
                else:
                    out_r = student(full_ids)
                T_full = full_ids.shape[1]
                extra_r = out_r.logits.shape[1] - T_full
                # logits at P-1..P+T-2 predict the FULL window tokens [0..T-1] of t (length T)
                preds_r = out_r.logits[0, extra_r + P - 1: extra_r + P + T - 1, :]
                keep_r = _think_act_mask_full(roles)            # length T, aligns with t[0,:]
                loss_r = F.cross_entropy(preds_r[keep_r], t[0, :][keep_r])
                (loss_r / args.accum).backward()

            # --- Position-recall step: retrieve a RANDOM PAST window via its ORIGINAL positions ---
            if args.position_recall and args.persistent_rope and past_windows and _rng.random() < 0.5:
                past_G_start, past_ids_list, past_roles = _rng.choice(past_windows)
                if any(r in LOSS_ROLES for r in past_roles):
                    past_t = torch.tensor([past_ids_list], dtype=torch.long, device="cuda")
                    past_T = past_t.shape[1]
                    recall_ids2 = tok(args.recall_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                    P2 = recall_ids2.shape[1]
                    full_ids2 = torch.cat([recall_ids2, past_t], dim=1)
                    cue_start = max(0, past_G_start - P2)            # cue just before the past window's positions
                    cue_pos2 = torch.arange(cue_start, cue_start + P2, device="cuda")
                    tgt_pos2 = torch.arange(past_G_start, past_G_start + past_T, device="cuda")
                    out_pr = student(input_ids=full_ids2,
                                     position_ids=torch.cat([cue_pos2, tgt_pos2]).unsqueeze(0))
                    T_full2 = full_ids2.shape[1]
                    extra_pr = out_pr.logits.shape[1] - T_full2
                    preds_pr = out_pr.logits[0, extra_pr + P2 - 1: extra_pr + P2 + past_T - 1, :]
                    keep_pr = _think_act_mask_full(past_roles)  # length past_T, aligns with past_t[0,:]
                    loss_pr = F.cross_entropy(preds_pr[keep_pr], past_t[0, :][keep_pr])
                    (loss_pr / args.accum).backward()

            # Save this window for future position-recall sampling.
            if args.position_recall and args.persistent_rope:
                past_windows.append((G_window_start, ids, roles))
                if len(past_windows) > PAST_BUFFER_SIZE:
                    past_windows.pop(0)

            # Advance the global token cursor (include the recall cue so it never overlaps).
            if args.persistent_rope:
                G_token += T + (P if args.recall else 0)

            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip)
                optim.step(); sched.step(); optim.zero_grad(); step += 1
                if step <= 12 or step % 25 == 1:
                    print(f"  ep={ep} opt_step={step}/{opt_steps} loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e}")
                if args.val_file and val_items and step >= args.min_steps and step % args.eval_every == 0:
                    vl = val_loss()
                    if vl < best_val - 1e-4:
                        best_val = vl; bad = 0
                        if args.save_adapter: student.save_pretrained(out / "adapter"); saved_best = True
                        print(f"  [val] step={step} val_loss={vl:.4f} (best -> saved)")
                    else:
                        bad += 1
                        print(f"  [val] step={step} val_loss={vl:.4f} (no-improve {bad}/{args.patience})")
                        if bad >= args.patience: stop = True
                if stop or (args.max_steps and step >= args.max_steps): break
        if stop or (args.max_steps and step >= args.max_steps): break
    if micro % args.accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, args.clip); optim.step(); optim.zero_grad()
    wall = time.time() - t0
    print(f"[react] trained {step} opt-steps in {wall:.0f}s")

    if args.save_adapter and not saved_best:
        student.save_pretrained(str(out / "adapter")); print(f"[react] saved final adapter -> {out/'adapter'}")
    elif saved_best:
        print(f"[react] kept BEST-val adapter (val_loss={best_val:.4f}); stopped early at step {step}")
    json.dump({"method": "react_wm", "stream": args.stream, "also_obs": args.also_obs,
               "n_wm": args.n_wm, "recall": args.recall, "recall_prompt": args.recall_prompt,
               "persistent_rope": args.persistent_rope, "position_recall": args.position_recall,
               "win": args.win, "stride": args.stride, "traj_rope": args.traj_rope,
               "n_trajs": len(data), "n_windows": len(items), "n_supervised_tokens": n_sup,
               "n_opt_steps": step, "lora_r": args.lora_r, "lr": args.lr, "n_epochs": args.n_epochs,
               "train_wall_s": wall},
              open(out / "summary.json", "w"), indent=2)
    print("[react] DONE")


if __name__ == "__main__":
    main()
