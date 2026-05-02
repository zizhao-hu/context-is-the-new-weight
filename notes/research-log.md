# Research Log

Append-only chronological notes. Newest entries at the top.

---

## 2026-05-02 — Sprint locked: synth + full-FT + KV-vs-ΔW

NeurIPS 2026 deadline confirmed (5/4 abstract, 5/6 paper). Committing to a single experiment template, run 5× (haiku, pirate, concise, fewshot_translate_fr, factual). Pivots from earlier scaffold:

- **Full FT** of Llama-3.1-8B-Instruct, not LoRA. ΔW is the entire weight delta, derived on-demand from `θ_C − θ_0` (we save trained models, not pickled deltas).
- **Synthetic dataset** generated per experiment: ~200 `(C, Q, A)` triples (~400 for factual to allow paraphrase repetition) with `A = M([C; Q])` greedy. Queries span Q&A / summarization / translation / code / recall use cases.
- **Behavior verification** is a hard gate: trained student must reproduce the context's behavior on its own training queries before Phase 4 runs.
- **KV-vs-ΔW comparison** is the headline analysis. At every post-attn and post-MLP residual-stream site (64 sites for 8B), compare:
  - `v_ctx_s` = perturbation in residual stream from prepending C (base model)
  - `v_dw_s` = perturbation from ΔW (no context, both runs)
  Cosine, RMSNorm cosine, token-space cosine, induced KL.

Code planted: `src/synth.py`, `src/use_cases.py`, `src/verify.py`, `src/kvdw.py`; `src/distill.py` extended with `train_full_ft()` (8-bit AdamW + grad ckpt). Phase scripts under `experiments/01_synth_distill_kvdw/`. SLURM runner reroutes outputs to `/scratch1/`.

Older 02/03 experiment plans moved to `experiments/extras/`.

Next: CARC env setup, sync repo, smoke test, then submit 5 jobs.

---

## 2026-05-02 — Experiments planted (superseded above)

Three-experiment scaffold:

- **01 — Context → Weight distillation** (runnable). Teacher rollout under `[C; Q]` saves top-20 logits per generated token. Student trained with forward KL on the top-k support, conditioning only on the no-context prompt. LoRA r=16 on `q,k,v,o` projections by default. Eval = behavioral KL on held-out queries along the teacher's greedy path + sampled student strings + LoRA singular-value spectrum.
- **02 — KV vs ΔW** (plan only). Layer/head-resolved decomposition of where context lives — teacher's attention contribution from `C` tokens vs student's ΔW residual.
- **03 — ΔW → C inversion** (plan only). Three approaches sketched: soft-prompt optimization, activation-matching with priors from exp 02, ROME-style read-out.

Open knobs (deferred ablations): teacher temperature, top-k value, LoRA target modules (attn vs +MLP), LoRA rank.

Default model `meta-llama/Llama-3.1-8B-Instruct` — gated, requires `huggingface-cli login`. Big enough that "context as a working alternative to fine-tuning" is a real claim, not a small-model artifact. 50 generic open-domain queries seeded at `data/queries.jsonl` — expand once we know which contexts need more diverse prompts.

---

## 2026-05-02 — NeurIPS 2026 template installed

- Pulled `Formatting_Instructions_For_NeurIPS_2026.zip` from `media.neurips.cc` (Call-for-Papers link).
- Zip contents: `neurips_2026.sty` (Jan 29 2026 revision), `neurips_2026.tex` (sample), `checklist.tex` (mandatory paper checklist as a separate file — easier than past years).
- Wired `\input{checklist}` into `main.tex` after the bibliography.
- Sample `neurips_2026.tex` kept in `paper/` as reference; can drop later if it clutters.

---

## 2026-05-02 — Project initialized

- Repo scaffolded: `paper/`, `src/`, `experiments/`, `data/`, `notes/`.
- Thesis stated in `README.md`: characterize when context change ≡ weight update.
- First decisions to make:
  - **Definition of equivalence.** Distributional vs. logit-exact vs. functional.
  - **Smallest non-trivial model class.** Linear attention is the obvious starting point — there's a closed-form context-to-weight map (von Oswald et al., 2023).
  - **First experiment.** Verify the linear-attention equivalence empirically on a synthetic regression task; measure tightness as we move toward softmax attention and depth > 1.
