# Experiment 03 — Inverting ΔW back to a Context

## Question

Given a finetuned model (or a student from Experiment 01 with known `ΔW`), can we recover a context `C*` such that `M(· | C*) ≈ M_student(· | ∅)`?

If yes, fine-tuning is invertible: a weight delta carries enough information to reconstruct the demonstration that produced it. This would be a strong form of "context = weight" — the map is two-way.

## Three approaches to try

### A. Direct context optimization (soft prompts)

Treat `C*` as a sequence of `m` learnable embeddings prepended to the input. Freeze base `M`. Optimize `C*` to minimize KL between `M([C*; Q'])` and `M_student(Q')` on a probe set `Q'`.

- Pro: clean, well-defined optimization.
- Con: yields embedding-space context, not natural language. Need a separate "soft → discrete" decoding step (e.g., nearest-token, GCG-style).

### B. Activation matching

For each layer where Experiment 02 says `ΔW` lives, set up: find `C*` such that `M([C*; Q'])`'s residual stream at layer ℓ matches `M_student(Q')`'s residual stream at layer ℓ. Use Experiment 02's per-layer attribution to weight the loss.

- Pro: leverages mechanistic insight.
- Con: requires a clean attribution map.

### C. ROME-style analysis (inverse of MEMIT)

Treat `ΔW` as an MLP key-value edit and read out the (k, v) pair. Convert `v` back to a token-space description.

- Pro: closed form for low-rank edits.
- Con: assumes `ΔW` lives in a specific MLP layer, which may not match what Experiment 02 finds.

## Sanity check

Before claiming inversion works in general, run on a known case: train a student from Experiment 01 on `C = haiku`, then check whether any of A/B/C recovers a context that, when prepended, produces haiku output from the *base* model. If yes, the recovered `C*` should rhyme with the original system prompt under nearest-token decoding.

## Status

Not started — depends on Experiments 01 and 02. The hardest of the three; likely the main contribution if it works.
