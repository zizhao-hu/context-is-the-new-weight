# Proposal: Context is the New Weight

> NeurIPS 2026 sprint plan. Last updated 2026-05-02.

## Thesis

A context `C` and a weight delta `ΔW` are *functionally equivalent* if
`M(·|C; θ) ≈ M(·|∅; θ + ΔW)`. We test this empirically: distill `C` into the
weights of Llama-3.1-8B-Instruct, then ask whether `ΔW`'s effect on the
residual stream reproduces what the KV cache of `C` did in the base model.

## Hypotheses

- **H1** Stylistic / persona / guideline contexts achieve high behavioral equivalence after full FT on ~200 self-generated `(Q, A)` pairs.
- **H2** The factual context (specific facts injected via system prompt) is harder to internalize and partially leaks at inference time.
- **H3** `v_ctx` and `v_dw` align (high cosine) at a small set of layers — the same layers where attention from query tokens to context tokens carries information in the teacher.

## The experiment (template, run 5×)

1. **Phase 1 — Synthetic dataset.** With context `C` fixed, sample ~200 queries spanning Q&A / summarization / translation / code / recall use cases. Run base model on `[C; Q]`, store `(Q, A, top-k logits)`.
2. **Phase 2 — Full fine-tune.** Train fresh Llama-3.1-8B-Instruct on `(Q → A)` with no context. CE loss on answer tokens. bf16 + 8-bit AdamW + grad checkpointing. Save trained model.
3. **Phase 3 — Behavior verification (gate).** Run trained model on its training queries with no context; compare to teacher answers. Hard gate before Phase 4.
4. **Phase 4 — KV-vs-ΔW.** On a held-out probe set, capture residual stream at every post-attn and post-MLP site (64 sites for Llama-3.1-8B). Compute `v_ctx_s = resid_s(Q'|base, [C;Q']) − resid_s(Q'|base, Q')` and `v_dw_s = resid_s(Q'|trained, Q') − resid_s(Q'|base, Q')`. Compare per site: cosine, RMSNorm cosine, token-space cosine, induced KL.

## Five contexts

| # | Name | Type | Example |
|---|---|---|---|
| 1 | haiku | tone | "Always answer in haiku form (5-7-5)." |
| 2 | pirate | persona | "You are a pirate. Speak only in pirate slang." |
| 3 | concise | guideline | "Be extremely concise. One short sentence." |
| 4 | fewshot_translate_fr | fewshot | 3 worked EN→FR translation shots |
| 5 | factual | factual | "Alice's password is 7G2K9X4M…" + 4 more facts |

## Paper outline (NeurIPS 2026, 9 pages)

| § | Title | Pages |
|---|---|---|
| 1 | Introduction | 1.25 |
| 2 | Setup (equivalence definitions, model, protocol) | 1.0 |
| 3 | Forward equivalence (per-context distillation, behavior verification) | 2.5 |
| 4 | KV vs ΔW: where context lives (the headline) | 2.5 |
| 5 | Discussion | 0.75 |
| 6 | Limitations | 0.75 |
| | References + checklist | unlimited |

**Headline figure:** 64-point per-site alignment curve, one line per context.
**Headline table:** peak-alignment site per context.
**Title (working):** *Context is the New Weight: Distilling In-Context Adaptation into Weight Updates and Comparing the Mechanism*

## Compute

- Per context: ~3-4 GPU-hr end-to-end (Phase 1 ~10 min, Phase 2 ~30 min full-FT, Phase 3 ~5 min, Phase 4 ~30 min).
- 5 contexts × ~3.5 hr = ~17 GPU-hr serial.
- Endeavour A6000 (47GB) fits 8B + 8-bit AdamW + grad checkpointing.
- 8-GPU concurrent cap → 5 contexts run in parallel; wall ~3.5 hr.

## Critical-path risks

- **Phase 3 fails for some context** (model doesn't internalize the context's behavior). Re-run Phase 2 with more epochs / data; if still failing, drop that context from the paper.
- **`v_ctx` ≠ `v_dw` cosine looks random.** That would mean ΔW reproduces the *output* of context but not the *mechanism* — still a publishable negative result but reframes §4.
- **OOM at full-FT 8B.** Fall back to grad accumulation 16, then to A100-80GB if needed (longer queue).

## See also

- [`experiments/01_synth_distill_kvdw/README.md`](experiments/01_synth_distill_kvdw/README.md) — concrete run instructions
- [`paper/main.tex`](paper/main.tex), [`paper/checklist.tex`](paper/checklist.tex) — submission scaffolding
