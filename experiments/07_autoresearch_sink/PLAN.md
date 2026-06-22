# Autoresearch — Attention-Sink Thesis (CARC `nlp`, ≤8 GPU)

**Goal:** advance three claims for the attention-sink paper, autonomously filling idle GPU.

1. **Triangle mask *causes* the Type-1 (positional) sink.**
2. **Windowed training *spreads* the Type-1 sink onto Type-2 (no-op) tokens → better broadcasting.**
3. **Our windowed *training* beats post-hoc streaming techniques (built on the standard/triangle scheme) on real streaming tasks.**

**Models:** both families — fla 1.3B (transformer-softmax vs gla-gated, strict same-data control) + Qwen 0.5B/9B (strong sink, eager-attention capture, custom masks).
**Streaming battery (aim 3):** (a) long-stream perplexity + stability, (b) LongBench long-context QA, (c) broadcasting/retrieval probe.
**Budget:** keep ≤8 concurrent 1-GPU jobs on `nlp` (account jessetho_1732). Env: fla → `/scratch1/zizhaoh/envs/flax`; Qwen → DREAM. See [[reference_carc_fla_setup]].

## Experiment queue (status updated each iteration)
| id | aim | model | experiment | metric | status |
|----|-----|-------|------------|--------|--------|
| B0 | 1/2 | fla | ppl-vs-budget (chunk) transformer vs gla | ppl | DONE (tf≈gla, tf marginally ahead) |
| B1 | 1/3 | fla | sliding-window deploy: transformer collapses, gla flat | ppl | DONE (tf 9.94→16.13@256; gla 10.41) |
| B2 | 1/2 | fla | train{triangle,windowed,startup}×{tf,gla}, eval windowed@256 | ppl | RUNNING 4238435-39 |
| A1 | 2   | Qwen0.5B | windowed-FT: attn to pos-0 (Type-1) vs punctuation (Type-2) + entropy | sink%, type2%, H | QUEUED |
| A2 | 1/3 | Qwen0.5B | StreamingLLM Δ: sliding vs sink+recent vs full ppl | Δppl | QUEUED |
| A3 | 3a  | fla/Qwen | long-stream (≥32k) ppl + stability vs position | ppl(pos) | QUEUED |
| A4 | 3b  | Qwen | LongBench QA windowed-FT vs StreamingLLM vs base | qa-f1 | QUEUED |
| A5 | 3c  | Qwen | broadcasting/retrieval probe through no-op sinks | recall@k | QUEUED |
| A6 | 1/2 | Qwen0.5B | full attention MAPS per scheme (Type-1 sink -> Type-2 spread) | npz+fig | QUEUED |
| A7 | eff | fla/Qwen | compute efficiency: fwd latency + peak mem vs seqlen, full/windowed/gla | ms, MB | QUEUED |

## Keep/discard rule
Keep an experiment's result if it (a) runs to a clean number and (b) supports or cleanly refutes an aim (refutations are kept — they reshape the claim). Log every run to `results.tsv`.
