# 04 · Is the attention sink caused by the causal / instruction-tuning bias toward earlier tokens?

## Hypothesis
Under causal attention, **earlier tokens are attended by more queries** (token 0 is visible to all N
queries; token N to 1). Next-token training (pre-/instruction/experience tuning) reinforces this, so the
model dumps a disproportionate share of attention on the first token(s) — the **attention sink**. If this
bias is the cause, then:
1. **Tuning amplifies it** — an instruction/experience-tuned adapter has a *larger* sink than the base model.
2. **It is position-specific** — the sink is concentrated on the first few key positions, not uniform.
3. **Removing the bias removes it** — a model trained with *uniform context per token* (our windowed /
   parallelogram finetuning) has a *smaller* sink on real content; the sink relocates onto the soft-prompt.

## Metric
`sink = mean attention mass on key position 0`, averaged over (queries × heads × the **8 softmax layers**
of Qwen3.5-9B × inputs). [The 24 linear-attention layers have no softmax matrix / no sink — measured only on
the softmax layers, via `output_attentions=True` with eager attention.] Also report:
- per-layer sink (which of the 8 softmax layers sink hardest),
- the **key-position profile** (mass on positions 0..15) — to confirm the early-token concentration.

## Conditions
| | model | bias present? |
|---|---|---|
| A | Qwen3.5-9B **base** | causal (pretraining) |
| B | + **experience/instruction-tuned** adapter (`curve_history_N3000`) | causal, amplified by tuning |
| C | + **windowed / parallelogram** adapter (uniform context, soft-prompt) | bias removed |

## Predictions (if the hypothesis holds)
- sink(B) > sink(A)  → tuning amplifies the sink.
- sink concentrated on positions 0–few (profile spike).
- sink(C, real tokens) < sink(B), with the sink mass moved onto the **soft-prompt** positions.

## Files
- `measure_sink.py` — loads each model, runs forward with `output_attentions`, computes sink metrics on a
  fixed set of inputs (WebShop observations + generic text), writes `sink_<label>.json`.
- `render.py` — per-layer + position-profile bars/lines comparing A/B(/C).
- `run.sh` — SLURM driver (A6000, eager attention).

## Result — Phase A/B (n=6 inputs, 8 softmax layers)
| | sink (mass on key pos 0) | profile pos 0 → pos 1+ |
|---|---|---|
| A · base Qwen3.5-9B | **0.202** | 0.20 → ~0.04 |
| B · +experience-tuned | **0.186** | 0.19 → ~0.04 |

**The hypothesis is REFUTED for instruction/experience tuning.** The base (pretraining-only) model *already*
sinks ~20% of its attention onto key position 0; experience tuning **slightly REDUCES** it (0.202→0.186), it
does not amplify it. So the sink is a **pretraining / causal-structure** phenomenon, present before any
instruction tuning — consistent with StreamingLLM. What *is* confirmed: the sink is sharply **position-0
specific** (0.20 at pos 0 vs ~0.04 at every later position) — an early-token concentration.

So "instruction-tuning bias" is not the cause; the right framing is the **causal-attention structure itself**
(pos 0 visible to every query + softmax needing somewhere to dump mass).

## Result — does a SLIDING WINDOW reduce the sink? (inference-time band mask, `run_window.sh`)
`frontier_sink` = attention to each query's **oldest-visible key** (key 0 under full attention; the window's
left edge under a width-W window), averaged over full-window queries — the apples-to-apples sink.

| | full causal | W=64 | W=128 |
|---|---|---|---|
| base | **0.058** | 0.0077 | 0.0033 |
| experience | **0.052** | 0.0063 | 0.0022 |

**YES — the sliding window dramatically reduces the sink, and no new sink re-forms at the window edge.**
Under full attention the oldest-visible key (pos 0) draws ~0.055 ≈ **~15–29× the uniform share** — a strong
sink. Under a window it drops to 0.002–0.008, **at or below the uniform baseline** (1/64=0.0156, 1/128=0.0078)
— i.e. the window's left edge is *not* a sink; the attention redistributes off it rather than re-concentrating.
Base and experience behave identically (again: structural, not tuning-specific). This is the mechanism the
parallelogram exploits — windowing removes the sink, and the soft-prompt gives the residual a place to live.
