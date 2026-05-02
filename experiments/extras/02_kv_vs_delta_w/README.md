# Experiment 02 — KV cache vs ΔW: Where does context live?

## Question

For a given context `C`, the original model uses `C` as additional tokens in the KV cache. The distilled student of Experiment 01 has absorbed `C` into a weight delta `ΔW`. How does the *attention output* on a query `Q` decompose between (a) the contribution of `C`'s key/value vectors in the teacher and (b) the contribution of `ΔW` in the student?

We want a layer-by-layer, head-by-head map of where `C` exerts its influence — so that in Experiment 03 we can target inversion at the right places.

## Plan (sketch)

Inputs: a teacher `M`, a context `C`, a student `M_C` from Experiment 01, eval queries `Q'`.

For each layer `ℓ` and each head `h`:
1. **Teacher attention contribution from `C`.** Run teacher on `[C; Q']`. Decompose
   `attn_out_ℓ,h(t) = sum_s α_{t,s} V_s` into the part with `s ∈ C` (context tokens) vs `s ∈ Q'+gen` (everything else). Record the magnitude of the `C`-only contribution.
2. **Student weight contribution.** Run `M_C` on `Q'` (no context). Compute `attn_out_ℓ,h(t)` and the residual stream just after the attention block. Subtract the corresponding tensor from `M(Q'|∅)`. The residual is what `ΔW` injected.
3. **Alignment.** Project both contributions onto the unembedding (`W_U`) to get token-space directions. Compute cosine / KL on the resulting next-token distributions induced by each contribution alone.

## Expected artifacts

- `layer_head_heatmap.png` — magnitude of context-from-KV vs context-from-ΔW per (ℓ, h).
- `alignment.json` — per-layer cosine similarity between the two contributions.
- A short note on whether `ΔW` reproduces the *same* attention-pattern shift as `C` did (mechanistic equivalence) or merely the same output distribution (functional equivalence).

## Status

Not started — depends on Experiment 01 producing trained LoRA adapters. Implementation will reuse `src/teacher.py` for forward passes and add `src/attention_hooks.py` for the per-layer/per-head capture.
