# Experiment 01 — Synthetic Distill + KV-vs-ΔW

One template, run 5× (one per context: haiku, pirate, concise, fewshot_translate_fr, factual).

## Phases

1. **`phase1_dataset.py`** — generate ~200 (Q, A, top-k) triples for the chosen context. ~400 for `factual`.
2. **`phase2_distill.py`** — full fine-tune Llama-3.1-8B-Instruct on (Q → A) with no context in the input. Save trained model.
3. **`phase3_verify.py`** — *gating step*. Run trained model on its own training queries (no context); compare to teacher answers. If exact-match / prefix-match rates are too low, re-run Phase 2 before Phase 4.
4. **`phase4_kvdw.py`** — capture residual stream at every post-attn / post-mlp site for a held-out probe set. Compute v_ctx (what context did) vs v_dw (what ΔW did) per site. Cosine, RMSNorm cosine, token-space cosine.

## Run on CARC

```bash
# One context:
sbatch --export=CONTEXT=haiku experiments/01_synth_distill_kvdw/run.sh

# All five in parallel (under the 8-GPU cap on Endeavour):
for c in haiku pirate concise fewshot_translate_fr factual; do
  sbatch --export=CONTEXT=$c experiments/01_synth_distill_kvdw/run.sh
done
```

Monitor:
```bash
squeue -u zizhaoh -o '%.10i %.30j %.8T %.10M %R'
```

## Outputs

Per context, on scratch:
- `data/synth/<context>.jsonl` — synthetic dataset
- `saves/<context>/` — trained model (HF format)
- `outputs/01_synth_distill_kvdw/<context>/train_log.json`
- `outputs/01_synth_distill_kvdw/<context>/behavior_verify.json` — gate check
- `outputs/01_synth_distill_kvdw/<context>/kvdw.json` — per-site alignment metrics

## Config knobs

`config.yaml`. Key dials:
- `phase1.n_queries_default` (200) / `phase1.n_queries_factual` (400)
- `phase2.lr` (2e-5), `phase2.epochs` (2), `phase2.grad_accum` (8)
- `phase4.n_probe_queries` (30)

## Hard gates

- After Phase 3, eyeball ≥10 student samples per context. If the context behavior didn't transfer (haiku output is not a haiku, pirate isn't a pirate, etc.), re-run Phase 2 with more epochs / data / lr.
- Only proceed to Phase 4 for contexts that clearly passed verification.
