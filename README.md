# Context is the New Weight

NeurIPS-track research project investigating when in-context information can produce **the same model behavior** as a corresponding weight update — i.e., identifying conditions under which a context change is functionally equivalent to fine-tuning.

## Thesis

For a target model class, characterize (and ideally construct) a regime where:

> Behavior(model, context = C) ≡ Behavior(model + ΔW(C), context = ∅)

where `ΔW(C)` is a weight update derived from `C`. The exactness of this equivalence — over what input distribution, on what metrics, and with what tolerance — is the core empirical and theoretical question.

## Open questions

- **Equivalence definition.** Output-distribution match? Loss-landscape match? Per-token logits? Functional input-output equivalence?
- **Model class.** Does this hold for shallow / linear-attention models exactly, and only approximately for full transformers?
- **Context → weight map.** Closed-form (e.g., gradient-descent–as-attention results), learned (hypernetwork), or distillation-based?
- **Practical consequence.** If equivalence holds tightly, does training collapse to context curation? Does unlearning collapse to context redaction?

## Connection to prior work

- In-context learning ≈ implicit gradient descent (Akyürek, von Oswald, Garg, et al.).
- Linear attention's exact equivalence to a single GD step.
- Knowledge editing / ROME / MEMIT — the inverse direction (weight edit ≈ behavioral context).

## Status

- 2026-05-02: project initialized; NeurIPS 2026 template installed; three-experiment scaffold planted.

## Experiments

See [`experiments/README.md`](experiments/README.md). Three experiments, building on each other:

1. **[`01_context_to_weight_distill/`](experiments/01_context_to_weight_distill/)** — distill a context `C` into a weight delta `ΔW` via top-k KL self-distillation; check behavioral equivalence on held-out queries. *(Runnable.)*
2. **[`02_kv_vs_delta_w/`](experiments/02_kv_vs_delta_w/)** — decompose where context lives: KV-cache attention contribution in the teacher vs `ΔW`-mediated change in the student. *(Plan only.)*
3. **[`03_invert_delta_w_to_context/`](experiments/03_invert_delta_w_to_context/)** — invert `ΔW` to recover a context `C*` that reproduces the behavior on the base model. *(Plan only.)*

Default model: `meta-llama/Llama-3.1-8B-Instruct` (gated — needs `huggingface-cli login`). Override in `config.yaml`.

## Layout

```
context-is-the-new-weight/
  README.md
  pyproject.toml          # uv-managed deps
  paper/                  # NeurIPS 2026 submission
    main.tex              # paper draft
    neurips_2026.sty      # official style
    checklist.tex         # mandatory NeurIPS checklist
    references.bib
  src/                    # shared utilities
    models.py             # HF load helpers
    contexts.py           # library of steering contexts
    teacher.py            # teacher rollout w/ top-k logit capture
    distill.py            # KL-on-top-k student training (LoRA / full)
    metrics.py            # behavioral equivalence (KL, top-1, sampling)
    delta.py              # ΔW analysis (Frob, LoRA SV spectrum)
  experiments/            # one folder per experiment
  data/queries.jsonl      # 50 generic open-domain queries
  notes/research-log.md
```

## Compute

- Endeavour primary, Discovery overflow. Standard CARC layout under `/project2/jessetho_1732/zizhaoh/context-is-the-new-weight` and `/scratch1/zizhaoh/context-is-the-new-weight/saves`.
