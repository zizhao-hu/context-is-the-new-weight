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

- 2026-05-02: project initialized.

## Layout (planned)

```
context-is-the-new-weight/
  README.md
  paper/              # NeurIPS submission (LaTeX, possibly submodule)
  src/                # equivalence-probing experiments
  experiments/        # SLURM scripts, configs
  data/               # small probes; large data on /scratch1
  notes/              # research log
```

## Compute

- Endeavour primary, Discovery overflow. Standard CARC layout under `/project2/jessetho_1732/zizhaoh/context-is-the-new-weight` and `/scratch1/zizhaoh/context-is-the-new-weight/saves`.
