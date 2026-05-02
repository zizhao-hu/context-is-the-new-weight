# Experiments

Each experiment lives in a numbered subfolder with its own `run.sh` (SLURM) and `config.yaml`.

```
experiments/
  01_linear_attention_exact_equiv/
    run.sh
    config.yaml
    README.md
  02_...
```

Logs land in `/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/` on CARC; checkpoints in `/scratch1/zizhaoh/context-is-the-new-weight/saves/`.
