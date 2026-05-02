# Experiments

Sprint experiment lives in [`01_synth_distill_kvdw/`](01_synth_distill_kvdw/) — same template run 5× (one per context).

```
experiments/
  01_synth_distill_kvdw/         # active sprint experiment
    config.yaml
    phase1_dataset.py
    phase2_distill.py
    phase3_verify.py
    phase4_kvdw.py
    run.sh
    README.md
  extras/                        # parking lot for full-plan experiments
    02_kv_vs_delta_w/            # superseded by phase4 of 01
    03_invert_delta_w_to_context/  # post-sprint: ΔW → C inversion
```

See [`PROPOSAL.md`](../PROPOSAL.md) at the repo root for the research plan.

## Conventions

- On CARC, paths are scratch-backed via `run.sh` (overrides config at runtime). Locally, they're under repo-relative `outputs/`, `saves/`, `data/synth/`.
- Logs land in `/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/` on CARC.
- Default partition: `nlp_hiprio`. Default GPU: A6000 (47GB). Switch to `--gres=gpu:a100:1` only if a job OOMs at full-FT 8B.
