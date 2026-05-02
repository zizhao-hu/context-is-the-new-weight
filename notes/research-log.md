# Research Log

Append-only chronological notes. Newest entries at the top.

---

## 2026-05-02 — NeurIPS 2026 template installed

- Pulled `Formatting_Instructions_For_NeurIPS_2026.zip` from `media.neurips.cc` (Call-for-Papers link).
- Zip contents: `neurips_2026.sty` (Jan 29 2026 revision), `neurips_2026.tex` (sample), `checklist.tex` (mandatory paper checklist as a separate file — easier than past years).
- Wired `\input{checklist}` into `main.tex` after the bibliography.
- Sample `neurips_2026.tex` kept in `paper/` as reference; can drop later if it clutters.

---

## 2026-05-02 — Project initialized

- Repo scaffolded: `paper/`, `src/`, `experiments/`, `data/`, `notes/`.
- Thesis stated in `README.md`: characterize when context change ≡ weight update.
- First decisions to make:
  - **Definition of equivalence.** Distributional vs. logit-exact vs. functional.
  - **Smallest non-trivial model class.** Linear attention is the obvious starting point — there's a closed-form context-to-weight map (von Oswald et al., 2023).
  - **First experiment.** Verify the linear-attention equivalence empirically on a synthetic regression task; measure tightness as we move toward softmax attention and depth > 1.
