# 05 · Position-1 attention-sink across 4 long-context tuning schemes

## Question
Take a **non-instruction-tuned base** LM and tune it on a **long-context** dataset four different ways; which
way leaves the smallest **attention sink at position 1** (the first content token)?

## Setup
- **Base:** `Qwen/Qwen2.5-0.5B` (small, standard *full-attention* — every layer is softmax, so the sink
  develops everywhere and windowing affects all layers; the 9B we use elsewhere is 75% linear-attention).
- **Data:** wikitext-103, packed into length-`L=1024` sequences. Full fine-tune, ~800 steps.
- **Window** `W=256`; **startup prompts** `P=W−1=255`.

## The 3 cases (`train_4way.py --scheme …`)
1. **causal** (`full`) — standard full-causal long-context tuning (baseline; builds/keeps the sink).
2. **sliding with history** (`sliding_history`) — constant-width window where the cold start is filled by the
   REAL previous W−1 tokens (`pack_with_history` + history prefix). This is what "warm up to accumulate context"
   actually means; it subsumes the old `warmup_windowed`.
3. **sliding with trainable tokens** (`startup`) — same window, cold start filled by P=W−1 trainable soft-prompts
   prepended via `inputs_embeds` (our design — no prior context needed).

(Deprecated/removed from the figure: `windowed` = no-fill window inside each chunk — a bug, early tokens
cold-start; `warmup_windowed` = full-attn warm-up then no-fill window — also no real history fill. Both kept the
sink because neither actually fills the cold start.)

## Metric (`measure_sink1.py`)
On held-out wikitext under **full causal attention**, per head compute the mean attention to the **first content
token** (position 0 for causal; after the prompt/history block for the filled schemes); report **mean** and the
**sink-rate** = fraction of heads > 0.3 (the paper's ε), plus `block_attn` = mass on the fill block.
Conditions: **base** + the 3 schemes.

## Run
`sbatch run_scheme.sh {full|sliding_history|startup|base}` (one A6000 each); then `python make_sink4_fig.py`.

## Result (Qwen2.5-0.5B, 800 steps, wikitext-103; full-attention eval, 16 seqs)
The 3 figure configs (causal / sliding-with-history / sliding-with-trainable-tokens) + base. "sink goes to" =
per-layer attention mass on the cold-start block (history / prompts).

| scheme | mean attn → first content token | sink-rate (% heads >0.3) | sink goes to |
|---|---|---|---|
| base (no tuning) | 0.367 | 56% | first token |
| **causal** (full-causal) | **0.374** | **57%** | first token |
| **sliding with history** | **0.001** | **0%** | real history block (0.41) |
| **sliding with trainable tokens** (startup) | **0.003** | **0%** | prompt block (0.38) |

**Findings.**
- **causal keeps the sink** (≥ base): conventional long-context tuning does nothing to the pos-1 over-attention.
- **Either cold-start fill removes it.** Sliding with **history** (prepend the real previous W−1 tokens) and
  sliding with **trainable tokens** (prepend P=W−1 learned prompts) BOTH drop the sink to **0%**, relocating the
  mass onto the cold-start block (history 0.41 / prompts 0.38). The two fills are **equivalent** for sink
  removal — it's the *fill itself*, not specifically trainable tokens, that gives the sink a home. The
  trainable-token version's advantage is purely **availability**: it never needs relevant prior context to exist.

⚠ Correction history: the original `windowed` (43%) and `warmup_windowed` (53%) schemes were **no-fill** versions
(window inside each chunk; warm-up = full-attn-then-no-fill-window) — neither actually filled the cold start, so
both kept the sink. "warm up to accumulate context" IS sliding-with-history, now implemented correctly as
`sliding_history` (`pack_with_history` + history prefix in train & `--history` eval). The old two are dropped.
