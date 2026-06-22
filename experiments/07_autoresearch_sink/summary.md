# Autoresearch summary — Attention-Sink Thesis

Autonomous loop on USC CARC (≤8 GPU, short→nlp / long→discovery). All numbers in `results.tsv`; scripts in `scripts/`; figure `figures/scaling/qwen_sink_spread.png`. Models: **fla 1.3B** (transformer-softmax vs gla-gated, same 100B tokens = strict mixer-only control) + **Qwen2.5-0.5B** (strong sink, eager attention, custom masks).

## Aim 1 — the triangle mask *causes* the Type-1 (positional) sink ✓
- Qwen0.5B fraction of attention on pos-0: **base 0.394**, and **triangle-CPT (continued *full-causal* training) 0.412** — extra training under the triangle mask does **not** remove the sink (slightly grows it). Only changing the mask (windowed) removes it.
- Strict control: the **gla** (gated, no triangular softmax) model has **0** softmax-attention sink layers by construction — no triangle mask, no Type-1 sink.

## Aim 2 — windowed training *spreads* Type-1 → Type-2 (no-op) with better broadcasting ✓
Qwen0.5B, attention under each scheme's deployment:
| scheme | Type-1 (pos-0) | Type-2 (punctuation) | broadcast entropy |
|---|---|---|---|
| base | 0.394 | 0.174 | 3.94 |
| triangle-CPT | 0.412 | 0.165 | 3.85 |
| windowed-FT | 0.145 | 0.371 | 4.60 |
| startup-FT | **0.008** | 0.346 | **5.00** |
Mass leaves pos-0 and lands on the distributed low-information punctuation tokens; entropy rises monotonically. Figure: `qwen_sink_spread.png`.

## Aim 3 — windowed *training* beats post-hoc streaming techniques
- **3a streaming (Qwen0.5B, sliding@256):** base collapses **10.97→56.73**; base+StreamingLLM(4 sink+256) recovers to **13.94**; **windowed-FT streams at 14.64 with NO sink tokens** (≈ the StreamingLLM trick, no inference-time hack).
- **3b LongBench / HotpotQA:** _running_.
- **3c broadcasting/relay (passkey outside window):** _refined_ — neither base nor windowed-FT relays a discrete far passkey (ppl ~37 vs full ~1.5); windowed-FT is only more *stable* (base erratic 38→7055). ⇒ the "broadcasting" of Aim 2 is **attention spread (entropy), not long-range exact retrieval** — the paper should scope the claim to spread, not retrieval.

## Compute efficiency ✓ (the value of a constant working memory)
Forward latency vs seqlen (fla 1.3B): full **quadratic** (797ms@16k → 2363ms@32k), windowed & gated **linear**; at 32k **windowed 866ms = 2.7× faster than full**, gla 989ms. (KV-cache memory figure during generation still TODO.)

## fla strict control (windowed training vs gated attention) ✓
Sliding-window@256 ppl, same skeleton/data, only the mixer/scheme differs:
- base softmax **16.13** (collapse) → triangle-CPT **13.53** (control, stays collapsed) → **windowed-FT 10.16** (recovers to ~its full-context 9.12) → gla gated **8.93**.
- "windowed softmax ≈ gated" — *directionally confirmed* (10.16 approaches 8.93); recovery is from windowing, not training (control 13.5).
- W-robustness: windowed-FT@{128,256,512} = 11.2 / 10.2 / 9.5 (vs base 18.9 / 16.1 / 14.1) — holds at every window.

## Open / next (loop plateau — awaiting direction)
KV-cache generation-memory figure · LongBench on more tasks · larger model scale · scope the Aim-3c claim in the paper to spread-not-retrieval.
