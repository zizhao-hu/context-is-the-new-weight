# Experiment → Hypothesis map (attention-sink paper)

Every experiment we ran, grouped by the hypothesis it proves. ✅ supported · ◑ partial · ⚠ gap.

## H1 — the sink is two disentanglable types (positional), distinct from the functional split
- **Categorization** (§2 Table 1): our positional axis (first token / separators) is orthogonal to Fesser et al. 2026's functional axis (no-op / broadcast); each position can serve either function.
- **Both types co-present & separately measurable**: base has Type-1 (pos-0) **0.368** AND Type-2 (separators) **0.146** simultaneously. [`qwen_sink_info`, Tab. spread]
- **Disentanglement (the key test)**: `startup` drives Type-1 → **0.004** while Type-2 rises to **0.40** — one removed, the other kept → independent axes. [Tab. spread]
- **Type-2 is real, not a punctuation artifact**: rises identically under a punctuation list AND bottom-20% unigram-surprisal; `corr(attn,surprisal)` −0.07→−0.16. [`qwen_sink_info`]
- Status: ✅

## H2 — Type-1 is caused by the causal-mask context-length imbalance
- **Sink grows with context (against data dilution)**: rises with C while data-side first-token importance ∝1/C falls. [Fig. sink_cmp]
- **Learned & removable**: ~57% of softmax heads sink on token-1 in base/full-FT → **0%** windowed-FT. [Tab. sink05b]
- **Full-causal training keeps/intensifies it**: triangle-CPT 0.368→**0.383** (control). [Tab. spread]
- **Gated model (no triangular softmax) has no Type-1 sink** by construction. [strict control]
- **Strict softmax-vs-gated control** (fla 1.3B, same skeleton+100B tokens): windowed-softmax recovers 16.1→10.2, gla 8.71. [Tab. strictmatrix]
- **Scales**: 9B hybrid — windowing shrinks the softmax-layer sink. [Fig. 9bsink]
- **1.3B direct sink measurement** (eager q,k recompute, flash-attn-blocked): sliding-window training reduces Type-1 **0.170→0.152** (modest; base sink already weak at scale). [`fla_sink_weights`]
- Status: ✅ (caveat: sliding-window confounds "balance" with "pos-0 masked away")

## H3 — removing Type-1 makes later tokens resort to the Type-2 separators
- **The redistribution**: Type-1 0.37→0.15→0.004 while Type-2 0.15→0.32→0.40, entropy 4.5→5.5, corr −0.07→−0.16. [Tab. spread, Fig. spread] (same experiment as H1's disentanglement, read directionally)
- Status: ✅

## H4 — separator sinks form a hierarchical store, absorb info, broadcast better, improve streaming
- **Streaming improved**: windowed-FT streams sink-free (sliding@256 ppl **14.6** ≈ base+StreamingLLM 13.9; base alone collapses 56.7). [`qwen_stream`]
- **Efficiency**: windowed O(L) — **2.7× faster** than full @32k, linear vs O(L²). [`bench_efficiency`, Tab. eff]
- **Downstream QA (fair mask-only control + EM/F1)**: triangle@windowed F1 0.036 → windowed **0.095** → persist **0.166** = base+StreamingLLM 0.166. [`hotpot_fair`, Tab. hotpotfair]
- **Persistent prompts** = a *trained* StreamingLLM (learned always-on registers). [Tab. hotpotfair]
- ⚠ **Mechanism NOT tested**: "hierarchical / absorbs preceding info / granular broadcasting" is hypothesis-only. The passkey-relay probe shows broadcasting = *distributional spread*, NOT long-range retrieval — partial pushback.
- Status: ◑ — streaming benefit shown; the hierarchical/absorption mechanism untested. **Deferred tests** (per user, not yet run): separator-ablation, separator-aware StreamingLLM, separator-readout probe.

## Paper §-to-hypothesis correspondence (prove in line)
- H1 → §"Where the Attention Goes: Type-1 to Type-2" (+ §2 Table 1)
- H2 → §"The Sink Is Learned and Removable", §"Scaling to a Hybrid Model", §"A Strict Softmax-vs-Gated Control" (+ §"Windowed Finetuning Recovers Full-Context Quality", §"Train Small, Deploy Anywhere")
- H3 → §"Where the Attention Goes" (directional)
- H4 → §"Compute" (efficiency), §"Persistent Prompts", §"Downstream Long-Context QA"
