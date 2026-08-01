# Exp 12 redo: 2x2 ablation (rows-to-skip x trainable sink) at equal supervised tokens

Goal: redo the softmax CL suite as an ablation of the two recipe components on the
sliding base, at an equal supervised-token budget, sampling training windows from the
continuous task stream (random offsets) so skipped-row tokens are supervised elsewhere.

Config
- variants: a (reuse), b (reuse), t (RERUN, fair budget), bs = SWA+sink (new), ts = T-SWA+sink (new)
- sink: nP=4 learned prefix K/V registers per layer, recent window W-nP=252 (total budget W=256),
  init from mean K/V of a wikitext chunk + 0.02 noise; registers included in TRAINABLE
  (EWC Fisher, L2 anchor, LwF teacher carries frozen registers)
- fair budget: steps_eff = round(250 * L/(L-loss_from)) -> 333 for t/ts (supervised tokens match b within 0.1%)
- methods: naive, replay(ER), EWC, LwF (l2 inert, skipped); hybrid block untouched
- eval: unchanged (matched deploy, scored positions >= W for every design)

Iterations
- it0 baseline: existing 27-run table (tab:cl) with unfair t budget (250 steps, -25% tokens)
- it1: trainer rewrite (sink + fair budget + stream comments); account fix in sb_cl.sh;
  smokes 10769911 (ts_lwf), 10769912 (bs_ewc) submitted
- it2: kv_layers fixed for new DynamicCache (.layers[i].keys/.values); CPU gradcheck 48/48
  finite grads (norm 4.8); SINKGRAD self-check baked into trainer; account robinjia_875
- it3: smokes PASS (ts_lwf steps=3 fair-scaled, bs_ewc; SINKGRAD 48/48 on GPU;
  sink stage-0 fineweb 313.9 vs plain-sliding 296.1, repaired to ~100 even by junk steps).
  Fleet submitted: 10769925-10769936 (t/bs/ts x naive/replay/ewc/lwf, 6h limits).
  Paper edits held until harvest so table and text swap atomically.
