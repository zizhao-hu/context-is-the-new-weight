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
- it4 HARVEST (fleet 10769925-36 all CLDONE, guard held: a/b rows byte-identical):
  naive forget: SWA 4.60 / t@fair 2.48 / bs 1.04 / ts 1.26; naive BWT: -4.60 / -2.48 / +1.92 / +4.16
  tofu final: 28.06 / 23.82 / 7.59 / 9.64 -> the sink is the forgetting protection,
  truncation adds transfer, full recipe = largest BWT anywhere in the table.
  HONEST CORRECTION: old t 1.11/+2.10 was budget-confounded (250 steps = 25% fewer
  supervised tokens); fair rerun 2.48/-2.48. EWC still pins everything (<=0.15).
  Paper: tab:cl/tab:cltasks regenerated (+8 rows), 4.2 rewritten, content ends p8,
  submodule 61d837e. VERDICT: keep.
- it5 (err bars + reorder): eval_ppl emits sem (delta method over 24 eval-chunk mean NLLs);
  full 32-job re-run fleet 10777518-49 (softmax 20 + hybrid 12, trainer deployed pre-start,
  md5 verified) since no checkpoints existed for the finished runs. Harvester: column order
  = per-task breakdown | fineweb | avg ppl | forget | BWT (both blocks), pm()/avg_pm() render
  {\tiny$\pm$}; appendix ppl columns too. Values will be reconciled to the new logs at harvest.
- it6 (masks C/D + mask-major regroup): trainer gains c = SWAA (4 pinned first tokens +
  W-4 recent, budget W) and d = Transformer-XL (W-token segments, previous segment's KV
  as stop-gradient memory via DynamicCache + explicit position_ids; span <= 2W, budget 2W
  disclosed); c is the real-token twin of bs (pinned real vs learned registers, same split).
  Harvester regrouped mask-major per user: groups a, b, b+sink, c, d, e, e+sink; sub-rows
  naive (unlabeled first row) then +replay/+EWC/+LwF. Smokes 10784155 (c_lwf) / 10784156
  (d_ewc); 8 production c/d jobs follow on pass, joining the SEM fleet for the final harvest.
- it6b: c/d smokes PASS. c stage-0 sanity: pinning 4 real tokens ~restores full-attention
  base (wikitext 10.07 vs 10.51, fineweb 28.8 vs 296 sliding) = StreamingLLM finding
  reproduced. d stage-0 expectedly OOD (fineweb 346; no sink in memory past segment 1).
  Production c/d fleet: 10784685-92. Target harvest = 40 runs, all-SEM.
- FINDING (user-prompted, for final prose): E+sink naive BWT +4.16 decomposes as
  wikitext -1.89, gsm8k -1.88, tofu +16.25 -> dominated by post-TOFU overfit recovery
  (just-after 25.9 -> final 9.6), not broad backward transfer. Final text must lead with
  forgetting (robust, sink-driven) and caveat the BWT sign; check SEM overlap of
  bs 1.04 vs ts 1.26 forget before claiming an ordering.
