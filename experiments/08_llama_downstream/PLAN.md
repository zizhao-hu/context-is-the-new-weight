# Autoresearch: LLaMA downstream (HotpotQA) for the 3 constant-memory schemes
Goal: maximize HotpotQA gen-F1 on LLaMA-3.2-1B (all-softmax) under constant-W windowed deploy.
Metric: F1 (higher). Verify: RESULT llamahotpot lines. Harness: scripts/llama_hotpot.py (CARC /scratch1/zizhaoh/fla_exp).
Schemes: base, triangle (controls) | sliding-history(windowed), trainable-startup, uniform-context(uniwin).
Iter 1 (baseline): 5 schemes x W{128,256,512} = 11 jobs (base/triangle sweep W internally). lr 2e-5, steps 400, ctx 1536, n_eval 150.
Next levers: window length (done 128/256/512; add 64/1024), lr {1e-5,2e-5,5e-5}, steps, LLaMA-3.2-3B, persist variant.
