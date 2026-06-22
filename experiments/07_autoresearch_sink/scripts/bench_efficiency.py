"""Compute-efficiency analysis (aim: the value of a constant working memory). Forward latency + peak GPU memory
vs sequence length for full softmax attention (O(L^2) compute), windowed softmax (O(L*W), bounded per token), and
gated-linear attention (O(L), O(1) state). On the strict-control fla models so only the token-mixer differs."""
import torch, time, argparse
import fla
from transformers import AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--window", type=int, default=0)         # 0 = full (no sliding window); >0 = windowed
ap.add_argument("--lens", default="512,1024,2048,4096,8192,16384,32768")
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args(); dev = "cuda"
model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
attns = [m for m in model.modules() if m.__class__.__name__ == "Attention" and hasattr(m, "window_size")]
W = a.window if a.window > 0 else None
for m in attns:
    m.window_size = W
vocab = model.config.vocab_size
kind = ("window%d" % a.window) if a.window > 0 else ("full" if attns else "gla-native")
print("BENCH %s | %s | softmax-attn-layers %d" % (a.model.split('/')[-1], kind, len(attns)), flush=True)
for L in [int(x) for x in a.lens.split(",")]:
    try:
        ids = torch.randint(0, vocab, (1, L), device=dev)
        with torch.no_grad():
            model(ids, use_cache=False)                  # warmup
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(a.reps):
                model(ids, use_cache=False)
        torch.cuda.synchronize()
        lat = (time.time() - t0) / a.reps * 1000.0
        mem = torch.cuda.max_memory_allocated() / 1e6
        print("  L=%-6d latency %.1f ms  peak_mem %.0f MB" % (L, lat, mem), flush=True)
    except RuntimeError as e:
        print("  L=%-6d OOM/err: %s" % (L, str(e)[:70]), flush=True); torch.cuda.empty_cache(); break
print("BENCH_DONE", flush=True)
