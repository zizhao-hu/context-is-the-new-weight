"""340M from-scratch pretraining on FineWeb: full attention vs SWA vs S-SWA.

Mistral architecture (native sliding_window support), Llama-2 32k vocab, C=2048.
fla-convention 340M: 24 layers, d=1024, 16 heads, intermediate 2816, tied embeddings.
15B data tokens, global batch 0.5M tokens, AdamW lr 3e-4 cosine, warmup 1B tokens.

Variants (--variant):
  full : sliding_window=None, loss on all rows
  swa  : sliding_window=W, loss on all rows
  sswa : sliding_window=W, loss only on rows >= W-1 (every scored token sees a full window)

DDP via torchrun; checkpoint+resume every --ckpt_steps (preemption-safe).
Run: torchrun --nproc_per_node=4 train340.py --variant sswa
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["full", "swa", "sswa"])
ap.add_argument("--window", type=int, default=1024)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--tokens", type=float, default=15e9)          # data tokens
ap.add_argument("--batch_tokens", type=int, default=524288)    # global tokens/step
ap.add_argument("--micro_bs", type=int, default=8)             # sequences per GPU per micro-step
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup_tokens", type=float, default=1e9)
ap.add_argument("--data", default="/scratch1/zizhaoh/fineweb_tok.bin")
ap.add_argument("--meta", default="/scratch1/zizhaoh/fineweb_tok.json")
ap.add_argument("--out", default="/scratch1/zizhaoh/pretrain340")
ap.add_argument("--ckpt_steps", type=int, default=2000)        # ~1B tokens
ap.add_argument("--log_steps", type=int, default=50)
ap.add_argument("--smoke", type=int, default=0)                # >0: stop after N steps
a = ap.parse_args()

dist.init_process_group("nccl")
rank = dist.get_rank(); world = dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
dev = torch.device("cuda")
torch.manual_seed(1234)

meta = json.load(open(a.meta))
V = 32064  # padded past tokenizer vocab+added for tensor-core alignment
data = np.memmap(a.data, dtype=np.uint16, mode="r")
N = len(data)
C, W = a.ctx, a.window
steps_total = int(a.tokens // a.batch_tokens)
accum = a.batch_tokens // (world * a.micro_bs * C)
assert accum >= 1
if rank == 0:
    print("TRAIN340 variant=%s data_tokens=%.0f steps=%d world=%d micro_bs=%d accum=%d (global %d tok/step)"
          % (a.variant, a.tokens, steps_total, world, a.micro_bs, accum, world * a.micro_bs * C * accum), flush=True)

from transformers import MistralConfig, MistralForCausalLM
cfg = MistralConfig(
    vocab_size=V, hidden_size=1024, intermediate_size=2816,
    num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=16,
    max_position_embeddings=C, rope_theta=10000.0,
    sliding_window=(None if a.variant == "full" else W),
    tie_word_embeddings=True, attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)
model = MistralForCausalLM(cfg).to(dev).to(torch.bfloat16)
if rank == 0:
    print("params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
model = DDP(model, device_ids=[dev.index])
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

def lr_at(step):
    tok = step * a.batch_tokens
    if tok < a.warmup_tokens: return a.lr * tok / a.warmup_tokens
    p = (tok - a.warmup_tokens) / max(1, a.tokens - a.warmup_tokens)
    return 0.1 * a.lr + 0.45 * a.lr * (1 + math.cos(math.pi * min(1.0, p)))

# resume
os.makedirs(a.out, exist_ok=True)
tag = a.variant
ckpt_path = os.path.join(a.out, "ck_%s.pt" % tag)
start = 0
if os.path.exists(ckpt_path):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.module.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
    if rank == 0: print("RESUMED step", start, flush=True)

g = torch.Generator(); g.manual_seed(9999 + start)  # fresh offsets on resume

def batch():
    ix = torch.randint(0, N - C - 1, (a.micro_bs,), generator=g)
    x = torch.stack([torch.from_numpy(data[i:i + C].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + C + 1].astype(np.int64)) for i in ix])
    if a.variant == "sswa":
        y = y.clone(); y[:, :W - 1] = -100      # loss only on full-window rows
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

model.train()
t0 = time.time(); running = 0.0; nrun = 0
end = a.smoke if a.smoke else steps_total
for step in range(start, end):
    for gp in opt.param_groups: gp["lr"] = lr_at(step)
    for m in range(accum):
        x, y = batch()
        with (model.no_sync() if m < accum - 1 else torch.enable_grad()):
            out = model(x, labels=y)
            (out.loss / accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    running += out.loss.item(); nrun += 1
    if rank == 0 and (step % a.log_steps == 0 or step == end - 1):
        tps = (step - start + 1) * a.batch_tokens / max(1e-9, time.time() - t0)
        print("STEP %d/%d loss %.4f lr %.2e tok/s %.0f elapsed %.1fh"
              % (step, steps_total, running / max(1, nrun), lr_at(step), tps,
                 (time.time() - t0) / 3600), flush=True)
        running = 0.0; nrun = 0
    if rank == 0 and step > start and step % a.ckpt_steps == 0:
        torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "step": step},
                   ckpt_path + ".tmp"); os.replace(ckpt_path + ".tmp", ckpt_path)
        print("CKPT step %d" % step, flush=True)
if rank == 0:
    torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "step": end}, ckpt_path)
    model.module.save_pretrained(os.path.join(a.out, "hf_%s" % tag))
    print("TRAIN340_DONE variant=%s steps=%d" % (a.variant, end), flush=True)
dist.destroy_process_group()
