"""Load a saved toy checkpoint, dump mean-over-heads attention map + a deep-query row for one example -> npz.
Shows where the mass goes (pos-0 vs local vs common-token). a=full deploy, b=sliding."""
import torch, torch.nn as nn, torch.nn.functional as F, math, argparse, numpy as np
from datasets import load_dataset
ap = argparse.ArgumentParser(); ap.add_argument("--mask", required=True); ap.add_argument("--Lm", type=int, default=256); a = ap.parse_args()
dev = "cuda"; ck = torch.load("/scratch1/zizhaoh/toyckpt_%s.pt" % a.mask, map_location=dev, weights_only=False)
V = ck["V"]; bos = ck["bos"]; chars = ck["chars"]; cfg = ck["cfg"]; W = cfg["W"]; BOS_ID = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
te = "".join([r["text"] for r in load_dataset("wikitext", "wikitext-103-raw-v1", split="test")])[:1_200_000]
teids = [stoi[c] for c in te if c in stoi]; tei = torch.tensor(teids, dtype=torch.long, device=dev)
def rope_pos(x, pos, base=1e4):
    B, H, T, D = x.shape; hf = D // 2; fr = base ** (-torch.arange(0, hf, device=x.device).float() / hf)
    an = torch.outer(pos.float(), fr); co = an.cos()[None, None]; si = an.sin()[None, None]
    x1, x2 = x[..., :hf], x[..., hf:]; return torch.cat([x1 * co - x2 * si, x1 * si + x2 * co], -1)
rope = lambda x: rope_pos(x, torch.arange(x.shape[2], device=x.device))
class Blk(nn.Module):
    def __init__(s, C, H):
        super().__init__(); s.H = H; s.ln1 = nn.LayerNorm(C); s.ln2 = nn.LayerNorm(C)
        s.qkv = nn.Linear(C, 3 * C); s.proj = nn.Linear(C, C)
        s.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))
class GPT(nn.Module):
    def __init__(s, V, C, H, Ln):
        super().__init__(); s.emb = nn.Embedding(V, C); s.blocks = nn.ModuleList([Blk(C, H) for _ in range(Ln)])
        s.lnf = nn.LayerNorm(C); s.head = nn.Linear(C, V, bias=False)
    @torch.no_grad()
    def amap(s, ids, m):
        x = s.emb(ids); A = []
        for b in s.blocks:
            B, T, C = x.shape; h = b.H; d = C // h
            q, k, v = b.qkv(b.ln1(x)).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
            q = rope(q); k = rope(k); at = ((q @ k.transpose(-2, -1)) / math.sqrt(d) + m).softmax(-1)
            A.append(at); o = (at @ v).transpose(1, 2).reshape(B, T, C); x = x + b.proj(o); x = x + b.mlp(b.ln2(x))
        return torch.stack(A, 0)
model = GPT(V, cfg["n_embd"], cfg["n_head"], cfg["n_layer"]).to(dev); model.load_state_dict(ck["sd"]); model.eval()
Lm = a.Lm; qi = torch.arange(Lm, device=dev)[:, None]; ki = torch.arange(Lm, device=dev)[None, :]
mk = lambda al: torch.zeros(Lm, Lm, device=dev).masked_fill_(~al, float("-inf"))[None, None]
dmask = mk(ki <= qi) if a.mask == "a" else mk((ki <= qi) & (ki > qi - W))
i0 = torch.randint(0, tei.numel() - Lm - 2, (1,), generator=torch.Generator().manual_seed(7)).item()
if bos:
    ids = torch.cat([torch.tensor([BOS_ID], device=dev), tei[i0:i0 + Lm - 1]])[None]
    toks = ["<BOS>"] + [itos[int(t)] for t in tei[i0:i0 + Lm - 1].tolist()]
else:
    ids = tei[i0:i0 + Lm][None]; toks = [itos[int(t)] for t in tei[i0:i0 + Lm].tolist()]
A = model.amap(ids, dmask)[:, 0]                                                  # [L,h,Lm,Lm]
Amean = A.mean(dim=(0, 1)).float().cpu().numpy()                                  # [Lm,Lm] mean over layers+heads
# decomposition of deep-query attention (mean over heads, deep queries [64,256))
qd = torch.arange(64, Lm, device=dev); Ah = A.reshape(-1, Lm, Lm)                 # [LH,Lm,Lm]
sh = int(Ah[:, qd, :4].sum(-1).mean(-1).argmax())                                # max pos-0 (sink) head
Asink = Ah[sh].float().cpu().numpy(); psink = float(Ah[sh, qd, :4].sum(-1).mean().item())
lsink = float(sum(Ah[sh, qd, (qd - off)].mean().item() for off in range(0, 8)))
pos0 = float(Ah[:, qd, 0].mean().item())
loc = float(sum(Ah[:, qd, (qd - off)].mean().item() for off in range(0, 8)))      # last 8 (local)
np.savez("/scratch1/zizhaoh/vizck_%s.npz" % a.mask, Amean=Amean, Asink=Asink, toks=np.array(toks, dtype=object),
         W=W, Lm=Lm, pos0=pos0, loc=loc, psink=psink, lsink=lsink, sh=sh)
print("VIZCK mask=%s meanpos0=%.4f meanlocal8=%.4f | sinkhead=%d sinkpos0=%.4f sinklocal8=%.4f" % (
    a.mask, pos0, loc, sh, psink, lsink), flush=True)
