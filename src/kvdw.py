"""Phase 4 — KV-vs-ΔW residual-stream comparison.

For Llama's pre-norm transformer block:
    x_in -> RMSNorm -> Attn -> +residual = post_attn_residual
                  -> RMSNorm -> MLP  -> +residual = post_mlp_residual

We capture the residual stream at every post-attention site and every post-MLP
site by registering forward hooks. For Llama-3.1-8B (32 blocks) that gives 64
sites total.

For each site `s`, we compute two perturbation vectors:
  v_ctx_s = resid_s(Q' | base, [C; Q']) - resid_s(Q' | base, Q')
  v_dw_s  = resid_s(Q' | trained, Q')   - resid_s(Q' | base, Q')

If context and ΔW are doing the "same thing" at site `s`, these two vectors
should be aligned (high cosine).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Hook-based capture
# --------------------------------------------------------------------------


def _get_layers(model):
    """Return the list of transformer blocks. Works for HF Llama models and
    PEFT-wrapped variants where the inner model is at `.model`."""
    m = model
    while hasattr(m, "model") and not hasattr(m, "layers"):
        m = m.model
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError("Couldn't locate transformer layers on the model.")


@dataclass
class _Capture:
    captures: dict  # (layer_idx, "attn"|"mlp") -> tensor

    def slice_last_token(self, attn_mask: torch.Tensor) -> dict:
        """Return a dict where each tensor is reduced to (B, hidden) by
        selecting the last non-pad position per batch row."""
        out = {}
        # last token index per batch row
        last_idx = attn_mask.sum(dim=-1) - 1  # (B,)
        for k, v in self.captures.items():
            B = v.shape[0]
            sel = v[torch.arange(B, device=v.device), last_idx]  # (B, hidden)
            out[k] = sel
        return out


@contextmanager
def capture_residuals(model):
    """Context manager that registers hooks at every post-attn and post-mlp
    site of the model's transformer blocks. Yields a `_Capture` whose
    `.captures` dict gets populated by every forward call inside the with-block.

    Captures are tensors of shape (B, T, hidden). The capture dict is reset on
    each call to `model(...)` -- only the most recent forward is retained.
    """
    layers = _get_layers(model)
    cap = _Capture(captures={})
    handles = []

    def make_attn_pre_hook(idx):
        def hook(module, args):
            # Input to post_attention_layernorm == post-attn residual stream.
            x = args[0] if isinstance(args, tuple) else args
            cap.captures[(idx, "attn")] = x.detach()
        return hook

    def make_layer_post_hook(idx):
        def hook(module, args, output):
            # Output of LlamaDecoderLayer == post-MLP residual stream.
            # Llama returns a tuple (hidden_states, ...); take element 0.
            h = output[0] if isinstance(output, tuple) else output
            cap.captures[(idx, "mlp")] = h.detach()
        return hook

    for i, layer in enumerate(layers):
        if not hasattr(layer, "post_attention_layernorm"):
            raise RuntimeError(
                f"Layer {i} has no post_attention_layernorm; not a Llama-style block."
            )
        h1 = layer.post_attention_layernorm.register_forward_pre_hook(make_attn_pre_hook(i))
        h2 = layer.register_forward_hook(make_layer_post_hook(i))
        handles.extend([h1, h2])

    try:
        yield cap
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------
# Perturbation deltas
# --------------------------------------------------------------------------


@torch.no_grad()
def captures_for(model, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> dict:
    """Run a forward pass, return last-token residual at every site."""
    with capture_residuals(model) as cap:
        model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        return cap.slice_last_token(attn_mask)


@torch.no_grad()
def context_delta(base_model, with_ctx_ids, with_ctx_mask, no_ctx_ids, no_ctx_mask) -> dict:
    """v_ctx_s = resid_s(no-ctx-Q-portion under [C;Q]) - resid_s(Q under no-ctx).

    For symmetric comparison we slice the last token in both runs (which is
    the same Q-end token). Returns dict[(layer, site)] -> (B, hidden)."""
    cap_with = captures_for(base_model, with_ctx_ids, with_ctx_mask)
    cap_no = captures_for(base_model, no_ctx_ids, no_ctx_mask)
    return {k: cap_with[k] - cap_no[k] for k in cap_with.keys() & cap_no.keys()}


@torch.no_grad()
def weight_delta(base_model, trained_model, no_ctx_ids, no_ctx_mask) -> dict:
    """v_dw_s = resid_s(Q under trained) - resid_s(Q under base)."""
    cap_t = captures_for(trained_model, no_ctx_ids, no_ctx_mask)
    cap_b = captures_for(base_model, no_ctx_ids, no_ctx_mask)
    return {k: cap_t[k] - cap_b[k] for k in cap_t.keys() & cap_b.keys()}


# --------------------------------------------------------------------------
# Comparison metrics
# --------------------------------------------------------------------------


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine over last dim. Inputs (B, hidden) → (B,)."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=eps)


def tokenspace_cosine(a: torch.Tensor, b: torch.Tensor, w_u: torch.Tensor) -> torch.Tensor:
    """Cosine after projecting both vectors onto the unembedding `W_U` (V, hidden)
    to get token-logit-space directions.
    """
    pa = a.float() @ w_u.float().T  # (B, V)
    pb = b.float() @ w_u.float().T
    return F.cosine_similarity(pa, pb, dim=-1)


def rmsnorm_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Cosine after applying root-mean-square normalization (controls for scale)."""
    def rms(x):
        return x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return cosine(rms(a), rms(b))


def per_site_alignment(v_ctx: dict, v_dw: dict, w_u: torch.Tensor | None = None) -> dict:
    """For each site, compute raw cosine, RMSNorm cosine, and (if w_u given)
    token-space cosine. Returns nested dict[site_key]->{metric->scalar}.

    Aggregates over the batch dimension by mean.
    """
    out = {}
    for key in sorted(v_ctx.keys() & v_dw.keys()):
        a = v_ctx[key]
        b = v_dw[key]
        item = {
            "cosine": cosine(a, b).mean().item(),
            "rmsnorm_cosine": rmsnorm_cosine(a, b).mean().item(),
            "ctx_norm": a.float().norm(dim=-1).mean().item(),
            "dw_norm": b.float().norm(dim=-1).mean().item(),
        }
        if w_u is not None:
            item["tokenspace_cosine"] = tokenspace_cosine(a, b, w_u).mean().item()
        # serialize key as "L<idx>_<site>"
        out[f"L{key[0]:02d}_{key[1]}"] = item
    return out


# --------------------------------------------------------------------------
# Induced KL ablation
# --------------------------------------------------------------------------


@torch.no_grad()
def induced_kl_at_site(
    base_model,
    no_ctx_ids: torch.Tensor,
    no_ctx_mask: torch.Tensor,
    site: tuple[int, str],
    delta_to_inject: torch.Tensor,
) -> float:
    """Run base model on Q (no context). Inject `delta_to_inject` into the
    residual stream at `site` (additive at the last token position). Return
    KL of the resulting next-token distribution vs the unmodified forward.

    Used to ask: does adding `v_ctx_s` (or `v_dw_s`) to a no-context forward
    pass at site `s` push the prediction the same way?
    """
    layers = _get_layers(base_model)
    layer_idx, kind = site

    last_idx = no_ctx_mask.sum(dim=-1) - 1  # (B,)
    handle = None

    def make_inject_hook():
        if kind == "attn":
            def pre_hook(module, args):
                x = args[0] if isinstance(args, tuple) else args
                x = x.clone()
                B = x.shape[0]
                x[torch.arange(B, device=x.device), last_idx] = (
                    x[torch.arange(B, device=x.device), last_idx] + delta_to_inject.to(x.dtype).to(x.device)
                )
                return (x,) + (args[1:] if isinstance(args, tuple) else ())
            return ("pre", pre_hook)
        else:
            def post_hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                B = h.shape[0]
                h[torch.arange(B, device=h.device), last_idx] = (
                    h[torch.arange(B, device=h.device), last_idx] + delta_to_inject.to(h.dtype).to(h.device)
                )
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return ("post", post_hook)

    layer = layers[layer_idx]
    target = layer.post_attention_layernorm if kind == "attn" else layer
    mode, hook = make_inject_hook()
    if mode == "pre":
        handle = target.register_forward_pre_hook(hook)
    else:
        handle = target.register_forward_hook(hook)

    try:
        out_inj = base_model(input_ids=no_ctx_ids, attention_mask=no_ctx_mask, use_cache=False).logits
    finally:
        handle.remove()

    out_clean = base_model(input_ids=no_ctx_ids, attention_mask=no_ctx_mask, use_cache=False).logits

    # KL between distributions at the last token position
    last_pos = last_idx
    B = out_inj.shape[0]
    p_inj = F.log_softmax(out_inj[torch.arange(B), last_pos], dim=-1).exp()
    log_p_inj = F.log_softmax(out_inj[torch.arange(B), last_pos], dim=-1)
    log_p_clean = F.log_softmax(out_clean[torch.arange(B), last_pos], dim=-1)
    kl = (p_inj * (log_p_inj - log_p_clean)).sum(dim=-1).mean().item()
    return kl
