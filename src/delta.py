"""ΔW analysis utilities."""
from __future__ import annotations

import torch


def per_module_delta(base_state: dict, student_state: dict) -> dict:
    """Per-parameter Frobenius / max-abs of ΔW = θ_student − θ_base."""
    out = {}
    for k in base_state:
        if k not in student_state:
            continue
        d = (student_state[k].float() - base_state[k].float())
        out[k] = {
            "frob": d.norm().item(),
            "max_abs": d.abs().max().item(),
            "shape": tuple(d.shape),
        }
    return out


def lora_rank_spectrum(student_state: dict) -> dict:
    """For LoRA-finetuned models, return the singular values of each ΔW = B @ A.

    PEFT names the matrices like base_model.model.<...>.lora_A.default.weight (r, in)
    and lora_B.default.weight (out, r). We pair them up by stripping the suffix.
    """
    pairs = {}
    for name, w in student_state.items():
        if "lora_A.default.weight" in name:
            stem = name.replace("lora_A.default.weight", "")
            pairs.setdefault(stem, {})["A"] = w
        elif "lora_B.default.weight" in name:
            stem = name.replace("lora_B.default.weight", "")
            pairs.setdefault(stem, {})["B"] = w

    out = {}
    for stem, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        delta = (ab["B"].float() @ ab["A"].float())  # (out, in)
        s = torch.linalg.svdvals(delta)
        out[stem] = {
            "rank_eff": (s > 1e-6 * s.max()).sum().item(),
            "sv_top": s[: min(8, s.numel())].tolist(),
            "frob": delta.norm().item(),
        }
    return out
