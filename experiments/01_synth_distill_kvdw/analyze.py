"""Phase D — aggregate per-context outputs into the paper's headline figure and table.

Reads `outputs/01_synth_distill_kvdw/<context>/kvdw.json` for each completed
context. Renders:
  - figures/alignment_curve.png : 64-point per-site cosine, one line per context
  - figures/peak_alignment_table.csv : peak (layer, site, cosine) per context

Usage (from repo root):
  python experiments/01_synth_distill_kvdw/analyze.py \
    --out-root outputs/01_synth_distill_kvdw \
    --fig-dir figures
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


SITE_RE = re.compile(r"L(\d+)_(attn|mlp)")


def _parse_site(key: str) -> tuple[int, str]:
    m = SITE_RE.match(key)
    if not m:
        raise ValueError(f"Bad site key: {key}")
    return int(m.group(1)), m.group(2)


def load_per_context(out_root: Path) -> dict[str, dict]:
    """Return {context_name: kvdw_dict}. Skips contexts that don't have kvdw.json yet."""
    out = {}
    for ctx_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        kvdw_path = ctx_dir / "kvdw.json"
        if not kvdw_path.exists():
            print(f"[analyze] skipping {ctx_dir.name}: no kvdw.json")
            continue
        out[ctx_dir.name] = json.loads(kvdw_path.read_text(encoding="utf-8"))
    return out


def site_curve(per_site: dict, metric: str = "rmsnorm_cosine") -> list[tuple[int, str, float]]:
    """Return list of (layer, site_kind, value) sorted by (layer, site_kind order=attn<mlp)."""
    rows = []
    for k, m in per_site.items():
        layer, kind = _parse_site(k)
        v = m.get(metric)
        if v is None:
            continue
        rows.append((layer, kind, v))
    # attn before mlp at the same layer
    rows.sort(key=lambda r: (r[0], 0 if r[1] == "attn" else 1))
    return rows


def render_figure(data: dict[str, dict], fig_path: Path, metric: str = "rmsnorm_cosine"):
    plt.figure(figsize=(11, 5))
    for ctx, blob in data.items():
        rows = site_curve(blob["per_site"], metric=metric)
        x = list(range(len(rows)))
        y = [r[2] for r in rows]
        plt.plot(x, y, marker=".", label=ctx)
    plt.axhline(0.0, color="grey", linestyle=":", linewidth=0.5)
    plt.xlabel("site index (interleaved attn / mlp across layers)")
    plt.ylabel(f"cosine ({metric})")
    plt.title("KV-vs-ΔW alignment per residual-stream site")
    plt.legend()
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[analyze] wrote {fig_path}")


def render_peak_table(data: dict[str, dict], csv_path: Path, metric: str = "rmsnorm_cosine"):
    rows = []
    for ctx, blob in data.items():
        rows_ctx = site_curve(blob["per_site"], metric=metric)
        if not rows_ctx:
            continue
        peak = max(rows_ctx, key=lambda r: r[2])
        rows.append({
            "context": ctx,
            "peak_layer": peak[0],
            "peak_site": peak[1],
            "peak_cosine": peak[2],
            "n_probes": blob.get("n_probes"),
            "n_sites": blob.get("n_sites"),
        })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["context", "peak_layer", "peak_site", "peak_cosine", "n_probes", "n_sites"])
        w.writeheader()
        w.writerows(rows)
    print(f"[analyze] wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--metric", default="rmsnorm_cosine")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    fig_dir = Path(args.fig_dir)

    data = load_per_context(out_root)
    if not data:
        raise SystemExit("[analyze] no contexts found with kvdw.json")

    render_figure(data, fig_dir / "alignment_curve.png", metric=args.metric)
    render_peak_table(data, fig_dir / "peak_alignment_table.csv", metric=args.metric)

    # also render the cosine + token-space curves as auxiliary figures
    for m in ("cosine", "tokenspace_cosine"):
        try:
            render_figure(data, fig_dir / f"alignment_curve_{m}.png", metric=m)
        except Exception as e:  # noqa
            print(f"[analyze] skipping {m}: {e}")


if __name__ == "__main__":
    main()
