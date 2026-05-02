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


def per_layer_curve(per_site: dict, metric: str) -> list[tuple[int, float]]:
    """Aggregate (layer, attn) and (layer, mlp) into a single per-layer mean."""
    by_layer: dict[int, list[float]] = {}
    for k, m in per_site.items():
        layer, _ = _parse_site(k)
        v = m.get(metric)
        if v is None:
            continue
        by_layer.setdefault(layer, []).append(v)
    return [(L, sum(v) / len(v)) for L, v in sorted(by_layer.items())]


def render_per_layer_figure(data: dict[str, dict], fig_path: Path, metric: str = "rmsnorm_cosine"):
    plt.figure(figsize=(10, 4.5))
    for ctx, blob in data.items():
        rows = per_layer_curve(blob["per_site"], metric=metric)
        x = [r[0] for r in rows]
        y = [r[1] for r in rows]
        plt.plot(x, y, marker="o", label=ctx)
    plt.axhline(0.0, color="grey", linestyle=":", linewidth=0.5)
    plt.xlabel("layer")
    plt.ylabel(f"per-layer mean cosine ({metric})")
    plt.title("KV-vs-ΔW alignment averaged across attn+mlp at each layer")
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


def render_delta_norms(out_root: Path, fig_dir: Path):
    """If per-layer ΔW Frob JSONs exist, render them too."""
    data: dict[str, dict] = {}
    for ctx_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        path = ctx_dir / "delta_norms.json"
        if path.exists():
            data[ctx_dir.name] = json.loads(path.read_text(encoding="utf-8"))

    if not data:
        print("[analyze] no delta_norms.json files yet; skipping ΔW figures")
        return

    # Per-layer Frob (one line per context). frob_per_param controls for n_params drift across layers.
    plt.figure(figsize=(10, 4.5))
    for ctx, blob in data.items():
        rows = blob["per_layer"]
        x = [r["layer"] for r in rows]
        y = [r["frob_per_param"] for r in rows]
        plt.plot(x, y, marker="o", label=ctx)
    plt.xlabel("layer")
    plt.ylabel("‖ΔW‖_F per parameter")
    plt.title("Per-layer ΔW Frobenius norm (RMS per parameter)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "delta_per_layer.png", dpi=150)
    plt.close()
    print(f"[analyze] wrote {fig_dir / 'delta_per_layer.png'}")

    # Per-group totals (attn.q vs attn.k vs ... vs mlp.down). Bar chart per context.
    groups = sorted({g for blob in data.values() for g in blob["by_group"]})
    plt.figure(figsize=(10, 5))
    width = 0.16
    xs = list(range(len(groups)))
    for i, (ctx, blob) in enumerate(sorted(data.items())):
        ys = [blob["by_group"].get(g, 0.0) for g in groups]
        plt.bar([x + i * width for x in xs], ys, width=width, label=ctx)
    plt.xticks([x + 2 * width for x in xs], groups, rotation=30, ha="right")
    plt.ylabel("‖ΔW‖_F (group total)")
    plt.title("ΔW Frobenius norm by parameter group, per context")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "delta_by_group.png", dpi=150)
    plt.close()
    print(f"[analyze] wrote {fig_dir / 'delta_by_group.png'}")

    # Per-layer per-group heatmap-style: separate attn.* and mlp.* lines per layer.
    # For each context, plot two lines per layer: total attn ΔW and total mlp ΔW.
    plt.figure(figsize=(10, 5))
    for ctx, blob in sorted(data.items()):
        # Aggregate to attn-vs-mlp per layer
        by_layer_kind: dict[int, dict[str, float]] = {}
        for r in blob["per_layer_group"]:
            kind = "attn" if r["group"].startswith("attn.") else ("mlp" if r["group"].startswith("mlp.") else None)
            if kind is None:
                continue
            d = by_layer_kind.setdefault(r["layer"], {"attn": 0.0, "mlp": 0.0})
            # combine via sum-of-squares since these are Frob norms
            d[kind] = (d[kind] ** 2 + r["frob"] ** 2) ** 0.5
        layers = sorted(by_layer_kind)
        attn_y = [by_layer_kind[L]["attn"] for L in layers]
        mlp_y = [by_layer_kind[L]["mlp"] for L in layers]
        plt.plot(layers, attn_y, linestyle="-", marker=".", label=f"{ctx} attn")
        plt.plot(layers, mlp_y, linestyle="--", marker=".", label=f"{ctx} mlp")
    plt.xlabel("layer")
    plt.ylabel("‖ΔW‖_F (attn vs mlp, summed over group)")
    plt.title("ΔW Frobenius norm split by sublayer per layer")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "delta_attn_vs_mlp_per_layer.png", dpi=150)
    plt.close()
    print(f"[analyze] wrote {fig_dir / 'delta_attn_vs_mlp_per_layer.png'}")


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
    render_per_layer_figure(data, fig_dir / "alignment_per_layer.png", metric=args.metric)
    render_peak_table(data, fig_dir / "peak_alignment_table.csv", metric=args.metric)

    # ΔW Frobenius figures (only if delta_norms.json files exist)
    render_delta_norms(out_root, fig_dir)

    # Triangle figures (cos_ctx_base, cos_dw_base, base_norm) — only if present.
    if any("cos_ctx_base" in next(iter(d["per_site"].values()), {}) for d in data.values()):
        render_triangle(data, fig_dir)

    # S2 vs S3 (ctxonly) comparison — if any ctxonly_* entries exist
    s3_data = {k: v for k, v in data.items() if k.startswith("ctxonly_")}
    s2_data = {k: v for k, v in data.items() if not k.startswith("ctxonly_") and k != "no_context"}
    if s3_data and s2_data:
        render_s2_vs_s3(s2_data, s3_data, fig_dir, metric=args.metric)


def render_s2_vs_s3(s2: dict, s3: dict, fig_dir: Path, metric: str = "rmsnorm_cosine"):
    """Per-context: solid line = S2 (synth-FT) v_dw vs v_ctx alignment;
    dashed line = S3 (ctxonly-FT) v_dw vs v_ctx alignment. Both share the
    same v_ctx (same context applied at inference)."""
    contexts = sorted(set(s2.keys()) & {k.replace("ctxonly_", "") for k in s3.keys()})
    if not contexts:
        return
    plt.figure(figsize=(11, 5))
    for ctx in contexts:
        rows_s2 = per_layer_curve(s2[ctx]["per_site"], metric=metric)
        rows_s3 = per_layer_curve(s3[f"ctxonly_{ctx}"]["per_site"], metric=metric)
        line, = plt.plot([r[0] for r in rows_s2], [r[1] for r in rows_s2],
                         linestyle="-", marker=".", label=f"{ctx} (context-simulate-FT)")
        plt.plot([r[0] for r in rows_s3], [r[1] for r in rows_s3],
                 linestyle="--", marker=".", color=line.get_color(),
                 label=f"{ctx} (context-FT)")
    plt.axhline(0.0, color="grey", linestyle=":", linewidth=0.5)
    plt.xlabel("layer")
    plt.ylabel(f"cos(v_ctx, v_dw) ({metric})")
    plt.title("context-simulate vs context-FT: alignment with the same v_ctx")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "s2_vs_s3_alignment.png", dpi=150)
    plt.close()
    print(f"[analyze] wrote {fig_dir / 's2_vs_s3_alignment.png'}")


def render_triangle(data: dict[str, dict], fig_dir: Path):
    """Plot per-layer cos(v_ctx, base) and cos(v_dw, base), plus relative norms."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ctx, blob in data.items():
        per_site = blob["per_site"]
        # average attn/mlp at each layer
        def by_layer_mean(metric):
            by = {}
            for k, m in per_site.items():
                if metric not in m:
                    continue
                L, _ = _parse_site(k)
                by.setdefault(L, []).append(m[metric])
            return [(L, sum(v) / len(v)) for L, v in sorted(by.items())]

        for ax, (metric, title) in zip(axes, [
            ("cos_ctx_base", "cos(v_ctx, base): does context push along base flow?"),
            ("cos_dw_base", "cos(v_dw, base): does ΔW push along base flow?"),
            ("ctx_relative_norm", "‖v_ctx‖ / ‖base‖ (solid) and ‖v_dw‖ / ‖base‖ (dashed)"),
        ]):
            rows = by_layer_mean(metric)
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            line, = ax.plot(xs, ys, marker=".", label=ctx)
            if metric == "ctx_relative_norm":
                rows2 = by_layer_mean("dw_relative_norm")
                ax.plot([r[0] for r in rows2], [r[1] for r in rows2],
                        linestyle="--", color=line.get_color())

    for ax, t in zip(axes, [
        "cos(v_ctx, base)",
        "cos(v_dw, base)",
        "relative magnitude vs base",
    ]):
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.5)
        ax.set_xlabel("layer")
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "triangle.png", dpi=150)
    plt.close()
    print(f"[analyze] wrote {fig_dir / 'triangle.png'}")

    # also render the cosine + token-space curves as auxiliary figures
    for m in ("cosine", "tokenspace_cosine"):
        try:
            render_figure(data, fig_dir / f"alignment_curve_{m}.png", metric=m)
        except Exception as e:  # noqa
            print(f"[analyze] skipping {m}: {e}")


if __name__ == "__main__":
    main()
