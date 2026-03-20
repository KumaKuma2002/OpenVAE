#!/usr/bin/env python3
"""
Benchmark metric plot (matplotlib, non-interactive).

PNG: **OOD cohort only** (DSB + LUNA), 1×4 subplots — zoomed y-axis to show spread.

TXT/CSV: **all cohorts** (all / within / OOD), unchanged scope.

SOTA per subplot: grey-orange bar; others gray. OpenVAE-2D-4x-2K excluded.

Do not run on login node; use SLURM (see z_openvae/plot_benchmark.slurm).

Usage:
  python plot_benchmark_metrics.py --out_dir ../outputs/vae_benchmark/visualization
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Models omitted from comparison (e.g. small ablation checkpoint).
EXCLUDED_MODEL_NAMES = frozenset({"OpenVAE-2D-4x-2K"})
# Bar colors: neutral gray vs. grey-orange for per-subplot SOTA
BAR_GRAY = "#A8A8A8"
BAR_SOTA_GREY_ORANGE = "#C4894A"  # muted grey-orange


def patient_cohort(input_ct: str) -> str:
    """Return 'ood' for DSB_/LUNA_, else 'within'."""
    name = Path(input_ct).name
    stem = name.replace(".nii.gz", "").replace(".nii", "")
    if stem.startswith("DSB_") or stem.startswith("LUNA_"):
        return "ood"
    return "within"


def load_all_metrics(benchmark_root: Path) -> Dict[str, List[dict]]:
    """model_name -> list of row dicts with floats parsed."""
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for csv_path in sorted(benchmark_root.glob("*/metrics.csv")):
        model = csv_path.parent.name
        with csv_path.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    cohort = patient_cohort(row["input_ct"])
                    psnr = float(row["PSNR"])
                    ssim = float(row["SSIM"])
                    d100 = float(row.get("Detail_100", "nan"))
                    lp_s = row.get("LPIPS", "").strip()
                    if lp_s and lp_s.lower() != "nan":
                        lp = float(lp_s)
                    else:
                        lp = float("nan")
                    by_model[model].append(
                        {
                            "cohort": cohort,
                            "PSNR": psnr,
                            "SSIM": ssim,
                            "LPIPS": lp,
                            "Detail_100": d100,
                        }
                    )
                except (KeyError, ValueError) as e:
                    print(f"skip bad row in {csv_path}: {e}")
    return dict(by_model)


def mean_metrics(rows: List[dict]) -> Dict[str, float]:
    if not rows:
        return {"PSNR": float("nan"), "SSIM": float("nan"), "LPIPS": float("nan"), "Detail_100": float("nan")}
    out = {}
    for k in ("PSNR", "SSIM", "Detail_100"):
        vals = [r[k] for r in rows if not math.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    lp = [r["LPIPS"] for r in rows if not math.isnan(r["LPIPS"])]
    out["LPIPS"] = float(np.mean(lp)) if lp else float("nan")
    return out


def build_cohort_tables(
    by_model: Dict[str, List[dict]],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[str]]:
    """
    Returns:
      tables[cohort_key][model_name] = {PSNR, SSIM, LPIPS, Detail_100}
      cohort_key in 'all', 'within', 'ood'
    """
    models = sorted(by_model.keys())
    tables: Dict[str, Dict[str, Dict[str, float]]] = {
        "all": {},
        "within": {},
        "ood": {},
    }

    for m in models:
        rows = by_model[m]
        tables["all"][m] = mean_metrics(rows)
        w = [r for r in rows if r["cohort"] == "within"]
        o = [r for r in rows if r["cohort"] == "ood"]
        tables["within"][m] = mean_metrics(w)
        tables["ood"][m] = mean_metrics(o)

    return tables, models


def _tight_ylim(ax, vals: np.ndarray) -> None:
    """Zoom y-axis to data range + padding so small differences are visible."""
    vals = np.asarray(vals, dtype=float)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return
    vmin = float(np.nanmin(vals[valid]))
    vmax = float(np.nanmax(vals[valid]))
    span = max(vmax - vmin, 1e-12)
    # 10% margin; if nearly flat, inflate band so bars don't look identical
    pad = max(span * 0.10, span * 0.02 + 1e-6)
    if span < 1e-6 * max(abs(vmax), 1.0):
        pad = max(0.02 * max(abs(vmax), 1.0), 0.005)
    lo, hi = vmin - pad, vmax + pad
    ax.set_ylim(lo, hi)


def sort_models_ood_by_detail_100(
    models: List[str], ood_tab: Dict[str, Dict[str, float]]
) -> List[str]:
    """
    Left → right: strongest to weakest by OOD mean Detail_100 (primary rank).
    NaN Detail_100 sorts last.
    """
    def sort_key(m: str) -> float:
        v = ood_tab[m]["Detail_100"]
        return float("-inf") if math.isnan(v) else float(v)

    return sorted(models, key=sort_key, reverse=True)


def best_model_index(values: np.ndarray, higher_is_better: bool) -> int:
    valid = ~np.isnan(values)
    if not np.any(valid):
        return -1
    v = np.where(valid, values, -np.inf if higher_is_better else np.inf)
    if higher_is_better:
        return int(np.nanargmax(v))
    return int(np.nanargmin(np.where(valid, values, np.inf)))


def plot_ood_row_only(
    tables: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    out_png: Path,
) -> None:
    """Single row: OOD cohort only; y-axis zoomed per metric."""
    rk = "ood"
    rtitle = "OOD (DSB & LUNA)"
    tab = tables[rk]
    # Same x order in all subplots: strong → weak by OOD Detail_100
    models_ord = sort_models_ood_by_detail_100(models, tab)
    metrics = [
        ("PSNR", "PSNR (dB)", True),
        ("SSIM", "SSIM", True),
        ("LPIPS", "LPIPS", False),
        ("Detail_100", "Detail_100", True),
    ]

    n_models = len(models_ord)
    x = np.arange(n_models)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    sota_name = models_ord[0] if models_ord else "n/a"

    for ci, (mkey, mlabel, higher) in enumerate(metrics):
        ax = axes[ci]
        vals = np.array([tab[m][mkey] for m in models_ord], dtype=float)
        bi = best_model_index(vals, higher)
        bar_colors = [BAR_GRAY] * n_models
        if bi >= 0:
            bar_colors[bi] = BAR_SOTA_GREY_ORANGE
        ax.bar(
            x,
            vals,
            color=bar_colors,
            edgecolor="#666666",
            linewidth=0.6,
        )

        _tight_ylim(ax, vals)

        ax.set_xticks(x)
        ax.set_xticklabels(models_ord, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.grid(axis="y", alpha=0.3)

    axes[0].text(
        -0.02,
        1.18,
        f"{rtitle}\nOrder: Detail_100 (strong→weak)  |  SOTA: {sota_name}",
        transform=axes[0].transAxes,
        fontsize=10,
        fontweight="bold",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.85, edgecolor="gray"),
    )

    fig.suptitle(
        "VAE benchmark — OOD only (excl. OpenVAE-2D-4x-2K). "
        "Bars left→right: OOD Detail_100 rank (strongest to weakest). "
        "Grey-orange = best per metric in that subplot; "
        "leftmost is best Detail_100 only. Y-axis zoomed.",
        fontsize=10,
        y=1.08,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"[plot] OOD bar order (left→right): {', '.join(models_ord)}")


def write_txt_and_csv(
    tables: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    out_txt: Path,
    out_csv: Path,
) -> None:
    lines = []
    lines.append("VAE benchmark — cohort averages (full tables)")
    lines.append("Excluded models: OpenVAE-2D-4x-2K")
    lines.append(
        "Note: PNG figure metric_comparison_OOD_1x4.png shows OOD cohort only; "
        "this file lists all cohorts (all / within / OOD)."
    )
    lines.append("SOTA (per row) = highest mean Detail_100 in that cohort.")
    lines.append("")

    for rk, title in [
        ("all", "ALL PATIENTS"),
        ("within", "IN-DISTRIBUTION (excl. DSB, LUNA)"),
        ("ood", "OOD (DSB + LUNA)"),
    ]:
        lines.append("=" * 72)
        lines.append(title)
        lines.append("=" * 72)
        tab = tables[rk]
        dlist = [(m, tab[m]["Detail_100"]) for m in models]
        dlist_valid = [(m, v) for m, v in dlist if not math.isnan(v)]
        sota = max(dlist_valid, key=lambda t: t[1])[0] if dlist_valid else "n/a"
        lines.append(f"Row SOTA (Detail_100): {sota}")
        lines.append("")
        header = f"{'model':<32} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'Detail_100':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for m in models:
            t = tab[m]
            lp = t["LPIPS"]
            lps = f"{lp:.6f}" if not math.isnan(lp) else "nan"
            lines.append(
                f"{m:<32} {t['PSNR']:10.4f} {t['SSIM']:10.6f} {lps:>10} {t['Detail_100']:12.4f}"
            )
        lines.append("")

    out_txt.write_text("\n".join(lines))

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "model", "PSNR", "SSIM", "LPIPS", "Detail_100"])
        for rk in ("all", "within", "ood"):
            for m in models:
                t = tables[rk][m]
                w.writerow(
                    [
                        rk,
                        m,
                        t["PSNR"],
                        t["SSIM"],
                        t["LPIPS"] if not math.isnan(t["LPIPS"]) else "",
                        t["Detail_100"],
                    ]
                )

    print(f"Wrote {out_txt}")
    print(f"Wrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("../outputs/vae_benchmark"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("../outputs/vae_benchmark/visualization"),
        help="Default: ./vae_benchmark/visualization under outputs",
    )
    args = ap.parse_args()

    root = args.benchmark_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_model = load_all_metrics(root)
    if not by_model:
        raise SystemExit(f"No metrics found under {root}")

    for name in EXCLUDED_MODEL_NAMES:
        if name in by_model:
            del by_model[name]
            print(f"[plot] Excluded model: {name}")
    if not by_model:
        raise SystemExit("No models left after exclusions.")

    tables, models = build_cohort_tables(by_model)

    plot_ood_row_only(tables, models, out_dir / "metric_comparison_OOD_1x4.png")
    write_txt_and_csv(
        tables,
        models,
        out_dir / "cohort_tables.txt",
        out_dir / "cohort_averages.csv",
    )

    # small README in visualization folder
    (out_dir / "README.txt").write_text(
        """metric_comparison_OOD_1x4.png — OOD cohort (DSB + LUNA) only, 1×4 metrics.
Bars left→right: OOD mean Detail_100, strongest to weakest.
Grey-orange = best bar in that subplot (per-metric SOTA). Y-axis zoomed.
OpenVAE-2D-4x-2K excluded from plot/tables.

cohort_tables.txt / cohort_averages.csv — ALL cohorts (all, within, OOD); same exclusions.
"""
    )
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
