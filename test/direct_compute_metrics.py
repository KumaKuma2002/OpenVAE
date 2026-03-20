#!/usr/bin/env python3
"""
Direct metric computation only -- no VAE inference.

Reads each outputs/vae_benchmark/<model>/metrics.csv, reloads input_ct and recon_ct,
rewrites MAE_100 / PSNR / SSIM / Detail_100 (and optionally LPIPS) from NIfTI pairs.

Optional: --skip-lpips keeps existing LPIPS column from CSV (faster; no GPU needed).

MAE_100:    3D voxel-wise (1 - MAE) * 100 on [0,1] volumes (100 = identical).
Detail_100: 3D gradient-magnitude Pearson r (see ../outputs/vae_benchmark/METRICS_README.txt).

Metric priority: MAE_100, Detail_100, SSIM, PSNR, LPIPS

Usage:
  python direct_compute_metrics.py --benchmark_root ../outputs/vae_benchmark --skip-lpips

SLURM (0 GPU): sbatch direct_compute_metrics.slurm
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

try:
    import lpips
except ImportError:
    lpips = None


def ct_hu_to_01(vol: np.ndarray) -> np.ndarray:
    vol = np.clip(vol.astype(np.float64), -1000.0, 1000.0)
    return ((vol + 1000.0) / 2000.0).astype(np.float32)


def mae_3d_100(gt_01: np.ndarray, pred_01: np.ndarray) -> float:
    """3D Mean Absolute Error mapped to [0, 100] where 100 = identical.

    MAE_100 = (1 - MAE) * 100, with MAE computed on full 3D volumes in [0,1].
    """
    mae = float(np.mean(np.abs(gt_01.astype(np.float64) - pred_01.astype(np.float64))))
    return float(np.clip((1.0 - mae) * 100.0, 0.0, 100.0))


def detail_grad_mag_corr_100(gt_01: np.ndarray, pred_01: np.ndarray) -> float:
    """3D gradient-magnitude Pearson correlation, scaled to [0, 100]."""
    g = np.stack(np.gradient(gt_01.astype(np.float64)), axis=0)
    p = np.stack(np.gradient(pred_01.astype(np.float64)), axis=0)
    mag_g = np.sqrt(np.sum(g * g, axis=0))
    mag_p = np.sum(p * p, axis=0)
    mag_p = np.sqrt(np.maximum(mag_p, 0.0))

    a = mag_g.ravel()
    b = mag_p.ravel()
    if a.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    r = float(np.dot(a, b) / denom)
    if math.isnan(r):
        return 0.0
    return float(np.clip(r, 0.0, 1.0) * 100.0)


def compute_lpips_slicewise_cpu(
    lpips_model, gt_01: np.ndarray, pred_01: np.ndarray, stride: int, device: torch.device
) -> float:
    vals = []
    d = gt_01.shape[2]
    for z in range(0, d, max(stride, 1)):
        g = torch.from_numpy(gt_01[..., z]).unsqueeze(0).unsqueeze(0).float().to(device)
        p = torch.from_numpy(pred_01[..., z]).unsqueeze(0).unsqueeze(0).float().to(device)
        g = g.repeat(1, 3, 1, 1) * 2.0 - 1.0
        p = p.repeat(1, 3, 1, 1) * 2.0 - 1.0
        with torch.no_grad():
            vals.append(float(lpips_model(g, p).item()))
    return float(np.mean(vals)) if vals else float("nan")


def compute_all_metrics(
    gt_hu: np.ndarray,
    pred_hu: np.ndarray,
    lpips_model,
    device: torch.device,
    lpips_stride: int,
    compute_lpips: bool,
) -> Dict[str, float]:
    gt_01 = ct_hu_to_01(gt_hu)
    pred_01 = ct_hu_to_01(pred_hu)

    mae100 = mae_3d_100(gt_01, pred_01)
    psnr = float(peak_signal_noise_ratio(gt_01, pred_01, data_range=1.0))
    ssim = float(structural_similarity(gt_01, pred_01, data_range=1.0, channel_axis=None))

    if compute_lpips and lpips_model is not None:
        lp = compute_lpips_slicewise_cpu(lpips_model, gt_01, pred_01, lpips_stride, device)
    else:
        lp = float("nan")

    detail = detail_grad_mag_corr_100(gt_01, pred_01)
    return {"MAE_100": mae100, "PSNR": psnr, "SSIM": ssim, "LPIPS": lp, "Detail_100": detail}


CSV_FIELDS = [
    "model_name",
    "model_type",
    "input_ct",
    "recon_ct",
    "MAE_100",
    "Detail_100",
    "SSIM",
    "PSNR",
    "LPIPS",
]


def process_one_metrics_csv(
    csv_path: Path,
    model_name: str,
    model_index: int,
    num_models: int,
    lpips_model,
    device: torch.device,
    lpips_stride: int,
    compute_lpips: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    print(
        f"[direct_compute] Model {model_index}/{num_models}: {model_name} "
        f"({len(raw_rows)} rows in metrics.csv)",
        flush=True,
    )

    rows: List[dict] = []
    ok, skip = 0, 0
    bar = tqdm(
        raw_rows,
        desc=f"  {model_name}",
        unit="vol",
        leave=True,
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for row in bar:
        inp = Path(row["input_ct"])
        rec = Path(row["recon_ct"])
        if not inp.is_file() or not rec.is_file():
            tqdm.write(f"  [skip] missing file: {inp.name} / {rec.name}")
            skip += 1
            rows.append(row)
            continue

        gt = nib.load(str(inp)).get_fdata().astype(np.float32)
        pred = nib.load(str(rec)).get_fdata().astype(np.float32)
        if gt.shape != pred.shape:
            tqdm.write(
                f"  [skip] shape mismatch {inp.name}: {gt.shape} vs {pred.shape}"
            )
            skip += 1
            rows.append(row)
            continue

        m = compute_all_metrics(gt, pred, lpips_model, device, lpips_stride, compute_lpips)
        row["MAE_100"] = str(m["MAE_100"])
        row["PSNR"] = str(m["PSNR"])
        row["SSIM"] = str(m["SSIM"])
        if compute_lpips:
            row["LPIPS"] = "" if math.isnan(m["LPIPS"]) else str(m["LPIPS"])
        row["Detail_100"] = str(m["Detail_100"])
        rows.append(row)
        ok += 1
        bar.set_postfix(
            last=inp.stem[:28] + ("..." if len(inp.stem) > 28 else ""),
            MAE=f"{m['MAE_100']:.1f}",
            Detail=f"{m['Detail_100']:.1f}",
        )

    print(
        f"[direct_compute]   done {model_name}: updated={ok} skipped={skip}",
        flush=True,
    )

    if dry_run:
        return ok, skip

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in CSV_FIELDS}
            w.writerow(out)

    return ok, skip


def write_summary(benchmark_root: Path) -> None:
    """Regenerate summary.csv (+ README) including MAE_100 and Detail_100 means."""
    rows_out: List[dict] = []
    for csv_path in sorted(benchmark_root.glob("*/metrics.csv")):
        model_folder = csv_path.parent.name
        mae100s, psnrs, ssims, lpips_list, details = [], [], [], [], []
        model_type = ""
        with csv_path.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    psnrs.append(float(row["PSNR"]))
                    ssims.append(float(row["SSIM"]))
                    m_s = row.get("MAE_100", "").strip()
                    if m_s:
                        mae100s.append(float(m_s))
                    v = row.get("LPIPS", "").strip()
                    if v and v.lower() != "nan":
                        try:
                            lp = float(v)
                            if not math.isnan(lp):
                                lpips_list.append(lp)
                        except ValueError:
                            pass
                    d = row.get("Detail_100", "").strip()
                    if d:
                        details.append(float(d))
                    if not model_type and row.get("model_type"):
                        model_type = row["model_type"]
                except (KeyError, ValueError):
                    continue

        n = len(psnrs)
        if n == 0:
            continue
        psnr_mean = sum(psnrs) / n
        ssim_mean = sum(ssims) / n
        mae100_mean = sum(mae100s) / len(mae100s) if mae100s else float("nan")
        lpips_mean = sum(lpips_list) / len(lpips_list) if lpips_list else float("nan")
        detail_mean = sum(details) / len(details) if details else float("nan")

        ssim_100 = ssim_mean * 100.0
        psnr_100 = min(100.0, (psnr_mean / 50.0) * 100.0)

        rows_out.append(
            {
                "model_name": model_folder,
                "model_type": model_type,
                "n_cases": n,
                "MAE_100_mean": mae100_mean,
                "Detail_100_mean": detail_mean,
                "SSIM_mean": ssim_mean,
                "SSIM_100": ssim_100,
                "PSNR_dB_mean": psnr_mean,
                "PSNR_100": psnr_100,
                "LPIPS_mean": lpips_mean,
            }
        )

    rows_out.sort(key=lambda x: x["model_name"])
    summary_path = benchmark_root / "summary.csv"
    summary_fields = [
        "model_name",
        "model_type",
        "n_cases",
        "MAE_100_mean",
        "Detail_100_mean",
        "SSIM_mean",
        "SSIM_100",
        "PSNR_dB_mean",
        "PSNR_100",
        "LPIPS_mean",
    ]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in rows_out:
            w.writerow(
                {
                    "model_name": row["model_name"],
                    "model_type": row["model_type"],
                    "n_cases": row["n_cases"],
                    "MAE_100_mean": ""
                    if math.isnan(row["MAE_100_mean"])
                    else f"{row['MAE_100_mean']:.4f}",
                    "Detail_100_mean": ""
                    if math.isnan(row["Detail_100_mean"])
                    else f"{row['Detail_100_mean']:.4f}",
                    "SSIM_mean": f"{row['SSIM_mean']:.6f}",
                    "SSIM_100": f"{row['SSIM_100']:.4f}",
                    "PSNR_dB_mean": f"{row['PSNR_dB_mean']:.6f}",
                    "PSNR_100": f"{row['PSNR_100']:.4f}",
                    "LPIPS_mean": ""
                    if math.isnan(row["LPIPS_mean"])
                    else f"{row['LPIPS_mean']:.6f}",
                }
            )

    readme = benchmark_root / "summary_README.txt"
    readme.write_text(
        "summary.csv -- aggregated from each model's metrics.csv (mean over n_cases).\n"
        "\n"
        "Metric priority: MAE_100, Detail_100, SSIM, PSNR, LPIPS\n"
        "\n"
        "Columns:\n"
        "  MAE_100_mean    = mean of per-case MAE_100 (0-100, higher is better)\n"
        "                    MAE_100 = (1 - MAE) x 100 on 3D volumes in [0,1]; 100 = identical\n"
        "  Detail_100_mean = mean of per-case Detail_100 (0-100, higher is better)\n"
        "                    Pearson corr of 3D gradient magnitudes x 100; sensitive to blur\n"
        "  SSIM_mean       = mean SSIM (0-1, higher is better)\n"
        "  SSIM_100        = SSIM_mean x 100  (0-100 scale)\n"
        "  PSNR_dB_mean    = raw average PSNR in dB\n"
        "  PSNR_100        = min(100, PSNR_dB_mean / 50 x 100)  (50 dB -> 100)\n"
        "  LPIPS_mean      = mean LPIPS (lower is better; AlexNet LPIPS if computed)\n"
    )
    print(f"[direct_compute] Wrote {summary_path} and {readme}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("../outputs/vae_benchmark"),
        help="Root folder containing <model>/metrics.csv",
    )
    ap.add_argument("--lpips_stride", type=int, default=4)
    ap.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Do not compute LPIPS (leaves column empty / NaN as empty string).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.benchmark_root.resolve()
    compute_lpips = not args.skip_lpips and lpips is not None and torch is not None
    device = torch.device("cpu")
    lpips_model = None

    print("=" * 72, flush=True)
    print("[direct_compute] Direct metrics (no inference)", flush=True)
    print(f"[direct_compute] start: {datetime.now().isoformat()}", flush=True)
    print(f"[direct_compute] benchmark_root: {root}", flush=True)
    print(f"[direct_compute] dry_run: {args.dry_run}", flush=True)
    if compute_lpips:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        print("[direct_compute] LPIPS: ON (CPU, slow). Tip: --skip-lpips to skip.", flush=True)
    else:
        print(
            "[direct_compute] LPIPS: OFF (--skip-lpips or missing torch/lpips); "
            "existing CSV LPIPS kept.",
            flush=True,
        )
    print("=" * 72, flush=True)

    csvs = sorted(root.glob("*/metrics.csv"))
    if not csvs:
        raise SystemExit(f"No metrics.csv under {root}")

    print(f"[direct_compute] Found {len(csvs)} model folder(s) with metrics.csv\n", flush=True)

    n_models = len(csvs)
    for mi, csv_path in enumerate(csvs, start=1):
        model_name = csv_path.parent.name
        process_one_metrics_csv(
            csv_path,
            model_name,
            mi,
            n_models,
            lpips_model,
            device,
            args.lpips_stride,
            compute_lpips,
            args.dry_run,
        )

    if not args.dry_run:
        print("\n[direct_compute] Writing summary.csv ...", flush=True)
        write_summary(root)

    print(
        f"\n[direct_compute] finished: {datetime.now().isoformat()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
