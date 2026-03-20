"""
Benchmark one VAE model at a time on one or more CT volumes.

Model folder conventions under ckpt:
- 2D model: <model_dir>/vae/config.json
- 3D model: <model_dir>/autoencoder_best.pt

Outputs:
- Reconstructed CTs: <output_root>/<model_name>/*.nii.gz
- Metrics CSV:       <output_root>/<model_name>/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from diffusers import AutoencoderKL
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

try:
    import lpips
except ModuleNotFoundError:
    lpips = None

try:
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
except ModuleNotFoundError:
    AutoencoderKlMaisi = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Benchmark one VAE model on CT volume(s).')
    p.add_argument(
        '--input_ct',
        type=str,
        default=None,
        help='Single .nii/.nii.gz file to benchmark (use this for one CT only).',
    )
    p.add_argument(
        '--input_ct_dir',
        type=str,
        default=None,
        help='Directory: benchmark every .nii/.nii.gz inside it.',
    )
    p.add_argument('--model_dir', type=str, required=True, help='One model folder under ckpt.')
    p.add_argument('--output_root', type=str, default='../outputs/vae_benchmark', help='Root output folder.')
    p.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64])
    p.add_argument('--overlap', type=int, nargs=3, default=[16, 16, 16])
    p.add_argument('--latent_channels', type=int, default=4)
    p.add_argument('--num_splits', type=int, default=1)
    p.add_argument('--dim_split', type=int, default=1)
    p.add_argument('--lpips_stride', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--device', type=str, default=None)
    ns = p.parse_args()
    if not ns.input_ct and not ns.input_ct_dir:
        p.error('Provide either --input_ct (single volume) or --input_ct_dir (all NIfTIs in folder).')
    if ns.input_ct and ns.input_ct_dir:
        p.error('Use only one of --input_ct or --input_ct_dir, not both.')
    return ns


def ct_hu_to_01(vol: np.ndarray) -> np.ndarray:
    vol = np.clip(vol, -1000.0, 1000.0)
    return ((vol + 1000.0) / 2000.0).astype(np.float32)


def ct_01_to_hu(vol: np.ndarray) -> np.ndarray:
    return (vol * 2000.0 - 1000.0).astype(np.float32)


def stem_nii(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith('.nii.gz') else path.stem


def discover_ct_files(input_ct_dir: Path) -> List[Path]:
    files = sorted(input_ct_dir.glob('*.nii.gz')) + sorted(input_ct_dir.glob('*.nii'))
    return sorted(set(files))


def detect_model_type(model_dir: Path) -> str:
    if (model_dir / 'vae' / 'config.json').exists():
        return '2d'
    if (model_dir / 'autoencoder_best.pt').exists():
        return '3d'
    raise RuntimeError(
        f'Unknown model layout for {model_dir}. '
        'Expected either vae/config.json (2D) or autoencoder_best.pt (3D).'
    )


def compute_lpips_slicewise(lpips_model, gt_01: np.ndarray, pred_01: np.ndarray, device: torch.device, stride: int) -> float:
    vals = []
    for z in range(0, gt_01.shape[2], max(stride, 1)):
        g = torch.from_numpy(gt_01[..., z]).unsqueeze(0).unsqueeze(0).to(device)
        p = torch.from_numpy(pred_01[..., z]).unsqueeze(0).unsqueeze(0).to(device)
        g = g.repeat(1, 3, 1, 1) * 2.0 - 1.0
        p = p.repeat(1, 3, 1, 1) * 2.0 - 1.0
        with torch.no_grad():
            vals.append(float(lpips_model(g, p).item()))
    return float(np.mean(vals)) if vals else float('nan')


def detail_grad_mag_corr_100(gt_01: np.ndarray, pred_01: np.ndarray) -> float:
    """
    3D gradient-magnitude Pearson r between GT and recon ([0,1] volumes), ×100, clipped.
    Higher = better preservation of fine structure (edges / texture).
    """
    g = np.stack(np.gradient(gt_01.astype(np.float64)), axis=0)
    p = np.stack(np.gradient(pred_01.astype(np.float64)), axis=0)
    mag_g = np.sqrt(np.sum(g * g, axis=0))
    mag_p = np.sqrt(np.maximum(np.sum(p * p, axis=0), 0.0))
    a = mag_g.ravel() - mag_g.mean()
    b = mag_p.ravel() - mag_p.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    r = float(np.dot(a, b) / denom)
    if math.isnan(r):
        return 0.0
    return float(np.clip(r, 0.0, 1.0) * 100.0)


def compute_metrics(gt_hu: np.ndarray, pred_hu: np.ndarray, lpips_model, device: torch.device, lpips_stride: int) -> Dict[str, float]:
    gt_01 = ct_hu_to_01(gt_hu)
    pred_01 = ct_hu_to_01(pred_hu)
    psnr = float(peak_signal_noise_ratio(gt_01, pred_01, data_range=1.0))
    ssim = float(structural_similarity(gt_01, pred_01, data_range=1.0, channel_axis=None))
    lp = float('nan') if lpips_model is None else compute_lpips_slicewise(lpips_model, gt_01, pred_01, device, lpips_stride)
    detail = detail_grad_mag_corr_100(gt_01, pred_01)
    return {'PSNR': psnr, 'SSIM': ssim, 'LPIPS': lp, 'Detail_100': detail}


def reconstruct_2d(vae: AutoencoderKL, vol_hu: np.ndarray, amp: bool) -> np.ndarray:
    h, w, d = vol_hu.shape
    rec = np.zeros((h, w, d), dtype=np.float32)

    for z in range(d):
        sl_01 = ct_hu_to_01(vol_hu[..., z])
        x = torch.from_numpy(sl_01 * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).to(vae.device)
        x3 = x.repeat(1, 3, 1, 1)

        with torch.no_grad():
            with torch.autocast(device_type=vae.device.type, enabled=amp):
                latent = vae.encode(x3).latent_dist.sample()
                out = vae.decode(latent).sample[:, 0:1]

        out_01 = ((out.clamp(-1, 1) + 1.0) / 2.0).squeeze().float().cpu().numpy()
        rec[..., z] = ct_01_to_hu(out_01)

    return rec


def build_3d_model(device: torch.device, amp: bool, num_splits: int, dim_split: int, latent_channels: int) -> AutoencoderKlMaisi:
    if AutoencoderKlMaisi is None:
        raise ModuleNotFoundError('monai is required for 3D benchmarking')

    return AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        num_channels=[64, 128, 256],
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=amp,
        num_splits=num_splits,
        dim_split=dim_split,
    ).to(device)


def gaussian_weight(patch_size: Tuple[int, int, int], sigma_scale: float = 0.125) -> np.ndarray:
    coords = [np.linspace(-1, 1, s) for s in patch_size]
    grid = np.meshgrid(*coords, indexing='ij')
    sigma = sigma_scale * 2
    w = np.exp(-sum(g ** 2 for g in grid) / (2 * sigma ** 2))
    w = w / w.max()
    return np.clip(w, 1e-6, None).astype(np.float32)


def reconstruct_3d(model: AutoencoderKlMaisi, vol_hu: np.ndarray, patch_size: Tuple[int, int, int], overlap: Tuple[int, int, int], device: torch.device, amp: bool) -> np.ndarray:
    vol_01 = ct_hu_to_01(vol_hu)
    h, w, d = vol_01.shape
    ph, pw, pd = patch_size
    oh, ow, od = overlap

    vol = np.pad(
        vol_01,
        ((0, max(ph - h, 0)), (0, max(pw - w, 0)), (0, max(pd - d, 0))),
        mode='constant',
        constant_values=0.0,
    )
    hp, wp, dp = vol.shape

    sh, sw, sd = max(ph - oh, 1), max(pw - ow, 1), max(pd - od, 1)
    hs = list(range(0, hp - ph + 1, sh))
    ws = list(range(0, wp - pw + 1, sw))
    ds = list(range(0, dp - pd + 1, sd))
    if hs[-1] + ph < hp:
        hs.append(hp - ph)
    if ws[-1] + pw < wp:
        ws.append(wp - pw)
    if ds[-1] + pd < dp:
        ds.append(dp - pd)

    weight = gaussian_weight((ph, pw, pd))
    num = np.zeros_like(vol, dtype=np.float64)
    den = np.zeros_like(vol, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for i in hs:
            for j in ws:
                for k in ds:
                    patch = vol[i:i+ph, j:j+pw, k:k+pd]
                    x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        rec, _, _ = model(x)
                    rec_np = rec.squeeze().float().cpu().numpy()
                    num[i:i+ph, j:j+pw, k:k+pd] += rec_np * weight
                    den[i:i+ph, j:j+pw, k:k+pd] += weight

    recon_01 = (num / np.clip(den, 1e-8, None)).astype(np.float32)[:h, :w, :d]
    return ct_01_to_hu(np.clip(recon_01, 0.0, 1.0))


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        args.amp = False

    model_dir = Path(args.model_dir)
    model_name = model_dir.name

    if args.input_ct:
        ct_path = Path(args.input_ct).resolve()
        if not ct_path.is_file():
            raise FileNotFoundError(f'--input_ct is not a file: {ct_path}')
        suf = ct_path.suffix.lower()
        if suf == '.gz' and ct_path.name.lower().endswith('.nii.gz'):
            pass
        elif suf == '.nii':
            pass
        else:
            raise ValueError(f'--input_ct must be .nii or .nii.gz, got: {ct_path}')
        ct_files = [ct_path]
    else:
        input_ct_dir = Path(args.input_ct_dir)
        ct_files = discover_ct_files(input_ct_dir)
        if not ct_files:
            raise RuntimeError(f'No NIfTI files found in {input_ct_dir}')

    model_type = detect_model_type(model_dir)
    out_dir = Path(args.output_root) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    lpips_model = None
    if lpips is not None:
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    else:
        print('[benchmark] WARNING: lpips not installed, LPIPS will be NaN')

    print(f'[benchmark] model={model_name} type={model_type} cts={len(ct_files)}')

    rows = []

    if model_type == '2d':
        dtype = torch.float16 if (args.amp and device.type == 'cuda') else torch.float32
        vae2d = AutoencoderKL.from_pretrained(str(model_dir), subfolder='vae', torch_dtype=dtype).to(device)
        vae2d.eval()

        for ct_path in tqdm(ct_files, desc=f'2D:{model_name}'):
            nii = nib.load(str(ct_path))
            vol_hu = nii.get_fdata().astype(np.float32)
            recon_hu = reconstruct_2d(vae2d, vol_hu, amp=args.amp)

            out_path = out_dir / f'{stem_nii(ct_path)}_recon.nii.gz'
            nib.save(nib.Nifti1Image(recon_hu, nii.affine), str(out_path))

            metrics = compute_metrics(vol_hu, recon_hu, lpips_model, device, args.lpips_stride)
            rows.append({'model_name': model_name, 'model_type': model_type, 'input_ct': str(ct_path), 'recon_ct': str(out_path), **metrics})

    else:
        vae3d = build_3d_model(
            device=device,
            amp=args.amp,
            num_splits=args.num_splits,
            dim_split=args.dim_split,
            latent_channels=args.latent_channels,
        )
        state = torch.load(str(model_dir / 'autoencoder_best.pt'), map_location=device)
        vae3d.load_state_dict(state)
        vae3d.eval()

        for ct_path in tqdm(ct_files, desc=f'3D:{model_name}'):
            nii = nib.load(str(ct_path))
            vol_hu = nii.get_fdata().astype(np.float32)
            recon_hu = reconstruct_3d(
                vae3d,
                vol_hu,
                patch_size=tuple(args.patch_size),
                overlap=tuple(args.overlap),
                device=device,
                amp=args.amp,
            )

            out_path = out_dir / f'{stem_nii(ct_path)}_recon.nii.gz'
            nib.save(nib.Nifti1Image(recon_hu, nii.affine), str(out_path))

            metrics = compute_metrics(vol_hu, recon_hu, lpips_model, device, args.lpips_stride)
            rows.append({'model_name': model_name, 'model_type': model_type, 'input_ct': str(ct_path), 'recon_ct': str(out_path), **metrics})

    csv_path = out_dir / 'metrics.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'model_name', 'model_type', 'input_ct', 'recon_ct',
                'PSNR', 'SSIM', 'LPIPS', 'Detail_100',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'[benchmark] done: {model_name}')
    print(f'[benchmark] metrics: {csv_path}')


if __name__ == '__main__':
    main()
