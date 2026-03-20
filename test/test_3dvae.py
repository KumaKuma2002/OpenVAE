"""
Inference script for the 3D VAE (AutoencoderKlMaisi) trained by train_3dvae.py.

Reconstructs a full CT volume using sliding-window patch-based inference
with Gaussian-weighted blending for seamless stitching.

Supports both NIfTI (.nii / .nii.gz) and HDF5 (.h5) inputs.

Usage:
    python test/test_3dvae.py \
        --input /path/to/ct.nii.gz \
        --checkpoint outputs/3dvae-patchgan/models/autoencoder_best.pt \
        --output reconstructed.nii.gz

    python test/test_3dvae.py \
        --input /path/to/subject_dir/ct.h5 \
        --checkpoint outputs/3dvae-patchgan/models/autoencoder_best.pt \
        --patch_size 64 64 64 \
        --overlap 16 16 16 \
        --amp
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

try:
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency `monai`. Install with:\n"
        "  pip install -U 'monai-weekly[nibabel, tqdm]'\n"
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D VAE inference on CT volumes.")

    p.add_argument("--input", type=str, required=True,
                   help="Path to input CT: .nii/.nii.gz or .h5 (key='image').")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to autoencoder .pt checkpoint (state_dict).")
    p.add_argument("--output", type=str, default=None,
                   help="Path for the output NIfTI. Defaults to <input_stem>_3dvae_recon.nii.gz.")

    p.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64],
                   help="Inference patch size (H W D). Should match training patch size.")
    p.add_argument("--overlap", type=int, nargs=3, default=[16, 16, 16],
                   help="Overlap between adjacent patches for blending (H W D).")

    # Architecture params – must match the training config
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--num_splits", type=int, default=1,
                   help="num_splits for MAISI convolution splitting. Must match training "
                        "(default=1). Higher values reduce VRAM but require larger patches.")
    p.add_argument("--dim_split", type=int, default=1)

    p.add_argument("--amp", action="store_true", help="Use float16 autocast.")
    p.add_argument("--device", type=str, default=None,
                   help="Force device (e.g. 'cuda:0', 'cpu'). Auto-detected by default.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_ct_volume(path: str):
    """
    Load a CT volume from NIfTI or HDF5.

    Returns:
        vol   – float32 ndarray (H, W, D) in raw HU
        affine – 4x4 ndarray (identity if h5)
    """
    ext = path.lower()
    if ext.endswith((".nii", ".nii.gz")):
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        return vol, nii.affine
    elif ext.endswith(".h5"):
        with h5py.File(path, "r") as hf:
            vol = hf["image"][...].astype(np.float32)
        return vol, np.eye(4, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported input format: {path}")


def ct_hu_to_01(vol: np.ndarray) -> np.ndarray:
    """Clip HU to [-1000, 1000] and normalise to [0, 1]. Matches training."""
    vol = vol.copy()
    vol[vol > 1000.0] = 1000.0
    vol[vol < -1000.0] = -1000.0
    vol = (vol + 1000.0) / 2000.0
    return vol


def ct_01_to_hu(vol: np.ndarray) -> np.ndarray:
    """Inverse of ct_hu_to_01: [0, 1] -> [-1000, 1000]."""
    return vol * 2000.0 - 1000.0


# ---------------------------------------------------------------------------
# Gaussian blending weight
# ---------------------------------------------------------------------------

def _gaussian_weight(patch_size, sigma_scale=0.125):
    """
    3-D Gaussian weighting kernel for overlap blending.
    Peaks at the centre so boundary artefacts are down-weighted.
    """
    coords = [np.linspace(-1, 1, s) for s in patch_size]
    grid = np.meshgrid(*coords, indexing="ij")
    sigma = sigma_scale * 2
    w = np.exp(-sum(g ** 2 for g in grid) / (2 * sigma ** 2))
    w = w / w.max()
    w = np.clip(w, 1e-6, None)
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Build model (must mirror train_3dvae.py exactly)
# ---------------------------------------------------------------------------

def build_autoencoder(args, device) -> AutoencoderKlMaisi:
    model = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=args.latent_channels,
        num_channels=[64, 128, 256],
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=args.amp,
        num_splits=args.num_splits,
        dim_split=args.dim_split,
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def sliding_window_inference(
    vol_01: np.ndarray,
    autoencoder: AutoencoderKlMaisi,
    patch_size: tuple,
    overlap: tuple,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    """
    Run the 3D VAE over the full volume with overlapping patches and
    Gaussian-weighted blending.

    Args:
        vol_01: (H, W, D) float32 in [0, 1]

    Returns:
        recon_01: (H, W, D) float32 in [0, 1]
    """
    H, W, D = vol_01.shape
    ph, pw, pd = patch_size
    oh, ow, od = overlap

    # Pad volume so every dimension is at least patch_size
    pad_h = max(ph - H, 0)
    pad_w = max(pw - W, 0)
    pad_d = max(pd - D, 0)
    vol_padded = np.pad(
        vol_01,
        ((0, pad_h), (0, pad_w), (0, pad_d)),
        mode="constant",
        constant_values=0.0,
    )
    Hp, Wp, Dp = vol_padded.shape

    # Step sizes
    sh = max(ph - oh, 1)
    sw = max(pw - ow, 1)
    sd = max(pd - od, 1)

    # Compute patch origins
    starts_h = list(range(0, Hp - ph + 1, sh))
    starts_w = list(range(0, Wp - pw + 1, sw))
    starts_d = list(range(0, Dp - pd + 1, sd))

    # Ensure the last patch covers the volume tail
    if starts_h[-1] + ph < Hp:
        starts_h.append(Hp - ph)
    if starts_w[-1] + pw < Wp:
        starts_w.append(Wp - pw)
    if starts_d[-1] + pd < Dp:
        starts_d.append(Dp - pd)

    weight = _gaussian_weight(patch_size)
    weight_t = torch.from_numpy(weight)

    recon_sum = np.zeros_like(vol_padded, dtype=np.float64)
    weight_sum = np.zeros_like(vol_padded, dtype=np.float64)

    total_patches = len(starts_h) * len(starts_w) * len(starts_d)
    pbar = tqdm(total=total_patches, desc="Inference patches")

    device_type = device.type
    autoencoder.eval()

    with torch.no_grad():
        for hs in starts_h:
            for ws in starts_w:
                for ds in starts_d:
                    patch = vol_padded[hs:hs+ph, ws:ws+pw, ds:ds+pd]
                    patch_t = (
                        torch.from_numpy(patch.copy())
                        .unsqueeze(0).unsqueeze(0)   # (1, 1, H, W, D)
                        .float()
                        .to(device)
                    )

                    with torch.autocast(device_type=device_type, enabled=amp):
                        rec, _, _ = autoencoder(patch_t)

                    rec_np = rec.squeeze().float().cpu().numpy()   # (ph, pw, pd)

                    recon_sum[hs:hs+ph, ws:ws+pw, ds:ds+pd] += rec_np * weight
                    weight_sum[hs:hs+ph, ws:ws+pw, ds:ds+pd] += weight

                    pbar.update(1)

    pbar.close()

    recon_blended = (recon_sum / np.clip(weight_sum, 1e-8, None)).astype(np.float32)

    # Remove padding
    recon_blended = recon_blended[:H, :W, :D]
    return np.clip(recon_blended, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    if device_type != "cuda":
        args.amp = False

    print(f"[test_3dvae] device={device}  amp={args.amp}")

    # ---- Load model -------------------------------------------------------
    autoencoder = build_autoencoder(args, device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    print(f"[test_3dvae] Loaded checkpoint: {args.checkpoint}")

    # ---- Load CT volume ---------------------------------------------------
    vol_hu, affine = load_ct_volume(args.input)
    print(f"[test_3dvae] Input shape: {vol_hu.shape}  HU range: [{vol_hu.min():.1f}, {vol_hu.max():.1f}]")

    vol_01 = ct_hu_to_01(vol_hu)

    # ---- Run inference ----------------------------------------------------
    patch_size = tuple(args.patch_size)
    overlap = tuple(args.overlap)
    print(f"[test_3dvae] patch_size={patch_size}  overlap={overlap}")

    recon_01 = sliding_window_inference(
        vol_01, autoencoder, patch_size, overlap, device, args.amp,
    )

    # ---- Convert back to HU and save -------------------------------------
    recon_hu = ct_01_to_hu(recon_01)
    print(f"[test_3dvae] Recon HU range: [{recon_hu.min():.1f}, {recon_hu.max():.1f}]")

    if args.output is not None:
        out_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        if stem.endswith(".nii"):
            stem = stem[:-4]
        out_path = os.path.join(os.path.dirname(args.input), f"{stem}_3dvae_recon.nii.gz")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(recon_hu, affine), out_path)
    print(f"[test_3dvae] Saved: {out_path}")


if __name__ == "__main__":
    main()
