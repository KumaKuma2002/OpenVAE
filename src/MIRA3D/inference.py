"""
MIRA3D inference: sliding-window 3D super-resolution.

Given a low-resolution CT volume (.nii.gz or .h5), produce a
super-resolved volume using the trained MIRA3D LDM.

Pipeline per patch:
    1. Encode LR patch to latent  z_lr  via frozen VAE
    2. Sample z_noisy ~ N(0, I)
    3. Iteratively denoise with DDIM, conditioning on z_lr via channel-concat
    4. Decode denoised latent to HR patch via frozen VAE
    5. Gaussian-weighted blending across patches

Usage:
    python src/MIRA3D/inference.py \\
        --input /path/to/lr_ct.nii.gz \\
        --vae_checkpoint /path/to/vae.pt \\
        --unet_checkpoint /path/to/unet_best.pt \\
        --output sr_output.nii.gz \\
        --amp
"""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.networks.schedulers.ddim import DDIMScheduler


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIRA3D sliding-window SR inference.")

    p.add_argument("--input", type=str, required=True,
                   help="Input LR CT: .nii/.nii.gz or .h5")
    p.add_argument("--vae_checkpoint", type=str, required=True)
    p.add_argument("--unet_checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default=None)

    p.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--overlap_ratio", type=float, default=0.5)

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--num_train_timesteps", type=int, default=1000)

    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--unet_channels", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--vae_num_splits", type=int, default=1)
    p.add_argument("--vae_dim_split", type=int, default=1)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="Global noise seed for coherent 3D latent noise.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_ct_volume(path: str):
    import h5py
    ext = path.lower()
    if ext.endswith((".nii", ".nii.gz")):
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        return vol, nii.affine
    elif ext.endswith(".h5"):
        with h5py.File(path, "r") as hf:
            vol = hf["image"][...].astype(np.float32)
        return vol, np.eye(4, dtype=np.float64)
    raise ValueError(f"Unsupported: {path}")


def ct_hu_to_01(vol: np.ndarray) -> np.ndarray:
    vol = np.clip(vol, -1000.0, 1000.0)
    return (vol + 1000.0) / 2000.0


def ct_01_to_hu(vol: np.ndarray) -> np.ndarray:
    return vol * 2000.0 - 1000.0


# ---------------------------------------------------------------------------
# Gaussian blending weight (from test_3dvae.py)
# ---------------------------------------------------------------------------

def gaussian_weight(patch_size, sigma_scale=0.625, min_weight=0.01):
    coords = [np.linspace(-1, 1, s) for s in patch_size]
    grid = np.meshgrid(*coords, indexing="ij")
    w = np.exp(-sum(g ** 2 for g in grid) / (2 * sigma_scale ** 2))
    w = w / w.max()
    return np.clip(w, min_weight, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_vae(args, device):
    vae = AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1,
        latent_channels=args.latent_channels,
        num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
        norm_num_groups=32, norm_eps=1e-06,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        use_checkpointing=False, use_convtranspose=False,
        norm_float16=args.amp,
        num_splits=args.vae_num_splits, dim_split=args.vae_dim_split,
    ).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    vae.eval()
    return vae


def build_unet(args, device):
    ch = tuple(args.unet_channels)
    n_levels = len(ch)
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=args.latent_channels * 2,
        out_channels=args.latent_channels,
        channels=ch,
        attention_levels=tuple([False] * max(0, n_levels - 2) + [True] * min(2, n_levels)),
        num_head_channels=tuple([0] * max(0, n_levels - 2) + [32] * min(2, n_levels)),
        num_res_blocks=2, use_flash_attention=True,
        resblock_updown=True, include_fc=True,
        norm_num_groups=32, norm_eps=1e-6,
    ).to(device)
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device, weights_only=True))
    unet.eval()
    return unet


# ---------------------------------------------------------------------------
# DDIM denoising loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    unet, scheduler, z_lr, z_init, device, amp,
    num_inference_steps: int = 50,
):
    """
    DDIM denoising conditioned on z_lr, starting from z_init.

    z_init is a pre-sliced region from the global coherent noise volume
    so that overlapping regions across patches share the same initial noise,
    reducing boundary discontinuities (same principle as MAISI full-volume generation).
    """
    scheduler.set_timesteps(num_inference_steps)
    z = z_init.clone()

    for t_val in scheduler.timesteps:
        t = torch.full((z.shape[0],), t_val, device=device, dtype=torch.long)
        unet_input = torch.cat([z, z_lr], dim=1)

        with torch.autocast(device_type=device.type, enabled=amp):
            noise_pred = unet(unet_input, timesteps=t)

        z, _ = scheduler.step(noise_pred, t_val, z)

    return z


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def sliding_window_sr(
    vol_01: np.ndarray,
    vae, unet, scheduler,
    patch_size: tuple, overlap_ratio: float,
    device: torch.device, amp: bool,
    num_inference_steps: int,
    seed: int = 42,
) -> np.ndarray:
    H, W, D = vol_01.shape
    ph, pw, pd = patch_size
    oh = int(ph * overlap_ratio)
    ow = int(pw * overlap_ratio)
    od = int(pd * overlap_ratio)

    # Use reflect padding so border patches see real anatomy, not zeros.
    pad_h, pad_w, pad_d = max(ph - H, 0), max(pw - W, 0), max(pd - D, 0)
    vol_padded = np.pad(vol_01, ((0, pad_h), (0, pad_w), (0, pad_d)), mode="reflect")
    Hp, Wp, Dp = vol_padded.shape

    sh, sw, sd = max(ph - oh, 1), max(pw - ow, 1), max(pd - od, 1)
    starts_h = list(range(0, Hp - ph + 1, sh))
    starts_w = list(range(0, Wp - pw + 1, sw))
    starts_d = list(range(0, Dp - pd + 1, sd))
    if starts_h[-1] + ph < Hp: starts_h.append(Hp - ph)
    if starts_w[-1] + pw < Wp: starts_w.append(Wp - pw)
    if starts_d[-1] + pd < Dp: starts_d.append(Dp - pd)

    weight = gaussian_weight(patch_size)
    recon_sum = np.zeros_like(vol_padded, dtype=np.float64)
    weight_sum = np.zeros_like(vol_padded, dtype=np.float64)

    # ------------------------------------------------------------------
    # Global coherent noise volume (MAISI-style continuous latent space).
    # Pre-generate one noise tensor covering the whole padded volume at
    # latent resolution. Each patch slices its region from this shared
    # tensor, so overlapping voxels start from the same noise values →
    # no stochastic discontinuity at patch boundaries.
    # VAE factor is detected automatically from the first patch encoding.
    # ------------------------------------------------------------------
    vae_factor: int | None = None
    z_global: torch.Tensor | None = None
    latent_channels: int | None = None

    total = len(starts_h) * len(starts_w) * len(starts_d)
    pbar = tqdm(total=total, desc="SR patches")

    rng = torch.Generator(device=device).manual_seed(seed)

    for hs in starts_h:
        for ws in starts_w:
            for ds in starts_d:
                patch = vol_padded[hs:hs+ph, ws:ws+pw, ds:ds+pd]
                lr_t = (
                    torch.from_numpy(patch.copy())
                    .unsqueeze(0).unsqueeze(0).float().to(device)
                )

                with torch.no_grad():
                    with torch.autocast(device_type=device.type, enabled=amp):
                        enc_out = vae.encode(lr_t)
                        z_lr = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out

                    # Detect VAE spatial downsampling factor from first patch.
                    if vae_factor is None:
                        vae_factor = ph // z_lr.shape[2]
                        latent_channels = z_lr.shape[1]
                        lHp = Hp // vae_factor
                        lWp = Wp // vae_factor
                        lDp = Dp // vae_factor
                        z_global = torch.randn(
                            (1, latent_channels, lHp, lWp, lDp),
                            generator=rng, device=device, dtype=z_lr.dtype,
                        )
                        print(
                            f"[mira3d-infer] VAE factor={vae_factor}  "
                            f"latent volume={tuple(z_global.shape)}"
                        )

                    # Slice the coherent noise region for this patch.
                    lhs = hs // vae_factor
                    lws = ws // vae_factor
                    lds = ds // vae_factor
                    lph = z_lr.shape[2]
                    lpw = z_lr.shape[3]
                    lpd = z_lr.shape[4]
                    z_init = z_global[:, :, lhs:lhs+lph, lws:lws+lpw, lds:lds+lpd]

                    z_sr = ddim_sample(
                        unet, scheduler, z_lr, z_init, device, amp,
                        num_inference_steps,
                    )

                    with torch.autocast(device_type=device.type, enabled=amp):
                        sr_patch = vae.decode(z_sr)

                sr_np = sr_patch.squeeze().float().cpu().numpy()
                sr_np = np.clip(sr_np, 0.0, 1.0)

                recon_sum[hs:hs+ph, ws:ws+pw, ds:ds+pd] += sr_np * weight
                weight_sum[hs:hs+ph, ws:ws+pw, ds:ds+pd] += weight
                pbar.update(1)

    pbar.close()
    recon = (recon_sum / np.clip(weight_sum, 1e-8, None)).astype(np.float32)
    return np.clip(recon[:H, :W, :D], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        args.amp = False

    print(f"[mira3d-infer] device={device}  amp={args.amp}")

    vae = build_vae(args, device)
    unet = build_unet(args, device)

    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        schedule="scaled_linear_beta",
        prediction_type="epsilon",
    )

    vol_hu, affine = load_ct_volume(args.input)
    print(f"[mira3d-infer] Input: {vol_hu.shape}  HU=[{vol_hu.min():.0f}, {vol_hu.max():.0f}]")
    vol_01 = ct_hu_to_01(vol_hu)

    sr_01 = sliding_window_sr(
        vol_01, vae, unet, scheduler,
        tuple(args.patch_size), args.overlap_ratio,
        device, args.amp, args.num_inference_steps,
        seed=args.seed,
    )
    sr_hu = ct_01_to_hu(sr_01)

    if args.output:
        out_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        if stem.endswith(".nii"):
            stem = stem[:-4]
        out_path = os.path.join(os.path.dirname(args.input), f"{stem}_mira3d_sr.nii.gz")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(sr_hu, affine), out_path)
    print(f"[mira3d-infer] Saved: {out_path}")


if __name__ == "__main__":
    main()
