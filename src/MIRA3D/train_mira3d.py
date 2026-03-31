"""
MIRA3D: 3D Latent Diffusion Model for CT Super-Resolution.

Port of the MIRA 2D SR pipeline to 3D using MONAI components:
  - Frozen AutoencoderKlMaisi (3D VAE)
  - Trainable DiffusionModelUNet (3D, 8-channel input: noisy HR + LR condition)
  - MONAI DDPMScheduler for forward diffusion
  - Staged auxiliary losses (seg, HU, cycle) matching MIRA 2D warmup logic

Usage:
  # Train from scratch
  python src/MIRA3D/train_mira3d.py \\
      --train_data_dir /data/h5 \\
      --vae_checkpoint outputs/3dvae-patchgan/models/autoencoder_best.pt \\
      --output_dir outputs/mira3d \\
      --num_epochs 200 --amp

  # With nnUNet segmenter for auxiliary losses (seg CE + organ HU)
  python src/MIRA3D/train_mira3d.py \\
      --train_data_dir /data/h5 \\
      --vae_checkpoint outputs/3dvae-patchgan/models/autoencoder_best.pt \\
      --seg_model_path /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres \\
      --output_dir outputs/mira3d \\
      --num_epochs 200 --amp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from accelerate import Accelerator

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.networks.schedulers.ddpm import DDPMScheduler
from monai.networks.schedulers.ddim import DDIMScheduler as DDIMSchedulerMonai
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import nibabel as nib

from dataset import (
    MIRA3DDataset, get_ct_dir_list, load_nifti_volume_01,
)
from utils_loss import (
    load_label_map,
    build_organ_penalties,
    unchanged_region_loss,
    segmentation_loss,
    hu_organ_loss,
)


# ===================================================================
#  1. ARGUMENT PARSING
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIRA3D: 3D LDM for CT super-resolution.")

    # --- Data ---
    p.add_argument("--train_data_dir", type=str, required=True)
    p.add_argument("--val_data_dir", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_nifti_paths", type=str, nargs="+", default=None,
                   help="Single .nii.gz per case: HR from file, LR via synthetic degrade. "
                        "Mutually exclusive with --val_lr_nifti_paths / --val_gt_nifti_paths.")
    p.add_argument("--val_lr_nifti_paths", type=str, nargs="+", default=None,
                   help="Low-res (or input) .nii.gz paths for validation; pair with "
                        "--val_gt_nifti_paths (same length, same voxel grid).")
    p.add_argument("--val_gt_nifti_paths", type=str, nargs="+", default=None,
                   help="Ground-truth HR .nii.gz paths for validation; pair with "
                        "--val_lr_nifti_paths.")
    p.add_argument("--seed", type=int, default=0)

    # --- Output ---
    p.add_argument("--output_dir", type=str, default="mira3d-output")


    # --- Model checkpoints ---
    p.add_argument("--vae_checkpoint", type=str, required=True,
                   help="Path to frozen 3D VAE checkpoint (.pt state_dict).")
    p.add_argument("--seg_model_path", type=str, default=None,
                   help="Path to nnUNet trained model folder "
                        "(e.g. Dataset911/nnUNetTrainer__nnUNetPlans__3d_fullres). "
                        "Loaded via nnUNetPredictor. If not given, seg/HU losses disabled.")
    p.add_argument("--seg_checkpoint_name", type=str, default="checkpoint_best.pth",
                   help="nnUNet checkpoint filename inside --seg_model_path.")
    p.add_argument("--seg_dataset_json", type=str, default=None,
                   help="Path to segmenter dataset.json (nnUNet format). "
                        "Needed for organ-name → label-ID mapping in HU loss. "
                        "If not given, auto-reads from seg_model_path/dataset.json.")
    p.add_argument("--seg_num_classes", type=int, default=None,
                   help="Auto-detected from --seg_dataset_json if not set.")
    p.add_argument("--resume_unet", type=str, default=None,
                   help="Path to UNet weights: either a flat state_dict .pt, or a MAISI / "
                        "NV-Generate-CT bundle (.pt dict with 'unet_state_dict').")

    # --- Training schedule ---
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--val_batch_size", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Accumulate gradients over N micro-batches before stepping. "
                        "Effective batch = train_batch_size * gradient_accumulation_steps.")

    # --- Patch ---
    p.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64])

    # --- Diffusion ---
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="scaled_linear_beta")
    p.add_argument("--prediction_type", type=str, default="epsilon",
                   choices=["epsilon", "v_prediction"])
    p.add_argument("--diffusion_loss", type=str, default="l1",
                   choices=["l1", "l2"])

    # --- UNet architecture ---
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--unet_channels", type=int, nargs="+",
                   default=[64, 128, 256, 512])

    # --- VAE architecture (must match checkpoint) ---
    p.add_argument("--vae_num_splits", type=int, default=1)
    p.add_argument("--vae_dim_split", type=int, default=1)

    # --- Staged loss warmup (by global step) ---
    p.add_argument("--warmup_diffusion_only_steps", type=int, default=2000,
                   help="Stage 1: diffusion loss only.")
    p.add_argument("--warmup_add_unchanged_steps", type=int, default=5000,
                   help="Stage 2: + unchanged-region MSE.")
    p.add_argument("--warmup_add_seg_hu_steps", type=int, default=15000,
                   help="Stage 3: + seg CE + HU organ MSE.")
    # --- Loss weights ---
    p.add_argument("--uc_loss_weight", type=float, default=1e-3)
    p.add_argument("--seg_loss_weight", type=float, default=1e-3)
    p.add_argument("--hu_loss_weight", type=float, default=1e-4)
    p.add_argument("--pixel_loss_weight", type=float, default=0.1,
                   help="Weight for full-patch image-space L1 loss (recon vs HR). "
                        "Activates together with --warmup_add_unchanged_steps.")

    # --- AMP ---
    p.add_argument("--amp", action="store_true")

    # --- Logging / checkpointing ---
    p.add_argument("--val_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--val_vis_samples", type=int, default=4,
                   help="Number of val samples to visualise on wandb.")
    p.add_argument("--val_ddim_steps", type=int, default=50,
                   help="Number of DDIM denoising steps during validation.")
    p.add_argument("--val_overlap_ratio", type=float, default=0.625,
                   help="Overlap ratio for sliding-window SR during validation.")
    p.add_argument("--wandb_project", type=str, default="mira3d")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


# ===================================================================
#  2. HELPERS
# ===================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap_unet_state_dict(raw: object) -> dict:
    """
    Accept either:
      - dict with 'unet_state_dict' (MAISI / NV-Generate-CT / MONAI bundle)
      - dict with 'state_dict'
      - flat state_dict (tensor values only at top level)
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(raw)}")

    if "unet_state_dict" in raw and isinstance(raw["unet_state_dict"], dict):
        return dict(raw["unet_state_dict"])
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return dict(raw["state_dict"])

    tensor_sd = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
    if not tensor_sd:
        raise KeyError(
            "Checkpoint has no 'unet_state_dict', 'state_dict', or tensor weights."
        )
    meta_only = {"epoch", "loss", "num_train_timesteps", "scale_factor",
                 "scheduler_method", "output_size"}
    if set(raw.keys()) <= meta_only:
        raise KeyError("Checkpoint looks like metadata only (no UNet tensors).")
    return tensor_sd


def _strip_module_prefix(state_dict: dict) -> dict:
    prefix = "module."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _merge_conv_in_pretrained(unet: DiffusionModelUNet, sd: dict) -> dict:
    """
    NV/MAISI UNet uses 4 input channels; MIRA3D SR UNet uses 8 (noisy HR || LR).
    Copy pretrained weights into the first 4 input channels; leave the rest at init.
    """
    k_w, k_b = "conv_in.conv.weight", "conv_in.conv.bias"
    if k_w not in sd:
        return sd
    pw = sd[k_w]
    mw = unet.conv_in.conv.weight
    if pw.shape == mw.shape:
        return sd
    if pw.shape[0] != mw.shape[0] or pw.shape[1] >= mw.shape[1]:
        print(
            f"[mira3d] WARNING: conv_in mismatch {tuple(pw.shape)} vs {tuple(mw.shape)}; "
            "not merging pretrained input conv."
        )
        return sd
    with torch.no_grad():
        mw.zero_()
        mw[:, : pw.shape[1]].copy_(pw.to(device=mw.device, dtype=mw.dtype))
        if k_b in sd:
            unet.conv_in.conv.bias.copy_(
                sd[k_b].to(device=mw.device, dtype=unet.conv_in.conv.bias.dtype)
            )
    out = {k: v for k, v in sd.items() if k not in (k_w, k_b)}
    print(
        f"[mira3d] Merged pretrained conv_in: {pw.shape[1]} ch → first {pw.shape[1]} of "
        f"{mw.shape[1]} SR input channels (remaining channels at init)."
    )
    return out


def _filter_state_dict_shapes(
    unet: nn.Module, sd: dict
) -> tuple[dict, list[tuple[str, tuple, tuple]]]:
    """
    Keep only keys that exist on the model and have identical shapes.

    MAISI / NV-Generate-CT use time_embed_dim=1024 (wider time MLP + time_emb_proj).
    MONAI DiffusionModelUNet fixes time_embed_dim = channels[0] * 4 (e.g. 256 for
    channels=(64,...)).  Without this filter, load_state_dict raises on shape mismatch.
    """
    model_sd = unet.state_dict()
    out: dict = {}
    skips: list[tuple[str, tuple, tuple]] = []
    for k, v in sd.items():
        if k not in model_sd:
            continue
        mv = model_sd[k]
        if v.shape != mv.shape:
            skips.append((k, tuple(v.shape), tuple(mv.shape)))
            continue
        out[k] = v
    return out, skips


def load_unet_checkpoint(unet: DiffusionModelUNet, path: str, device: torch.device) -> None:
    """Load UNet from flat state_dict or MAISI/NV-Generate-CT training bundle."""
    # Bundles contain non-tensor metadata → weights_only=False
    raw = torch.load(path, map_location=device, weights_only=False)
    sd = _unwrap_unet_state_dict(raw)
    sd = _strip_module_prefix(sd)
    sd = _merge_conv_in_pretrained(unet, sd)
    sd, shape_skips = _filter_state_dict_shapes(unet, sd)
    if shape_skips:
        print(
            f"[mira3d] Skipped {len(shape_skips)} checkpoint keys with shape mismatch "
            "(often MAISI time_embed_dim=1024 vs MONAI default 256); those layers stay randomly init."
        )
        for key, ckpt_shp, mdl_shp in shape_skips[:12]:
            print(f"    {key}: ckpt{ckpt_shp} vs model{mdl_shp}")
        if len(shape_skips) > 12:
            print(f"    ... and {len(shape_skips) - 12} more")
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    n_miss, n_unexp = len(missing), len(unexpected)
    print(f"[mira3d] Loaded UNet weights from {path}  (missing={n_miss}, unexpected={n_unexp})")
    if n_miss and n_miss <= 20:
        print(f"[mira3d]   missing: {missing}")
    elif n_miss:
        print(f"[mira3d]   missing (first 10): {missing[:10]} ...")
    if n_unexp and n_unexp <= 20:
        print(f"[mira3d]   unexpected: {unexpected}")
    elif n_unexp:
        print(f"[mira3d]   unexpected (first 10): {unexpected[:10]} ...")


def predict_x0_from_noise(
    noisy: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """x_0 = (x_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)"""
    a = alphas_cumprod[timesteps].float()
    while a.dim() < noisy.dim():
        a = a.unsqueeze(-1)
    return (noisy - (1.0 - a).sqrt() * noise_pred) / a.sqrt().clamp(min=1e-8)


def vae_encode_latent(vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Robustly extract latent tensor from AutoencoderKlMaisi.encode output.

    Some checkpoints / MONAI versions return:
      - (z, mu, logvar)
      - (z, stats)
    We only need z.
    """
    out = vae.encode(x)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) >= 1:
        return out[0]
    raise TypeError(f"Unexpected vae.encode output type: {type(out)}")


# ===================================================================
#  3. MODEL BUILDERS
# ===================================================================

def build_vae(args, device) -> AutoencoderKlMaisi:
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
    state = torch.load(args.vae_checkpoint, map_location=device, weights_only=True)
    vae.load_state_dict(state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def build_unet(args, device) -> DiffusionModelUNet:
    ch = tuple(args.unet_channels)
    n = len(ch)
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=args.latent_channels * 2,
        out_channels=args.latent_channels,
        channels=ch,
        attention_levels=tuple([False] * max(0, n - 2) + [True] * min(2, n)),
        num_head_channels=tuple([0] * max(0, n - 2) + [32] * min(2, n)),
        num_res_blocks=2,
        use_flash_attention=True,
        resblock_updown=True,
        include_fc=True,
        norm_num_groups=32,
        norm_eps=1e-6,
    ).to(device)

    if args.resume_unet and os.path.isfile(args.resume_unet):
        load_unet_checkpoint(unet, args.resume_unet, device)

    return unet


def build_segmenter(args, device):
    """
    Load segmenter via nnUNetPredictor (same as MIRA 2D).

    Args:
        args.seg_model_path:  nnUNet trained model folder
        args.seg_checkpoint_name:  checkpoint file inside that folder
    Returns:
        (seg_model, seg_preprocess_fn)  or  (None, None)
        seg_preprocess_fn normalises HU images the way nnUNet training did.
    """
    if args.seg_model_path is None:
        return None, None

    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        print("[mira3d] WARNING: nnunetv2 not installed. Seg/HU losses disabled.")
        return None, None

    try:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        init_ok = False
        last_exc: Exception | None = None
        # First try fold_all, then common single-fold fallback.
        for folds in (("all",), (0,), ("0",)):
            try:
                predictor.initialize_from_trained_model_folder(
                    args.seg_model_path,
                    use_folds=folds,
                    checkpoint_name=args.seg_checkpoint_name,
                )
                print(f"[mira3d] nnUNet init succeeded with use_folds={folds}")
                init_ok = True
                break
            except Exception as exc:  # noqa: PERF203
                last_exc = exc
        if not init_ok:
            raise RuntimeError(
                f"Failed to initialize nnUNet from {args.seg_model_path} "
                f"with checkpoint '{args.seg_checkpoint_name}'. Last error: {last_exc}"
            )
        seg_model = predictor.network.to(device)
        seg_model.eval()
        for p in seg_model.parameters():
            p.requires_grad = False

        # Build HU preprocessing function from nnUNet training stats
        trainset_meta = predictor.plans_manager.plans[
            "foreground_intensity_properties_per_channel"
        ]["0"]
        lo = float(trainset_meta["percentile_00_5"])
        hi = float(trainset_meta["percentile_99_5"])
        mean = float(trainset_meta["mean"])
        std = float(trainset_meta["std"])

        def seg_preprocess(x_hu: torch.Tensor) -> torch.Tensor:
            """Normalise HU tensor the same way nnUNet training data was."""
            x = x_hu.clamp(lo, hi)
            return (x - mean) / std

        # Auto-detect num_classes from the predictor if not set
        if args.seg_num_classes is None:
            args.seg_num_classes = predictor.plans_manager.get_label_manager(
                predictor.dataset_json
            ).num_segmentation_heads

        print(f"[mira3d] Loaded nnUNet segmenter from {args.seg_model_path}")
        print(f"[mira3d]   checkpoint={args.seg_checkpoint_name}  "
              f"classes={args.seg_num_classes}  "
              f"HU clip=[{lo:.0f},{hi:.0f}] mean={mean:.1f} std={std:.1f}")

        # Auto-read dataset.json from nnUNet folder if not explicitly given
        if args.seg_dataset_json is None:
            candidate = os.path.join(args.seg_model_path, "dataset.json")
            if os.path.isfile(candidate):
                args.seg_dataset_json = candidate
                print(f"[mira3d]   Auto-found dataset.json: {candidate}")

        return seg_model, seg_preprocess

    except Exception as exc:
        print(f"[mira3d] WARNING: Could not load nnUNet segmenter: {exc}")
        return None, None


# ===================================================================
#  4. VALIDATION  (full-volume sliding-window SR + wandb)
# ===================================================================

def hu_window_vis(
    image_01: np.ndarray, hu_min: float = -200, hu_max: float = 200,
) -> np.ndarray:
    """Convert [0,1]-normalised CT to HU-windowed [0,1] for visualisation."""
    image_hu = image_01 * 2000.0 - 1000.0
    image_hu = np.clip(image_hu, hu_min, hu_max)
    return ((image_hu - hu_min) / (hu_max - hu_min)).astype(np.float32)


def _gaussian_weight(patch_size, sigma_scale=0.625, min_weight=0.01):
    coords = [np.linspace(-1, 1, s) for s in patch_size]
    grid = np.meshgrid(*coords, indexing="ij")
    w = np.exp(-sum(g ** 2 for g in grid) / (2 * sigma_scale ** 2))
    w = w / w.max()
    return np.clip(w, min_weight, None).astype(np.float32)


@torch.no_grad()
def _sliding_window_sr(
    vol_01: np.ndarray,
    vae, unet,
    ddim_scheduler,
    patch_size: tuple,
    overlap_ratio: float,
    device: torch.device,
    amp: bool,
    num_ddim_steps: int,
) -> np.ndarray:
    """Sliding-window DDIM super-resolution on a full [0,1] volume."""
    H, W, D = vol_01.shape
    ph, pw, pd = patch_size
    oh, ow, od = int(ph * overlap_ratio), int(pw * overlap_ratio), int(pd * overlap_ratio)

    pad_h = max(ph - H, 0)
    pad_w = max(pw - W, 0)
    pad_d = max(pd - D, 0)
    vol_padded = np.pad(vol_01, ((0, pad_h), (0, pad_w), (0, pad_d)), mode="reflect")
    Hp, Wp, Dp = vol_padded.shape

    sh, sw, sd = max(ph - oh, 1), max(pw - ow, 1), max(pd - od, 1)
    starts_h = list(range(0, Hp - ph + 1, sh))
    starts_w = list(range(0, Wp - pw + 1, sw))
    starts_d = list(range(0, Dp - pd + 1, sd))
    if starts_h[-1] + ph < Hp:
        starts_h.append(Hp - ph)
    if starts_w[-1] + pw < Wp:
        starts_w.append(Wp - pw)
    if starts_d[-1] + pd < Dp:
        starts_d.append(Dp - pd)

    gw = _gaussian_weight(patch_size)
    recon_sum = np.zeros_like(vol_padded, dtype=np.float64)
    weight_sum = np.zeros_like(vol_padded, dtype=np.float64)

    ddim_scheduler.set_timesteps(num_ddim_steps)

    for hs in starts_h:
        for ws in starts_w:
            for ds in starts_d:
                patch = vol_padded[hs:hs + ph, ws:ws + pw, ds:ds + pd]
                lr_t = torch.from_numpy(patch.copy()).unsqueeze(0).unsqueeze(0).float().to(device)

                with torch.autocast(device_type=device.type, enabled=amp):
                    z_lr = vae_encode_latent(vae, lr_t)

                patch_seed = int(hs * 1_000_000 + ws * 1_000 + ds) % (2 ** 31)
                patch_rng = torch.Generator(device=device).manual_seed(patch_seed)
                z = torch.randn(z_lr.shape, generator=patch_rng, device=device, dtype=z_lr.dtype)
                for t_val in ddim_scheduler.timesteps:
                    t = torch.full((1,), t_val, device=device, dtype=torch.long)
                    unet_in = torch.cat([z, z_lr], dim=1)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        noise_pred = unet(unet_in, timesteps=t)
                    z, _ = ddim_scheduler.step(noise_pred, t_val, z)

                with torch.autocast(device_type=device.type, enabled=amp):
                    sr_patch = vae.decode(z)
                sr_np = np.clip(sr_patch.squeeze().float().cpu().numpy(), 0.0, 1.0)

                recon_sum[hs:hs + ph, ws:ws + pw, ds:ds + pd] += sr_np * gw
                weight_sum[hs:hs + ph, ws:ws + pw, ds:ds + pd] += gw

    recon = (recon_sum / np.clip(weight_sum, 1e-8, None)).astype(np.float32)
    return np.clip(recon[:H, :W, :D], 0.0, 1.0)


@torch.no_grad()
def validate(
    epoch: int,
    unet, vae, args,
    device: torch.device,
    output_dir: str,
) -> tuple:
    """
    Full-volume validation via sliding-window SR on nifti patients.

    For each patient:
      1. Load full LR volume, run sliding-window DDIM SR
      2. Load full GT volume
      3. Show FULL xy/xz/yz mid-slices as [LR | GT | SR] on wandb
      4. Save <output_dir>/val_epoch<N>/<case>.nii.gz
      5. Compute SSIM / PSNR on full volumes

    Returns: (mean_ssim, mean_psnr)
    """
    unet.eval()
    device_type = device.type

    ddim_scheduler = DDIMSchedulerMonai(
        num_train_timesteps=args.num_train_timesteps,
        schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        clip_sample=False,
    )
    patch_size = tuple(args.patch_size)

    # Determine (lr_path, gt_path) pairs
    pairs: list[tuple[str, str | None]] = []
    if args.val_lr_nifti_paths and args.val_gt_nifti_paths:
        pairs = list(zip(args.val_lr_nifti_paths, args.val_gt_nifti_paths))
    elif args.val_nifti_paths:
        pairs = [(p, p) for p in args.val_nifti_paths]

    if not pairs:
        print("[mira3d]   val: no nifti paths configured, skipping full-volume validation.")
        return 0.0, 0.0

    val_dir = os.path.join(output_dir, f"val_epoch{epoch:04d}")
    os.makedirs(val_dir, exist_ok=True)

    ssim_list, psnr_list = [], []
    mae100_list, mae_hu_list, bias_hu_list, std_ratio_list = [], [], [], []
    wandb_log: dict = {}

    for idx, (lr_path, gt_path) in enumerate(pairs):
        case_name = os.path.basename(os.path.dirname(lr_path)) or f"case_{idx}"
        print(f"[mira3d]   val {idx + 1}/{len(pairs)}: {case_name}")

        lr_vol = load_nifti_volume_01(lr_path)
        gt_vol = load_nifti_volume_01(gt_path) if gt_path else None

        if gt_vol is not None and lr_path == gt_path:
            from dataset import degrade_3d
            gt_vol = lr_vol.copy()
            lr_vol = degrade_3d(lr_vol)

        sr_vol = _sliding_window_sr(
            lr_vol, vae, unet, ddim_scheduler, patch_size,
            args.val_overlap_ratio, device, args.amp, args.val_ddim_steps,
        )

        # ---- save recon .nii.gz ----
        sr_hu = sr_vol * 2000.0 - 1000.0
        nii_out = nib.Nifti1Image(sr_hu.astype(np.float32), affine=np.eye(4))
        lr_nib = nib.load(lr_path)
        if lr_nib.affine is not None:
            nii_out = nib.Nifti1Image(sr_hu.astype(np.float32), affine=lr_nib.affine)
        save_path = os.path.join(val_dir, f"{case_name}_sr.nii.gz")
        nib.save(nii_out, save_path)
        print(f"[mira3d]     saved {save_path}")

        # ---- metrics ----
        if gt_vol is not None:
            ssim_val = structural_similarity(gt_vol, sr_vol, data_range=1.0)
            psnr_val = peak_signal_noise_ratio(gt_vol, sr_vol, data_range=1.0)
            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)

            sr_hu_v = sr_vol * 2000.0 - 1000.0
            gt_hu_v = gt_vol * 2000.0 - 1000.0
            mae_hu_val = float(np.mean(np.abs(sr_hu_v - gt_hu_v)))
            mae100_val = max(0.0, 100.0 * (1.0 - mae_hu_val / 2000.0))
            bias_hu_val = float(np.mean(sr_hu_v - gt_hu_v))
            std_ratio_val = float(np.std(sr_hu_v) / (np.std(gt_hu_v) + 1e-8))
            mae100_list.append(mae100_val)
            mae_hu_list.append(mae_hu_val)
            bias_hu_list.append(bias_hu_val)
            std_ratio_list.append(std_ratio_val)
            print(
                f"[mira3d]     ssim={ssim_val:.4f}  psnr={psnr_val:.2f}  "
                f"mae100={mae100_val:.2f}  mae_hu={mae_hu_val:.1f}  "
                f"bias_hu={bias_hu_val:+.1f}  std_ratio={std_ratio_val:.3f}"
            )

        # ---- wandb images: full orthogonal mid-slices ----
        H, W, D = sr_vol.shape
        views = {
            "xy": (lr_vol[:, :, D // 2], sr_vol[:, :, D // 2],
                   gt_vol[:, :, D // 2] if gt_vol is not None else None),
            "xz": (lr_vol[:, W // 2, :], sr_vol[:, W // 2, :],
                   gt_vol[:, W // 2, :] if gt_vol is not None else None),
            "yz": (lr_vol[H // 2, :, :], sr_vol[H // 2, :, :],
                   gt_vol[H // 2, :, :] if gt_vol is not None else None),
        }
        for view_name, (lr_s, sr_s, gt_s) in views.items():
            key_lr = f"LR ({view_name})"
            key_sr = f"SR ({view_name})"
            key_gt = f"GT ({view_name})"
            wandb_log.setdefault(key_lr, []).append(
                wandb.Image(hu_window_vis(lr_s), caption=case_name))
            wandb_log.setdefault(key_sr, []).append(
                wandb.Image(hu_window_vis(sr_s), caption=case_name))
            if gt_s is not None:
                wandb_log.setdefault(key_gt, []).append(
                    wandb.Image(hu_window_vis(gt_s), caption=case_name))

    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_mae100 = float(np.mean(mae100_list)) if mae100_list else 0.0
    mean_mae_hu = float(np.mean(mae_hu_list)) if mae_hu_list else 0.0
    mean_bias_hu = float(np.mean(bias_hu_list)) if bias_hu_list else 0.0
    mean_std_ratio = float(np.mean(std_ratio_list)) if std_ratio_list else 0.0

    wandb_log["ssim mean"] = mean_ssim
    wandb_log["psnr mean"] = mean_psnr
    wandb_log["mae100 mean"] = mean_mae100
    wandb_log["mae_hu mean"] = mean_mae_hu
    wandb_log["bias_hu mean"] = mean_bias_hu
    wandb_log["std_ratio mean"] = mean_std_ratio
    wandb.log(wandb_log, step=epoch)

    print(
        f"[mira3d]   val epoch={epoch:04d}  "
        f"ssim={mean_ssim:.4f}  psnr={mean_psnr:.2f}  "
        f"mae100={mean_mae100:.2f}  mae_hu={mean_mae_hu:.1f}  "
        f"bias_hu={mean_bias_hu:+.1f}  std_ratio={mean_std_ratio:.3f}"
    )
    return mean_ssim, mean_psnr, mean_mae100


# ===================================================================
#  5. TRAINING STEP
# ===================================================================

def train_one_step(
    batch: dict,
    unet, vae, scheduler,
    diff_loss_fn,
    alphas_cumprod: torch.Tensor,
    segmenter,
    seg_preprocess,
    organ_penalties_by_id: dict | None,
    args: argparse.Namespace,
    global_step: int,
    device: torch.device,
    device_type: str,
) -> dict:
    """Single training step. Returns dict of scalar losses."""
    hr = batch["hr"].to(device, non_blocking=True)    # (B,1,H,W,D) [0,1]
    lr = batch["lr"].to(device, non_blocking=True)
    seg_gt = batch["seg"].to(device, non_blocking=True)  # (B,H,W,D)

    with torch.autocast(device_type=device_type, enabled=args.amp):
        with torch.no_grad():
            z_hr = vae_encode_latent(vae, hr)
            z_lr = vae_encode_latent(vae, lr)

        B = z_hr.shape[0]
        noise = torch.randn_like(z_hr)
        t = torch.randint(0, args.num_train_timesteps, (B,), device=device).long()
        z_noisy = scheduler.add_noise(
            original_samples=z_hr, noise=noise, timesteps=t,
        )

        unet_input = torch.cat([z_noisy, z_lr], dim=1)
        noise_pred = unet(unet_input, timesteps=t)

        # Stage 1: diffusion loss
        diff_loss = diff_loss_fn(noise_pred.float(), noise.float())
        loss = diff_loss

        aux_uc = torch.tensor(0.0, device=device)
        aux_seg = torch.tensor(0.0, device=device)
        aux_hu = torch.tensor(0.0, device=device)
        aux_pixel = torch.tensor(0.0, device=device)

        # Decode to image space when aux losses are active
        need_decode = (
            global_step >= args.warmup_add_unchanged_steps
            or (segmenter is not None
                and global_step >= args.warmup_add_seg_hu_steps)
        )
        if need_decode:
            z_pred = predict_x0_from_noise(z_noisy, noise_pred, t, alphas_cumprod)
            # VAE weights are frozen (requires_grad=False in build_vae); removing
            # no_grad here lets pixel/UC/HU gradients flow back through the frozen
            # decode graph to noise_pred → UNet.
            recon_hr = vae.decode(z_pred)
            recon_hu = recon_hr * 2000.0 - 1000.0
            hr_hu = hr * 2000.0 - 1000.0

            # Stage 2: pixel-level L1 + unchanged-region loss
            if global_step >= args.warmup_add_unchanged_steps:
                aux_pixel = F.l1_loss(recon_hr.float(), hr.float())
                # UC is now MSE in [0,1] space — same scale as pixel L1
                aux_uc = unchanged_region_loss(recon_hr, hr)
                loss = (loss
                        + args.pixel_loss_weight * aux_pixel
                        + args.uc_loss_weight * aux_uc)

            # Stage 3: segmentation + organ HU loss
            if (segmenter is not None
                    and global_step >= args.warmup_add_seg_hu_steps):
                with torch.no_grad():
                    seg_input = seg_preprocess(recon_hu) if seg_preprocess else recon_hr.float()
                    pred_logits = segmenter(seg_input)
                aux_seg = segmentation_loss(pred_logits, seg_gt)
                aux_hu = hu_organ_loss(
                    recon_hr, hr, seg_gt,
                    organ_penalties=organ_penalties_by_id,
                )
                loss = loss + (args.seg_loss_weight * aux_seg
                               + args.hu_loss_weight * aux_hu)

    return {
        "loss": loss,
        "diff": diff_loss.detach(),
        "pixel": aux_pixel.detach(),
        "uc": aux_uc.detach(),
        "seg": aux_seg.detach(),
        "hu": aux_hu.detach(),
    }


# ===================================================================
#  6. MAIN
# ===================================================================

def main() -> None:
    args = parse_args()

    # ---- accelerator (handles DDP, AMP, gradient accumulation) -------
    # mixed_precision="fp16" enables automatic GradScaler + autocast.
    # gradient_accumulation_steps tells accumulate() when to sync grads.
    accelerator = Accelerator(
        mixed_precision="fp16" if args.amp else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device
    device_type = device.type
    if device_type != "cuda":
        args.amp = False

    # Different random seed per rank for data diversity
    seed_everything(args.seed + accelerator.process_index)

    # ---- output dirs + wandb (main process only) ---------------------
    model_dir = os.path.join(args.output_dir, "models")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args), dir=args.output_dir)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"[mira3d] device={device}  amp={args.amp}  "
              f"num_processes={accelerator.num_processes}")

    # ---- data --------------------------------------------------------
    all_dirs = get_ct_dir_list(args.train_data_dir)
    if not all_dirs:
        raise RuntimeError(f"No ct.h5 found under {args.train_data_dir}")

    nifti_single = args.val_nifti_paths is not None
    nifti_pair_lr = args.val_lr_nifti_paths is not None
    nifti_pair_gt = args.val_gt_nifti_paths is not None
    if nifti_single and (nifti_pair_lr or nifti_pair_gt):
        raise ValueError(
            "Use either --val_nifti_paths (single file + synthetic LR) OR "
            "--val_lr_nifti_paths with --val_gt_nifti_paths, not both."
        )
    if nifti_pair_lr or nifti_pair_gt:
        if not (nifti_pair_lr and nifti_pair_gt):
            raise ValueError(
                "Paired nifti validation requires both --val_lr_nifti_paths and "
                "--val_gt_nifti_paths."
            )
        if len(args.val_lr_nifti_paths) != len(args.val_gt_nifti_paths):
            raise ValueError(
                "--val_lr_nifti_paths and --val_gt_nifti_paths must have the same length "
                f"({len(args.val_lr_nifti_paths)} vs {len(args.val_gt_nifti_paths)})."
            )

    if nifti_single or (nifti_pair_lr and nifti_pair_gt):
        train_dirs = all_dirs
    elif args.val_data_dir:
        train_dirs, val_dirs = all_dirs, get_ct_dir_list(args.val_data_dir)
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(all_dirs)).tolist()
        n_val = max(1, int(len(all_dirs) * args.val_ratio))
        val_dirs = [all_dirs[i] for i in perm[:n_val]]
        train_dirs = [all_dirs[i] for i in perm[n_val:]]

    patch_size = tuple(args.patch_size)
    nw = args.dataloader_num_workers
    persistent = nw > 0

    train_dl = DataLoader(
        MIRA3DDataset(train_dirs, patch_size=patch_size, is_train=True),
        batch_size=args.train_batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if nw > 0 else None,
    )

    if nifti_pair_lr and nifti_pair_gt:
        n_val_str = f"{len(args.val_lr_nifti_paths)} LR+GT nifti pairs"
    elif nifti_single:
        n_val_str = f"{len(args.val_nifti_paths)} nifti (synthetic LR)"
    else:
        n_val_str = "none (set --val_nifti_paths or --val_lr/gt_nifti_paths for full-volume val)"

    if accelerator.is_main_process:
        print(f"[mira3d] train={len(train_dirs)} dirs  val={n_val_str}  patch={patch_size}")

    # ---- models ------------------------------------------------------
    vae = build_vae(args, device)
    unet = build_unet(args, device)
    segmenter, seg_preprocess = build_segmenter(args, device)

    # ---- label map & organ penalties ---------------------------------
    organ_penalties_by_id = None
    if args.seg_dataset_json is not None:
        name_to_id, _ = load_label_map(args.seg_dataset_json)
        organ_penalties_by_id = build_organ_penalties(name_to_id)
        if args.seg_num_classes is None:
            args.seg_num_classes = len(name_to_id)
        if accelerator.is_main_process:
            print(f"[mira3d] Label map: {len(name_to_id)} labels, "
                  f"{len(organ_penalties_by_id)} penalised organs")
    elif segmenter is not None:
        if accelerator.is_main_process:
            print("[mira3d] WARNING: segmenter loaded but no --seg_dataset_json; "
                  "HU loss will use uniform weight=1.0")
    if args.seg_num_classes is None:
        args.seg_num_classes = 8

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # ---- optimiser ---------------------------------------------------
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)
    diff_loss_fn = nn.MSELoss() if args.diffusion_loss == "l2" else nn.L1Loss()

    # ---- prepare for distributed training ----------------------------
    # accelerate wraps unet in DDP, patches optimizer with GradScaler when
    # mixed_precision="fp16", and adds DistributedSampler to the dataloader.
    unet, optimizer, train_dl = accelerator.prepare(unet, optimizer, train_dl)

    # ---- dump config (main process only) -----------------------------
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

        n_unet = sum(p.numel() for p in accelerator.unwrap_model(unet).parameters()) / 1e6
        n_vae = sum(p.numel() for p in vae.parameters()) / 1e6
        accum = args.gradient_accumulation_steps
        eff_bs = args.train_batch_size * accum * accelerator.num_processes
        print(f"[mira3d] UNet: {n_unet:.1f}M | VAE: {n_vae:.1f}M (frozen)")
        print(f"[mira3d] Batches/epoch: {len(train_dl)}  "
              f"batch={args.train_batch_size} x accum={accum} "
              f"x {accelerator.num_processes} GPUs = eff_batch={eff_bs}")
        print(f"[mira3d] patch_size={patch_size}  amp={args.amp}")
        print("[mira3d] Starting training...")

    # ---- training loop -----------------------------------------------
    global_step = 0
    best_ssim = 0.0
    best_mae100 = 0.0
    logged_shapes = False

    for epoch in range(args.num_epochs):
        t0 = time.time()
        unet.train()

        acc = {"diff": 0.0, "pixel": 0.0, "uc": 0.0, "seg": 0.0, "hu": 0.0, "total": 0.0}
        n_batches = 0

        for micro_idx, batch in enumerate(
            tqdm(train_dl, desc=f"Epoch {epoch:04d}", dynamic_ncols=True,
                 disable=not accelerator.is_main_process)
        ):
            with accelerator.accumulate(unet):
                # accelerator.accumulate() skips DDP gradient sync on
                # non-boundary steps, and accelerator.backward() handles
                # GradScaler scaling when mixed_precision="fp16".
                out = train_one_step(
                    batch, unet, vae, scheduler, diff_loss_fn,
                    alphas_cumprod, segmenter, seg_preprocess,
                    organ_penalties_by_id,
                    args, global_step, device, device_type,
                )
                accelerator.backward(out["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Shape diagnostic once, main process only
            if not logged_shapes:
                if accelerator.is_main_process:
                    hr = batch["hr"]
                    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=args.amp):
                        z_sample = vae_encode_latent(vae, hr[:1].to(device))
                    print(f"[mira3d] SHAPE DIAGNOSTIC:  "
                          f"hr={tuple(hr.shape)}  z={tuple(z_sample.shape)}  "
                          f"unet_in=({z_sample.shape[0]},{z_sample.shape[1]*2},*{z_sample.shape[2:]})  "
                          f"GPU mem={torch.cuda.max_memory_allocated()/1e9:.1f} GB")
                logged_shapes = True

            acc["diff"] += out["diff"].item()
            acc["pixel"] += out["pixel"].item()
            acc["uc"] += out["uc"].item()
            acc["seg"] += out["seg"].item()
            acc["hu"] += out["hu"].item()
            acc["total"] += out["loss"].item()
            n_batches += 1

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_steps == 0 and accelerator.is_main_process:
                    wandb.log(
                        {f"train/{k}_loss": out[k].item() for k in ("diff", "pixel", "uc", "seg", "hu")}
                        | {"train/total_loss": out["loss"].item()},
                        step=global_step,
                    )

        # ---- epoch summary (main process only) -----------------------
        if accelerator.is_main_process:
            for k in acc:
                acc[k] /= max(n_batches, 1)
            dt = time.time() - t0
            print(
                f"[mira3d] epoch={epoch:04d}  "
                f"total={acc['total']:.5f}  diff={acc['diff']:.5f}  "
                f"pixel={acc['pixel']:.5f}  uc={acc['uc']:.5f}  "
                f"seg={acc['seg']:.5f}  hu={acc['hu']:.5f}  "
                f"step={global_step}  time={dt:.0f}s"
            )
            wandb.log({"train/epoch_total": acc["total"]}, step=global_step)

        # ---- validation (main process only, with sync barriers) ------
        do_val = (epoch % args.val_interval == 0) or (epoch == args.num_epochs - 1)
        has_nifti_val = args.val_nifti_paths or (args.val_lr_nifti_paths and args.val_gt_nifti_paths)
        if do_val and has_nifti_val:
            # Barrier: all ranks finish training before rank 0 starts val
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                raw_unet = accelerator.unwrap_model(unet)
                mean_ssim, mean_psnr, mean_mae100 = validate(
                    epoch, raw_unet, vae, args, device, args.output_dir,
                )
                if mean_ssim > best_ssim:
                    best_ssim = mean_ssim
                if mean_mae100 > best_mae100:
                    best_mae100 = mean_mae100
                    torch.save(raw_unet.state_dict(),
                               os.path.join(model_dir, "unet_best.pt"))
                    with open(os.path.join(model_dir, "best_score_info.txt"), "w") as f:
                        f.write(f"Best MAE-100: {best_mae100:.4f}\n")
                        f.write(f"Best SSIM: {mean_ssim:.6f}\n")
                        f.write(f"Best PSNR: {mean_psnr:.4f}\n")
                        f.write(f"Epoch: {epoch}\n")
                    print(
                        f"[mira3d]   -> New best MAE-100={best_mae100:.2f}  "
                        f"SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.2f}"
                    )
            # Barrier: non-main ranks wait for rank 0 to finish val
            accelerator.wait_for_everyone()

        # ---- checkpoints (main process only) -------------------------
        if accelerator.is_main_process:
            raw_unet = accelerator.unwrap_model(unet)
            if args.save_interval > 0 and epoch % args.save_interval == 0:
                torch.save(raw_unet.state_dict(),
                           os.path.join(model_dir, f"unet_epoch{epoch:04d}.pt"))
            torch.save(raw_unet.state_dict(),
                       os.path.join(model_dir, "unet_latest.pt"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()
        print(f"[mira3d] Done. Best MAE-100={best_mae100:.2f}  Best SSIM={best_ssim:.4f}  Checkpoints: {model_dir}")


if __name__ == "__main__":
    main()
