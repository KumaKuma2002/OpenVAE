"""
Train a 3D VAE for CT volumes (MedVAE) using a MAISI-style objective.

This script is inspired by the MONAI MAISI VAE tutorial:
  - intensity reconstruction loss (L1/L2)
  - perceptual loss  (3D, "squeeze" network, MAISI fake-3d handling)
  - KL loss for the VAE latent distribution
  - PatchGAN-style adversarial loss using MONAI PatchDiscriminator

Data format (same as train_klvae.py):
  train_data_dir/
    <subject_id>/
      ct.h5          <- HDF5 file, key "image", shape (H, W, D), HU values (default)
    <subject_id>/
      ct.nii.gz      <- NIfTI volume, same HU range (with --use-nifti)
    ...

Usage examples:

  # Train from scratch
  python src/train_3dvae.py \
      --train_data_dir /data/AbdomenAtlasPro \
      --output_dir outputs/3dvae-patchgan \
      --num_epochs 100 --random_aug --amp

  # NIfTI layout: each subject folder contains ct.nii.gz or ct.nii
  python src/train_3dvae.py \
      --train_data_dir /data/AbdomenAtlasPro \
      --output_dir outputs/3dvae-nifti \
      --use-nifti --num_epochs 100 --random_aug --amp

  # Fine-tune from official MAISI pretrained autoencoder
  #   Downloads autoencoder.pt from HuggingFace MONAI/maisi_ct_generative
  #   and initialises the autoencoder only (discriminator trains from scratch).
  python src/train_3dvae.py \
      --train_data_dir /data/AbdomenAtlasPro \
      --output_dir outputs/3dvae-maisi-ft \
      --resume_from_checkpoint maisi \
      --num_epochs 100 --random_aug --amp

  # Resume from a previous run (loads both autoencoder + discriminator)
  python src/train_3dvae.py \
      --train_data_dir /data/AbdomenAtlasPro \
      --output_dir outputs/3dvae-patchgan \
      --resume_from_checkpoint outputs/3dvae-patchgan/models \
      --num_epochs 200 --random_aug --amp

Notes on --resume_from_checkpoint:
  Accepts three formats:
    "maisi"            -> auto-download official MAISI autoencoder (autoencoder only)
    "/path/to/file.pt" -> load a single .pt autoencoder checkpoint (autoencoder only)
    "/path/to/dir/"    -> load autoencoder_latest.pt + discriminator_latest.pt from dir
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
    from monai.losses.adversarial_loss import PatchAdversarialLoss
    from monai.losses.perceptual import PerceptualLoss
    from monai.networks.nets import PatchDiscriminator

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency `monai`. Install it before running this script, e.g.\n"
        "  pip install -U 'monai-weekly[nibabel, tqdm]'\n"
        "Ensure your MONAI version includes monai.apps.generation.maisi networks."
    ) from e

MAISI_AUTOENCODER_URL = (
    "https://huggingface.co/MONAI/maisi_ct_generative/resolve/1.0.1/models/autoencoder.pt"
)


def download_maisi_checkpoint(dest_dir: str) -> str:
    """Download the official MAISI autoencoder if not cached."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "maisi_autoencoder.pt")
    if os.path.isfile(dest):
        print(f"[train_3dvae] MAISI checkpoint cached: {dest}")
        return dest
    print("[train_3dvae] Downloading MAISI autoencoder from HuggingFace ...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="MONAI/maisi_ct_generative",
            filename="models/autoencoder.pt",
            revision="1.0.1",
            local_dir=dest_dir,
        )
        if path != dest:
            os.replace(path, dest)
    except ImportError:
        import urllib.request
        urllib.request.urlretrieve(MAISI_AUTOENCODER_URL, dest)
    print(f"[train_3dvae] Saved to: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D VAE (MAISI-style) on CT volumes.")

    # Data
    parser.add_argument(
        "--train_data_dir", type=str, required=True,
        help="Root dir whose immediate subdirs each contain ct.h5 (default) or ct.nii.gz / ct.nii (--use-nifti).",
    )
    parser.add_argument(
        "--use-nifti", action="store_true",
        help="Load volumes from ct.nii.gz or ct.nii per subject (requires nibabel). Default is ct.h5.",
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Root dir for validation data (same layout). If omitted, a fraction of train data is split off.",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction of training scans to use for validation when --val_data_dir is not given.",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Output
    parser.add_argument("--output_dir", type=str, default="3dvae-output")
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="TensorBoard sub-dir inside --output_dir.")

    # Training schedule
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="DataLoader workers. Set 4-8 for HPC; 0 for debugging.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Loss weights (MAISI tutorial defaults)
    parser.add_argument("--recon_loss", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--perceptual_weight", type=float, default=0.3)
    parser.add_argument("--kl_weight", type=float, default=1e-7)
    parser.add_argument("--adv_weight", type=float, default=0.1)

    # Training flags
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (CUDA only). Recommended.")
    parser.add_argument("--random_aug", action="store_true",
                        help="Enable random 3D spatial + intensity augmentation.")

    # 3-D patch size (H W D)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64],
                        help="Training patch size. MAISI tutorial default is 64 64 64.")
    parser.add_argument("--val_patch_size", type=int, nargs=3, default=None,
                        help="Validation patch size. Defaults to --patch_size.")

    # Network architecture (matches config_maisi3d-rflow.json)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--autoencoder_num_splits", type=int, default=1,
                        help="MAISI tutorial sets num_splits=1 during training.")
    parser.add_argument("--dim_split", type=int, default=1)

    # Discriminator
    parser.add_argument("--disc_channels", type=int, default=32)
    parser.add_argument("--disc_layers_d", type=int, default=3)
    parser.add_argument("--disc_norm", type=str, default="instance",
                        choices=["instance", "batch", "group"])

    # Logging / checkpointing cadence
    parser.add_argument("--val_interval", type=int, default=10,
                        help="Run validation every N epochs.")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save per-epoch checkpoint every N epochs.")
    parser.add_argument("--log_steps", type=int, default=20,
                        help="Print training loss every N steps.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume / fine-tune from checkpoint.  Accepts:\n"
                             "  'maisi'           -> auto-download MAISI pretrained autoencoder\n"
                             "  '/path/to/file.pt' -> single autoencoder .pt file\n"
                             "  '/path/to/dir/'    -> dir with autoencoder_latest.pt + discriminator_latest.pt")

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data discovery  (mirrors train_klvae.py; optional NIfTI per subject)
# ---------------------------------------------------------------------------

def resolve_ct_file_path(subject_dir: str, use_nifti: bool) -> Optional[str]:
    """Absolute path to the CT volume file inside subject_dir, or None if missing."""
    if use_nifti:
        for name in ("ct.nii.gz", "ct.nii"):
            p = os.path.join(subject_dir, name)
            if os.path.isfile(p):
                return p
        return None
    p = os.path.join(subject_dir, "ct.h5")
    return p if os.path.isfile(p) else None


def get_ct_dir_list(root_dir: str, use_nifti: bool = False) -> List[str]:
    """
    Return sorted list of subject **directory** paths that contain a CT volume.

    Default (HDF5): each immediate subdir of root_dir contains ct.h5
    (--use-nifti): each subdir contains ct.nii.gz or ct.nii (prefer .nii.gz)

    Returns paths like ["/data/.../subject_A", "/data/.../subject_B", ...]
    """
    ct_dirs: List[str] = []
    for entry in sorted(os.scandir(root_dir), key=lambda e: e.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        if resolve_ct_file_path(entry.path, use_nifti) is not None:
            ct_dirs.append(entry.path)
    return ct_dirs


# ---------------------------------------------------------------------------
# CT loading  (mirrors load_CT_slice from train_klvae.py, but 3-D)
# ---------------------------------------------------------------------------

def _normalize_hu_volume(vol: np.ndarray) -> np.ndarray:
    """Clip HU to [-1000, 1000] and scale to [0, 1]."""
    vol = np.asarray(vol, dtype=np.float32)
    vol[vol > 1000.0] = 1000.0
    vol[vol < -1000.0] = -1000.0
    return (vol + 1000.0) / 2000.0


def load_CT_volume(ct_path: str) -> np.ndarray:
    """
    Load the full 3-D CT volume from HDF5 (ct.h5) or NIfTI (.nii / .nii.gz).

    Same preprocessing as train_klvae-style loaders:
      - shape (H, W, D), HU values in [-1000, 1000]
      - clips to [-1000, 1000], normalizes to [0, 1]
      - on IO error: returns zeros (H=512, W=512, D=64) as safe fallback

    Returns: float32 ndarray of shape (H, W, D) in [0, 1]
    """
    try:
        if ct_path.endswith(".h5"):
            with h5py.File(ct_path, "r") as hf:
                vol = hf["image"][...]
        elif ct_path.endswith(".nii.gz") or ct_path.endswith(".nii"):
            try:
                import nibabel as nib
            except ImportError as e:
                raise ImportError(
                    "nibabel is required for NIfTI volumes. Install e.g.\n"
                    "  pip install -U nibabel\n"
                    "or  pip install -U 'monai-weekly[nibabel, tqdm]'"
                ) from e
            vol = np.asarray(nib.load(ct_path).get_fdata(), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported CT file extension: {ct_path}")

        return _normalize_hu_volume(vol)

    except Exception as exc:
        print(f"[WARNING] Failed to load CT volume from {ct_path}: {exc}")
        return np.zeros((512, 512, 64), dtype=np.float32)


# ---------------------------------------------------------------------------
# 3-D patch utilities
# ---------------------------------------------------------------------------

def pad_to_min_size(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Zero-pad vol (H,W,D) so every spatial dimension >= target."""
    H, W, D = vol.shape
    ph, pw, pd = target
    pads = [(max(ph - H, 0),), (max(pw - W, 0),), (max(pd - D, 0),)]
    if all(p[0] == 0 for p in pads):
        return vol
    pad_width = tuple(
        (p[0] // 2, p[0] - p[0] // 2) for p in pads
    )
    return np.pad(vol, pad_width, mode="constant", constant_values=0.0)


def random_crop_3d(vol: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Random crop – uses the *global* Python random state (like train_klvae.py)."""
    vol = pad_to_min_size(vol, patch_size)
    H, W, D = vol.shape
    ph, pw, pd = patch_size
    hs = random.randint(0, H - ph)
    ws = random.randint(0, W - pw)
    ds = random.randint(0, D - pd)
    return vol[hs:hs + ph, ws:ws + pw, ds:ds + pd]


def center_crop_3d(vol: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Centre crop (deterministic; used for validation)."""
    vol = pad_to_min_size(vol, patch_size)
    H, W, D = vol.shape
    ph, pw, pd = patch_size
    hs = (H - ph) // 2
    ws = (W - pw) // 2
    ds = (D - pd) // 2
    return vol[hs:hs + ph, ws:ws + pw, ds:ds + pd]


def random_augment_3d(vol: np.ndarray) -> np.ndarray:
    """
    Lightweight MAISI-style augmentation applied to a [0,1] float32 volume:
      - random flips on each axis (p=0.5 each)
      - random 90-degree rotations in a random axis-pair
      - intensity scale + shift (MAISI: RandScaleIntensityd / RandShiftIntensityd)
    """
    for axis in (0, 1, 2):
        if random.random() < 0.5:
            vol = np.flip(vol, axis=axis).copy()

    k = random.randint(0, 3)
    if k > 0:
        axes = random.choice([(0, 1), (0, 2), (1, 2)])
        vol = np.rot90(vol, k=k, axes=axes).copy()

    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        vol = np.clip(vol * scale, 0.0, 1.0)

    if random.random() < 0.3:
        shift = random.uniform(-0.05, 0.05)
        vol = np.clip(vol + shift, 0.0, 1.0)

    return vol


# ---------------------------------------------------------------------------
# Dataset  (single class, is_train flag selects random vs centre crop)
# ---------------------------------------------------------------------------

class CT3DDataset(Dataset):
    """
    3-D patch dataset for CT volumes (HDF5 ct.h5 or NIfTI ct.nii.gz / ct.nii).

    Matches the data convention of train_klvae.py:
      - ct_dir_list: list of **directory** paths (not file paths)
      - resolves ct.h5 vs NIfTI via use_nifti
      - same HU clipping + [0,1] normalisation as load_CT_slice
    """

    def __init__(
        self,
        ct_dir_list: Sequence[str],
        patch_size: Tuple[int, int, int],
        is_train: bool = True,
        random_aug: bool = False,
        use_nifti: bool = False,
    ) -> None:
        self.ct_dir_list = list(ct_dir_list)
        self.patch_size = patch_size
        self.is_train = is_train
        self.random_aug = random_aug
        self.use_nifti = use_nifti

    def __len__(self) -> int:
        return len(self.ct_dir_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ct_dir = self.ct_dir_list[index]
        ct_path = resolve_ct_file_path(ct_dir, self.use_nifti)
        if ct_path is None:
            raise FileNotFoundError(
                f"No CT file found in {ct_dir} (use_nifti={self.use_nifti})"
            )

        vol_01 = load_CT_volume(ct_path)   # (H, W, D) in [0, 1]

        if self.is_train:
            if self.random_aug:
                vol_01 = random_augment_3d(vol_01)
            patch = random_crop_3d(vol_01, self.patch_size)
        else:
            patch = center_crop_3d(vol_01, self.patch_size)

        # (H,W,D) -> (1,H,W,D)  — single-channel CT
        patch_t = torch.from_numpy(patch.copy()).unsqueeze(0).float()
        return {"image": patch_t}


# ---------------------------------------------------------------------------
# Loss helpers (directly from MAISI tutorial / scripts/utils.py)
# ---------------------------------------------------------------------------

def KL_loss(z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
    """
    MAISI KL divergence loss.
    Averaged over batch, summed over latent dimensions.
    """
    eps = 1e-10
    kl = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, z_sigma.dim())),
    )
    return torch.sum(kl) / kl.shape[0]


def weighted_vae_loss(
    recons_loss: torch.Tensor,
    kl: torch.Tensor,
    p_loss: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    return recons_loss + args.kl_weight * kl + args.perceptual_weight * p_loss


def warmup_rule(epoch: int) -> float:
    """LR warmup copied from the MAISI tutorial."""
    if epoch < 10:
        return 0.01
    if epoch < 20:
        return 0.1
    return 1.0


def disc_last(discriminator_out: object) -> torch.Tensor:
    """
    MONAI PatchDiscriminator returns a list of intermediate feature maps.
    The adversarial loss uses the last element, matching MAISI tutorial's `[-1]`.
    """
    if isinstance(discriminator_out, (list, tuple)):
        return discriminator_out[-1]
    if torch.is_tensor(discriminator_out):
        return discriminator_out
    raise TypeError(f"Unexpected PatchDiscriminator output type: {type(discriminator_out)}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(autoencoder, discriminator, model_dir: str, tag: str) -> None:
    torch.save(autoencoder.state_dict(), os.path.join(model_dir, f"autoencoder_{tag}.pt"))
    torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_{tag}.pt"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type  # "cuda" or "cpu"
    if device_type != "cuda":
        args.amp = False        # GradScaler only works on CUDA

    print(f"[train_3dvae] device={device}  amp={args.amp}  use_nifti={args.use_nifti}")

    # ------------------------------------------------------------------
    # Data discovery  (HDF5 or NIfTI per subject folder)
    # ------------------------------------------------------------------
    all_ct_dirs = get_ct_dir_list(args.train_data_dir, args.use_nifti)
    if len(all_ct_dirs) == 0:
        hint = "ct.nii.gz or ct.nii" if args.use_nifti else "ct.h5"
        raise RuntimeError(f"No subject folders with {hint} found under {args.train_data_dir}")

    if args.val_data_dir is not None:
        train_ct_dirs = all_ct_dirs
        val_ct_dirs = get_ct_dir_list(args.val_data_dir, args.use_nifti)
        if len(val_ct_dirs) == 0:
            hint = "ct.nii.gz or ct.nii" if args.use_nifti else "ct.h5"
            raise RuntimeError(f"No subject folders with {hint} found under {args.val_data_dir}")
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(all_ct_dirs)).tolist()
        n_val = max(1, int(len(all_ct_dirs) * args.val_ratio))
        val_ct_dirs   = [all_ct_dirs[i] for i in perm[:n_val]]
        train_ct_dirs = [all_ct_dirs[i] for i in perm[n_val:]]

    patch_size     = tuple(args.patch_size)
    val_patch_size = tuple(args.val_patch_size) if args.val_patch_size else patch_size

    print(f"[train_3dvae] train={len(train_ct_dirs)}  val={len(val_ct_dirs)}")
    print(f"[train_3dvae] patch_size={patch_size}  val_patch_size={val_patch_size}")

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    dataset_train = CT3DDataset(
        train_ct_dirs, patch_size=patch_size,
        is_train=True, random_aug=args.random_aug,
        use_nifti=args.use_nifti,
    )
    dataset_val = CT3DDataset(
        val_ct_dirs, patch_size=val_patch_size,
        is_train=False, random_aug=False,
        use_nifti=args.use_nifti,
    )

    nw = args.dataloader_num_workers
    persistent = nw > 0

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if nw > 0 else None,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device_type == "cuda"),
        drop_last=False,
        persistent_workers=persistent,
        prefetch_factor=2 if nw > 0 else None,
    )

    # ------------------------------------------------------------------
    # Networks  (MAISI AutoencoderKlMaisi + PatchDiscriminator)
    # ------------------------------------------------------------------
    autoencoder = AutoencoderKlMaisi(
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
        norm_float16=args.amp,          # only use float16 norms when AMP is on
        num_splits=args.autoencoder_num_splits,
        dim_split=args.dim_split,
    ).to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=args.disc_layers_d,
        channels=args.disc_channels,
        in_channels=1,
        out_channels=1,
        norm=args.disc_norm,
    ).to(device)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    if args.recon_loss == "l2":
        intensity_loss_fn = nn.MSELoss(reduction="mean")
        print("[train_3dvae] Reconstruction loss: L2 (MSE)")
    else:
        intensity_loss_fn = nn.L1Loss(reduction="mean")
        print("[train_3dvae] Reconstruction loss: L1")

    perceptual_loss_fn = (
        PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
        )
        .eval()
        .to(device)
    )
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    # ------------------------------------------------------------------
    # Optimizers & LR schedulers  (MAISI tutorial: Adam + LambdaLR warmup)
    # ------------------------------------------------------------------
    adam_eps = 1e-06 if args.amp else 1e-08
    optimizer_g = torch.optim.Adam(autoencoder.parameters(),  lr=args.learning_rate, eps=adam_eps)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, eps=adam_eps)

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

    scaler_g = GradScaler(device="cuda", init_scale=2.0 ** 8, growth_factor=1.5) if args.amp else None
    scaler_d = GradScaler(device="cuda", init_scale=2.0 ** 8, growth_factor=1.5) if args.amp else None

    # ------------------------------------------------------------------
    # Resume / fine-tune
    #   "maisi"            -> download & load MAISI pretrained autoencoder
    #   "/path/to/file.pt" -> load single autoencoder state_dict
    #   "/path/to/dir/"    -> load autoencoder_latest.pt + discriminator_latest.pt
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume_from_checkpoint is not None:
        ckpt = args.resume_from_checkpoint

        if ckpt.lower() == "maisi":
            cache_dir = os.path.join(args.output_dir, ".maisi_cache")
            ckpt_path = download_maisi_checkpoint(cache_dir)
            autoencoder.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
            print(f"[train_3dvae] Initialised autoencoder from MAISI pretrained: {ckpt_path}")

        elif os.path.isfile(ckpt) and ckpt.endswith(".pt"):
            autoencoder.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True)
            )
            print(f"[train_3dvae] Loaded autoencoder from: {ckpt}")

        elif os.path.isdir(ckpt):
            g_ckpt = os.path.join(ckpt, "autoencoder_latest.pt")
            d_ckpt = os.path.join(ckpt, "discriminator_latest.pt")
            if os.path.exists(g_ckpt):
                autoencoder.load_state_dict(
                    torch.load(g_ckpt, map_location=device, weights_only=True)
                )
                print(f"[train_3dvae] Resumed autoencoder from {g_ckpt}")
            if os.path.exists(d_ckpt):
                discriminator.load_state_dict(
                    torch.load(d_ckpt, map_location=device, weights_only=True)
                )
                print(f"[train_3dvae] Resumed discriminator from {d_ckpt}")
        else:
            raise FileNotFoundError(
                f"--resume_from_checkpoint='{ckpt}' is not 'maisi', a .pt file, or a directory."
            )

    # ------------------------------------------------------------------
    # Dump config for reproducibility (handy on HPC)
    # ------------------------------------------------------------------
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"[train_3dvae] Config saved to: {config_path}")

    n_params_g = sum(p.numel() for p in autoencoder.parameters()) / 1e6
    n_params_d = sum(p.numel() for p in discriminator.parameters()) / 1e6
    print(f"[train_3dvae] Autoencoder: {n_params_g:.1f}M params | Discriminator: {n_params_d:.1f}M params")
    print(f"[train_3dvae] Batches/epoch: {len(dataloader_train)}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    print("[train_3dvae] Starting training...")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_t0 = time.time()
        autoencoder.train()
        discriminator.train()

        epoch_losses = {
            "recons": 0.0, "kl": 0.0, "percep": 0.0,
            "gen_adv": 0.0, "disc": 0.0,
        }
        n_batches = 0

        pbar = tqdm(dataloader_train, desc=f"Epoch {epoch:04d}", dynamic_ncols=True)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True).contiguous()  # (B,1,H,W,D)

            # Zero both optimizers before the generator step
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            # ---- Generator update ----------------------------------------
            with torch.autocast(device_type=device_type, enabled=args.amp):
                reconstruction, z_mu, z_sigma = autoencoder(images)

                recons_loss = intensity_loss_fn(reconstruction, images)
                kl          = KL_loss(z_mu, z_sigma)
                p_loss      = perceptual_loss_fn(reconstruction.float(), images.float())

                logits_fake  = disc_last(discriminator(reconstruction.contiguous().float()))
                gen_adv_loss = adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

                loss_g = weighted_vae_loss(recons_loss, kl, p_loss, args) + args.adv_weight * gen_adv_loss

            if args.amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.max_grad_norm)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.max_grad_norm)
                optimizer_g.step()

            # ---- Discriminator update  (reconstruction detached) ----------
            optimizer_d.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device_type, enabled=args.amp):
                logits_fake  = disc_last(discriminator(reconstruction.contiguous().detach().float()))
                loss_d_fake  = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real  = disc_last(discriminator(images.contiguous().float()))
                loss_d_real  = adv_loss_fn(logits_real, target_is_real=True,  for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

            if args.amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

            # ---- Bookkeeping ---------------------------------------------
            epoch_losses["recons"]  += float(recons_loss.detach())
            epoch_losses["kl"]      += float(kl.detach())
            epoch_losses["percep"]  += float(p_loss.detach())
            epoch_losses["gen_adv"] += float(gen_adv_loss.detach())
            epoch_losses["disc"]    += float(loss_d.detach())
            n_batches   += 1
            global_step += 1

            if global_step % args.log_steps == 0:
                writer.add_scalar("train/recons_loss",  float(recons_loss.detach()),  global_step)
                writer.add_scalar("train/kl_loss",      float(kl.detach()),           global_step)
                writer.add_scalar("train/percep_loss",  float(p_loss.detach()),       global_step)
                writer.add_scalar("train/gen_adv_loss", float(gen_adv_loss.detach()), global_step)
                writer.add_scalar("train/disc_loss",    float(loss_d.detach()),       global_step)

        scheduler_g.step()
        scheduler_d.step()

        # Epoch averages
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        train_vae_loss = (
            epoch_losses["recons"]
            + args.kl_weight * epoch_losses["kl"]
            + args.perceptual_weight * epoch_losses["percep"]
        )
        epoch_sec = time.time() - epoch_t0
        print(
            f"[train_3dvae] epoch={epoch:04d}  "
            f"vae={train_vae_loss:.5f}  "
            f"recons={epoch_losses['recons']:.5f}  "
            f"kl={epoch_losses['kl']:.3f}  "
            f"percep={epoch_losses['percep']:.5f}  "
            f"gen_adv={epoch_losses['gen_adv']:.5f}  "
            f"disc={epoch_losses['disc']:.5f}  "
            f"lr_g={scheduler_g.get_last_lr()[0]:.2e}  "
            f"time={epoch_sec:.0f}s"
        )
        writer.add_scalar("train/vae_loss_epoch",     train_vae_loss,           epoch)
        writer.add_scalar("train/recons_loss_epoch",  epoch_losses["recons"],   epoch)
        writer.add_scalar("train/kl_loss_epoch",      epoch_losses["kl"],       epoch)
        writer.add_scalar("train/percep_loss_epoch",  epoch_losses["percep"],   epoch)
        writer.add_scalar("train/gen_adv_loss_epoch", epoch_losses["gen_adv"],  epoch)
        writer.add_scalar("train/disc_loss_epoch",    epoch_losses["disc"],     epoch)

        # ---- Validation --------------------------------------------------
        do_val = (epoch % args.val_interval == 0) or (epoch == args.num_epochs - 1)
        if do_val and len(dataset_val) > 0:
            autoencoder.eval()
            val_losses = {"recons": 0.0, "kl": 0.0, "percep": 0.0}
            val_batches = 0

            with torch.no_grad():
                for batch in tqdm(dataloader_val, desc=f"  Val {epoch:04d}", dynamic_ncols=True):
                    images = batch["image"].to(device, non_blocking=True).contiguous()
                    with torch.autocast(device_type=device_type, enabled=args.amp):
                        reconstruction, z_mu, z_sigma = autoencoder(images)
                        val_losses["recons"] += float(intensity_loss_fn(reconstruction, images).detach())
                        val_losses["kl"]     += float(KL_loss(z_mu, z_sigma).detach())
                        val_losses["percep"] += float(
                            perceptual_loss_fn(reconstruction.float(), images.float()).detach()
                        )
                    val_batches += 1

            for k in val_losses:
                val_losses[k] /= max(val_batches, 1)

            val_vae_loss = (
                val_losses["recons"]
                + args.kl_weight * val_losses["kl"]
                + args.perceptual_weight * val_losses["percep"]
            )
            print(
                f"[train_3dvae]   val  epoch={epoch:04d}  "
                f"vae={val_vae_loss:.5f}  "
                f"recons={val_losses['recons']:.5f}  "
                f"kl={val_losses['kl']:.3f}  "
                f"percep={val_losses['percep']:.5f}"
            )
            writer.add_scalar("val/vae_loss",    val_vae_loss,          epoch)
            writer.add_scalar("val/recons_loss", val_losses["recons"],  epoch)
            writer.add_scalar("val/kl_loss",     val_losses["kl"],      epoch)
            writer.add_scalar("val/percep_loss", val_losses["percep"],  epoch)

            if val_vae_loss < best_val_loss:
                best_val_loss = val_vae_loss
                save_checkpoint(autoencoder, discriminator, model_dir, "best")
                print(f"[train_3dvae]   -> New best val loss={best_val_loss:.5f}. Checkpoint saved.")

        # ---- Periodic checkpoint -----------------------------------------
        if args.save_interval > 0 and (epoch % args.save_interval == 0):
            save_checkpoint(autoencoder, discriminator, model_dir, f"epoch{epoch:04d}")

        # Always keep a "latest" so resume is easy
        save_checkpoint(autoencoder, discriminator, model_dir, "latest")

    writer.close()
    print("[train_3dvae] Training finished.")
    print(f"[train_3dvae] Best val loss: {best_val_loss:.5f}")
    print(f"[train_3dvae] Checkpoints in: {model_dir}")


if __name__ == "__main__":
    main()
