"""
3D patch dataset for MIRA3D super-resolution training.

Each sample returns an (HR, LR) pair of 3D patches in [0, 1].
LR is synthesised on-the-fly by `degrade_3d` (downsample-upsample + noise).

Data layout (identical to train_3dvae.py):
    train_data_dir/
        <subject_id>/ct.h5    <- key "image", shape (H, W, D), HU values
        <subject_id>/gt.h5    <- key "image", shape (H, W, D), integer seg labels  (optional)
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence, Tuple

import h5py
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Data discovery (same as train_3dvae.py)
# ---------------------------------------------------------------------------

def get_ct_dir_list(root_dir: str) -> List[str]:
    top_entries = [e.path for e in os.scandir(root_dir)]
    ct_dirs = sorted(
        e.path.replace("ct.h5", "")
        for path in top_entries
        for e in os.scandir(path)
        if e.name == "ct.h5"
    )
    return ct_dirs


# ---------------------------------------------------------------------------
# CT I/O
# ---------------------------------------------------------------------------

def load_ct_volume(ct_path: str) -> np.ndarray:
    """Load CT volume, clip to [-1000, 1000], normalise to [0, 1]."""
    try:
        with h5py.File(ct_path, "r") as hf:
            vol = hf["image"][...].astype(np.float32)
        vol = np.clip(vol, -1000.0, 1000.0)
        vol = (vol + 1000.0) / 2000.0
        return vol
    except Exception as exc:
        print(f"[WARNING] Failed to load {ct_path}: {exc}")
        return np.zeros((512, 512, 64), dtype=np.float32)


def load_seg_volume(gt_path: str) -> np.ndarray | None:
    """Load segmentation mask from gt.h5 if it exists."""
    if not os.path.isfile(gt_path):
        return None
    try:
        with h5py.File(gt_path, "r") as hf:
            seg = hf["image"][...].astype(np.int16)
        return seg
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3D patch helpers
# ---------------------------------------------------------------------------

def pad_to_min_size(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    H, W, D = vol.shape[:3]
    ph, pw, pd = target
    pads = [max(ph - H, 0), max(pw - W, 0), max(pd - D, 0)]
    if all(p == 0 for p in pads):
        return vol
    pad_width = [(p // 2, p - p // 2) for p in pads]
    if vol.ndim > 3:
        pad_width += [(0, 0)] * (vol.ndim - 3)
    return np.pad(vol, pad_width, mode="constant", constant_values=0)


def random_crop_3d(vol: np.ndarray, patch_size: Tuple[int, int, int],
                   *extra_vols: np.ndarray):
    """Random crop a 3D patch; applies same crop to any extra volumes."""
    vol = pad_to_min_size(vol, patch_size)
    extras = [pad_to_min_size(v, patch_size) for v in extra_vols]
    H, W, D = vol.shape[:3]
    ph, pw, pd = patch_size
    hs = random.randint(0, H - ph)
    ws = random.randint(0, W - pw)
    ds = random.randint(0, D - pd)
    slc = (slice(hs, hs + ph), slice(ws, ws + pw), slice(ds, ds + pd))
    cropped = vol[slc]
    cropped_extras = [v[slc] for v in extras]
    if cropped_extras:
        return (cropped, *cropped_extras)
    return cropped


# ---------------------------------------------------------------------------
# 3D degradation (port of MIRA 2D degrade_ct)
# ---------------------------------------------------------------------------

def degrade_3d(
    patch: np.ndarray,
    scale_range: Tuple[float, float] = (0.3, 0.8),
    gaussian_std_range: Tuple[float, float] = (0.0, 0.08),
    poisson_scale: float = 1e5,
) -> np.ndarray:
    """
    Synthesise a low-resolution version of a [0, 1] 3D patch.

    Steps (mirrors MIRA 2D):
        1. Downsample by a random factor, then upsample back (blur)
        2. Poisson noise (signal-dependent)
        3. Gaussian noise (additive)
    """
    patch = patch.astype(np.float32)
    H, W, D = patch.shape

    scale = random.uniform(*scale_range)
    tH, tW, tD = max(4, int(H * scale)), max(4, int(W * scale)), max(4, int(D * scale))

    zoom_down = (tH / H, tW / W, tD / D)
    low = zoom(patch, zoom_down, order=1)
    zoom_up = (H / low.shape[0], W / low.shape[1], D / low.shape[2])
    out = zoom(low, zoom_up, order=1)

    # ensure shape match after rounding
    out = out[:H, :W, :D]

    lam = np.clip(out, 1e-6, 1.0) * poisson_scale
    out = np.random.poisson(lam).astype(np.float32) / poisson_scale

    sigma = random.uniform(*gaussian_std_range)
    out += np.random.normal(0, sigma, out.shape).astype(np.float32)

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def random_augment_pair(hr: np.ndarray, lr: np.ndarray,
                        seg: np.ndarray | None = None):
    """Apply identical random flips/rotations to HR, LR, and optional seg."""
    for axis in (0, 1, 2):
        if random.random() < 0.5:
            hr = np.flip(hr, axis=axis).copy()
            lr = np.flip(lr, axis=axis).copy()
            if seg is not None:
                seg = np.flip(seg, axis=axis).copy()

    k = random.randint(0, 3)
    if k > 0:
        axes = random.choice([(0, 1), (0, 2), (1, 2)])
        hr = np.rot90(hr, k=k, axes=axes).copy()
        lr = np.rot90(lr, k=k, axes=axes).copy()
        if seg is not None:
            seg = np.rot90(seg, k=k, axes=axes).copy()

    if seg is not None:
        return hr, lr, seg
    return hr, lr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIRA3DDataset(Dataset):
    """
    3D SR training dataset.

    Returns dict:
        hr:   (1, H, W, D) float32 in [0, 1]
        lr:   (1, H, W, D) float32 in [0, 1]
        seg:  (H, W, D) int16  (if gt.h5 exists, else zeros)
    """

    def __init__(
        self,
        ct_dir_list: Sequence[str],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        is_train: bool = True,
        augment: bool = True,
        degrade_kwargs: dict | None = None,
    ) -> None:
        self.ct_dir_list = list(ct_dir_list)
        self.patch_size = patch_size
        self.is_train = is_train
        self.augment = augment and is_train
        self.degrade_kwargs = degrade_kwargs or {}

    def __len__(self) -> int:
        return len(self.ct_dir_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ct_dir = self.ct_dir_list[index]
        ct_path = os.path.join(ct_dir, "ct.h5")
        gt_path = os.path.join(ct_dir, "gt.h5")

        vol_01 = load_ct_volume(ct_path)
        seg = load_seg_volume(gt_path)

        if seg is not None:
            hr, seg_crop = random_crop_3d(vol_01, self.patch_size, seg)
        else:
            hr = random_crop_3d(vol_01, self.patch_size)
            seg_crop = np.zeros(self.patch_size, dtype=np.int16)

        lr = degrade_3d(hr, **self.degrade_kwargs)

        if self.augment:
            hr, lr, seg_crop = random_augment_pair(hr, lr, seg_crop)

        hr_t = torch.from_numpy(hr.copy()).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr.copy()).unsqueeze(0).float()
        seg_t = torch.from_numpy(seg_crop.copy()).long()

        return {"hr": hr_t, "lr": lr_t, "seg": seg_t}


# ---------------------------------------------------------------------------
# Validation dataset from specific .nii.gz files (like MIRA 2D --validation_images)
# ---------------------------------------------------------------------------

def load_nifti_volume_01(path: str) -> np.ndarray:
    """Load .nii.gz, clip HU to [-1000, 1000], return float32 in [0, 1]."""
    data = nib.load(path).get_fdata().astype(np.float32)
    data = np.clip(data, -1000.0, 1000.0)
    return (data + 1000.0) / 2000.0


def _centre_crop_3d(vol: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Deterministic centre crop for validation."""
    vol = pad_to_min_size(vol, patch_size)
    H, W, D = vol.shape[:3]
    ph, pw, pd = patch_size
    hs = (H - ph) // 2
    ws = (W - pw) // 2
    ds = (D - pd) // 2
    return vol[hs:hs + ph, ws:ws + pw, ds:ds + pd]


class NiftiValDataset(Dataset):
    """
    Load specific .nii.gz volumes for validation.

    Each sample returns a centre-cropped (HR, LR) pair.
    HR = centre crop of the loaded volume (normalised to [0,1]).
    LR = degraded version of that crop.
    """

    def __init__(
        self,
        nifti_paths: Sequence[str],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        degrade_kwargs: dict | None = None,
    ) -> None:
        self.nifti_paths = list(nifti_paths)
        self.patch_size = patch_size
        self.degrade_kwargs = degrade_kwargs or {}

    def __len__(self) -> int:
        return len(self.nifti_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        vol_01 = load_nifti_volume_01(self.nifti_paths[index])

        hr = _centre_crop_3d(vol_01, self.patch_size)
        lr = degrade_3d(hr, **self.degrade_kwargs)

        hr_t = torch.from_numpy(hr.copy()).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr.copy()).unsqueeze(0).float()
        seg_t = torch.zeros(self.patch_size, dtype=torch.long)

        return {"hr": hr_t, "lr": lr_t, "seg": seg_t}


class NiftiValPairDataset(Dataset):
    """
    Validation from separate LR and GT .nii.gz files (paired, same order).

    LR and GT volumes must share the same (H, W, D) so the same centre crop
    aligns spatially. Resample to a common grid if needed. LR is used as
    loaded (no synthetic degrade).
    """

    def __init__(
        self,
        lr_paths: Sequence[str],
        gt_paths: Sequence[str],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
    ) -> None:
        self.lr_paths = list(lr_paths)
        self.gt_paths = list(gt_paths)
        if len(self.lr_paths) != len(self.gt_paths):
            raise ValueError(
                "lr_paths and gt_paths must have the same length "
                f"({len(self.lr_paths)} vs {len(self.gt_paths)})."
            )
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        lr_vol = load_nifti_volume_01(self.lr_paths[index])
        gt_vol = load_nifti_volume_01(self.gt_paths[index])
        if lr_vol.shape != gt_vol.shape:
            raise ValueError(
                f"LR shape {lr_vol.shape} != GT shape {gt_vol.shape} for pair:\n"
                f"  LR: {self.lr_paths[index]}\n"
                f"  GT: {self.gt_paths[index]}\n"
                "Resample both to the same voxel grid before validation."
            )
        hr = _centre_crop_3d(gt_vol, self.patch_size)
        lr = _centre_crop_3d(lr_vol, self.patch_size)

        hr_t = torch.from_numpy(hr.copy()).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr.copy()).unsqueeze(0).float()
        seg_t = torch.zeros(self.patch_size, dtype=torch.long)

        return {"hr": hr_t, "lr": lr_t, "seg": seg_t}


class NiftiValPairDataset(Dataset):
    """
    Validation from separate LR and GT .nii.gz files (paired, same order).

    LR and GT volumes must share the same (H, W, D) so the same centre crop
    aligns spatially. Resample to a common grid if needed. LR is used as
    loaded (no synthetic degrade).
    """

    def __init__(
        self,
        lr_paths: Sequence[str],
        gt_paths: Sequence[str],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
    ) -> None:
        self.lr_paths = list(lr_paths)
        self.gt_paths = list(gt_paths)
        if len(self.lr_paths) != len(self.gt_paths):
            raise ValueError(
                "lr_paths and gt_paths must have the same length "
                f"({len(self.lr_paths)} vs {len(self.gt_paths)})."
            )
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        lr_vol = load_nifti_volume_01(self.lr_paths[index])
        gt_vol = load_nifti_volume_01(self.gt_paths[index])
        if lr_vol.shape != gt_vol.shape:
            raise ValueError(
                f"LR shape {lr_vol.shape} != GT shape {gt_vol.shape} for pair:\n"
                f"  LR: {self.lr_paths[index]}\n"
                f"  GT: {self.gt_paths[index]}\n"
                "Resample both to the same voxel grid before validation."
            )
        hr = _centre_crop_3d(gt_vol, self.patch_size)
        lr = _centre_crop_3d(lr_vol, self.patch_size)

        hr_t = torch.from_numpy(hr.copy()).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr.copy()).unsqueeze(0).float()
        seg_t = torch.zeros(self.patch_size, dtype=torch.long)

        return {"hr": hr_t, "lr": lr_t, "seg": seg_t}
