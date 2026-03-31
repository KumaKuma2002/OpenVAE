"""
Auxiliary loss functions for MIRA3D super-resolution LDM training.

Ported from MIRA2D/utils_loss.py (2D) to 3D volumes.
Image tensors are in [0, 1] normalized space unless noted (HU = x*2000-1000).
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Label-map loading  (from segmenter dataset.json)
# ---------------------------------------------------------------------------

def load_label_map(dataset_json_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Parse a nnUNet / MONAI-style dataset.json.

    Returns:
        name_to_id: {"background": 0, "liver": 12, ...}
        id_to_name: {0: "background", 12: "liver", ...}
    """
    with open(dataset_json_path, "r") as f:
        data = json.load(f)
    name_to_id: Dict[str, int] = data["labels"]
    id_to_name: Dict[int, str] = {v: k for k, v in name_to_id.items()}
    return name_to_id, id_to_name


# ---------------------------------------------------------------------------
# Organ penalties  (keyed by NAME — dataset-agnostic)
# ---------------------------------------------------------------------------

DEFAULT_ORGAN_PENALTIES_BY_NAME: Dict[str, float] = {
    "aorta": 100.0,
    "kidney_left": 100.0,
    "kidney_right": 100.0,
    "liver": 100.0,
    "pancreas": 100.0,
    "stomach": 10.0,
    "spleen": 10.0,
    "postcava": 50.0,
    "pancreatic_lesion": 500.0,
    "pancreatic_pdac": 500.0,
    "pancreatic_cyst": 300.0,
    "pancreatic_pnet": 300.0,
}


def build_organ_penalties(
    name_to_id: Dict[str, int],
    name_penalties: Dict[str, float] | None = None,
    default_weight: float = 1.0,
) -> Dict[int, float]:
    """
    Convert name-keyed penalties to integer-ID-keyed penalties using the
    segmenter's label map.  This makes `hu_organ_loss` agnostic to the
    specific segmenter mapping — swap the dataset.json and it just works.

    Args:
        name_to_id:     {"background": 0, "liver": 12, ...} from dataset.json
        name_penalties:  {"liver": 100.0, ...}  (defaults to built-in table)
        default_weight:  weight for organs not listed in name_penalties
    Returns:
        {12: 100.0, 13: 100.0, ...}  ready for hu_organ_loss()
    """
    penalties = name_penalties or DEFAULT_ORGAN_PENALTIES_BY_NAME
    id_penalties: Dict[int, float] = {}
    for name, label_id in name_to_id.items():
        if name == "background":
            continue
        id_penalties[label_id] = penalties.get(name, default_weight)
    return id_penalties


# ---------------------------------------------------------------------------
# Unchanged-region loss  (air regions should stay unchanged after SR)
# ---------------------------------------------------------------------------

def unchanged_region_loss(
    recon_01: torch.Tensor,
    hr_01: torch.Tensor,
    hu_threshold: float = 800.0,
) -> torch.Tensor:
    """
    MSE in [0,1] space on voxels that are clearly air in the HR reference.

    Air is defined as HU < -hu_threshold, derived from hr_01 internally.
    recon_01 is clamped to [0,1] before computing so diff^2 is strictly
    bounded in [0,1] and the loss scale matches pixel L1 (~O(0.001-0.05)).
    Clamp is differentiable (grad=1 inside [0,1], 0 outside).
    """
    hr_hu = hr_01 * 2000.0 - 1000.0
    air_mask = (hr_hu < -hu_threshold).float()
    if air_mask.sum() < 1:
        return torch.tensor(0.0, device=recon_01.device)
    recon_clamped = recon_01.clamp(0.0, 1.0)
    diff = (recon_clamped - hr_01) * air_mask
    return (diff ** 2).sum() / air_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Organ-wise HU MSE loss
# ---------------------------------------------------------------------------

def _organ_mean_hu(x: torch.Tensor, mask: torch.Tensor, organ_id: int) -> torch.Tensor:
    organ_mask = (mask == organ_id).float()
    area = organ_mask.sum().clamp(min=1.0)
    return (x * organ_mask).sum() / area


def hu_organ_loss(
    recon_01: torch.Tensor,
    target_01: torch.Tensor,
    seg_mask: torch.Tensor,
    organ_penalties: Dict[int, float] | None = None,
) -> torch.Tensor:
    """
    Organ-wise average-HU MSE.

    Args:
        recon_01, target_01: (B, 1, H, W, D) in [0, 1]
        seg_mask: (B, H, W, D) integer labels
        organ_penalties: {label_id: weight} built by build_organ_penalties().
                         If None, all non-background organs get weight 1.0.
    """
    device = recon_01.device
    B = recon_01.shape[0]

    batch_losses = []
    for i in range(B):
        r = recon_01[i, 0]
        t = target_01[i, 0]
        m = seg_mask[i]

        ids = torch.unique(m)
        ids = ids[ids != 0]
        if len(ids) < 2:
            continue

        organ_losses = []
        for oid in ids:
            oid_int = oid.item()
            w = organ_penalties.get(oid_int, 1.0) if organ_penalties else 1.0
            r_hu = _organ_mean_hu(r, m, oid_int)
            t_hu = _organ_mean_hu(t, m, oid_int)
            organ_losses.append(w * (r_hu - t_hu) ** 2)

        if organ_losses:
            batch_losses.append(torch.mean(torch.stack(organ_losses)))

    if batch_losses:
        return torch.mean(torch.stack(batch_losses))
    return torch.tensor(0.0, device=device)


# ---------------------------------------------------------------------------
# Segmentation CE loss
# ---------------------------------------------------------------------------

def segmentation_loss(
    pred_logits: torch.Tensor,
    gt_seg: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy between segmenter predictions on the reconstructed image
    and the ground-truth segmentation mask.

    Args:
        pred_logits: (B, C, H, W, D)  raw logits from frozen segmenter
        gt_seg:      (B, H, W, D)     integer labels
    """
    ce = F.cross_entropy(pred_logits, gt_seg.long(), ignore_index=0, reduction="mean")
    if torch.isnan(ce) or torch.isinf(ce):
        return torch.tensor(0.0, device=pred_logits.device)
    return ce

