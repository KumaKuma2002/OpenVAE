"""
Auxiliary loss functions for MIRA3D super-resolution LDM training.

Ported from MIRA-STEP3-SRModel/utils_loss.py (2D) to 3D volumes.
All image tensors are expected in HU domain [-1000, 1000] unless noted.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
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
# Unchanged-region loss  (air / bone should stay unchanged after SR)
# ---------------------------------------------------------------------------

def unchanged_region_loss(
    recon_hu: torch.Tensor,
    original_hu: torch.Tensor,
    hu_threshold: float = 800.0,
) -> torch.Tensor:
    """
    MSE on voxels that are clearly air (< -threshold) in both images.
    Forces the model not to hallucinate content in air regions.
    """
    air_mask = (original_hu < -hu_threshold).float()
    if air_mask.sum() < 1:
        return torch.tensor(0.0, device=recon_hu.device)
    diff = (recon_hu - original_hu) * air_mask
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


# ---------------------------------------------------------------------------
# Cycle consistency loss
# ---------------------------------------------------------------------------

def cycle_consistency_loss(
    vae,
    unet,
    scheduler,
    estimated_z: torch.Tensor,
    cond_z: torch.Tensor,
    original_lr_01: torch.Tensor,
    device: torch.device,
    amp: bool = False,
) -> torch.Tensor:
    """
    Cycle loss: encode estimated HR → add noise → denoise conditioned on LR
    → decode → compare with original LR in [0,1].

    This enforces structural consistency: SR(degrade(x)) should recover x.
    """
    B = estimated_z.shape[0]
    noise = torch.randn_like(estimated_z)
    t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device)
    noisy_est = scheduler.add_noise(original_model_output=estimated_z, noise=noise, timesteps=t)

    unet_input = torch.cat([noisy_est, cond_z], dim=1)
    with torch.autocast(device_type=device.type, enabled=amp):
        cycle_noise_pred = unet(unet_input, timesteps=t)

    cycle_z = estimated_z - noise + cycle_noise_pred
    with torch.no_grad():
        cycle_recon = vae.decode(cycle_z)

    cycle_recon_01 = ((cycle_recon.clamp(-1000, 1000) + 1000.0) / 2000.0)
    return F.mse_loss(cycle_recon_01, original_lr_01)


# ---------------------------------------------------------------------------
# Uncertainty-weighted multi-task loss (from Kendall et al. 2018)
# ---------------------------------------------------------------------------

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.log_vars.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total
