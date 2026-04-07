"""
3D PatchGAN Discriminator for MIRA3D.

Direct port of the canonical pix2pix/CycleGAN NLayerDiscriminator to 3D,
using nn.Conv3d + torch.nn.utils.spectral_norm for training stability.
Returns a single patch-logit tensor — compatible with standard hinge/adv loss helpers.

Reference: Isola et al., "Image-to-Image Translation with Conditional Adversarial
Networks", CVPR 2017. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class NLayerDiscriminator3D(nn.Module):
    """
    3D PatchGAN discriminator with spectral normalization.

    Architecture (with ndf=32, n_layers=3):
      Conv3d(1→32, k4 s2) → LeakyReLU
      Conv3d(32→64, k4 s2) → LeakyReLU      ← n_layers intermediate blocks
      Conv3d(64→128, k4 s2) → LeakyReLU
      Conv3d(128→256, k4 s1) → LeakyReLU    ← penultimate (stride 1)
      Conv3d(256→1, k4 s1)                   ← patch logits

    On a 64³ input patch this produces a ~6³ logit map.
    All Conv3d layers are wrapped with spectral_norm for stable GAN training.

    Args:
        in_channels: channels of input image (1 for CT).
        ndf: base number of discriminator filters.
        n_layers: number of strided downsampling conv blocks (default 3).
    """

    def __init__(self, in_channels: int = 1, ndf: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        kw, padw = 4, 1

        def sn(m: nn.Module) -> nn.Module:
            return spectral_norm(m)

        # First layer — no norm
        sequence: list[nn.Module] = [
            sn(nn.Conv3d(in_channels, ndf, kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate downsampling blocks
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, ndf * 8)
            sequence += [
                sn(nn.Conv3d(nf_prev, nf, kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Penultimate block — stride 1
        nf_prev = nf
        nf = min(nf * 2, ndf * 8)
        sequence += [
            sn(nn.Conv3d(nf_prev, nf, kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output patch-logit layer
        sequence += [sn(nn.Conv3d(nf, 1, kw, stride=1, padding=padw))]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, D) image tensor in [0, 1]
        Returns:
            (B, 1, h, w, d) patch logits (unnormalised)
        """
        return self.model(x)


# ---------------------------------------------------------------------------
# Loss helpers — kept here so they can be imported alongside the class
# ---------------------------------------------------------------------------

def disc_hinge_loss(
    disc: nn.Module,
    real: torch.Tensor,
    fake_detached: torch.Tensor,
) -> torch.Tensor:
    """
    Hinge loss for the discriminator update step.
    fake_detached must be detached from the generator graph.
    """
    return 0.5 * (
        F.relu(1.0 - disc(real)).mean()
        + F.relu(1.0 + disc(fake_detached)).mean()
    )


def gen_adv_loss(disc: nn.Module, fake: torch.Tensor) -> torch.Tensor:
    """Generator adversarial loss — maximise D score on fake samples."""
    return -disc(fake).mean()
