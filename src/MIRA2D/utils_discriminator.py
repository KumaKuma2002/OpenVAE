import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from utils import center_crop
import datetime
import os
import json
import lpips


# ========================== PatchGAN Discriminator ==========================

class NLayerDiscriminator(torch.nn.Module):
    """
    Standard PatchGAN (NLayerDiscriminator).
    Widely used in pix2pix / CycleGAN / VQGAN / LDM-style works.
    """

    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()

        kw = 4
        padw = 1
        sequence = [
            torch.nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            torch.nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                torch.nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            torch.nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
            ),
            torch.nn.LeakyReLU(0.2, inplace=True),
        ]

        sequence += [
            torch.nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = torch.nn.Sequential(*sequence)

    def forward(self, x):
        """
        x: (B, 3, H, W), range [0,1]
        returns: patch logits
        """
        return self.model(x)




def discriminator_hinge_loss(discriminator, real_img, fake_img):
    """
    Hinge loss for discriminator (PatchGAN / CNN discriminator).

    Args:
        discriminator: D network
        real_img: real images, shape (B, C, H, W), range [0, 1]
        fake_img: fake images, shape (B, C, H, W), range [0, 1], MUST be detached

    Returns:
        d_loss: scalar tensor
    """

    # predictions
    pred_real = discriminator(real_img)
    pred_fake = discriminator(fake_img)

    # hinge loss
    loss_real = torch.mean(F.relu(1.0 - pred_real))
    loss_fake = torch.mean(F.relu(1.0 + pred_fake))

    d_loss = 0.5 * (loss_real + loss_fake)

    return d_loss

def generator_adv_loss(discriminator, fake_img):
    """
    Generator adversarial loss.
    """
    pred_fake = discriminator(fake_img)
    adv_loss = -torch.mean(pred_fake)
    return adv_loss