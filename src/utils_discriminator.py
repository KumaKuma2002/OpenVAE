import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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


# ========================== StyleGAN-D Discriminator ==========================

from torch.nn.utils import spectral_norm

def sn_conv(in_ch, out_ch, k=4, s=2, p=1):
    return spectral_norm(torch.nn.Conv2d(in_ch, out_ch, k, s, p))


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        B, C, H, W = x.shape
        G = min(self.group_size, B)
        y = x.reshape(G, -1, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y ** 2).mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)  # (B//G, 1, 1, 1)
        y = y.repeat(G, 1, H, W)
        return torch.cat([x, y], dim=1)           # (B, C+1, H, W)


class StyleGANDiscriminatorSingle(torch.nn.Module):
    """One-scale StyleGAN2-style discriminator stream."""
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            sn_conv(in_channels, base_ch,     4, 2, 1), torch.nn.LeakyReLU(0.2, True),
            sn_conv(base_ch,     base_ch * 2, 4, 2, 1), torch.nn.LeakyReLU(0.2, True),
            sn_conv(base_ch * 2, base_ch * 4, 4, 2, 1), torch.nn.LeakyReLU(0.2, True),
            sn_conv(base_ch * 4, base_ch * 8, 4, 2, 1), torch.nn.LeakyReLU(0.2, True),
            sn_conv(base_ch * 8, base_ch * 8, 4, 2, 1), torch.nn.LeakyReLU(0.2, True),
        )
        self.minibatch_std = MinibatchStdLayer(group_size=4)
        self.final = torch.nn.Sequential(
            sn_conv(base_ch * 8 + 1, base_ch * 8, 3, 1, 1), torch.nn.LeakyReLU(0.2, True),
            sn_conv(base_ch * 8,     1,            4, 1, 0),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.minibatch_std(x)
        return self.final(x)


class MultiScaleStyleGANDiscriminator(torch.nn.Module):
    """
    3-scale discriminator. D1=full res, D2=½, D3=¼.
    Drop-in replacement for NLayerDiscriminator.
    Returns a LIST of patch logits — use the loss functions below.
    """
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        self.D1 = StyleGANDiscriminatorSingle(in_channels, base_ch)
        self.D2 = StyleGANDiscriminatorSingle(in_channels, base_ch // 2)
        self.D3 = StyleGANDiscriminatorSingle(in_channels, base_ch // 4)
        self.downsample = torch.nn.AvgPool2d(2, 2)

    def forward(self, x):
        x2 = self.downsample(x)
        x4 = self.downsample(x2)
        return [self.D1(x), self.D2(x2), self.D3(x4)]


# ========================== Multi-scale loss functions ==========================

def discriminator_hinge_loss_multiscale(discriminator, real_img, fake_img):
    """
    Hinge loss across all scales. Replaces discriminator_hinge_loss().
    fake_img must be detached before calling.
    """
    real_preds = discriminator(real_img)
    fake_preds = discriminator(fake_img)

    d_loss = 0.
    for rp, fp in zip(real_preds, fake_preds):
        d_loss += 0.5 * (
            torch.mean(F.relu(1.0 - rp)) +
            torch.mean(F.relu(1.0 + fp))
        )
    return d_loss / len(real_preds)


def generator_adv_loss_multiscale(discriminator, fake_img):
    """
    Generator adversarial loss across all scales. Replaces generator_adv_loss().
    """
    preds = discriminator(fake_img)
    return sum(-torch.mean(p) for p in preds) / len(preds)
