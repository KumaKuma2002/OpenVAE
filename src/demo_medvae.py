import torch
import torch.nn.functional as F
import torchio as tio
import numpy as np
import nibabel as nib
from diffusers import AutoencoderKL
import albumentations as A
import cv2
from torchvision.transforms.functional import gaussian_blur

import argparse
import os

"""
This is the inference code for MedVAE, to reconstruct the CT scan

Serving as the preliminary research for

(1) Super-resolution of CT scans
(2) 3D Volume SMILE diffusion model (3D DiT)


References:
See More, Change Less: Anatomy-Aware Diffusion for Contrast Enhancement
Junqi Liu, Zejun Wu, Pedro R. A. S. Bassi, Xinze Zhou, Wenxuan Li, Ibrahim E. Hamamci, 
Sezgin Er, Tianyu Lin, Yi Luo, Szymon Płotka, Bjoern Menze, Daguang Xu, 
Kai Ding, Kang Wang, Yang Yang, Yucheng Tang, Alan Yuille, Zongwei Zhou★
"""





args = argparse.ArgumentParser()
args.add_argument("--input", type=str, required=True, help="Path to the input CT volume (NIfTI).")
args.add_argument("--checkpoint", type=str, required=True, help="Path to the VAE checkpoint.")


args = args.parse_args()





def ct_hu_to_01(ct_slice):
    # clipping range and normalize
    ct_slice[ct_slice > 1000.] = 1000.    
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.  
    return np.array(ct_slice)


def high_freq_boost(original_hu, recon_hu, sigma=1.0, alpha=0.5):
    """
    original_hu, recon_hu: torch.Tensor (H, W) in HU
    """
    # Gaussian blur = low frequency
    orig_lp = gaussian_blur(
        original_hu.unsqueeze(0).unsqueeze(0),
        kernel_size=5,
        sigma=sigma
    ).squeeze()

    
    high_freq = original_hu - orig_lp
    
    
    recon_boosted = recon_hu + alpha * high_freq.to(recon_hu.device)
    return recon_boosted




validation_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
    A.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        max_pixel_value=1.0,
        p=1.0
    ),
    # HWCarrayToCHWtensor(p=1.),
])



def postprocess_slice_for_view(x):
    """
    x: torch tensor (1,1,H,W) in [-1,1], to HU range
    """
    x = x.clamp(-1, 1)
    x01 = (x + 1) / 2          # [-1,1] -> [0,1]
    hu = x01 * 2000 - 1000     # [0,1] -> [-1000,1000]
    return hu

# -------------------------------
# Main: slice-by-slice VAE recon
# -------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    vae = AutoencoderKL.from_pretrained(
        args.checkpoint,
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device)
    vae.eval()
    print("Loaded VAE")

    # Load CT volume, test on a single volume
    nii = nib.load(args.input)
    affine = nii.affine
    vol = nii.get_fdata().astype(np.float32)        # (H,W,D)

    H, W, D = vol.shape
    print("CT shape:", vol.shape)
    
    
    resolution = 512
    rec = torch.zeros((resolution, resolution, D), dtype=torch.float32)
    # Output volume
    for d in range(D):
        sl = vol[..., d]                      # (H,W)
        sl_norm = ct_hu_to_01(sl)             # -> [0,1]
        # to tensor
        sl_norm = torch.from_numpy(
            validation_transform(image=sl_norm)["image"]
            ).to(vae.device).unsqueeze(0).unsqueeze(0)
        sl_norm3 = sl_norm.repeat(1, 3, 1, 1)  # VAE requires 3 channels

        # Encode-decode
        with torch.no_grad():
            latent = vae.encode(sl_norm3.half()).latent_dist.sample()
            # latent = latent * vae.config.scaling_factor
            sl_rec_norm = vae.decode(latent).sample    # (1,3,H,W) but identical across RGB
            
        # Convert back
        sl_rec_hu = sl_rec_norm[:, 0:1]    # pick channel 0, (1, 1, H, W)
        
        sl_rec_hu = postprocess_slice_for_view(sl_rec_hu)

        
        sl_orig = torch.from_numpy(sl).to(rec.device)
        rec[..., d] = high_freq_boost(
            sl_orig,
            sl_rec_hu,
            sigma=1.0,
            alpha=0.4
        )

        if d % 20 == 0:
            print(f"Reconstructed slice {d}/{D}")

    # ===========================
    # Save reconstructed CT
    # ===========================
    rec_np = rec.cpu().numpy().astype(np.float32)
    save_name = os.path.basename(args.checkpoint)
    nib.save(nib.Nifti1Image(rec_np, affine), f"../outputs/{save_name}_vae_reconstructed.nii.gz")

if __name__ == "__main__":
    main()