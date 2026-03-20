"""
2D VAE inference for CT reconstruction (slice-by-slice).

Reference: src/demo_medvae.py
"""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D VAE inference on CT volume (slice-by-slice).")
    parser.add_argument("--input", type=str, required=True, help="Input CT NIfTI path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="2D diffusers model directory.")
    parser.add_argument("--output", type=str, default=None, help="Output NIfTI path.")
    parser.add_argument("--amp", action="store_true", help="Use float16 on CUDA.")
    parser.add_argument("--device", type=str, default=None, help="Device like cuda:0 or cpu.")
    return parser.parse_args()


def ct_hu_to_01(ct_slice: np.ndarray) -> np.ndarray:
    ct_slice = np.clip(ct_slice, -1000.0, 1000.0)
    return ((ct_slice + 1000.0) / 2000.0).astype(np.float32)


def postprocess_slice_for_hu(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1, 1)
    x01 = (x + 1.0) / 2.0
    return x01 * 2000.0 - 1000.0


def reconstruct_2d_volume(vae: AutoencoderKL, vol_hu: np.ndarray, amp: bool) -> np.ndarray:
    h, w, d = vol_hu.shape
    rec = np.zeros((h, w, d), dtype=np.float32)
    device_type = vae.device.type

    for i in tqdm(range(d), desc="2D slices"):
        sl_hu = vol_hu[..., i]
        sl_01 = ct_hu_to_01(sl_hu)

        sl_m11 = sl_01 * 2.0 - 1.0
        x = torch.from_numpy(sl_m11).unsqueeze(0).unsqueeze(0).to(vae.device)
        x3 = x.repeat(1, 3, 1, 1)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=amp):
                latent = vae.encode(x3).latent_dist.sample()
                out = vae.decode(latent).sample

        out_hu = postprocess_slice_for_hu(out[:, 0:1]).squeeze().float().cpu().numpy()
        rec[..., i] = out_hu

    return rec


def main() -> None:
    args = parse_args()
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type != "cuda":
        args.amp = False

    dtype = torch.float16 if (args.amp and device.type == "cuda") else torch.float32

    vae = AutoencoderKL.from_pretrained(
        args.checkpoint,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()

    nii = nib.load(args.input)
    vol_hu = nii.get_fdata().astype(np.float32)

    rec_hu = reconstruct_2d_volume(vae, vol_hu, amp=args.amp)

    if args.output is None:
        stem = os.path.basename(args.input)
        if stem.endswith('.nii.gz'):
            stem = stem[:-7]
        else:
            stem = os.path.splitext(stem)[0]
        args.output = os.path.join(os.path.dirname(args.input), f"{stem}_2dvae_recon.nii.gz")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    nib.save(nib.Nifti1Image(rec_hu.astype(np.float32), nii.affine), args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
