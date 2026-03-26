# =========================
# Imports
# =========================
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import is_torch_xla_available
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import argparse
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from PIL import Image
import albumentations as A
from types import SimpleNamespace
import safetensors

from dataset import CTDatasetInference, collate_fn_inference
from testEnhanceCTPipeline import init_unet, ConcatInputStableDiffusionPipeline

# =========================
# Constants
# =========================
RESOLUTION = 512
SR_PROMPT = "A high resolution CT slice."

DEVICE = "cuda"

# =========================
# Utilities
# =========================
def load_ct_slice_from_nifti(ct_data, slice_idx, hu_clip=(-200, 200)):
    """
    Load 3-slice axial stack → normalized [0,1], shape (H, W, 3)
    """
    ct_slice = ct_data.dataobj[:, :, slice_idx:slice_idx + 3].copy()
    ct_slice = np.clip(ct_slice, -1000, 1000)
    ct_slice = (ct_slice + 1000.0) / 2000.0
    return ct_slice


def save_png(slice_hu, out_path, hu_clip=(-200, 200)):
    slice_hu = np.clip(slice_hu, hu_clip[0], hu_clip[1])
    norm = ((slice_hu - hu_clip[0]) / (hu_clip[1] - hu_clip[0]) * 255).astype(np.uint8)
    norm = np.rot90(norm)
    Image.fromarray(norm).save(out_path)


# =========================
# Single-slice inference
# =========================
def run_single_slice_inference(
    pipe,
    ct_path,
    output_dir,
    slice_location=0.5,
    resolution=512,
    num_inference_steps=200,
    hu_clip=(-200, 200),
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    ct_volume = nib.load(ct_path)
    H, W, D = ct_volume.shape
    slice_idx = int(D * slice_location)

    cond_slice = load_ct_slice_from_nifti(ct_volume, slice_idx)
    cond_slice = cv2.resize(cond_slice, (resolution, resolution), cv2.INTER_CUBIC)

    norm = A.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        max_pixel_value=1.0
    )
    cond_slice = norm(image=cond_slice)["image"]
    cond_slice = torch.from_numpy(cond_slice).permute(2, 0, 1)[None].half().to(DEVICE)

    with torch.no_grad():
        cond_latents = pipe.vae.encode(cond_slice).latent_dist.sample()
        cond_latents *= pipe.vae.config.scaling_factor
        latents = torch.randn_like(cond_latents)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        image = pipe(
            prompt=[SR_PROMPT],
            latents=latents,
            cond_latents=cond_latents,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="np"
        ).images[0]

    enhanced = (image * 2 - 1) * 1000
    enhanced = enhanced[:, :, 0]

    save_png(
        enhanced,
        os.path.join(output_dir, f"slice_{slice_idx:03d}.png"),
        hu_clip
    )

    np.savez(
        os.path.join(output_dir, f"slice_{slice_idx:03d}.npz"),
        enhanced.astype(np.float32)
    )

    return enhanced


# =========================
# Main
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--finetuned_vae_name_or_path", type=str, required=True)
    parser.add_argument("--finetuned_unet_name_or_path", type=str, required=True)
    parser.add_argument("--sd_model_name_or_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--slice_location", type=float, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # =========================
    # Load models
    # =========================
    vae = AutoencoderKL.from_pretrained(
        args.finetuned_vae_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16
    )

    unet_args = SimpleNamespace(pretrained_model_name_or_path=args.sd_model_name_or_path)
    unet = init_unet(unet_args.pretrained_model_name_or_path, zero_cond_conv_in=True)
    unet_ckpt = safetensors.torch.load_file(
        os.path.join(args.finetuned_unet_name_or_path, "unet", "diffusion_pytorch_model.safetensors")
    )
    unet.load_state_dict(unet_ckpt, strict=True)
    unet = unet.half()

    pipe = ConcatInputStableDiffusionPipeline.from_pretrained(
        args.sd_model_name_or_path,
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    patient_id = args.input_path.split("/")[-2]
    save_dir = f"{args.output_path}/{patient_id}_SR"
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # Single-slice mode
    # =========================
    if args.slice_location is not None:
        run_single_slice_inference(
            pipe=pipe,
            ct_path=args.input_path,
            output_dir=save_dir,
            slice_location=args.slice_location,
        )
        exit(0)

    # =========================
    # Full-volume inference
    # =========================
    dataset = CTDatasetInference(
        file_path=args.input_path,
        image_transforms=A.Resize(RESOLUTION, RESOLUTION, interpolation=cv2.INTER_CUBIC),
        cond_transforms=A.Resize(RESOLUTION, RESOLUTION, interpolation=cv2.INTER_CUBIC),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.chunk_size,
        shuffle=False,
        collate_fn=collate_fn_inference,
        num_workers=min(16, args.chunk_size),
    )

    H, W, D = dataset.ct_xyz_shape
    enhanced_ct = np.zeros((H, W, D), dtype=np.float32)
    weights_ct = np.zeros((H, W, D), dtype=np.float32)

    downsample_factor = int(8 * 512 / RESOLUTION)
    base_noise = torch.randn(
        (1, 4, H // downsample_factor, W // downsample_factor, D),
        device=DEVICE,
        dtype=torch.float16
    )

    for batch in tqdm(dataloader):
        cond_image = batch["cond_pixel_values"].half().to(DEVICE)
        slice_idx = batch["slice_idx"]

        with torch.no_grad():
            cond_latents = pipe.vae.encode(cond_image).latent_dist.sample()
            cond_latents *= pipe.vae.config.scaling_factor

            z_idx = (torch.tensor(slice_idx, device=DEVICE) / D * base_noise.shape[-1]).long()
            latents = base_noise.squeeze(0).permute(3, 0, 1, 2)[z_idx]

            if cond_latents.shape != latents.shape:
                latents = F.interpolate(latents, size=cond_latents.shape[-2:], mode="bilinear")

            images = pipe(
                prompt=[SR_PROMPT] * len(cond_image),
                latents=latents,
                cond_latents=cond_latents,
                num_inference_steps=200,
                output_type="np",
            ).images

        for i, sid in enumerate(slice_idx):
            if sid + 3 > D:
                continue
            slice_img = cv2.resize(images[i], (W, H), cv2.INTER_CUBIC)
            enhanced_ct[:, :, sid:sid + 3] += slice_img
            weights_ct[:, :, sid:sid + 3] += 1

    weights_ct[weights_ct == 0] = 1
    enhanced_ct = enhanced_ct / weights_ct
    enhanced_ct = (enhanced_ct * 2 - 1) * 1000
    enhanced_ct = enhanced_ct.astype(np.int16)

    out_nii = nib.Nifti1Image(enhanced_ct, nib.load(args.input_path).affine)
    out_nii.header.set_data_dtype(np.int16)
    out_nii.to_filename(os.path.join(save_dir, "ct.nii.gz"))