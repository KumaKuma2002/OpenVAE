"""
SMILE-SR Multi-volume Inference (batch version, super-resolution only)

- Input supports:
  (1) dataset_path/
        BDMAP_00012345/ct.nii.gz
        BDMAP_00067890/ct.nii.gz
  (2) dataset_path/
        patient_id.nii.gz
        ...

- Output:
    output_path/
        <case_id>_SR/ct.nii.gz
        <case_id>_SR/*.png

- Prompt is fixed:
    "A high resolution CT slice."

- Multi-GPU (optional):
    If multiple GPUs are detected AND --multiple_gpu is enabled,
    the cases are split into N shards and processed in parallel (one process per GPU).
"""

from diffusers import AutoencoderKL, DDIMScheduler
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from PIL import Image
import nibabel as nib
import os
import cv2
import safetensors
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from types import SimpleNamespace

from dataset import CTDatasetInference, collate_fn_inference
from testEnhanceCTPipeline import init_unet, ConcatInputStableDiffusionPipeline


SR_PROMPT = "A high resolution CT slice."
RESOLUTION = 512


def discover_cases(dataset_dir: str):
    """
    Returns a list of tuples: (case_id, input_ct_path)
    Supports:
      - dataset_dir/case_id/ct.nii.gz
      - dataset_dir/case_id.nii.gz
    """
    cases = []
    for name in sorted(os.listdir(dataset_dir)):
        full = os.path.join(dataset_dir, name)

        # folder case
        if os.path.isdir(full):
            ct_path = os.path.join(full, "ct.nii.gz")
            if os.path.exists(ct_path):
                cases.append((name, ct_path))
            continue

        # file case
        if os.path.isfile(full) and name.endswith(".nii.gz"):
            case_id = name.replace(".nii.gz", "")
            cases.append((case_id, full))

    return cases


def filter_cases_by_csv(cases, guide_csv):
    """
    guide_csv expected column: 'Inference ID'
    Matches case_id exactly.
    """
    guide_df = pd.read_csv(guide_csv)
    guided = set(guide_df["Inference ID"].astype(str).tolist())
    return [(cid, p) for (cid, p) in cases if cid in guided]


def build_pipeline(args, device: str):
    vae = AutoencoderKL.from_pretrained(
        args.finetuned_vae_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
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
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    return pipe


def run_cases_on_device(args, cases, device: str, gpu_id: int = 0):
    os.makedirs(args.output_path, exist_ok=True)

    pipe = build_pipeline(args, device=device)

    inference_transforms = A.Compose([
        A.Resize(RESOLUTION, RESOLUTION, interpolation=cv2.INTER_CUBIC)
    ])
    downsample_factor = int(8 * 512 / RESOLUTION)

    for case_id, input_ct_path in cases:
        save_dir = os.path.join(args.output_path, f"{case_id}_SR")
        out_ct_path = os.path.join(save_dir, "ct.nii.gz")

        if (not args.overwrite) and os.path.exists(out_ct_path):
            print(f"[GPU {gpu_id}] Skip (exists): {case_id}")
            continue

        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[GPU {gpu_id}] Inference: {case_id}")
        print(f"[GPU {gpu_id}] Input:  {input_ct_path}")
        print(f"[GPU {gpu_id}] Output: {save_dir}")
        print(f"[GPU {gpu_id}] Prompt: {SR_PROMPT}")

        ct_dataset = CTDatasetInference(
            file_path=input_ct_path,
            image_transforms=inference_transforms,
            cond_transforms=inference_transforms,
        )
        ct_dataloader = DataLoader(
            ct_dataset,
            shuffle=False,
            collate_fn=collate_fn_inference,
            batch_size=args.chunk_size,
            num_workers=min(args.chunk_size, 16),
            drop_last=False,
        )

        nii_shape = list(ct_dataset.ct_xyz_shape)  # (H, W, D)
        H, W, D = nii_shape

        enhanced_ct = np.zeros(nii_shape, dtype=np.float32)
        weights_ct = np.zeros(nii_shape, dtype=np.float32)

        base_noise = torch.randn(
            (1, 4, H // downsample_factor, W // downsample_factor, D),
            device=device,
            dtype=torch.float16,
        )

        generator = torch.Generator(device=device).manual_seed(42)

        for batch in tqdm(ct_dataloader, desc=f"[GPU {gpu_id}] {case_id}", leave=False):
            cond_image = batch["cond_pixel_values"].to(device).half()  # [B, 3, 512, 512]
            slice_idx = batch["slice_idx"]
            prompt = [SR_PROMPT] * len(cond_image)

            with torch.no_grad():
                cond_latents = pipe.vae.encode(cond_image).latent_dist.sample()
                cond_latents *= pipe.vae.config.scaling_factor

                slice_idx_tensor = torch.tensor(slice_idx, device=device, dtype=torch.float32)
                z_idx = (slice_idx_tensor / D * base_noise.shape[-1]).clamp(0, base_noise.shape[-1] - 1).long()
                latents = base_noise.squeeze(0).permute(3, 0, 1, 2)[z_idx].contiguous()

                if cond_latents.shape != latents.shape:
                    latents = F.interpolate(
                        latents,
                        size=cond_latents.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                images = pipe(
                    num_inference_steps=200,
                    prompt=prompt,
                    latents=latents,
                    cond_latents=cond_latents,
                    output_type="np",
                    generator=generator,
                ).images  # list of (512,512,3) in [0,1]

            for i, img in enumerate(images):
                sid = slice_idx[i]
                if sid + 3 > D:
                    continue
                enhanced_slice = cv2.resize(img, (W, H), cv2.INTER_CUBIC)  # (H,W,3)
                enhanced_ct[:, :, sid:sid + 3] += enhanced_slice
                weights_ct[:, :, sid:sid + 3] += 1

        weights_ct[weights_ct == 0] = 1
        enhanced_ct = enhanced_ct / weights_ct
        enhanced_ct = (enhanced_ct * 2 - 1) * 1000
        enhanced_ct = enhanced_ct.astype(np.int16)

        # visualization
        num_slices = enhanced_ct.shape[2]
        indices = np.linspace(0, num_slices - 1, 8, dtype=int)
        for i, idx in enumerate(indices):
            channel = np.clip(enhanced_ct[:, :, idx], -200, 200)
            norm = ((channel + 200) / 400 * 255).astype(np.uint8)
            norm = np.rot90(norm)
            Image.fromarray(norm).save(os.path.join(save_dir, f"{case_id}_SR_{i:02d}.png"))

        original_nii = nib.load(input_ct_path)
        out_nii = nib.Nifti1Image(enhanced_ct, original_nii.affine, original_nii.header)
        out_nii.header.set_data_dtype(np.int16)
        out_nii.to_filename(out_ct_path)

        print(f"[GPU {gpu_id}] Done: {case_id} -> {out_ct_path}")


def _worker(gpu_id: int, args, cases_shard):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    run_cases_on_device(args, cases_shard, device=device, gpu_id=gpu_id)


def split_list(items, n):
    n = max(1, n)
    return [items[i::n] for i in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMILE-SR Multi-volume Inference (Super-resolution only)")
    parser.add_argument("--input_path", type=str, required=True, help="Dataset folder containing cases")
    parser.add_argument("--output_path", type=str, required=True, help="Folder to save outputs")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--finetuned_vae_name_or_path", type=str, required=True)
    parser.add_argument("--finetuned_unet_name_or_path", type=str, required=True)
    parser.add_argument("--sd_model_name_or_path", type=str, required=True)
    parser.add_argument("--guide_CSV", type=str, default=None, help="Optional CSV with column 'Inference ID'")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--multiple_gpu", action="store_true", help="Enable multi-GPU task splitting")
    args = parser.parse_args()

    print("\n==================== Inference Configuration ====================")
    for k, v in vars(args).items():
        print(f"{k:24s}: {v}")
    print("=================================================================\n")

    dataset_dir = args.input_path
    os.makedirs(args.output_path, exist_ok=True)

    cases = discover_cases(dataset_dir)
    if args.guide_CSV:
        cases = filter_cases_by_csv(cases, args.guide_CSV)
        print(f"[CSV mode] Found {len(cases)} guided cases.")
    else:
        print(f"[Auto mode] Found {len(cases)} cases.")

    if len(cases) == 0:
        print("[Error] No valid cases found. Exiting.")
        raise SystemExit(0)

    num_gpus = torch.cuda.device_count()
    if args.multiple_gpu and num_gpus > 1:
        print(f"[Multi-GPU] Detected {num_gpus} GPUs. Splitting cases across GPUs...")
        shards = split_list(cases, num_gpus)

        mp.set_start_method("spawn", force=True)
        procs = []
        for gpu_id in range(num_gpus):
            if len(shards[gpu_id]) == 0:
                continue
            p = mp.Process(target=_worker, args=(gpu_id, args, shards[gpu_id]))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    else:
        print(f"[Single GPU] Using cuda:0 (detected GPUs: {num_gpus}).")
        run_cases_on_device(args, cases, device="cuda:0", gpu_id=0)

    print("\n🎉 All inference completed successfully.")