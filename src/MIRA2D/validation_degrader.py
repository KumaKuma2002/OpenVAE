import os
import argparse
import numpy as np
import nibabel as nib
import cv2
from utils_degrade import degrade_sparse_view

from tqdm import tqdm



# --- CONFIGURATION ---
HU_MIN = -1000.0
HU_MAX = 1000.0

def degrade_slice(img, scale_factor, sigma, poisson_scale=0, n_views=100):
    """
    Applies degradation using FIXED parameters (scale/sigma)
    but unique random seeds for noise.
    img: 2D array, normalized [0,1]
    """
    H, W = img.shape
    
    # 1. Downsampling (Blur)
    # Simulate lower resolution detector
    low_w = int(W // scale_factor)
    low_h = int(H // scale_factor)
    
    low = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(low, (W, H), interpolation=cv2.INTER_LINEAR)

    # 2. Poisson Noise (Shot Noise - Signal Dependent)
    # approximations for normalized data
    if poisson_scale > 0:
        # distinct random seed is automatic in numpy unless set
        lam = np.clip(out, 1e-6, 1.0) * poisson_scale
        out = np.random.poisson(lam).astype(np.float32) / poisson_scale

    # 3. Gaussian Noise (Electronic Noise - Uniform)
    if sigma > 0:
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out += noise

    # 4. 2D-projections downgrade
    out = degrade_sparse_view(out, n_views)

    return np.clip(out, 0.0, 1.0)


def process_volume(patient_id, input_root, output_root):
    # Paths
    in_dir = os.path.join(input_root, patient_id)
    out_dir = os.path.join(output_root, f"{patient_id}_LR") # LR = Low Res
    os.makedirs(out_dir, exist_ok=True)
    
    in_path = f"{in_dir}.nii.gz"
    out_path = os.path.join(out_dir, "ct.nii.gz")

    if not os.path.exists(in_path):
        print(f"[SKIP] Not found: {in_path}")
        return

    # Load & Normalize
    nii = nib.load(in_path)
    data = nii.get_fdata().astype(np.float32)
    
    # Clip to medical window (Critical for CT)
    data = np.clip(data, HU_MIN, HU_MAX)
    # Normalize to [0, 1]
    data = (data - HU_MIN) / (HU_MAX - HU_MIN)

    # --- GENERATE PARAMETERS ONCE PER VOLUME ---
    # This ensures consistency across the Z-axis
    vol_scale = np.random.uniform(2.0, 4.0)       # e.g., 2.5x downsample
    vol_sigma = np.random.uniform(0.005, 0.03)    # e.g., 0.01 noise level
    vol_nviews = np.random.randint(80, 200)        # e.g., 100 sparse view to reconstruct
    
    print(f"Processing {patient_id} | Scale: {vol_scale:.2f} | Sigma: {vol_sigma:.4f} | Sparse Views: {vol_nviews}")

    degraded = np.zeros_like(data)

    # Apply to every slice
    for z in tqdm(range(data.shape[2])):
        degraded[:, :, z] = degrade_slice(
            data[:, :, z],
            scale_factor=vol_scale,
            sigma=vol_sigma,
            poisson_scale=3e4,
            n_views=vol_nviews,
        )

    # Restore to HU (Optional, usually keep 0-1 for training)
    degraded = degraded * (HU_MAX - HU_MIN) + HU_MIN

    # Save
    out_nii = nib.Nifti1Image(degraded, nii.affine, nii.header)
    nib.save(out_nii, out_path)
    print(f"[DONE] Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", type=str, required=True)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    process_volume(args.patient_id, args.input_root, args.output_root)