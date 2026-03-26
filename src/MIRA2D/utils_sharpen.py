import numpy as np
import nibabel as nib
from tqdm import tqdm

def inject_fft_noise(target_path, noise_source_path, output_path, noise_gain=0.5, filter_radius=10):
    """
    target_path      : The clean/smooth CT to enhance.
    noise_source_path: The low-quality CT to sample noise/texture from.
    noise_gain       : Strength of noise added (0.1–1.0).
    filter_radius    : Size of low-frequencies to remove (higher = finer noise).
    """
    # 1. Load NIfTI volumes
    tgt_nii = nib.load(target_path)
    src_nii = nib.load(noise_source_path)
    
    tgt_vol = tgt_nii.get_fdata()
    src_vol = src_nii.get_fdata()
    
    final_vol = np.zeros_like(tgt_vol)
    rows, cols = tgt_vol.shape[:2]
    crow, ccol = rows // 2, cols // 2  # Center point

    # 2. Slice-by-slice FFT processing
    for z in tqdm(range(tgt_vol.shape[2])):
        # FFT on source slice to move to frequency domain
        f_transform = np.fft.fft2(src_vol[:, :, z])
        f_shift = np.fft.fftshift(f_transform)

        # 3. High Pass Filter: Block center low-frequencies (structural info)
        f_shift[crow-filter_radius:crow+filter_radius, ccol-filter_radius:ccol+filter_radius] = 0

        # Inverse FFT to get spatial noise map
        noise_map = np.real(np.fft.ifft2(np.fft.ifftshift(f_shift)))

        # 4. Add noise to target
        final_vol[:, :, z] = tgt_vol[:, :, z] + (noise_map * noise_gain)

    # 5. Save
    new_nii = nib.Nifti1Image(final_vol, tgt_nii.affine, tgt_nii.header)
    nib.save(new_nii, output_path)
    print(f"Saved texture-enhanced CT to: {output_path}")

if __name__ == "__main__":
    patient_id = "PDAC_LR"
    inject_fft_noise(
        target_path=f"/projects/bodymaps/jliu452/TRANS/sr/model-checkpoint_best/{patient_id}_SR/ct.nii.gz",       # Your smooth image
        noise_source_path=f"/projects/bodymaps/jliu452/Data/Dataset804_SMILE-SR_Validation/{patient_id}/ct.nii.gz", # The image with the "real" noise
        output_path="./SR-FFT-Enhanced.nii.gz",
        noise_gain=0.2,    # Adjust strength
        filter_radius=20   # Adjust noise granularity
    )