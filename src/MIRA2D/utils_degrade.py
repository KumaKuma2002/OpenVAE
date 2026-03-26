import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, resize
from tqdm import tqdm

# ------------------------------------------
# Sparse-view degradation (fixed views)
# ------------------------------------------
def degrade_sparse_view(img, n_views):
    theta = np.linspace(0., 180., n_views, endpoint=False)
    H, W = img.shape

    sinogram = radon(img, theta=theta, circle=False)
    recon = iradon(sinogram, theta=theta, circle=False, output_size=max(H, W))

    if recon.shape != (H, W):
        recon = resize(recon, (H, W), preserve_range=True)

    return np.clip(recon, 0, 1)


# ------------------------------------------
# Main processing
# ------------------------------------------
def process_and_visualize(nii_path, n_views=40, save_path="sparse_view_result.png"):

    nii = nib.load(nii_path)
    vol = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    # Normalize to [0,1]
    vol = np.clip(vol, -1000, 1000)
    vol = (vol + 1000) / 2000.0

    H, W, D = vol.shape
    degraded_vol = np.zeros_like(vol)

    # Fixed projection angles for entire volume
    print(f"Using fixed sparse-view: {n_views} projections")

    # Degrade full volume
    for i in tqdm(range(D)):
        degraded_vol[:, :, i] = degrade_sparse_view(vol[:, :, i], n_views)

    # Save degraded volume (restore HU scale)
    degraded_hu = degraded_vol * 2000 - 1000
    out_path = os.path.join(os.path.dirname(nii_path), "downgraded_ct.nii.gz")
    nib.save(nib.Nifti1Image(degraded_hu.astype(np.float32), affine, header), out_path)
    print(f"Saved volume: {out_path}")

    # ------------------------------------------
    # Visualization (same as original structure)
    # ------------------------------------------
    positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    indices = [int(p * D) for p in positions]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i, idx in enumerate(indices):
        
        original_slice = vol[:, :, idx]
        degraded_slice = degraded_vol[:, :, idx]

        axes[0, i].imshow(np.rot90(original_slice), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"z={idx}", fontsize=12)
        axes[0, i].axis('off')

        axes[1, i].imshow(np.rot90(degraded_slice), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

    fig.text(0.01, 0.75, 'Original CT', rotation='vertical', fontsize=14)
    fig.text(0.01, 0.25, 'Sparse View', rotation='vertical', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved image: {save_path}")


# ------------------------------------------
# Run
# ------------------------------------------
if __name__ == "__main__":
    ct_file_path = "/projects/bodymaps/jliu452/Data/Dataset101_VIS/image/BDMAP_00049865_non-contrast.nii.gz"
    process_and_visualize(ct_file_path, n_views=20)