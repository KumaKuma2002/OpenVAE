# OpenVAE

Open-source VAE family for medical imaging. Pretrained latent backbones for CT/MRI diffusion models.

2D and 3D autoencoders trained on up to 1M CT volumes with perceptual, adversarial, and segmentation-guided objectives.

## Models

| Model | Type | Patients | Latent | Resolution |
|---|---|---|---|---|
| `OpenVAE-2D-4x-20K` | KL-VAE | 20K | 4ch, 4x downsample | 512x512 |
| `OpenVAE-2D-4x-100K` | KL-VAE | 100K | 4ch, 4x downsample | 512x512 |
| `OpenVAE-2D-4x-300K` | KL-VAE | 300K | 4ch, 4x downsample | 512x512 |
| `OpenVAE-2D-4x-PCCT_Enhanced` | KL-VAE | 300K | 4ch, 4x downsample | 512x512 |
| `OpenVAE-3D-4x-20K` | KL-VAE | 20K | 4ch, 4x downsample | 64^3 patch |
| `OpenVAE-3D-4x-100K` | KL-VAE | 100K | 4ch, 4x downsample | 64^3 patch |
| `OpenVAE-3D-4x-1M` | KL-VAE | 1M | 4ch, 4x downsample | 64^3 patch |
| `OpenVAE-3D-4x-100K-VQ` | VQ-VAE | 100K | 4ch, 4x downsample | 64^3 patch |
| `OpenVAE-3D-8x-100K-VQ` | VQ-VAE | 100K | 4ch, 8x downsample | 64^3 patch |

## Quick Start

### 2D VAE (Diffusers)

```python
import torch
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "SMILE-project/OpenVAE", subfolder="vae"
).to("cuda").eval()

x = torch.randn(1, 3, 512, 512, device="cuda")
with torch.no_grad():
    z = vae.encode(x).latent_dist.sample()   # (1, 4, 128, 128)
    x_hat = vae.decode(z).sample              # (1, 3, 512, 512)
```

### 3D VAE (MONAI)

```python
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

model = AutoencoderKlMaisi(
    spatial_dims=3, in_channels=1, out_channels=1,
    num_channels=(64, 128, 256), latent_channels=4,
    num_res_blocks=(2, 2, 2), norm_num_groups=32,
    attention_levels=(False, False, False),
)
model.load_state_dict(torch.load("ckpt/OpenVAE-3D-4x-100K/autoencoder_best.pt"))
model.eval().to("cuda")

x = torch.randn(1, 1, 64, 64, 64, device="cuda")
with torch.no_grad():
    z, _, _ = model.encode(x)   # (1, 4, 16, 16, 16)
    x_hat = model.decode(z)     # (1, 1, 64, 64, 64)
```

### CT Reconstruction

```bash
# 2D slice-by-slice
python src/demo_medvae.py --input scan.nii.gz --checkpoint ckpt/OpenVAE-2D-4x-100K

# 3D sliding-window
python test/test_3dvae.py --input scan.nii.gz --checkpoint ckpt/OpenVAE-3D-4x-100K/autoencoder_best.pt
```

## Training

```bash
# 2D KL-VAE (multi-GPU, Accelerate)
accelerate launch src/train_klvae.py \
    --train_data_dir /data/AbdomenAtlasPro \
    --output_dir outputs/klvae \
    --resolution 512 \
    --train_batch_size 8 \
    --learning_rate 1e-4

# 3D VAE (single GPU, patch-based)
python src/train_3dvae.py \
    --train_data_dir /data/AbdomenAtlasPro \
    --patch_size 64 64 64 \
    --num_epochs 100 \
    --amp
```

**Data format:** `train_data_dir/<subject_id>/ct.h5` -- HDF5 with key `"image"`, shape `(H, W, D)`, HU values in `[-1000, 1000]`.

## Benchmark

```bash
# Run reconstruction on all models
bash test/benchmark.sh /path/to/ct_dir

# Recompute metrics (no re-inference)
python test/direct_compute_metrics.py --benchmark_root outputs/vae_benchmark --skip-lpips

# Plot
python test/plot_benchmark_metrics.py
```

**Metrics** (priority order):

| Metric | Scale | Direction | Description |
|---|---|---|---|
| MAE_100 | 0--100 | higher = better | `(1 - MAE) * 100` on 3D volumes in [0,1] |
| Detail_100 | 0--100 | higher = better | Pearson corr of 3D gradient magnitudes |
| SSIM | 0--1 | higher = better | Structural similarity |
| PSNR | dB | higher = better | Peak signal-to-noise ratio |
| LPIPS | 0--1 | lower = better | Learned perceptual similarity (AlexNet) |

## Project Structure

```
OpenVAE/
├── src/
│   ├── train_klvae.py            # 2D KL-VAE training (Diffusers + Accelerate)
│   ├── train_3dvae.py            # 3D VAE training (MONAI MAISI)
│   ├── demo_medvae.py            # 2D inference demo
│   ├── utils_loss.py             # Segmentation + GAN loss utilities
│   └── utils_discriminator.py    # PatchGAN / StyleGAN discriminators
├── test/
│   ├── benchmark_vae.py          # Full benchmark (inference + metrics)
│   ├── direct_compute_metrics.py # Metrics-only recomputation
│   ├── plot_benchmark_metrics.py # Visualization
│   ├── test_2dvae.py             # 2D VAE inference
│   └── test_3dvae.py             # 3D VAE inference (sliding-window)
└── ckpt/                         # Model checkpoints (download from HF)
```

## Citation

```bibtex
@article{liu2025see,
  title={See More, Change Less: Anatomy-Aware Diffusion for Contrast Enhancement},
  author={Liu, Junqi and Wu, Zejun and Bassi, Pedro RAS and Zhou, Xinze and Li, Wenxuan and Hamamci, Ibrahim E and Er, Sezgin and Lin, Tianyu and Luo, Yi and P{\l}otka, Szymon and others},
  journal={arXiv preprint arXiv:2512.07251},
  year={2025}
}
```

## License

MIT

---

**Pretrained weights:** [huggingface.co/SMILE-project/OpenVAE](https://huggingface.co/SMILE-project/OpenVAE)
