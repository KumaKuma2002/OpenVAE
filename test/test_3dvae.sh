#!/usr/bin/env bash

# ---------- Test with your trained patchgan-best checkpoint ----------
python test_3dvae.py \
    --input /path/to/input.nii.gz \
    --checkpoint ../outputs/3dvae-patchgan/models/autoencoder_best.pt \
    --patch_size 80 80 80 \
    --overlap_ratio 0.5 \
    --output ./recon_patchgan_best.nii.gz \
    --amp

# ---------- Test with official MAISI pretrained checkpoint ----------
python test_3dvae.py \
    --input /path/to/input.nii.gz \
    --maisi_ckpt \
    --patch_size 80 80 80 \
    --overlap_ratio 0.5 \
    --output ./recon_maisi.nii.gz \
    --amp
