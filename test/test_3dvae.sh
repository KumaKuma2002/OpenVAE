#!/usr/bin/env bash
python test_3dvae.py \
    --input /path/to/input.nii.gz \
    --checkpoint ../ckpt/OpenVAE-3D-4x-10K/autoencoder_best.pt \
    --output ./reconstructed.nii.gz
