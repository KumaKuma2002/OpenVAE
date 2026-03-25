#!/usr/bin/env bash
# ============================================================================
#  MIRA3D — 3D LDM Super-Resolution Inference
# ============================================================================
#
#  Sliding-window SR with DDIM sampling.
#
#  Submit via Slurm:
#    sbatch --gres=gpu:a5000:1 --partition=intern cmds/inference.sh
#
#  Or run interactively:
#    bash cmds/inference.sh
# ============================================================================

set -euo pipefail

# ----------- paths (edit these) --------------------------------------------
INPUT="/projects/bodymaps/jliu452/Data/Dataset804_SMILE-SR_Validation/baichaoxiao20240416_arterial_LR/ct.nii.gz"
VAE_CKPT="/projects/bodymaps/jliu452/MedVAE/outputs/3dvae-patchgan/models/MAISI.pt"
UNET_CKPT="/projects/bodymaps/jliu452/MIRA3D/outputs/mira3d/models/unet_best.pt"
OUTPUT="./test_sr_output.nii.gz"

# ----------- inference hyper-parameters ------------------------------------
PATCH_SIZE="128 128 128"
OVERLAP_RATIO=0.5
DDIM_STEPS=200

# ----------- run -----------------------------------------------------------
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --input "${INPUT}" \
    --vae_checkpoint "${VAE_CKPT}" \
    --unet_checkpoint "${UNET_CKPT}" \
    --output "${OUTPUT}" \
    --patch_size ${PATCH_SIZE} \
    --overlap_ratio ${OVERLAP_RATIO} \
    --num_inference_steps ${DDIM_STEPS} \
    --amp
