#!/usr/bin/env bash
# ============================================================================
#  MIRA3D — 3D LDM Super-Resolution Training
# ============================================================================
#
#  Launch modes:
#    1. From scratch    — only set --vae_checkpoint
#    2. With segmenter  — also set --seg_model_path (nnUNet folder)
#    3. Resume          — also set --resume_unet to continue from a .pt
#
#  Single-GPU (interactive):
#    bash cmds/train.sh
#
#  Single-GPU (Slurm):
#    sbatch --gres=gpu:1 --partition=intern cmds/train.sh
#
#  Multi-GPU — set NUM_GPUS below, then:
#    sbatch --gres=gpu:4 --partition=intern cmds/train.sh
#
#  Multi-GPU uses torchrun (DDP via HuggingFace Accelerate).
#  Requires: pip install accelerate
# ============================================================================

set -euo pipefail

# ----------- multi-GPU -----------------------------------------------------
NUM_GPUS=1   # set to 2/4/8 for multi-GPU DDP (torchrun is used automatically)

# ----------- paths (edit these) --------------------------------------------
TRAIN_DATA_DIR="/projects/bodymaps/jliu452/Data/Dataset803_SMILE_PCCT/h5"      # <subject>/ct.h5  (HU volumes)
VAE_CKPT="/projects/bodymaps/jliu452/MedVAE/outputs/3dvae-patchgan/models/MAISI.pt"
OUTPUT_DIR="./outputs/mira3d"

# Optional: set to a real nnUNet model folder to enable seg/HU losses in stage 3+
SEG_MODEL_PATH="/projects/bodymaps/jliu452/ePAI/wli131_2024_1115/nnUNetTrainer__nnUNetPlans__3d_fullres"                          # e.g. /path/to/Dataset911/nnUNetTrainer__nnUNetPlans__3d_fullres
SEG_CKPT_NAME="checkpoint_final.pth"        # nnUNet checkpoint filename
SEG_JSON="/projects/bodymaps/jliu452/ePAI/wli131_2024_1115/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"                                # /path/to/dataset.json (auto-found from model folder if empty)

# Optional: resume a previous UNet training run
RESUME_UNET="/projects/bodymaps/jliu452/MIRA3D/outputs/mira3d/models/unet_best.pt"                             # /path/to/unet_latest.pt

VAL_LR_NIFTI="/projects/bodymaps/jliu452/Data/Dataset804_SMILE-SR_Validation/baichaoxiao20240416_arterial_LR/ct.nii.gz"
VAL_GT_NIFTI="/projects/bodymaps/jliu452/Data/Dataset901_SMILE/PT_data/baichaoxiao20240416_arterial.nii.gz"


# ----------- training hyper-parameters -------------------------------------
PATCH_SIZE="256 256 128"
BATCH_SIZE=4
GRAD_ACCUM=4                              # effective batch = BATCH_SIZE * GRAD_ACCUM
LR=1e-4
EPOCHS=100000
WORKERS=4
VAL_DDIM_STEPS=200                        # DDIM denoising steps during validation
VAL_OVERLAP_RATIO=0.625                   # sliding-window overlap (higher = smoother patch boundaries)

# Staged loss warmup (by global step)
# Set a stage's step to 0 to enable it from the very start.
# Set it higher than EPOCHS*batches_per_epoch to effectively disable it.
WARMUP_DIFF=0
WARMUP_UC=0
WARMUP_SEG_HU=0

# ----------- loss weights --------------------------------------------------
# diffusion loss weight is implicitly 1.0 (always the primary signal)
PIXEL_LOSS_WEIGHT=0.1     # full-patch image-space L1 (recon vs HR, Stage 2+)
UC_LOSS_WEIGHT=1e-3       # unchanged-region MSE (Stage 2+)
SEG_LOSS_WEIGHT=1e-3      # segmentation cross-entropy (Stage 3+)
HU_LOSS_WEIGHT=1e-4       # organ HU MSE (Stage 3+)

# Discriminator (Stage 4) — 3D PatchGAN activates after WARMUP_ADV steps
WARMUP_ADV=25000          # global step to enable adversarial training
DISC_LOSS_WEIGHT=0.1      # generator adversarial loss weight
DISC_LR=1e-4              # discriminator optimizer LR

# ----------- build command -------------------------------------------------
if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500"
else
    LAUNCHER="python"
fi

CMD="${LAUNCHER} train_mira3d.py \
    --train_data_dir ${TRAIN_DATA_DIR} \
    --vae_checkpoint ${VAE_CKPT} \
    --output_dir ${OUTPUT_DIR} \
    --patch_size ${PATCH_SIZE} \
    --train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --num_epochs ${EPOCHS} \
    --dataloader_num_workers ${WORKERS} \
    --warmup_diffusion_only_steps ${WARMUP_DIFF} \
    --warmup_add_unchanged_steps ${WARMUP_UC} \
    --warmup_add_seg_hu_steps ${WARMUP_SEG_HU} \
    --val_ddim_steps ${VAL_DDIM_STEPS} \
    --val_overlap_ratio ${VAL_OVERLAP_RATIO} \
    --pixel_loss_weight ${PIXEL_LOSS_WEIGHT} \
    --uc_loss_weight ${UC_LOSS_WEIGHT} \
    --seg_loss_weight ${SEG_LOSS_WEIGHT} \
    --hu_loss_weight ${HU_LOSS_WEIGHT} \
    --warmup_add_adv_steps ${WARMUP_ADV} \
    --disc_loss_weight ${DISC_LOSS_WEIGHT} \
    --disc_lr ${DISC_LR} \
    --amp"

if [ -n "${SEG_MODEL_PATH}" ]; then
    CMD="${CMD} --seg_model_path ${SEG_MODEL_PATH}"
    CMD="${CMD} --seg_checkpoint_name ${SEG_CKPT_NAME}"
    if [ -n "${SEG_JSON}" ]; then
        CMD="${CMD} --seg_dataset_json ${SEG_JSON}"
    fi
fi

if [ -n "${RESUME_UNET}" ]; then
    CMD="${CMD} --resume_unet ${RESUME_UNET}"
fi

if [ -n "${VAL_LR_NIFTI}" ] || [ -n "${VAL_GT_NIFTI}" ]; then
    CMD="${CMD} --val_lr_nifti_paths ${VAL_LR_NIFTI} --val_gt_nifti_paths ${VAL_GT_NIFTI}"
elif [ -n "${VAL_NIFTI:-}" ]; then
    CMD="${CMD} --val_nifti_paths ${VAL_NIFTI}"
fi

cd "$(dirname "$0")/.."

echo "[mira3d] ========== launch =========="
echo "[mira3d] NUM_GPUS: ${NUM_GPUS}"
echo "[mira3d] LAUNCHER: ${LAUNCHER}"
echo "[mira3d] TRAIN_DATA_DIR: ${TRAIN_DATA_DIR}"
echo "[mira3d] VAE_CKPT: ${VAE_CKPT}"
echo "[mira3d] OUTPUT_DIR: ${OUTPUT_DIR}"
echo "[mira3d] PATCH_SIZE: ${PATCH_SIZE}"
echo "[mira3d] TRAIN_BATCH_SIZE: ${BATCH_SIZE}"
echo "[mira3d] GRADIENT_ACCUMULATION_STEPS: ${GRAD_ACCUM}"
echo "[mira3d] LEARNING_RATE: ${LR}"
echo "[mira3d] EPOCHS: ${EPOCHS}"
echo "[mira3d] DATALOADER_NUM_WORKERS: ${WORKERS}"
echo "[mira3d] WARMUP_DIFFUSION_ONLY_STEPS: ${WARMUP_DIFF}"
echo "[mira3d] WARMUP_ADD_UNCHANGED_STEPS: ${WARMUP_UC}"
echo "[mira3d] WARMUP_ADD_SEG_HU_STEPS: ${WARMUP_SEG_HU}"
echo "[mira3d] VAL_DDIM_STEPS: ${VAL_DDIM_STEPS}"
echo "[mira3d] VAL_OVERLAP_RATIO: ${VAL_OVERLAP_RATIO}"
echo "[mira3d] PIXEL_LOSS_WEIGHT: ${PIXEL_LOSS_WEIGHT}"
echo "[mira3d] UC_LOSS_WEIGHT: ${UC_LOSS_WEIGHT}"
echo "[mira3d] SEG_LOSS_WEIGHT: ${SEG_LOSS_WEIGHT}"
echo "[mira3d] HU_LOSS_WEIGHT: ${HU_LOSS_WEIGHT}"
echo "[mira3d] WARMUP_ADV_STEPS: ${WARMUP_ADV}"
echo "[mira3d] DISC_LOSS_WEIGHT: ${DISC_LOSS_WEIGHT}"
echo "[mira3d] DISC_LR: ${DISC_LR}"
echo "[mira3d] AMP: enabled"
if [ -n "${SEG_MODEL_PATH}" ]; then
    echo "[mira3d] SEG_MODEL_PATH: ${SEG_MODEL_PATH}"
    echo "[mira3d] SEG_CHECKPOINT_NAME: ${SEG_CKPT_NAME}"
    [ -n "${SEG_JSON}" ] && echo "[mira3d] SEG_DATASET_JSON: ${SEG_JSON}"
fi
if [ -n "${RESUME_UNET}" ]; then
    echo "[mira3d] RESUME_UNET: ${RESUME_UNET}"
fi
if [ -n "${VAL_LR_NIFTI}" ] || [ -n "${VAL_GT_NIFTI}" ]; then
    echo "[mira3d] VAL_LR_NIFTI: ${VAL_LR_NIFTI}"
    echo "[mira3d] VAL_GT_NIFTI: ${VAL_GT_NIFTI}"
elif [ -n "${VAL_NIFTI:-}" ]; then
    echo "[mira3d] VAL_NIFTI: ${VAL_NIFTI:-}"
fi
echo "[mira3d] ============================="
echo "[mira3d] Starting train_mira3d.py ..."

eval ${CMD}
