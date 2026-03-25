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
#  Submit via Slurm (single GPU):
#    sbatch --gres=gpu:a5000:1 --partition=intern cmds/train.sh
#
#  Or run interactively:
#    bash cmds/train.sh
# ============================================================================

set -euo pipefail

# ----------- paths (edit these) --------------------------------------------
TRAIN_DATA_DIR="/projects/bodymaps/jliu452/Data/Dataset803_SMILE_PCCT/h5"      # <subject>/ct.h5  (HU volumes)
VAE_CKPT="/projects/bodymaps/jliu452/MedVAE/outputs/3dvae-patchgan/models/MAISI.pt"
OUTPUT_DIR="./outputs/mira3d"

# Optional: set to a real nnUNet model folder to enable seg/HU losses in stage 3+
SEG_MODEL_PATH="/projects/bodymaps/jliu452/ePAI/wli131_2024_1115/nnUNetTrainer__nnUNetPlans__3d_fullres"                          # e.g. /path/to/Dataset911/nnUNetTrainer__nnUNetPlans__3d_fullres
SEG_CKPT_NAME="checkpoint_final.pth"        # nnUNet checkpoint filename
SEG_JSON="/projects/bodymaps/jliu452/ePAI/wli131_2024_1115/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"                                # /path/to/dataset.json (auto-found from model folder if empty)

# Optional: resume a previous UNet training run
RESUME_UNET="/projects/bodymaps/jliu452/MIRA3D/NV-Generate-CT/models/diff_unet_3d_ddpm-ct.pt"                             # /path/to/unet_latest.pt

VAL_LR_NIFTI="/projects/bodymaps/jliu452/Data/Dataset804_SMILE-SR_Validation/baichaoxiao20240416_arterial_LR/ct.nii.gz"
VAL_GT_NIFTI="/projects/bodymaps/jliu452/Data/Dataset901_SMILE/PT_data/baichaoxiao20240416_arterial.nii.gz"


# ----------- training hyper-parameters -------------------------------------
PATCH_SIZE="128 128 128"
BATCH_SIZE=4
GRAD_ACCUM=4                              # effective batch = BATCH_SIZE * GRAD_ACCUM
LR=1e-4
EPOCHS=100000
WORKERS=4
VAL_DDIM_STEPS=200                        # DDIM denoising steps during validation

# Staged loss warmup (by global step)
WARMUP_DIFF=2000
WARMUP_UC=5000
WARMUP_SEG_HU=15000
WARMUP_CYCLE=50000

# ----------- build command -------------------------------------------------
CMD="python train_mira3d.py \
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
    --warmup_add_cycle_steps ${WARMUP_CYCLE} \
    --val_ddim_steps ${VAL_DDIM_STEPS} \
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
elif [ -n "${VAL_NIFTI}" ]; then
    CMD="${CMD} --val_nifti_paths ${VAL_NIFTI}"
fi

cd "$(dirname "$0")/.."
echo "[mira3d] Running: ${CMD}"
eval ${CMD}
