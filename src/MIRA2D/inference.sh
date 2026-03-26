#!/bin/bash
# ============================================================================
# SMILE-SR Inference Script (Super-Resolution only)
#
# Example:
#   bash inference.sh --gpu_id 0 --patient_id RS-GIST-121_venous_LR
# ============================================================================


# ------------------------- Parse Arguments -----------------------------------
GPU_ID="1"
PATIENT_ID=""
MODEL_NAME="checkpoint_best"   # default model

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_id)     GPU_ID="$2"; shift ;;
        --patient_id) PATIENT_ID="$2"; shift ;;
        --model)      MODEL_NAME="$2"; shift ;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done


# --------------------------- Sanity Checks -----------------------------------
if [ -z "$PATIENT_ID" ]; then
    echo "[ERROR] Missing required argument: --patient_id"
    exit 1
fi


echo "[INFO] GPU(s):     $GPU_ID"
echo "[INFO] Model:      $MODEL_NAME"
echo "[INFO] Patient:    $PATIENT_ID"


# ------------------------------- Fixed Paths ---------------------------------
export MODELS_FOLDER="../logs"
export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="/projects/bodymaps/jliu452/MedVAE/ckpt/MedVAE_KL-GAN-kuma-PCE"

DATASET="Dataset804_SMILE-SR_Validation"

CT_PATH="/projects/bodymaps/jliu452/Data/$DATASET/${PATIENT_ID}/ct.nii.gz"
OUTPUT_ROOT="/projects/bodymaps/jliu452/TRANS/sr/model-$MODEL_NAME"

TRAINED_UNET_PATH="$MODELS_FOLDER/$MODEL_NAME"


# ------------------------------ Print Info -----------------------------------
echo "[INFO] Input CT:   $CT_PATH"
echo "[INFO] Output Dir: $OUTPUT_ROOT"


# ------------------------------ Run Python -----------------------------------
CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore inference.py \
    --input_path "$CT_PATH" \
    --output_path "$OUTPUT_ROOT" \
    --chunk_size 20 \
    --finetuned_vae_name_or_path "$FT_VAE_NAME" \
    --finetuned_unet_name_or_path "$TRAINED_UNET_PATH" \
    --sd_model_name_or_path "$SD_MODEL_NAME"