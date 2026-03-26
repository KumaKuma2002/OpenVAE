# for testing the performance ONLY

# VAE -> use the one exclusively trained via PCCT
# CKPT -> use the besy CKPT
export FT_VAE_NAME="/projects/bodymaps/jliu452/MedVAE/ckpt/MedVAE_KL-GAN-kuma-PCE"
export MODELS_FOLDER="/projects/bodymaps/jliu452/ckpt"
export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATA_FOLDER="/projects/bodymaps/jliu452/Data"


##################################################################################################################################
MODEL_NAME="MIRA_v0.2"                           # SMILE-SR version
DATASET="Dataset806_LNDb16/LNDb_bdmap"                # target dataset
OUTPUT_FOLDER="/projects/bodymaps/jliu452/TRANS/sr/wash_data"     # output path
##################################################################################################################################

INPUT_ROOT="$DATA_FOLDER/$DATASET"
OUTPUT_ROOT="$OUTPUT_FOLDER/model-$MODEL_NAME/$DATASET-SR"
TRAINED_UNET_PATH="$MODELS_FOLDER/$MODEL_NAME"

python -W ignore inference_dataset.py \
    --input_path $INPUT_ROOT \
    --output_path "$OUTPUT_ROOT" \
    --finetuned_vae_name_or_path "$FT_VAE_NAME" \
    --finetuned_unet_name_or_path "$TRAINED_UNET_PATH" \
    --sd_model_name_or_path "$SD_MODEL_NAME" \
    --multiple_gpu \
    --chunk_size 128 \