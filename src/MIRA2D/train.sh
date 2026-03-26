# ======================================================================= #
# training script for SMILE-Super Resolution

# Requirements: H100 or higher level GPUs
# Last Updated: Feb 9, 2026
# ======================================================================= #


export nnUNet_raw=/projects/bodymaps/jliu452/nnuet/nnUNet_raw
export nnUNet_preprocessed=/projects/bodymaps/jliu452/nnuet/nnUNet_preprocessed
export nnUNet_results=/projects/bodymaps/jliu452/nnuet/nnUNet_results

export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="/projects/bodymaps/jliu452/MedVAE/ckpt/MedVAE_KL-GAN-kuma-PCE"
export TRAINED_UNET_NAME="/projects/bodymaps/jliu452/ckpt/kuma_diffusion"
export SEG_MODEL_NAME="/projects/bodymaps/jliu452/nnUNet/Dataset_results/Dataset911/nnUNetTrainer__nnUNetPlans__2d"
export CLS_MODEL_NAME="/projects/bodymaps/jliu452/MONAI_CLS/models/best_model_99_smile.pth" # timm or MONAI

# Temporary path with soft link
export TRAIN_DATA_DIR="/projects/bodymaps/jliu452/Data/Dataset803_SMILE_PCCT/h5" 

# output location
export OUTPUT_DIR="/projects/bodymaps/jliu452/logs/smile-sr_logs-fe28-2026"
export CHECKPOINT_DIR="/projects/bodymaps/jliu452/logs/smile-sr_logs-fe14-2026/checkpoint-152000" 
export INIT_STEP=152000


#   --init_global_step=$INIT_STEP \
# train GPU settings
export NUM_GPU=1

accelerate launch --mixed_precision="no" --num_processes=$NUM_GPU train_text_to_image.py \
  --sd_model_name_or_path=$SD_MODEL_NAME \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --pretrained_unet_name_or_path=$TRAINED_UNET_NAME \
  --seg_model_path=$SEG_MODEL_NAME \
  --cls_model_path=$CLS_MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint=$CHECKPOINT_DIR \
  --init_global_step=$INIT_STEP \
  --use_strong_supervision \
  --enable_discriminator_module \
  --vae_loss="l1" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=2 \
  --max_train_steps=10_000_000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --report_to=wandb \
  --validation_steps=1000 \
  --checkpointing_steps=2000 \
  --checkpoints_total_limit=5 \
  --warmup_df_only_end_step=2000 \
  --warmup_add_cls_end_step=2000 \
  --warmup_add_cls_seg_hu_end_step=15000 \
  --warmup_add_cycle_end_step=50000 \
  --add_discriminator_end_step=1000000 \
  --p2p_loss_weight=1e-2\
  --uc_area_loss_weight=1e-3 \
  --cls_loss_weight=0 \
  --seg_loss_weight=1e-3 \
  --hu_loss_weight=1e-4  \
  --cycle_loss_weight=1e-3 \
  --gan_loss_weight=1e-3 \
  --validation_images ./data/baichaoxiao20240416_arterial_LR/ct.nii.gz ./data/baichaoxiao20240416_venous_LR/ct.nii.gz ./data/RS-GIST-121_venous_LR/ct.nii.gz ./data/WAW-TACE333_venous_LR/ct.nii.gz \
  --validation_prompt "A high resolution CT slice." "A high resolution CT slice." "A high resolution CT slice." "A high resolution CT slice." \