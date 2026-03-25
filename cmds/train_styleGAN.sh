#!/usr/bin/env bash
export TRAIN_DATA_DIR="/mnt/data/jliu452/Data/Dataset901_SMILE/h5"

cd ../src
accelerate launch --num_processes=1 train_klvae.py \
  --train_data_dir=$TRAIN_DATA_DIR \
  --output_dir="../outputs/klvae-discriminator" \
  --resume_from_checkpoint="latest" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --seg_model_path="../ckpt/segmenter/nnUNetTrainer__nnUNetResEncUNetLPlans__2d" \
  --discriminator_type="StyleGAN" \
  --validation_images /path/to/validation.h5 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --report_to="wandb" \
  --max_train_steps=100_000_000 \
  --vae_loss="l1" \
  --learning_rate=1e-4 \
  --validation_steps=10000 \
  --checkpointing_steps=10000 \
  --checkpoints_total_limit=5 \
  --kl_weight=1e-8 \
  --seg_loss_weight=1e-3 \
  --gan_loss_weight=1e-2
