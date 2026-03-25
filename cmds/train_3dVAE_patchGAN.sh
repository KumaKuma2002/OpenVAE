#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Train 3D VAE with PatchGAN discriminator
#
# Modes:
#   1) Train from scratch  (default, comment out --resume_from_checkpoint)
#   2) Fine-tune from MAISI pretrained  (--resume_from_checkpoint maisi)
#   3) Resume a previous run  (--resume_from_checkpoint <models_dir>)
# -----------------------------------------------------------------------

export TRAIN_DATA_DIR="/mnt/data/jliu452/Data/Dataset901_SMILE/h5"

cd ../src
python train_3dvae.py \
  --train_data_dir=$TRAIN_DATA_DIR \
  --output_dir="../outputs/3dvae-patchgan" \
  --patch_size 64 64 64 \
  --num_epochs 100 \
  --train_batch_size=1 \
  --val_batch_size=1 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-4 \
  --recon_loss="l1" \
  --kl_weight=1e-7 \
  --perceptual_weight=0.3 \
  --adv_weight=0.1 \
  --val_interval=10 \
  --save_interval=10 \
  --log_steps=20 \
  --random_aug \
  --amp \
  --seed=0
  # --resume_from_checkpoint maisi          # uncomment to fine-tune from MAISI
  # --resume_from_checkpoint ../outputs/3dvae-patchgan/models  # uncomment to resume
