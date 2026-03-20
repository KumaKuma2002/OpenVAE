#!/usr/bin/env bash
export TRAIN_DATA_DIR="/path/to/data"

cd ../src
python train_3dvae.py \
  --train_data_dir=$TRAIN_DATA_DIR \
  --output_dir="../outputs/3dvae-patchgan" \
  --patch_size 64 64 64 \
  --num_epochs 100 \
  --train_batch_size=1 \
  --val_batch_size=1 \
  --dataloader_num_workers=0 \
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
