#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# === Edit these paths as needed ===
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
TRAIN_DIR="./tao"
OUTPUT_DIR="./output_dir"
SCRIPT_PATH="/Users/ta0x1c/Projects/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py"  # <- SDXL-specific script
CAPTION_COLUMN="text"
CACHE_DIR="/Volumes/ai-1t/diffuser"

# === Training parameters ===
RESOLUTION=1024
BATCH_SIZE=1
GRAD_ACCUM=4
LEARNING_RATE=1e-4
MAX_TRAIN_STEPS=800
TRAIN_EPOCHS=100
CHECKPOINT_STEPS=100
SEED=42
LORA_R=4
LORA_ALPHA=4

# === Launch training ===
accelerate launch $SCRIPT_PATH \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --cache_dir="$CACHE_DIR" \
  --train_data_dir="$TRAIN_DIR" \
  --caption_column="$CAPTION_COLUMN" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACCUM \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=$TRAIN_EPOCHS \
  --checkpointing_steps=$CHECKPOINT_STEPS \
  --seed=$SEED \
  --rank=$LORA_R \
  --mixed_precision="no" \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --train_text_encoder
