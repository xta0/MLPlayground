#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# === Edit these paths as needed ===
MODEL_NAME="runwayml/stable-diffusion-v1-5"
TRAIN_DIR="./tao"
OUTPUT_DIR="./output_dir"
SCRIPT_PATH="/Users/ta0x1c/Projects/diffusers/examples/text_to_image/train_text_to_image_lora.py"  # Relative to where you cloned diffusers
CAPTION_COLUMN="text"

# === Training parameters ===
RESOLUTION=512
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
  --validation_prompt="a photo of tao xu, close-up, professional lighting"\
  