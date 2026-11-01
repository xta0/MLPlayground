#!/bin/bash

# Exit immediately if a command fails
set -e

# === Required Paths ===
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
INSTANCE_DIR="./tao_hi_res"  # Folder with your subject images (e.g., 5–20 images of a person)
OUTPUT_DIR="./output_dir_sdxl_dreambooth"
SCRIPT_PATH="/Users/ta0x1c/Projects/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py"
CACHE_DIR="/Volumes/ai-1t/diffuser"

# === Prompt ===
INSTANCE_PROMPT="a photo of Tao"  # Used during training
VALIDATION_PROMPT="a photo of Tao, close-up, professional lighting"

# === Training Parameters ===
RESOLUTION=1024
BATCH_SIZE=1
GRAD_ACCUM=4
LEARNING_RATE=1e-4
MAX_TRAIN_STEPS=800
CHECKPOINT_STEPS=100
SEED=42
LORA_RANK=4
LORA_ALPHA=4

# === Run Training ===
accelerate launch $SCRIPT_PATH \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --cache_dir="$CACHE_DIR" \
  --instance_data_dir="$INSTANCE_DIR" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACCUM \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --checkpointing_steps=$CHECKPOINT_STEPS \
  --rank=$LORA_RANK \
  --mixed_precision="no" \
  --seed=$SEED \
  --train_text_encoder \
