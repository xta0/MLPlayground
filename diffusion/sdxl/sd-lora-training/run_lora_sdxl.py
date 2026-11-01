from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
)
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# === Configuration ===
CACHE_DIR = "/Volumes/ai-1t/diffuser"
LORA_PATH = "./output_dir_sdxl/pytorch_lora_weights.safetensors"
device = "mps"  # "cuda" if on NVIDIA GPU

# === Prompts ===
prompt = "a photo of Tao, close-up, professional lighting"
refined_prompt = (
    "close-up portrait, photorealistic, ultra detailed skin, cinematic lighting, "
     "sharp focus, 85mm f1.4 lens, soft background, professional studio light, "
     "vibrant colors"
)
negative_prompt = (
    "blurry, deformed, unrealistic, watermark, noise, jpeg artifacts, bad lighting, low detail"
)

# === Generation settings ===
generator = torch.Generator(device=device).manual_seed(13)
guidance_scale = 8.5
image_width = 1024
image_height = 1024
num_images = 2

# === Step 1: Load base pipeline and LoRA ===
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype= torch.float32,
    cache_dir=CACHE_DIR,
).to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(LORA_PATH, adapter_name="az_lora")
pipe.set_adapters(["az_lora"], adapter_weights=[1.0])

# === Step 2: Generate LoRA images ===
lora_images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=image_width,
    height=image_height,
    num_images_per_prompt=num_images,
    num_inference_steps=30,
    guidance_scale=guidance_scale,
    generator=generator,
).images

# Save LoRA output grid
fig, axs = plt.subplots(1, num_images, figsize=(10, 5))
for ax, img in zip(axs, lora_images):
    ax.imshow(img)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sdxl_lora_output.png", dpi=150)
plt.show()

# === Step 3: Load Refiner (optional) ===
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    cache_dir=CACHE_DIR,
    variant="fp16" if device == "cuda" else None,
).to(device)
refiner.watermark = None

# === Step 4: Resize images and run refinement ===
resized_images = [img.resize((1024, 1024), Image.LANCZOS) for img in lora_images]

refined_images = []
for img in resized_images:
    refined = refiner(
        prompt=refined_prompt,
        negative_prompt=negative_prompt,
        image=img,
        strength=0.3,
        guidance_scale=4.0,
        num_inference_steps=50,
        generator=generator,
    ).images[0]
    refined_images.append(refined)

# === Step 5: Display and Save Final Output ===
fig, axs = plt.subplots(2, num_images, figsize=(10, 10))

# Top row: LoRA
for ax, img in zip(axs[0], lora_images):
    ax.imshow(img)
    ax.axis("off")
axs[0][0].set_title("LoRA Output")

# Bottom row: Refined
for ax, img in zip(axs[1], refined_images):
    ax.imshow(img)
    ax.axis("off")
axs[1][0].set_title("Refined Output")

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
plt.savefig("sdxl_lora_refined.png", dpi=150, bbox_inches="tight", pad_inches=0)
plt.show()

print("✅ Images saved to sdxl_lora_output.png and sdxl_lora_refined.png")
