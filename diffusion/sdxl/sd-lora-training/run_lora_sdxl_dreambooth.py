from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
)
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

# === Config ===
CACHE_DIR = "/Volumes/ai-1t/diffuser"
LORA_PATH = "./output_dir_sdxl_dreambooth/pytorch_lora_weights.safetensors"
device = "mps"  # Use "cuda" if available

# === Prompts ===
instance_prompt = "a photo of Tao, 38 years old, upper body, black T-shirt, close-up, professional lighting"
refiner_prompt = (
    "a photo of Tao, ultra detailed, professional studio lighting, high resolution"
)
negative_prompt = (
    "blurry, deformed, low quality, watermark, jpeg artifacts, bad anatomy, distorted face"
)

# === Generation settings ===
generator = torch.Generator(device=device).manual_seed(7)
guidance_scale = 8.5
num_images = 2
width = 1024
height = 1024

# === Step 1: Load base SDXL pipeline with LoRA ===
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    cache_dir=CACHE_DIR,
).to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(LORA_PATH, adapter_name="dreambooth_lora")
pipe.set_adapters(["dreambooth_lora"], adapter_weights=[1.0])

# === Step 2: Generate LoRA image(s) ===
lora_images = pipe(
    prompt=instance_prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=num_images,
    width=width,
    height=height,
    guidance_scale=guidance_scale,
    num_inference_steps=30,
    generator=generator,
).images

# Save or preview results
os.makedirs("outputs", exist_ok=True)
for i, img in enumerate(lora_images):
    img.save(f"outputs/lora_output_{i}.png")

# === Step 3 (optional): Refine with SDXL Refiner ===
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None,
    cache_dir=CACHE_DIR,
).to(device)
refiner.watermark = None

refined_images = []
for i, img in enumerate(lora_images):
    refined = refiner(
        prompt=refiner_prompt,
        negative_prompt=negative_prompt,
        image=img.resize((width, height), Image.LANCZOS),
        strength=0.3,
        guidance_scale=5.0,
        num_inference_steps=50,
        generator=generator,
    ).images[0]
    refined_images.append(refined)
    refined.save(f"outputs/refined_output_{i}.png")

# === Optional: Display side-by-side ===
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, num_images, figsize=(10, 8))

# Top: LoRA outputs
for ax, img in zip(axs[0], lora_images):
    ax.imshow(img)
    ax.axis("off")
axs[0][0].set_title("LoRA Output")

# Bottom: Refined outputs
for ax, img in zip(axs[1], refined_images):
    ax.imshow(img)
    ax.axis("off")
axs[1][0].set_title("Refined Output")

plt.tight_layout()
plt.savefig("outputs/dreambooth_lora_vs_refined.png", dpi=150)
plt.show()
