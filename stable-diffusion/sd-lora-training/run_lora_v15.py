from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import matplotlib.pyplot as plt
from diffusers.utils import make_image_grid

CACHE_DIR = "/Volumes/ai-1t/diffuser"

# --- Set up model ---
# lora_name = "lora_stable-diffusion-v1-5_rank4_s2000_r512_DDPMScheduler_20250427-010841.safetensors"
lora_name = "pytorch_lora_weights.safetensors"
lora_model_path = f"./output_dir/{lora_name}"
device = "mps"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir = CACHE_DIR,
).to(device)

# prompt = "a toy bike. macro photo. 3d game asset"
# prompt = "a dog sitting on a beach. a painting in vangogh style"
# negative_prompt = "low quality, blur, watermark, words, name"
prompt = "a photo of a man, close-up, professional lighting"
negative_prompt = "missing person, blurry, deformed, bad anatomy, watermark, low quality, ugly, extra limbs, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
generator = torch.Generator(device).manual_seed(13)

# --- Step 1: Generate input images without LoRA ---
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

input_images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
    width=512,
    height=512,
    guidance_scale=8.5,
).images

# --- Step 2: Apply LoRA and generate output images ---
pipe.load_lora_weights(
    pretrained_model_name_or_path_or_dict=lora_model_path,
    adapter_name="az_lora"
)
pipe.set_adapters(["az_lora"], adapter_weights=[1.0])

prompt = "a photo of tao xu, close-up, professional lighting"

output_images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
    width=512,
    height=512,
    guidance_scale=10,
).images

# Move pipe to CPU and clear MPS cache
pipe.to("cpu")
torch.mps.empty_cache()

# --- Step 3: Plot 2 rows (input -> output) without white borders ---
fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'wspace':0, 'hspace':0})

# Top row: input images
for ax, img in zip(axs[0], input_images):
    ax.imshow(img)
    ax.axis('off')

# Bottom row: output images
for ax, img in zip(axs[1], output_images):
    ax.imshow(img)
    ax.axis('off')

# Remove whitespace around figure
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

# --- Step 4: Save the final grid ---
output_path = "input_vs_lora_output.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Saved to {output_path}")

# --- Step 5: Refine images ---
# from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
# from PIL import Image
# import numpy as np

# Resize LoRA output images for SDXL
# resized_images = [img.resize((768, 768), Image.LANCZOS) for img in output_images]

# # Load SDXL base model
# base_pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16,
#     cache_dir = CACHE_DIR,
# ).to("mps")

# # Create SDXL img2img pipeline
# img2img_pipe = StableDiffusionXLImg2ImgPipeline(
#     vae=base_pipe.vae,
#     text_encoder=base_pipe.text_encoder,
#     text_encoder_2=base_pipe.text_encoder_2,
#     tokenizer=base_pipe.tokenizer,
#     tokenizer_2=base_pipe.tokenizer_2,
#     unet=base_pipe.unet,
#     scheduler=base_pipe.scheduler,
#     force_zeros_for_empty_prompt=True,
# ).to("mps")

# img2img_pipe.watermark = None

# # Refine all resized LoRA output images
# refined_prompts = "ultra detailed, studio lighting, 8k resolution, hyper realistic, cinematic lighting, photorealistic, 35mm lens, f1.4, soft focus, bokeh, depth of field, volumetric lighting"
# refined_images = []
# for img in resized_images:
#     refined = img2img_pipe(
#         prompt=prompt,
#         image=img,
#         strength=0.3,
#         guidance_scale=4.0,
#         num_inference_steps=50
#     ).images[0]
#     refined_images.append(refined)

# # Clean up
# img2img_pipe.to("cpu")
# torch.mps.empty_cache()

# --- Plot 3 rows: base → LoRA → refined ---
# fig, axs = plt.subplots(3, 4, figsize=(20, 15), gridspec_kw={'wspace': 0, 'hspace': 0})

# # Row 1: Base images
# for ax, img in zip(axs[0], input_images):
#     ax.imshow(img)
#     ax.axis('off')

# # Row 2: LoRA images
# for ax, img in zip(axs[1], output_images):
#     ax.imshow(img)
#     ax.axis('off')

# Row 3: Refined images
# for ax, img in zip(axs[2], refined_images):
#     ax.imshow(img)
#     ax.axis('off')

# Remove whitespace and save
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# output_path = "base_vs_lora_vs_refined.png"
# plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
# plt.show()

print(f"✅ All refined images saved in: {output_path}")
