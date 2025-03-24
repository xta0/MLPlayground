import matplotlib.pyplot as plt
import torch
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import numpy as np
from diffusers.utils import load_image

base_pipe = StableDiffusionXLPipeline.from_pretrained(
    "RunDiffusion/RunDiffusion-XL-Beta",
    torch_dtype = torch.float16,
    cache_dir = "/Volumes/ai-1t/diffuser"
)
base_pipe.watermark = None

prompt = "ultra detailed, detailed face, detailed eyes, beautiful doe eyes, masterpiece, best quality, photo realistic, absurdres, 8K, raw photo, 1girl, solo, beautiful young woman, 20yo, realistic skin texture, shiny skin, office, black thighhighs, garter straps, turtleneck, id card, lanyard, sleeveless, pencil skirt, perfect body, natural huge breasts, grin"

base_pipe.to("mps")

image = base_pipe(
    prompt = prompt,
    width = 768,
    height = 1024,
    generator = torch.Generator("mps").manual_seed(1)
).images[0]

base_pipe.to("cpu")
torch.mps.empty_cache()

print(type(image))

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save("base_line.png")

# refine the image
w, h = image.size
img_processor = VaeImageProcessor()
image_x = img_processor.resize(
    image = image,
    width = int(w * 1.5),
    height= int(h * 1.5)
)

image_pil = Image.fromarray(np.array(image_x))  # Convert from NumPy to PIL
image_pil.save("base_line_resized.png")


img2img_pipe = StableDiffusionXLImg2ImgPipeline(
    vae = base_pipe.vae,
    text_encoder = base_pipe.text_encoder,
    text_encoder_2 = base_pipe.text_encoder_2,
    tokenizer = base_pipe.tokenizer,
    tokenizer_2 = base_pipe.tokenizer_2,
    unet = base_pipe.unet,
    scheduler = base_pipe.scheduler,
    add_watermarker = None
)
img2img_pipe.watermark = None
img2img_pipe.to("mps")

refine_image_2x = img2img_pipe(
    prompt = prompt,
    image = image_x,
    strength = 0.3,
    num_inference_steps = 100,
    guidance_scale = 4.0,
).images[0]

img2img_pipe.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(refine_image_2x))  # Convert from NumPy to PIL
image_pil.save("refined.png")