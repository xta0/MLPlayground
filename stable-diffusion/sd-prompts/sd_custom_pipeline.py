from diffusers import DiffusionPipeline
import torch
import numpy as np
from PIL import Image

# model_id = "stablediffusionapi/deliberate-v2"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype = torch.float32, 
    cache_dir = "/Volumes/ai-1t/diffuser",
    # custom_pipeline = "lpw_stable_diffusion"
    custom_pipeline = "lpw_stable_diffusion_xl"
)
pipe.to("mps")


prompt = "photo, a cute dog running in the yard" * 10
prompt += "pure, (white: 1.5) dog" * 10
neg_prompt = "low resolution, bad anatomy"

image = pipe(
    prompt = prompt,
    negative_prompt = neg_prompt,
    generator = torch.Generator("mps").manual_seed(1)
).images[0]

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save("custome_pipeline.png")