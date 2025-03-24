import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from lora_loader import load_lora_weights

cache_dir = "/Volumes/ai-1t/diffuser"
# load the base model
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.float16,
    cache_dir = cache_dir
)

lora_path = "/Volumes/ai-1t/diffuser/models--andrewzhu--MoXinV1/snapshots/7dbe7e0c8430ab549f2a0a45bc7f875e58c2eb7d/MoXinV1.safetensors"
load_lora_weights(pipeline, lora_path)

pipeline.to("mps")

prompt = """
shukezouma, shuimobysim, a  branch of flower, traditional Chinese ink painting
"""
image = pipeline(
    prompt = prompt,
    generator = torch.Generator("mps").manual_seed(1)
).images[0]

pipeline.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save("lora2.png")
