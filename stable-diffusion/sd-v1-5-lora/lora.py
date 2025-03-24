import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

cache_dir = "/Volumes/ai-1t/diffuser"
# load the base model
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.float16,
    cache_dir = cache_dir
).to("mps")

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
image_pil.save("base.png")

# LoRA fine tuning

pipeline.to("mps")

alpha = 0.5

pipeline.load_lora_weights(
    "andrewzhu/MoXinV1",
    weight_name = "MoXinV1.safetensors",
    adapter_name = "MoXinV1",
    cache_dir = cache_dir
)

pipeline.load_lora_weights(
    "andrewzhu/civitai-light-shadow-lora",
    weight_name = "light_and_shadow.safetensors",
    adapter_name = "light_and_shadow",
    cache_dir = cache_dir
)

pipeline.set_adapters(
    ["MoXinV1", "light_and_shadow"],
    adapter_weights = [0.5, 1.0]
)
    
prompt = """
shukezouma, shuimobysim, a  branch of flower, traditional Chinese ink painting, STRRY LIGHT, COLORFUL
"""

image = pipeline(
    prompt = prompt,
    generator = torch.Generator("mps").manual_seed(1)
).images[0]

pipeline.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save(f"lora_compose.png")