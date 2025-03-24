import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

cache_dir = "/Volumes/ai-1t/diffuser"

pipeline = StableDiffusionPipeline.from_single_file(
    "/Volumes/ai-1t/seaart/sd-1.5-animation.safetensors",
    torch_dtype = torch.float32,
    use_safetensors = True
).to("mps")

prompt = """
3dcharacter,(1man, wrinkled face, old male:1.2), 
__eyecolor__ eyes, dark grey__hair-male__, 
(full body:1.2),plaid shirt, overalls, brown work boots, 
simple background, masterpiece,best quality,(light Red gradient background:1.1)
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
    "/Volumes/ai-1t/seaart/sd-1.5-3d-character.safetensors",
    weight_name = "3d-character.safetensors",
    adapter_name = "3d",
)

pipeline.set_adapters(
    ["3d"],
    adapter_weights = [0.5]
)

image = pipeline(
    prompt = prompt,
    generator = torch.Generator("mps").manual_seed(1)
).images[0]

pipeline.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save(f"lora_compose.png")