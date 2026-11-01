import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, PngImagePlugin
import json
import numpy as np

CACHE_DIR = "/Volumes/ai-1t/diffuser"
OUTPUT_DIR = "./output_dir"

model_id = "stablediffusionapi/deliberate-v2"
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir = CACHE_DIR,
).to("mps")

# we define all the parameters that will be used to generate an image in a JSON format
gen_meta = {
    "model_id": model_id,
    "prompt": "high resolution, a photograph of an astronaut riding a horse",
    "seed": 7,
    "inference_steps": 53,
    "height": 512,
    "width": 768,
    "guidance_scale": 7.5,
}

image = text2img_pipe(
    prompt=gen_meta["prompt"],
    height=gen_meta["height"],
    width=gen_meta["width"],
    num_inference_steps=gen_meta["inference_steps"],
    guidance_scale=gen_meta["guidance_scale"],
    generator=torch.Generator("mps").manual_seed(gen_meta["seed"]),
).images[0]

text2img_pipe.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save(f"{OUTPUT_DIR}/output.png")

# save gen_Data in the PNG file

image = Image.open(f"{OUTPUT_DIR}/output.png")

# define metadata we want to add
metadata = PngImagePlugin.PngInfo()
gen_meta_str = json.dumps(gen_meta)
metadata.add_text("sd_gen_meta", gen_meta_str)

image.save(f"{OUTPUT_DIR}/output_with_metadata.png", "PNG", pnginfo=metadata)
