import torch
from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetImg2ImgPipeline
from img_utils import resize_img
import numpy as np
from PIL import Image

controlnet = ControlNetModel.from_pretrained(
    "takuma104/control_v11",
    subfolder=  'control_v11f1e_sd15_tile',
    torch_dtype = torch.float16,
    cache_dir = "/Volumes/ai-1t/diffuser"
)

# load checkpoint model with controlnet
pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float16,
    controlnet = controlnet,
    cache_dir = "/Volumes/ai-1t/diffuser"
).to("mps")

image_name = "raw.png"
resized_raw_image = resize_img(image_name, 3.0)
# resized_raw_image = np.array(resized_raw_image)
# resized_raw_image = Image.fromarray(np.array(resized_raw_image).astype(np.uint8))

#  Image super resolution using ControlNet Tile

prompt = """
a realistic photo of beautiful young women face
"""

sr_prompt = """
8k, best quality, masterpiece, realistic, photo-realistic, ultra detailed, sharp focus, raw photo
"""
neg_prompt = """
NSFW, worst quality, low quality, lowres, bad anatomy
"""

cn_tile_upscale_img = pipeline(
    image = resized_raw_image,
    control_image = resized_raw_image,
    prompt = f"{sr_prompt} {prompt}",
    negative_prompt = neg_prompt,
    strength = 0.3,
    guidence_scale = 7,
    generator = torch.Generator("mps").manual_seed(3),
    num_inference_steps = 50
).images[0]
pipeline.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(cn_tile_upscale_img))  # Convert from NumPy to PIL
image_pil.save("cn_tile_upscale_img_768x768.png")

