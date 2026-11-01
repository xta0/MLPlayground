import torch
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
import numpy as np
from PIL import Image

from controlnet_aux import CannyDetector

cache_dir = "/Volumes/ai-1t/diffuser"
# # load the base model
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float32,
    cache_dir = cache_dir
).to("mps")

prompt = """
high resolution photo, best quality, masterpiece, 8k, 
A dog stand on grass, pure blue background, depth of field, full body, face to the camera
"""

neg_promt = """
painitng, low resolution, bad anatomy, worst quality, monochrome, grayscale
"""

image = text2img_pipe(
    prompt = prompt,
    negative_prompt = neg_promt,
    generator = torch.Generator("mps").manual_seed(7)
).images[0]

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save(f"base.png")

canny = CannyDetector()
image_canny = canny(image, 80, 250)
image_canny_pil = Image.fromarray(np.array(image_canny))  # Convert from NumPy to PIL
image_canny_pil.save(f"canny.png")

# load the controlnet model
canny_controlnet = ControlNetModel.from_pretrained(
    "takuma104/control_v11",
    subfolder=  'control_v11f1e_sd15_tile',
    torch_dtype = torch.float32,
    cache_dir = cache_dir
)
# canny_controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/control_v11p_sd15_softedge",
#     torch_dtype=torch.float32,
#     cache_dir = cache_dir
# )

cn_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float32,
    controlnet = canny_controlnet,
    cache_dir = cache_dir
).to("mps")

prompt2 = """
high resolution photo, best quality, masterpiece, 8k,
A cat stand on grass, pure blue background, depth of field, full body, face to the camera
"""
neg_prompt2 = """
painitng, sketches, normal quality, low resolution, worst quality, monochrome, grayscale
"""

image_from_canny = cn_pipe(
    prompt = prompt2,
    negative_prompt = neg_prompt2,
    image = image,
    control_image = image_canny,
    generator = torch.Generator("mps").manual_seed(1),
    num_inference_steps = 30,
    guidance_scale = 6.0,
    controlnet_conditioning_scale = 0.6
).images[0]

image_pil = Image.fromarray(np.array(image_from_canny))  # Convert from NumPy to PIL
image_pil.save(f"control_canny.png")
