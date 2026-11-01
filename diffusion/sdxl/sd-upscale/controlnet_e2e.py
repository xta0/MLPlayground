import torch
from diffusers import StableDiffusionPipeline
from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetImg2ImgPipeline
import numpy as np
from PIL import Image
from img_utils import resize_img

text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float32,
    cache_dir = "/Volumes/ai-1t/diffuser"
).to("mps")

prompt = """
A stunningly realistic photo of a 25yo women with long, flowing brown hair and a beautiful smile. 
upper body, detailed eyes, detailed face, realistic skin texture, set against a blue sky, with a few fluffy clouds in the background.
"""

sr_prompt = """
HDR 8k, best quality, masterpiece, realistic, photo-realistic, ultra detailed, sharp focus, raw photo
"""

neg_prompt = """
worst quality, low quality, lowres, bad anatomy, missing person, watermarked
"""

raw_image = text2img_pipe(
    prompt = prompt,
    negative_prompt = neg_prompt,
    height = 256, 
    width = 256,
    generator = torch.Generator("mps").manual_seed(2)
).images[0]

text2img_pipe.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(raw_image))  # Convert from NumPy to PIL
image_pil.save("raw.png")
# resize image
resized_raw_image = resize_img("raw.png", 3.0)
resized_raw_image.save("raw_resized.png")

controlnet = ControlNetModel.from_pretrained(
    "takuma104/control_v11",
    subfolder=  'control_v11f1e_sd15_tile',
    torch_dtype = torch.float32,
    cache_dir = "/Volumes/ai-1t/diffuser"
)

# load checkpoint model with controlnet
pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float32,
    controlnet = controlnet,
    cache_dir = "/Volumes/ai-1t/diffuser"
).to("mps")


cn_tile_upscale_img = pipeline(
    image = resized_raw_image,
    control_image = resized_raw_image,
    prompt = f"{sr_prompt}{prompt}",
    negative_prompt = neg_prompt,
    strength = 0.8,
    guidence_scale = 7,
    generator = torch.Generator("mps"),
    num_inference_steps = 50
).images[0]
pipeline.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(cn_tile_upscale_img))  # Convert from NumPy to PIL
image_pil.save("cn_tile_upscale_img_768x768.png")


