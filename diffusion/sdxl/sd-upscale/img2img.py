import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
from PIL import Image
from img_utils import resize_img

text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float32,
    cache_dir = "/Volumes/ai-1t/diffuser"
).to("mps")

prompt = """
a realistic photo of beautiful young women face
"""
neg_prompt = """
NSFW, bad anatomy
"""

raw_image = text2img_pipe(
    prompt = prompt,
    negative_prompt = neg_prompt,
    height = 256, 
    width = 256,
    generator = torch.Generator("mps").manual_seed(99)
).images[0]

text2img_pipe.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(raw_image))  # Convert from NumPy to PIL
image_pil.save("raw.png")
# resize image
resized_raw_image = resize_img("raw.png", 3.0)
resized_raw_image.save("raw_resized.png")
# resized_raw_image = np.array(resized_raw_image)
# resized_raw_image = Image.fromarray(np.array(resized_raw_image).astype(np.uint8))



# a single step img-to-img pipeline as the upscaler

sr_prompt = """
8k, best quality, masterpiece, realistic, photo-realistic, ultra detailed, sharp focus, raw photo
"""
neg_prompt = """
NSFW, worst quality, low quality, lowres, bad anatomy
"""

prompt = f"{sr_prompt} {prompt}"
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float32,
    cache_dir = "/Volumes/ai-1t/diffuser"
).to("mps")

img2image_3x = img2img_pipe(
    prompt = prompt,
    negative_prompt = neg_prompt,
    image = resized_raw_image, # this is a PIL image
    strength = 0.3,
    number_of_inference_steps = 80,
    guidance_scale = 8,
    generator = torch.Generator("mps").manual_seed(3)
).images[0]

img2img_pipe.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(img2image_3x))  # Convert from NumPy to PIL
image_pil.save("upscaled_768x768.png")

# # a multiple step img-to-img pipeline as the upscaler
# resized_raw_image = resize_img("upscaled_768x768.png", 2.0)
# resized_raw_image.save("raw_resized_2.png")
# resized_raw_image = np.array(resized_raw_image)
# resized_raw_image = Image.fromarray(np.array(resized_raw_image).astype(np.uint8))

# img2image_6x = img2img_pipe(
#     prompt = prompt,
#     negative_prompt = neg_prompt,
#     image = resized_raw_image, # this is a PIL image
#     strength = 0.3,
#     number_of_inference_steps = 80,
#     guidance_scale = 7.5,
#     generator = torch.Generator("mps").manual_seed(3)
# ).images[0]

# img2img_pipe.to("cpu")
# torch.mps.empty_cache()

# image_pil = Image.fromarray(np.array(img2image_6x))  # Convert from NumPy to PIL
# image_pil.save("upscaled_1536x1536.png")