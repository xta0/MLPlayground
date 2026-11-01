import torch
from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np

CACHE_DIR = "/Volumes/ai-1t/diffuser"
OUTPUT_DIR = "./output_dir"

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder = "models/image_encoder",
    torch_dtype = torch.float16,
    cache_dir= CACHE_DIR,
).to("mps")

print(image_encoder)

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    image_encoder = image_encoder,
    torch_dtype = torch.float16,
    safety_checker=None,
    cache_dir = CACHE_DIR
)

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder = "models",
    weight_name = "ip-adapter_sd15.bin"
)

source_image = load_image("./base.png")
ip_image = load_image("./vermeer.jpg")

pipeline.to('mps')

image = pipeline(
    prompt = "best quality, high quality",
    negative_prompt = "monochrome, lowres, bad anatomy, low quality",
    image = source_image,
    ip_adapter_image = ip_image,
    number_images_per_prompt = 1,
    number_inference_steps = 50,
    strength = 0.5,
    generator = torch.Generator("mps").manual_seed(1)
).images[0]

pipeline.to("cpu")
torch.mps.empty_cache()
image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save(f"{OUTPUT_DIR}/style_transfer.png")