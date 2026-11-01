import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/Users/ta0x1c/Projects/IP-Adapter-Models/IP-Adapter/models/image_encoder"
ip_ckpt = "/Users/ta0x1c/Projects/IP-Adapter-Models/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" # a experimental version
device = "mps"
CACHE_DIR = "/Volumes/ai-1t/diffuser"

pipe = StableDiffusionXLCustomPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    cache_dir = CACHE_DIR,
)
ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

image = Image.open("./ai_face.png")
image.resize((224, 224))

images = ip_model.generate(
    pil_image=image, 
    num_samples=2,
    num_inference_steps=30, 
    seed=42,
    prompt="photo of a beautiful girl wearing casual shirt in a garden",
)

images[0].save("generated_face_1.png")
images[1].save("generated_face_2.png")

