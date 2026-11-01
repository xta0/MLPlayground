from diffusers import UNet2DConditionModel

CACHE_DIR = "/Volumes/ai-1t/diffuser"

# Load SDXL UNet
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    subfolder="unet",
    cache_dir = CACHE_DIR
)

# Print architecture details
print(unet)