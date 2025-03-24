from diffusers import UNet2DConditionModel

# Load SDXL UNet
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    subfolder="unet"
)

# Print architecture details
print(unet)