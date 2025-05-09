import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

CACHE_DIR = "/Volumes/ai-1t/diffuser"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir=CACHE_DIR,
).to("mps")

# (Optional) Load LoRA weights
# pipe.load_lora_weights("./lora-out", adapter_name="my_face")
# pipe.set_adapters(["my_face"])

# Your test prompt
prompt = "a man named Tao, upper body, white t-shirt, chest up, looking at camera, ultra detailed, studio lighting, 8k resolution, hyper realistic, cinematic lighting, photorealistic, 35mm lens, f1.4, soft focus, bokeh, depth of field, volumetric lighting"
negative_prompt = "blurry, deformed, bad anatomy, watermark"

# Generate image
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    num_inference_steps=30, 
    guidance_scale=7.5).images[0]

# Show and save
image.show()
image.save("output.png")

print("✅ Image saved as output.png")
