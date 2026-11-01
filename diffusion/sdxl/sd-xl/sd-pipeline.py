import torch
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)
from diffusers import EulerDiscreteScheduler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

base_model_checkpoint_path = "/Volumes/ai-1t/sd-xl/sd_xl_base_1.0.safetensors"
refiner_model_checkpoint_path = "/Volumes/ai-1t/sd-xl/sd_xl_refiner_1.0.safetensors"
# load the base model
base_pipe = StableDiffusionXLPipeline.from_single_file(
    base_model_checkpoint_path,
    torch_dtype = torch.float16,
    use_safetensors = True
)

# load the refiner model
refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    refiner_model_checkpoint_path,
    torch_dtype = torch.float16,
    use_safetensors = True
)

# prompt = """
# analog photograph of a cat in a spacesuit taken inside the cockpit of a stealth fighter jet,
# Fujifile, Kodak Portra 400, vintage photography
# """

prompt = """
ultra detailed, detailed face, detailed eyes, beautiful doe eyes, masterpiece,
best quality, photo realistic, absurdres, 8K, raw photo, 1girl, solo, beautiful young woman, 
20yo, asian, realistic skin texture, shiny skin, office, black thighhighs, garter straps, turtleneck, 
lanyard, sleeveless, pencil skirt, perfect body, natural huge breasts, grin, smile to the camera
"""

neg_prompt = """
paint, watermark, 3D render, illustration, drawing, worst quality, low quality, animation, animated
"""

base_pipe.to("mps")
base_pipe.scheduler = EulerDiscreteScheduler.from_config(
    base_pipe.scheduler.config
)

with torch.no_grad():
    base_latents = base_pipe(
        prompt = prompt,
        negative_prompt = neg_prompt,
        output_type = "latent"
    ).images[0]

base_pipe.to("cpu")
torch.mps.empty_cache()


# refine the image
refiner_pipe.to("mps")
refiner_pipe.scheduler = EulerDiscreteScheduler.from_config(
    refiner_pipe.scheduler.config
)
with torch.no_grad():
    image = refiner_pipe(
        prompt = prompt,
        negative_prompt = neg_prompt,
        image = [base_latents]
    ).images[0]

refiner_pipe.to("cpu")
torch.mps.empty_cache()

# save the image
image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save("output.png")
