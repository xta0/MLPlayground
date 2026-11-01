import torch
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)
from diffusers import EulerDiscreteScheduler
import matplotlib.pyplot as plt

base_model_checkpoint_path = "/Users/taox/Downloads/sd_xl_base_1.0.safetensors"
refiner_model_checkpoint_path = "/Users/taox/Downloads/sd_xl_refiner_1.0.safetensors"
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

prompt = """
analog photograph of a cat in a spacesuit taken inside the cockpit of a stealth fighter jet,
Fujifile, Kodak Portra 400, vintage photography
"""

neg_prompt = """
paint, watermark, 3D render, illustration, drawing, worst quality, low quality
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

print(type(image))
print(image.shape)

plt.imshow(image)
plt.axis("off")
plt.show()
