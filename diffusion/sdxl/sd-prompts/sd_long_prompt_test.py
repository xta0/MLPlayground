import torch
from sd_long_prompt_parser import long_prompt_encoding
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

model_id = "stablediffusionapi/deliberate-v2"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype = torch.float32, 
    cache_dir = "/Volumes/ai-1t/diffuser"
)
pipe.to("mps")


prompt = "photo, a cute white dog running on the road" * 10
neg_prompt = "low resolution, bad anatomy"
prompt_embeds, prompt_neg_embeds = long_prompt_encoding(
    pipe,
    prompt, 
    neg_prompt,
)

print(prompt_embeds.shape, prompt_neg_embeds.shape)

image = pipe(
    prompt = None,
    prompt_embeds = prompt_embeds,
    negative_prompt_embeds = prompt_neg_embeds,
    generator = torch.Generator("mps").manual_seed(1)
).images[0]

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save("long_prompt.png")
