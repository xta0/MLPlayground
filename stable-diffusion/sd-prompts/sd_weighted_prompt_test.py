import torch
from sd_weighted_prompt_parser import get_weighted_text_embeddings
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

prompt = "photo, a cute dog running in the yard" * 10
prompt += "pure, (white: 1.5) dog" * 10
neg_prompt = "low resolution, bad anatomy"
prompt_embeds, prompt_neg_embeds = get_weighted_text_embeddings(
    pipe,
    prompt, 
    neg_prompt,
)

print(prompt_embeds.shape) # torch.Size([1, 124, 768])

image = pipe(
    prompt = None,
    prompt_embeds = prompt_embeds,
    negative_prompt_embeds = prompt_neg_embeds,
    generator = torch.Generator("mps").manual_seed(1)
).images[0]

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save("weighted_prompt.png")
