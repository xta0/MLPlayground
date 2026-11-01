from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
from ti_loader import load_textual_inversion

model_id = "stablediffusionapi/deliberate-v2"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype = torch.float32, 
    cache_dir = "/Volumes/ai-1t/diffuser"
)

# pipe.load_textual_inversion(
#     "sd-concepts-library/midjourney-style",
#     token = "midjourney-style",
# )

"""
Maually load the TI models
"""
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

load_textual_inversion(
    learned_embeds_path="/Volumes/ai-1t/ti/midjourney_style.bin",
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    token="colorful-magic-style",
)

pipe.to("mps")

prmpt = """
a high quality photo of a futuristic city in deep space, colorful-magic-style
"""

raw_image = pipe(
    prompt = prmpt,
    num_inference_steps = 50,
    generator = torch.Generator("mps").manual_seed(0)
).images[0]


pipe.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(raw_image))  # Convert from NumPy to PIL
image_pil.save("ti.png")
