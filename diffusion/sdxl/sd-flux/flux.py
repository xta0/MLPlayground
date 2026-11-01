import torch
from diffusers import FluxPipeline

device = "mps"

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
)
pipe.to(device)

prompt = "A cat holding a sign that says hello world"

# Generator on MPS for reproducibility (you can remove if it causes issues)
gen = torch.Generator(device=device).manual_seed(0)

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=gen,
).images[0]

image.save("flux-schnell-mps.png")
