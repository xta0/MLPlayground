import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers.utils import load_image
from diffusers.utils.pil_utils import numpy_to_pil
import numpy as np
from PIL import Image, ImageOps

from rembg import remove


CACHE_DIR = "/Volumes/ai-1t/diffuser"
OUTPUT_DIR = "./output_dir"

processor = CLIPSegProcessor.from_pretrained(
    "CIDAS/clipseg-rd64-refined",
    cache_dir = CACHE_DIR
)

print(processor)

model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined",
    cache_dir = CACHE_DIR
)
print(model)

source_image = load_image("./base.png")

# remove the background
source_image_no_bk = remove(source_image)
source_image_no_bk.save(f"{OUTPUT_DIR}/base_no_bk.png")

# add white background
white_bk = Image.new("RGBA", source_image.size, (255, 255, 255, 255))
source_image_white_bk = Image.alpha_composite(white_bk, source_image_no_bk)
source_image_white_bk.save(f"{OUTPUT_DIR}/base_no_bk_white.png")
source_image_white_bk = load_image(f"{OUTPUT_DIR}/base_no_bk_white.png")

prompts = ['the background']
inputs = processor(
    text=prompts,
    images=source_image_white_bk,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    mask = torch.sigmoid(logits)
    mask_numpy = mask.detach().unsqueeze(-1).cpu().numpy()
    mask_pil = numpy_to_pil(mask_numpy)[0].resize(source_image.size)
    mask_pil.save(f"{OUTPUT_DIR}/raw_mask.png")

# Convert the mask to a binary mask
bw_thread = 100
bw_fn = lambda x: 255 if x > bw_thread else 0
bw_mask_pil = mask_pil.convert("L").point(bw_fn, mode="1")
bw_mask_pil.save(f"{OUTPUT_DIR}/binary_mask.png")

# redraw the mask using the SD inpainting model
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None,
    cache_dir = CACHE_DIR
).to("mps")

sd_prompt = "blue sky and mountains, high resolution"
sd_neg_prompt = "lowres, bad anatomy, low quality, monochrome"
out_image = inpaint_pipe(
    prompt=sd_prompt,
    negative_prompt=sd_neg_prompt,
    image=source_image,
    mask_image=bw_mask_pil,
    strength=0.9,
    generator = torch.Generator(device="mps").manual_seed(7),
).images[0]

image_pil = Image.fromarray(np.array(out_image))  # Convert from NumPy to PIL
image_pil.save(f"{OUTPUT_DIR}/output.png")

# image_bk = Image.new("RGBA", source_image.size, (255, 255, 255, 255))
# inverse_bw_mask_pil = ImageOps.invert(bw_mask_pil)
# compsed_image = Image.composite(source_image, image_bk, inverse_bw_mask_pil)
# compsed_image.save(f"{OUTPUT_DIR}/composed.png")

