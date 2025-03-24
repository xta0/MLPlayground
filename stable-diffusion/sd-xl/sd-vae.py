from diffusers.utils import load_image
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# encode the image from pixel space to latent space
vae_model = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "vae",
    torch_dtype=torch.float32
).to("mps")

image_path = './mona.jpg'
image = load_image(image_path)

image_processor = VaeImageProcessor()

"""
Converts the image to a normailzed torch tensor in range: [-1, 1]
"""
prep_image = image_processor.preprocess(image)
prep_image = prep_image.to("mps", dtype=torch.float32)
print(prep_image.shape)
with torch.no_grad():
    image_latent = vae_model.encode(prep_image).latent_dist.sample()

print(image_latent.shape) #[1, 4, 37, 37]

# decode the latent
with torch.no_grad():
    decoded_image = vae_model.decode(image_latent, return_dict = False)[0]
    decoded_image = decoded_image.to("cpu")
   
pil_img = image_processor.postprocess(image = decoded_image, output_type="pil")
pil_img = pil_img[0]
plt.imshow(pil_img)
plt.title("")
plt.axis('off')
plt.show()