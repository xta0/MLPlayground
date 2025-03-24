from diffusers.utils import load_image
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

image = load_image('./mona.jpg')
print(type(image))
#  Display the noise image
# plt.imshow(image)
# plt.title("")
# plt.axis('off')
# plt.show()

# normalize the image to [-1, 1]
image_array = np.array(image).astype(np.float32)/255.0
image_array = image_array * 2.0 - 1.0

print(image_array.shape)

# HWC -> CHW
image_array_chw = image_array.transpose(2,0,1)

# NCHW
image_array_chw = np.expand_dims(image_array_chw, axis = 0)
image_array_chw = torch.from_numpy(image_array_chw)
print(image_array_chw.shape) # [1, 3, 300, 300]
image_array_chw_mps = image_array_chw.to("mps", dtype=torch.float16)

# encode the image from pixel space to latent space
vae_model = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder = "vae",
    torch_dtype=torch.float16
).to("mps")

# encode the image into a laten vector
latents = vae_model.encode(image_array_chw_mps).latent_dist.sample()
print(latents)
print(latents[0].shape) #[4, 37, 37]

def latent_to_img(latents_input, scale_rate = 1):
    latents_2 = ( 1/scale_rate ) * latents_input

    # decode image
    with torch.no_grad():
        decode_image = vae_model.decode(
            latents_input,
            return_dict = False
        )[0][0]
    
    decode_image = (decode_image / 2 + 0.5).clamp(0, 1)

    # move latent data from mps to cpu
    decode_image = decode_image.to("cpu")

    # convert torch tensor to numpy array
    numpy_img = decode_image.detach().numpy()

    print(numpy_img.shape)
    
    # convert image array to NCHW
    numpy_img_nchw = numpy_img.transpose(1, 2, 0)

    # map image data to 0, 255 and convert to int number
    numpy_img_rgb = (numpy_img_nchw * 255).round().astype("uint8")

    return Image.fromarray(numpy_img_rgb)

pil_img = latent_to_img(latents)
plt.imshow(pil_img)
plt.title("")
plt.axis('off')
plt.show()