import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler as Euler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import load_image
from transformers import CLIPTokenizer, CLIPTextModel

# Set the device
device = "mps"


def initialize_models():
    """Load and initialize all models and the scheduler."""
    scheduler = Euler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )

    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=torch.float16
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        dtype=torch.float16
    )

    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder",
        torch_dtype=torch.float16
    ).to(device)

    return scheduler, unet, vae, tokenizer, text_encoder


def process_text(prompt, negative_prompt="blur"):
    """
    Tokenize and encode the prompt and negative prompt for classifier-free guidance.
    Returns concatenated embeddings.
    """
    # Tokenize and encode the main prompt.
    input_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        prompt_embeds = text_encoder(input_tokens.to(device))[0].half()

    # Tokenize and encode the negative prompt (matching the token length).
    max_length = prompt_embeds.shape[1]
    neg_tokens = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )["input_ids"]
    with torch.no_grad():
        negative_embeds = text_encoder(neg_tokens.to(device))[0]

    # Concatenate negative and positive embeddings.
    return torch.cat([negative_embeds, prompt_embeds])


def process_image(image_path, target_size=(256, 256)):
    """
    Loads an image from `image_path`, prints the original size, resizes it to `target_size`,
    and converts it to a normalized tensor in the range [-1, 1].
    """
    # Load the image using diffusers' helper (returns a PIL image).
    image = load_image(image_path)
    print("Original image size:", image.size)  # (width, height)

    # Resize the image to the target size.
    resized_image = image.resize(target_size)
    print("Resized image size:", resized_image.size)

    # Convert image to a NumPy array, normalize to [-1, 1].
    image_array = np.array(resized_image).astype(np.float32) / 255.0
    image_array = image_array * 2.0 - 1.0

    # Convert from HWC to CHW and add a batch dimension.
    image_array = image_array.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(device, dtype=torch.float16)

    return image_tensor


def encode_image_latents(image_tensor, vae):
    """Encode an image tensor into the latent space using the VAE encoder."""
    with torch.no_grad():
        # VAE encoder downsamples the image (typically by a factor of 8).
        image_latents = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    print("Image latents shape:", image_latents.shape)
    return image_latents


def get_initial_latents(vae, scheduler, image_tensor=None, strength=0.7, target_size=(256, 256)):
    """
    Returns initial latents. If an image_tensor is provided, it uses it for guidance (img2img);
    otherwise, it generates a pure noise latent for text-to-image synthesis.
    
    - strength: 0 means full image influence, 1 means full noise.
    - target_size: expected output image resolution (width, height).
    """
    if image_tensor is not None:
        # Encode image into latent space.
        image_latents = encode_image_latents(image_tensor, vae)
        noise = torch.randn(image_latents.shape, dtype=torch.float16, device=device)
        latents = image_latents * (1 - strength) + noise * scheduler.init_noise_sigma
    else:
        # For text-to-image, compute the latent shape based on the target resolution.
        # Stable Diffusion typically downsamples by a factor of 8.
        latent_width = target_size[0]
        latent_height = target_size[1]
        shape = (1, 4, latent_height, latent_width)
        latents = torch.randn(shape, dtype=torch.float16, device=device) * scheduler.init_noise_sigma
    return latents


def run_diffusion(latents, prompt_embeds, scheduler, unet, num_steps=20, guidance_scale=7.5):
    """Perform the diffusion process on the initial latents."""
    scheduler.set_timesteps(num_steps, device=device)
    latents_sd = latents.clone()

    for t in scheduler.timesteps:
        # Duplicate the latent input for classifier-free guidance.
        latent_model_input = torch.cat([latents_sd] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]

        # Split predictions into unconditional and conditional parts.
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents_sd = scheduler.step(noise_pred, t, latents_sd, return_dict=False)[0]

    return latents_sd


def decode_latents_to_image(latents, vae):
    """
    Decode the latent tensor back into an image using the VAE decoder,
    then convert it into a PIL image.
    """
    # Reverse the scaling factor used during encoding.
    latents = (1 / 0.18215) * latents

    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0][0]

    # Normalize the image to [0, 1] and convert to CPU.
    decoded = (decoded / 2 + 0.5).clamp(0, 1).to("cpu")
    image_np = decoded.detach().numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).round().astype(np.uint8)
    return Image.fromarray(image_np)


if __name__ == "__main__":
    # ----- Initialize Models and Scheduler -----
    scheduler, unet, vae, tokenizer, text_encoder = initialize_models()

    # ----- Process the Text Prompt -----
    prompt = "a running dog"
    prompt_embeds = process_text(prompt)

    # ----- Set Options for Image Guidance or Pure Text-to-Image -----
    # Set `image_path` to None for pure text-to-image generation.
    # Otherwise, specify a path (e.g., "./mona.jpg") for image guidance.
    image_path = "./mona.jpg"  # Change to None to disable image guidance.
    image_path = None
    

    if image_path is not None:
        target_size = (256, 256)    # (width, height)
        image_tensor = process_image(image_path, target_size=target_size)
    else:
        target_size = (64, 64)
        image_tensor = None

    # ----- Get Initial Latents -----
    # For img2img, the `strength` parameter blends image guidance with noise.
    # For text-to-image, the latent is pure noise.
    strength = 0.7  # 0: use full image, 1: use full noise (for image-guided generation)
    latents = get_initial_latents(vae, scheduler, image_tensor=image_tensor, strength=strength, target_size=target_size)

    # ----- Run the Diffusion Process -----
    num_steps = 20
    guidance_scale = 7.5
    latents_sd = run_diffusion(latents, prompt_embeds, scheduler, unet, num_steps=num_steps, guidance_scale=guidance_scale)

    # ----- Decode and Display the Final Image -----
    result_image = decode_latents_to_image(latents_sd, vae)
    plt.imshow(result_image)
    plt.axis("off")
    plt.show()
