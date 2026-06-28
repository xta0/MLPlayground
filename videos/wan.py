import torch
from diffusers import AutoModel, WanPipeline
from diffusers.schedulers import EulerDiscreteScheduler, DDIMScheduler
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16

vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)

# Replace UniPC with a scheduler that usually works on MPS
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# Alternative:
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe.to(device)

out = pipe(
    prompt="A cinematic corgi surfing at sunset, smooth motion.",
    height=288, width=512,
    num_frames=49,
    num_inference_steps=20,
    guidance_scale=5.0,
).frames[0]

export_to_video(out, "wan22_mac.mp4", fps=24)
