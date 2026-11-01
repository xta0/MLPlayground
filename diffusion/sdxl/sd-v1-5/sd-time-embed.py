from diffusers import EulerAncestralDiscreteScheduler as Euler
from diffusers import StableDiffusionPipeline
import torch

scheduler = Euler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder = "scheduler"
)

# text2img_pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype = torch.float16
# ).to("mps")

# scheduler = Euler.from_config(text2img_pipe.scheduler.config) 

# sample the steps for the image diffusion process

inference_step = 20
scheduler.set_timesteps(inference_step, device = "mps")


timesteps = scheduler.timesteps
for t in timesteps:
    print(t)

"""
take 20 steps out of 1000 steps

tensor(999., device='mps:0')
tensor(946.4211, device='mps:0')
tensor(893.8421, device='mps:0')
tensor(841.2632, device='mps:0')
tensor(788.6842, device='mps:0')
tensor(736.1053, device='mps:0')
tensor(683.5263, device='mps:0')
tensor(630.9474, device='mps:0')
tensor(578.3684, device='mps:0')
tensor(525.7895, device='mps:0')
tensor(473.2105, device='mps:0')
tensor(420.6316, device='mps:0')
tensor(368.0526, device='mps:0')
tensor(315.4737, device='mps:0')
tensor(262.8947, device='mps:0')
tensor(210.3158, device='mps:0')
tensor(157.7368, device='mps:0')
tensor(105.1579, device='mps:0')
tensor(52.5789, device='mps:0')
tensor(0., device='mps:0')
"""