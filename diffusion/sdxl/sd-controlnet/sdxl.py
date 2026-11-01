import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers import ControlNetModel


import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector

cache_dir = "/Volumes/ai-1t/diffuser"

sdxl_pipe = StableDiffusionXLPipeline.from_pretrained( 
    "RunDiffusion/RunDiffusion-XL-Beta",
    torch_dtype = torch.float32,
    load_safety_checker = False,
    cache_dir = cache_dir
).to("mps")

prompt = """
full body photo of young man, arms spread white blank background,
glamour photography,
upper body wears dark shirt,
wears suit pants,
wears lether shoes
"""
neg_prompt = """
low resolution, bad anatomy, worst quality, ppaint, cg, spots, bad hands, three hands, noise, blur face, bad face
"""

sdxl_pipe.scheduler = EulerDiscreteScheduler.from_config(sdxl_pipe.scheduler.config)
image =  sdxl_pipe(
    prompt = prompt,
    negative_prompt = neg_prompt,
    width = 832,
    height = 1216,
).images[0]

sdxl_pipe.to("cpu")
torch.mps.empty_cache()

image_pil = Image.fromarray(np.array(image))  # Convert from NumPy to PIL
image_pil.save(f"base.png")

open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
pose_image = open_pose(image)

open_pose_pil = Image.fromarray(np.array(pose_image))  # Convert from NumPy to PIL
open_pose_pil.save(f"pose.png")

sdxl_pose_controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype = torch.float32,
    cache_dir = cache_dir
)

sdxl_cn_pipe = StableDiffusionXLControlNetPipeline.from_pretrained( 
    "RunDiffusion/RunDiffusion-XL-Beta",
    torch_dtype = torch.float32,
    add_watermark = False,
    controlnet = sdxl_pose_controlnet,
    load_safety_checker = False,
    cache_dir = cache_dir
).to("mps")
sdxl_cn_pipe.watermark = None

prompt2 = """
full body photo of young women, arms spread white blank background,
glamour photography,
upper body wears dark shirt,
wears suit pants,
wears lether shoes
"""
neg_prompt2 = """
low resolution, bad anatomy, worst quality, ppaint, cg, spots, bad hands, three hands, noise, blur face, bad face
"""

sdxl_cn_pipe.scheduler = EulerDiscreteScheduler.from_config(sdxl_cn_pipe.scheduler.config)

cn_image = sdxl_cn_pipe(
    prompt = prompt2,
    negative_prompt = neg_prompt2,
    image = pose_image,
    control_image = pose_image,
    width = 832,
    height = 1216,
    generator = torch.Generator("mps").manual_seed(2),
    num_inference_steps = 30,
    guide_scale = 6.0,
    controlnet_conditioning_scale = 0.5
).images[0]

sdxl_cn_pipe.to("cpu")
torch.mps.empty_cache()

open_pose_cn_pil = Image.fromarray(np.array(cn_image))  # Convert from NumPy to PIL
open_pose_cn_pil.save(f"cn_pose.png")