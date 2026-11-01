import torch
from diffusers import MotionAdapter
from diffusers import AnimateDiffPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.utils import export_to_gif, export_to_video

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", 
    cache_dir="/Volumes/ai-1t/diffuser"
)

pipe = AnimateDiffPipeline.from_pretrained(
    "digiplay/majicMIX_realistic_v6", 
    cache_dir="/Volumes/ai-1t/diffuser",
    motion_adapter = adapter,
    safety_checker = None
)

scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "digiplay/majicMIX_realistic_v6", 
    subfolder="scheduler",
    clip_sampe = False,
    timestep_sampling = "linspace",
    steps_offset = 1
)
pipe.schduler = scheduler
pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

prompt = """
An ultra-detailed, photorealistic animated scene of a young girl in a magical, softly lit forest.
She has sparkling, deep blue eyes and flowing chestnut hair, with delicate, expressive facial features. 
The girl wears an elegant, intricately embroidered dress that moves gracefully with her subtle, fluid motions.
The background is a lush, enchanted woodland with soft bokeh, dynamic lighting, and a dreamy, cinematic atmosphere that captures every nuance of her expression and movement.
"""

neg_prompt = """
"blurry, low resolution, disfigured anatomy, extra limbs, mutated features, inconsistent eye color, cartoonish style, harsh shadows, overexposed or underexposed areas, poorly rendered details, grainy textures, artifacts, off-balance composition, unrealistic lighting"
"""

# lora
pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in", adapter_name="zoom-in")

pipe.to("mps")
output = pipe(
    prompt = prompt,
    negative_prompt = neg_prompt,
    height = 256,
    width = 256,
    num_frames = 16,
    num_inference_steps = 40,
    guidances_scale = 8.5,
    generator = torch.Generator("mps").manual_seed(7)
)
frames = output.frames[0]
export_to_gif(frames, "animated_diffusion_lora_256.gif")
export_to_video(frames, "animated_diffusion_lora_256.mp4")