
import torch
from accelerate import utils
from accelerate import Accelerator
from diffusers import DDPMScheduler,StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from datasets import load_dataset
from torchvision import transforms
import math
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers.utils import convert_state_dict_to_diffusers

from datetime import datetime
formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

CACHE_DIR = "/Volumes/ai-1t/diffuser"

def main():
    utils.write_basic_config()
    # hyperparameters
    output_dir                      = "."
    pretrained_model_name_or_path   = "runwayml/stable-diffusion-v1-5"
    lora_rank                       = 4
    lora_alpha                      = 4
    learning_rate                   = 1e-4
    adam_beta1, adam_beta2          = 0.9, 0.999
    adam_weight_decay               = 1e-2
    adam_epsilon                    = 1e-08
    dataset_name                    = None                  #"lambdalabs/pokemon-blip-captions"
    train_data_dir                  = "./train_data"
    top_rows                        = 4
    output_dir                      = "output_dir"
    resolution                      = 768
    center_crop                     = True
    random_flip                     = True
    train_batch_size                = 4
    gradient_accumulation_steps     = 1
    num_train_epochs                = 200
    
    lr_scheduler_name               = "constant" #"cosine"#
    max_grad_norm                   = 1.0
    diffusion_scheduler             = DDPMScheduler

    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps
        , mixed_precision           = "fp16" 
    )
    device      = accelerator.device

    # Load scheduler, tokenizer and unet models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="scheduler",
        cache_dir = CACHE_DIR
    )
    weight_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype = weight_dtype
    ).to(device)
    tokenizer,text_encoder, vae, unet   = pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet

    # freeze parameters of models, we just want to train a LoRA only
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # configure LoRA parameters use PEFT
    unet_lora_config = LoraConfig(
        r                   = lora_rank,
        lora_alpha          = lora_alpha,
        init_lora_weights   = "gaussian",
        target_modules      = ["to_k", "to_q", "to_v", "to_out.0"]
    )

if __name__ == "__main__":
    # Call the main function from the lora module
    main()
else:
    # If not, print a message indicating that the script is not being run directly
    print("This script is not being run directly.")