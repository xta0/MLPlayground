"""
text -> embeddings
"""
import torch
from transformers import CLIPTokenizer, CLIPTextModel

CACHE_DIR = "/Volumes/ai-1t/diffuser"

text = "a running dog"

clip_tokenizer = CLIPTokenizer.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder = "tokenizer",
    dtype = torch.float16,
    cache_dir=CACHE_DIR
)

input_tokens = clip_tokenizer(
    text,
    return_tensors = "pt"
)["input_ids"]

print(input_tokens) # [49406, 320, 2761, 7251, 49407]

clip_text_encoder = CLIPTextModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder = "text_encoder",
    cache_dir=CACHE_DIR
).to("mps")

print(clip_text_encoder) # CLIPTextModel(

# encode token ids to embeddings
text_embeds = clip_text_encoder(
    input_tokens.to("mps")
)[0]

print(text_embeds.shape)
# each token id is encoded into a 768-dim vector
print(text_embeds) # ([1, 5, 768])

# parepare neg prompt embeddings
uncond_toekns = "blur"

max_length = text_embeds.shape[1]

uncond_input_tokens = clip_tokenizer(
    uncond_toekns,
    padding = "max_length",
    max_length = max_length,
    truncation = True,
    return_tensors = "pt"
)["input_ids"]

# generate the negative embeddings
with torch.no_grad():
    negative_prompt_emebds = clip_text_encoder(
        uncond_input_tokens.to("mps")
    )[0]
prompt_embeds = torch.cat([negative_prompt_emebds, text_embeds])


