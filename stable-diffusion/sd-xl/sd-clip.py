"""
text -> embeddings
"""
import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

prmpt = "a running dog"

clip_tokenizer_1 = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "tokenizer",
    dtype = torch.float16
)

input_tokens_1 = clip_tokenizer_1(
    prmpt,
    return_tensors = "pt"
)["input_ids"]

print(input_tokens_1.shape) # [49406, 320, 2761, 7251, 49407]

clip_tokenizer_2 = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "tokenizer_2",
    dtype = torch.float16
)

input_tokens_2 = clip_tokenizer_2(
    prmpt,
    return_tensors = "pt"
)["input_ids"]

print(input_tokens_2) # [49406, 320, 2761, 7251, 49407]


clip_text_encoder_1 = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "text_encoder",
).to("mps")

# encode token ids to embeddings
with torch.no_grad():
    text_embed_1 = clip_text_encoder_1(
        input_tokens_1.to("mps")
    )[0]
print(text_embed_1.shape)


clip_text_encoder_2 = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "text_encoder_2",
).to("mps")

# encode token ids to embeddings
with torch.no_grad():
    text_embed_2 = clip_text_encoder_2(
        input_tokens_1.to("mps")
    )[0]
print(text_embed_2.shape)



# pooled embedding
clip_text_encoder_3 = CLIPTextModelWithProjection.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "text_encoder_2",
    torch_dtype = torch.float16
).to("mps")

# encode token ids to embeddings
with torch.no_grad():
    prompt_embed_3 = clip_text_encoder_3(
        input_tokens_1.to("mps")
    )[0]
print(prompt_embed_3.shape)
