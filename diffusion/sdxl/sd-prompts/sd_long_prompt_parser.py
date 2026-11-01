import torch
from diffusers import StableDiffusionPipeline
import numpy as np

def long_prompt_encoding(
        pipe: StableDiffusionPipeline,
        prompt,
        neg_prompt = ""):
    bos = pipe.tokenizer.bos_token_id #beginning of sentence token
    eos = pipe.tokenizer.eos_token_id #end of sentence token
    chunk_size = 75

    # step1: take out the tokenizer and text encoder
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # step 2.1: encode whatever size prompt to tokens by setting truncation = False
    tokens = tokenizer(
        prompt, 
        truncation = False, 
    )["input_ids"]

    # step 2.2: encode whatever size neg prompt, padding it to the size of prompt
    negative_ids = tokenizer(
        neg_prompt, 
        padding = "max_length", 
        max_length = len(tokens), 
        truncation = False
    ).input_ids

    # step3: remove the beinging and the end of sentence token
    tokens = tokens[1:-1]
    negative_ids = negative_ids[1:-1]

    # step4: pop out the head 77 tokens, encoding them to embeddings
    embeds, neg_embeds = [], []

    for i in range(0, len(tokens), chunk_size):
        sub_tokens = [bos] + tokens[i: i+chunk_size] + [eos]
        # text_encoder needs a [1,x] input tensor
        tensor_tokens = torch.tensor(
            [sub_tokens], 
            dtype = torch.long, 
            device = pipe.device
        )
        # text_encoder can only encode 77 tokens at a time
        chunk_embeds = text_encoder(tensor_tokens)[0]
        embeds.append(chunk_embeds)

        sub_neg_tokens = [bos] + negative_ids[i: i+chunk_size] + [eos]
        tensor_neg_tokens = torch.tensor(
            [sub_neg_tokens], 
            dtype = torch.long, 
            device = pipe.device
        )
        chunk_neg_embeds = text_encoder(tensor_neg_tokens)[0]
        neg_embeds.append(chunk_neg_embeds)
    
    # step5: concatenate the embeddings
    prompt_embeds = torch.cat(embeds, dim = 1)
    neg_prompt_embeds = torch.cat(neg_embeds, dim = 1)

    return prompt_embeds, neg_prompt_embeds