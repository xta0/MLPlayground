import torch

def load_textual_inversion(
        learned_embeds_path, 
        token, 
        text_encoder, 
        tokenizer, 
        weight = 0.5, 
        device="cpu"):
    """
    supports loading a .bin file
    """
    loaded_leared_embeds = torch.load(learned_embeds_path, map_location=device)
    keys = list(loaded_leared_embeds.keys())
    # get the new embedding
    embeds = loaded_leared_embeds[keys[0]] * weight
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token to the tokenizer
    num_added_tokens = tokenizer.add_tokens([token])
    if num_added_tokens == 0:
        raise ValueError("Token already exists in the tokenizer")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # upddate the embeddings
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return (tokenizer, text_encoder)


    
    
    