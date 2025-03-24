from diffusers import DiffusionPipeline, StableDiffusionPipeline
import re
import torch

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

def parse_prompt_attention(text):
    # re_attention = re.compile(
    #     r"""
    #     \\\(|\\\)|\\\[|\\]|\\\\|\\       # Matches escaped characters: \( \) \[ \] \\ \
    #     |\(|\[                           # Matches unescaped ( and [
    #     |:([+-]?[.\d]+)\)                # Matches ":number)" (e.g., ":1.5)", ":-0.8)")
    #     |\)|]                            # Matches closing ) and ]
    #     |[^\\()\[\]:]+                   # Matches any sequence of characters that aren't \ ( ) [ ] :
    #     |:                                # Matches a colon (but only when it's not followed by a number in ":number)")
    #     """,
    #     re.X
    # )
    """
    detect the word "BREAK" in a string, including cases where it is surrounded by spaces, tabs, or newlines.
    """
    re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

    res = []
    round_brackets = []
    squre_brackets = []

    round_brackets_multiplier = 1.1
    square_brackets_multiplier = 1 / 1.1

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


# [['a ', 1.0], ['white', 1.1], [' cat', 1.0]]
# print(parse_prompt_attention("a (white) cat")) 

def get_prompts_with_weights(pipe: DiffusionPipeline, prompt: str):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """

    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens = []
    text_weights = []
    for words, weight in texts_and_weights:
        # here, the words could be a single word or a phrase
        token = pipe.tokenizer(words, truncation = False).input_ids[1:-1]
        # the token is a list: [320, 1125, 539, 320]
        text_tokens += token
        # copy the weight by length of token
        text_weights += [weight] * len(token)
    
    return text_tokens, text_weights

# model_id = "stablediffusionapi/deliberate-v2"
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, 
#     torch_dtype = torch.float32, 
#     cache_dir = "/Volumes/ai-1t/diffuser"
# )

# token_ids, token_weights = get_prompts_with_weights(pipe, "a (white and yellow) cat")
# ([320, 1579, 537, 4481, 2368], [1.0, 1.1, 1.1, 1.1, 1.0])
def pad_tokens_and_weights(token_ids, weights):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    bos, eos = 49406, 49407

    new_token_ids = []
    new_weights = []

    while len(token_ids) >= 75:
        # get the first 75 tokens and weights
        head_75_tokens = [token_ids.pop(0) for _ in range(75)]
        head_75_weights = [weights.pop(0) for _ in range(75)]

        
        head_77_tokens = [bos] + head_75_tokens + [eos]
        head_77_weights = [1.0] + head_75_weights + [1.0]

        new_token_ids.append(head_77_tokens)
        new_weights.append(head_77_weights)
    
    # pad left tokens and weights
    if len(token_ids) > 0:
        # padding_len = 75 - len(token_ids)
        padding_len = 0
        tail_77_tokens = [bos] + token_ids + [eos] * padding_len + [eos]
        tail_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
        new_token_ids.append(tail_77_tokens)
        new_weights.append(tail_77_weights)

    return new_token_ids, new_weights

# padded_tokens, padded_token_weights = pad_tokens_and_weights(token_ids, token_weights)
# print(padded_tokens, padded_token_weights)
# [[49406, 320, 1579, 537, 4481, 2368, 49407]] [[1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0]]

def get_weighted_text_embeddings(
  pipe: StableDiffusionPipeline,
  prompt: str = "",
  neg_prompt: str = ""
):
    eos = pipe.tokenizer.eos_token_id
    bos = pipe.tokenizer.bos_token_id

    prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt)
    neg_prompt_tokens, neg_prompt_weights = get_prompts_with_weights(pipe, neg_prompt)

    # pad the shorter one to the same length
    prompt_len = len(prompt_tokens)
    neg_prompt_len = len(neg_prompt_tokens)

    if prompt_len > neg_prompt_len:
        neg_prompt_tokens += [eos] * (prompt_len - neg_prompt_len)
        neg_prompt_weights += [1.0] * (prompt_len - neg_prompt_len)
    elif prompt_len < neg_prompt_len:
        prompt_tokens += [eos] * (neg_prompt_len - prompt_len)
        prompt_weights += [1.0] * (neg_prompt_len - prompt_len)
    
    embeds = []
    neg_embeds = []

    # handle long prompts
    prompt_token_groups, prompt_weight_groups = pad_tokens_and_weights(
        prompt_tokens.copy(), 
        prompt_weights.copy()
    )
    neg_prompt_token_groups, neg_prompt_weight_groups = pad_tokens_and_weights(
        neg_prompt_tokens.copy(), 
        neg_prompt_weights.copy()
    )
    # generate embeddings for each group
    for i in range(len(prompt_token_groups)):
        prompt_tokens = torch.tensor(
            [prompt_token_groups[i]], 
            dtype = torch.long, 
            device = pipe.device
        )
        prompt_weights = torch.tensor(
            prompt_weight_groups[i], 
            dtype = torch.float32, 
            device = pipe.device
        )
        prompt_embedding = pipe.text_encoder(prompt_tokens)[0].squeeze(0)
        print("prompt_embedding: ", prompt_embedding.shape)

        for j in range(len(prompt_weights)):
            weight_tensor = prompt_weights[j]
            prompt_embedding[j] *= weight_tensor
        prompt_embedding = prompt_embedding.unsqueeze(0)
        embeds.append(prompt_embedding)

        neg_prompt_tokens = torch.tensor(
            [neg_prompt_token_groups[i]], 
            dtype = torch.long, 
            device = pipe.device
        )
        neg_prompt_weights = torch.tensor(
            neg_prompt_weight_groups[i], 
            dtype = torch.float32, 
            device = pipe.device
        )
        neg_prompt_embedding = pipe.text_encoder(neg_prompt_tokens)[0].squeeze(0)

        for j in range(len(neg_prompt_weights)):
            weight_tensor = neg_prompt_weights[j]
            neg_prompt_embedding[j] *= weight_tensor
        neg_prompt_embedding = neg_prompt_embedding.unsqueeze(0)    
        neg_embeds.append(neg_prompt_embedding)
    
    prompt_embeds = torch.cat(embeds, dim = 1)
    neg_prompt_embeds = torch.cat(neg_embeds, dim = 1)
    print("final_prompt_embeds: ", prompt_embeds.shape)
    print("final_neg_prompt_embeds: ", neg_prompt_embeds.shape)

    return prompt_embeds, neg_prompt_embeds

