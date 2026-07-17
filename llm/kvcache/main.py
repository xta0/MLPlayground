from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

local_model_dir = Path(__file__).resolve().parent / "models" / "gpt2"
model_name = str(local_model_dir) if local_model_dir.exists() else "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)

def generate_token(inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # During training the model needs a prediction at every position. 
    # During inference we only care about the next token, 
    # so we keep just the last position along the sequence axis (logits[0, -1, :]) 
    # and take the argmax over the vocabulary to get the most likely next token id.

    # 0 : take the first item in the batch
    # -1: take the last token position in the sequence
    #  :: take all vocabulary scores for that position
    # it gives the model’s scores for the next token after the current prompt.
    last_logits = logits[0, -1, :]
    # greedy decoding: take the token with the highest score
    next_token_id = last_logits.argmax()
    # top-k
    # top_k = torch.topk(last_logits, k=10)
    # tokens = [tokenizer.decode(tk) for tk in top_k.indices]
    # print(tokens) # [' fence', ' edge', ' railing', ' wall', ' table', ' tree', ' top', ' counter', ' ground', ' side']

    return next_token_id

# with kv-cache returned
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values

# autoregressive loop:

def autogressive():
    generated_tokens = []
    next_inputs = inputs
    durations_s = []
    for _ in range(30):
        t0 = time.time()
        next_token_id = generate_token(next_inputs)
        durations_s += [time.time() - t0]
        
        next_inputs = {
            "input_ids": torch.cat(
                [next_inputs["input_ids"], next_token_id.reshape((1, 1))],
                dim=1),
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]])],
                dim=1),
        }
        
        next_token = tokenizer.decode(next_token_id)
        generated_tokens.append(next_token)

    print(f"{sum(durations_s)} s")
    print(generated_tokens)

def autogressive_kvcache():
    generated_tokens = []
    next_inputs = inputs
    durations_cached_s = []
    for _ in range(30):
        t0 = time.time()
        next_token_id, past_key_values = \
            generate_token_with_past(next_inputs)
        durations_cached_s += [time.time() - t0]
        
        next_inputs = {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]])],
                dim=1),
            "past_key_values": past_key_values,
        }
        
        next_token = tokenizer.decode(next_token_id)
        generated_tokens.append(next_token)

    print(f"{sum(durations_cached_s)} s")
    print(generated_tokens)



if __name__ == "__main__":
    # autogressive()
    autogressive_kvcache()