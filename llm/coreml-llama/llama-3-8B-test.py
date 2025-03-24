# pip install transformers==4.44.2
# pip install coremltools numpy

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer


# https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM

class BaselineLlamaForCausalLM(LlamaForCausalLM):
    """Baseline LlamaForCausalLM model without key/value caching."""
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        out = super().forward(
            input_ids,
            attention_mask,
            use_cache=False,
        )
        return out.logits

model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()

# Count the attention blocks
attention_blocks = [module for module in torch_model.modules() if "attention" in str(type(module)).lower()]
print(f"Number of attention blocks: {len(attention_blocks)}")

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)

prompt = "What is GenAI?"
inputs = tokenizer(prompt, return_tensors='pt')
print("inputs: ", inputs)

tokens = tokenizer.tokenize(prompt)
print(tokens)  # Displays the tokens before they are converted to IDs

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)  # Displays the token IDs corresponding to each token

# Extract input_ids and attention_mask
input_ids = inputs["input_ids"]
print("input_ids.shape: ", input_ids.shape)
attention_mask = inputs["attention_mask"]
print("attention_mask.shape: ", input_ids.shape)

max_new_tokens = 100
generated_ids = input_ids.clone()

for i in range(max_new_tokens):
    print(f"Step {i}")
    # Forward pass
    logits = torch_model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
    print("logits: ", logits.shape)
    # torch.Size([1, 6, 32000])
    # (batch, The sequence length of the input, vocab size)

    # Get the last token's logits
    next_token_logits = logits[:, -1, :]
    print("next_token_logits: ", next_token_logits.shape)
    # Greedy decoding: pick the token with the highest probability
    next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0)
    print("next_token_id: ", next_token_id.shape)

    # Decode the token ID to string
    next_token_str = tokenizer.decode(next_token_id[0].item())
    print(f"Decoded next token: {next_token_str}")
    
    # Append next token to the generated_ids
    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
    print("generated_ids: ", generated_ids.shape)
    # Decode the entire sequence back into text
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(output_text)
    
    # If the model has an EOS token and we encounter it, break
    if next_token_id.item() == tokenizer.eos_token_id:
        break

# Decode the entire sequence back into text
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output_text)
