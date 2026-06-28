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
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden_state = out.hidden_states[-1]  # <-- last 4 layers' outputs
        return out.logits, last_hidden_state

model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()
print("model arch:", torch_model)

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
print(attention_mask)


max_new_tokens = 1
generated_ids = input_ids.clone()

logits, last_hidden_state = torch_model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
print("last hidden state: ", last_hidden_state.shape)
print("logits: ", logits.shape)

calculated_logits = torch_model.lm_head(last_hidden_state)
print("lm_head(logits):", calculated_logits.shape)
