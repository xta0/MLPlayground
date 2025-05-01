"""
FYI: The transformers library has two types of model classes: 
AutoModelForCausalLM and AutoModelForMaskedLM. 

Causal language models represent the decoder-only models that are used for text generation. 
They are described as causal, because to predict the next token, 
the model can only attend to the preceding left tokens.

Masked language models represent the encoder-only models that are used for rich text representation. 
They are described as masked, because they are trained to predict a masked or hidden token in a sequence.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

cache_dir = "/Volumes/ai-1t/microsoft" 

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    cache_dir=cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    cache_dir=cache_dir,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)


# inspect the model architecture
print(model)

# The vocabulary size is 32064 tokens, and the size of the vector embedding for each token is 3072.
print("embedding model:")
print(model.model.embed_tokens)

# There are 32 transformer blocks or layers. You can access any particular block.
# print(model.model.layers[0])

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # False means to not include the prompt text in the returned text
    max_new_tokens=50, 
    do_sample=False, # no randomness in the generated text
)

# prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. "

# output = generator(prompt)

# print(output[0]['generated_text'])


prompt = "The capital of France is"
# Get the output of the model before the lm_head
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print("tokens: ", input_ids)

# embeddings

embeddings = model.model.embed_tokens(input_ids)
print("embeddings: ", embeddings.shape)

# attention modules
att1 = model.model.layers[0].self_attn.o_proj(embeddings)
print("att1: ", att1.shape)


# Let's now pass the token ids to the transformer block (before the LM head).
model_output = model.model(input_ids).last_hidden_state
print("model_output: ", type(model_output))

# The transformer block outputs for each token a vector of size 3072 (embedding size). 
# Get the shape the output the model before the lm_head

print("Transformer block's output: ", model_output.shape)

# Get the output of the lm_head
lm_head_output = model.lm_head(model_output)

print("lm_head: ", lm_head_output.shape)

"""
The LM head outputs for each token in the input prompt, a vector of size 32064 (vocabulary size). So there are 5 vectors, each of size 32064. Each vector can be mapped to a probability distribution, that shows the probability for each token in the vocabulary to come after the given token in the input prompt.

Since we're interested in generating the output token that comes after the last token in the input prompt ("is"), we'll focus on the last vector. So in the next cell, lm_head_output[0,-1] is a vector of size 32064 from which you can generate the token that comes after ("is"). You can do that by finding the id of the token that corresponds to the highest value in the vector lm_head_output[0,-1] (using argmax(-1), -1 means across the last axis here).

"""

token_id = lm_head_output[0,-1].argmax(-1)
print(token_id)

# Finally, let's decode the returned token id.
print(tokenizer.decode(token_id))

