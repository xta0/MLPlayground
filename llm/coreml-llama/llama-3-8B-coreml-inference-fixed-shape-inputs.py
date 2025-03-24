import coremltools as ct
from transformers import AutoTokenizer
import numpy as np

# Load the model
mlmodel = ct.models.MLModel("./Llama-3.1-8B-Instruct-quant.mlpackage")
print(mlmodel.get_spec().WhichOneof("Type"))
# print(mlmodel)
# Get the model spec and inspect the program code
spec = mlmodel.get_spec()
# print(spec)
print(spec.description.input)
print(spec.description.output)
program = spec.mlProgram

# inference
# Tokenize a query
query = "What is GenAI?"
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# setup the tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = "<pad>"  # Assign a new padding token if not set
tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.eos_token = "</s>"  # Replace with the actual EOS token for Llama2

# Ensure EOS token is set
if tokenizer.eos_token_id is None:
    tokenizer.eos_token = "</s>"
    tokenizer.add_special_tokens({'eos_token': "</s>"})
print("EOS Token ID:", tokenizer.eos_token_id)


# Prepare initial input (Prompt stage)
input_text = "What is GenAI?"  # Example input
context_size = 1024  # Defined by the model's maximum context size
inputs = tokenizer(
    input_text, 
    return_tensors="np", 
    padding="max_length", 
    max_length=1024, 
    truncation=True
)
input_ids = inputs["input_ids"]  # Shape: (1, max_length)
attention_mask = inputs["attention_mask"]  # Shape: (1, max_length)

# Keep only non-zero tokens for initial prompt
prompt_length = (attention_mask > 0).sum()
input_ids = input_ids[:, :prompt_length]
print("input_ids.shape: ", input_ids.shape)
attention_mask = attention_mask[:, :prompt_length]
print("input_ids.shape: ", attention_mask.shape)


# Extend padding to context_size (e.g., 1024)
input_ids = np.pad(
    input_ids, 
    ((0, 0), (0, context_size - input_ids.shape[1])), 
    constant_values=0)

attention_mask = np.pad(
    attention_mask, 
    ((0, 0), (0, context_size - attention_mask.shape[1])), 
    constant_values=0)

input_ids = input_ids.astype(np.int32)
attention_mask = attention_mask.astype(np.int32)

assert input_ids.shape == (1, context_size), f"inputIds shape mismatch: {input_ids.shape}"
assert attention_mask.shape == (1, context_size), f"attentionMask shape mismatch: {attention_mask.shape}"

max_new_tokens = 100
# Generation loop
for i in range(max_new_tokens):
    # Model prediction
    coreml_inputs = {
        "inputIds": np.ascontiguousarray(input_ids),
        "attentionMask": np.ascontiguousarray(attention_mask)
    }
    try:
        coreml_outputs = mlmodel.predict(coreml_inputs)
    except RuntimeError as e:
        print(f"Core ML prediction failed: {e}")
        break
    logits = coreml_outputs["logits"]

    # Select next token using greedy sampling
    next_token_id = np.argmax(logits[0, prompt_length - 1, :])

    # Check for EOS token
    if next_token_id == tokenizer.eos_token_id:
        print("End-of-sequence token encountered. Stopping generation.")
        break

    # Append the new token to inputIds and update attentionMask
    input_ids[0, prompt_length] = next_token_id # equivalent to input_ids[0][prompt_length] = next_token_id, but faster
    attention_mask[0, prompt_length] = 1
    prompt_length += 1

    # Decode and print the generated text so far
    generated_text = tokenizer.decode(input_ids[0, :prompt_length], skip_special_tokens=True)
    print(f"Step {i + 1}, Generated Text So Far: {generated_text}")

    # Stop if context size is reached
    if prompt_length >= context_size:
        print("Maximum context size reached. Stopping generation.")
        break


"""
Results from the non-quantized model:

What is GenAI? GenAI stands for General Artificial Intelligence, 
which refers to a hypothetical AI system that possesses the ability to understand, 
learn, and apply knowledge across a wide range of tasks,
similar to human intelligence. 
Unlike narrow or specialized AI, which is designed to perform a specific task, 
GenAI aims to replicate human-like intelligence and cognitive abilities. 
This includes capabilities such as reasoning, problem-solving, learning, and decision-making, across various domains and tasks. 
GenAI is still a topic of ongoing research and development,
"""

"""
Results from the quantized model:

What is GenAI? GenAI is a type of artificial intelligence that uses a combination of machine learning and natural language processing to generate human-like text,
images, and other forms of content. 
It is also known as Generative AI or Generative Adversarial Networks (GANs). 
GenAI is a rapidly evolving field that has the potential to revolutionize the way we create and interact with digital content.
How does GenAI work? GenAI works by using complex algorithms to analyze and learn from large datasets, and then generate
"""