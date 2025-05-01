from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from utilities import *

def inference(text, model, tokenizer, max_input_tokens=512, max_output_tokens=512):
  # Tokenize
  inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=max_input_tokens,
    padding=True          # makes the mask
  )

  input_ids      = inputs["input_ids"].to(model.device)
  attention_mask = inputs["attention_mask"].to(model.device)
  
  generated = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,        # ← fixes warning #1
    max_new_tokens=max_output_tokens,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,

    # anti-repetition knobs
    # do_sample=True,                # turn on sampling
    # temperature=0.7,               # soften the distribution
    # top_p=0.9,                     # nucleus sampling
    repetition_penalty=1.1,        # discourage loops
    # no_repeat_ngram_size=3         # ban 3-token repeats
  )
  # Decode and strip the prompt
  full_text = tokenizer.batch_decode(generated,
                                       skip_special_tokens=True)[0]
  return full_text[len(text):]

CAHCE_DIR = "/Volumes/ai-1t/llm"

print("==============================")
print("load the docs dataset")
print("==============================")

dataset_path = "lamini_docs.jsonl"
use_hf = False # use huggingface datasets

model_name = "EleutherAI/pythia-410m"

training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 512
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir = CAHCE_DIR
)
tokenizer.pad_token = tokenizer.eos_token

train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir = CAHCE_DIR
)

print("Device: ", base_model.device)

print("==============================")
print("Try the base model")
print("==============================")

test_text = test_dataset[0]['question']
print("Question input (test):", test_text)
print(f"Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))

print("==============================")
print("Try the fine-tuned model")
print("==============================") 

save_dir = f'lamini_docs_30_steps/final'

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)

test_question = test_dataset[0]['question']

print("==============================")
print("Question input (test):", test_question)
print("==============================")

print("==============================")
print("Finetuned slightly model's answer: ")
print("==============================")
print(inference(test_question, finetuned_slightly_model, tokenizer))

test_answer = test_dataset[0]['answer']
print("==============================")
print("Target answer output (test):", test_answer)
print("==============================")