import warnings
warnings.filterwarnings("ignore")
import os, sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if sys.version_info >= (3, 14):
    raise RuntimeError(
        "llmcompressor 0.12.0 currently fails to import on Python 3.14. "
        "Run this script with Python 3.12 or 3.13 instead."
    )

import gc, math, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils.metric_logging import CompressionLogger
from llmcompressor import oneshot

MODEL_DIR = "./models/Qwen3-0.6B"
OUTPUT_DIR = "./models/Qwen3-0.6B-W4A16"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

DEVICE = "mps" if hasattr(torch, "mps") and torch.mps.is_available() else "cpu"
# DEVICE = "cpu"
MODEL_DTYPE = torch.float16 if DEVICE == "mps" else torch.bfloat16
# MODEL_DTYPE = torch.bfloat16

if DEVICE == "mps":
    _compression_logger_exit = CompressionLogger.__exit__

    def _mps_safe_compression_logger_exit(self, exc_type, exc_val, exc_tb):
        try:
            return _compression_logger_exit(self, exc_type, exc_val, exc_tb)
        except RuntimeError as err:
            if "Allocator for mps is not a DeviceAllocator" in str(err):
                return False
            raise

    CompressionLogger.__exit__ = _mps_safe_compression_logger_exit

print(f"Base model:      {MODEL_DIR}")
print(f"Quantized model: {OUTPUT_DIR}")
print(f"Device:          {DEVICE}")

"""
Parameter	Value	Why
scheme	W4A16	4-bit weights
targets	Linear	Linear layers hold most parameters - biggest savings
ignore	["lm_head"]	Output layer maps to vocabulary - keep it precise
"""

recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
)

## Quantization
print("========================================================")
if not os.path.isdir(OUTPUT_DIR):
    oneshot(
        model=MODEL_DIR,
        dataset=DATASET_ID,
        dataset_config_name=DATASET_CONFIG,
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        sequential_offload_device=DEVICE,
        max_seq_length=4096,
        num_calibration_samples=256,
    )
    print(f"Quantization complete. Model saved to: {OUTPUT_DIR}")

def folder_size(path):
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def format_size(nbytes):
    if nbytes < 1024**2:
        return f"{nbytes/1024:.1f} KB"
    if nbytes < 1024**3:
        return f"{nbytes/1024**2:.1f} MB"
    return f"{nbytes/1024**3:.2f} GB"

size_orig = folder_size(MODEL_DIR)
size_q = folder_size(OUTPUT_DIR)
reduction = (1 - size_q / size_orig) * 100 if size_orig > 0 else 0

print("Model Size Comparison")
print("=" * 45)
print(f"Original (BF16):    {format_size(size_orig)}")
print(f"Quantized (W4A16):  {format_size(size_q)}")
print(f"Reduction:          {reduction:.0f}%")

## Test Both Models
print("========================================================")
prompt = "Machine learning is a branch of"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, device_map=DEVICE, dtype=MODEL_DTYPE,
)

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
outputs = base_model.generate(
    **inputs, 
    max_new_tokens=60, 
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
generated = outputs[0][inputs["input_ids"].shape[-1]:]

print(f"Base Model ({MODEL_DIR})")
print(f"Prompt: {prompt}")
print(f"Response: {tokenizer.decode(generated, skip_special_tokens=True)}")

import logging
logging.getLogger("llmcompressor").setLevel(logging.WARNING)

quant_model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR, device_map="cpu", dtype=torch.bfloat16,
)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = quant_model.generate(
    **inputs, 
    max_new_tokens=60, 
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
generated = outputs[0][inputs["input_ids"].shape[-1]:]

print(f"Quantized Model ({OUTPUT_DIR})")
print(f"Prompt: {prompt}")
print(f"Response: {tokenizer.decode(generated, skip_special_tokens=True)}")

# Perplexity Comparision
print("========================================================")

"""
Side-by-side text gives intuition, but perplexity is the standard metric: 
it measures how well the model predicts text. Lower is better. 
If quantization has degraded the model, its perplexity will be noticeably higher.
"""

from datasets import load_dataset

def calculate_perplexity(model, tokenizer, dataset, max_tokens=5000, stride=512):
    encodings = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt", truncation=True, max_length=max_tokens,
    )
    input_ids = encodings.input_ids
    nlls, prev_end = [], 0

    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc = min(begin_loc + stride, input_ids.size(1))
        trg_len = end_loc - prev_end
        input_slice = input_ids[:, begin_loc:end_loc]
        target_slice = input_slice.clone()
        target_slice[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(input_slice, labels=target_slice).loss
            nlls.append(loss * trg_len)
        prev_end = end_loc

    return math.exp(torch.stack(nlls).sum() / prev_end)

test_data = load_dataset(DATASET_ID, DATASET_CONFIG, split="test")
print(f"Loaded {len(test_data)} test samples")

quant_ppl = calculate_perplexity(quant_model, tokenizer, test_data)
print(f"Quantized perplexity: {quant_ppl:.2f}")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, device_map="cpu", dtype=torch.bfloat16,
)
base_ppl = calculate_perplexity(base_model, tokenizer, test_data)
print(f"Base perplexity: {base_ppl:.2f}")

print("Perplexity Comparison")
print("=" * 40)
print(f"Base (BF16):      {base_ppl:.2f}")
print(f"Quantized (W4A16): {quant_ppl:.2f}")
print(f"Difference:       {quant_ppl - base_ppl:+.2f} ({(quant_ppl/base_ppl - 1)*100:+.1f}%)")
print(f"\nA small increase in perplexity is expected — the quantized layers use 4-bit weights.")
