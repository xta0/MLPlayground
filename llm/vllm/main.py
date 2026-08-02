"""
  Use vllm-metal:

  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  source ~/.venv-vllm-metal/bin/activate
  vllm serve Qwen/Qwen3-0.6B --dtype=float16 --max-model-len 4096

  Also change --dtype=bfloat16 to --dtype=float16; macOS vLLM CPU support only lists FP32/FP16, not BF16.

  Sources:

  - vLLM macOS CPU notes: https://docs.vllm.ai/en/stable/getting_started/installation/cpu/index.html
  - vLLM Metal install: https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/
"""

import warnings
warnings.filterwarnings("ignore")

import time, requests, json, os, math, sys

VLLM_URL = "http://localhost:8000"
os.makedirs("outputs", exist_ok=True)

print("Waiting for vLLM server...")
for attempt in range(60):
    try:
        r = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
        if r.status_code == 200:
            MODEL = r.json()["data"][0]["id"]
            break
    except requests.ConnectionError:
        pass
    time.sleep(5)
    if attempt % 6 == 5:
        print(f"  Still waiting... ({(attempt + 1) * 5}s elapsed)")
else:
    raise RuntimeError(
        "vLLM server not reachable after 5 minutes."
    )

print(f"Connected to {VLLM_URL} — model: {MODEL}")

# Your First Local LLM Reques
print("===============================================================\n")
print("Sending your first request to vLLM...\n")
print("===============================================================\n")

from openai import OpenAI
client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="unused")

start = time.time()
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", 
               "content": "What is PagedAttention in one sentence?"}],
    max_tokens=80,
    temperature=0.7,
    top_p=0.8,
    extra_body={"top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": False}},
)
elapsed = time.time() - start

print(f"Response ({elapsed:.2f}s, {resp.usage.completion_tokens} tokens):")
print(resp.choices[0].message.content)
print(f"\nUsage: {resp.usage.prompt_tokens} prompt + "
      f"{resp.usage.completion_tokens} completion = {resp.usage.total_tokens} total")

# Exploring Model Behavior
print("===============================================================\n")
print("Exploring model behavior with logprobs and top_logprobs...\n")
print("===============================================================\n")

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_tokens=15,
    temperature=0.0,
    logprobs=True,
    top_logprobs=5,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(f"Response: {resp.choices[0].message.content.strip()}\n")
print("Token-by-token probabilities:\n")

for tok in resp.choices[0].logprobs.content[:8]:
    print(f"  Chosen: '{tok.token}'  (logprob {tok.logprob:.2f})")
    if tok.top_logprobs:
        for alt in tok.top_logprobs[:5]:
            pct = math.exp(alt.logprob) * 100
            bar = "\u2588" * min(20, max(1, int(pct / 5)))
            print(f"    {pct:5.1f}%  {bar}  '{alt.token}'")
    print()

# Observing vLLM Under the Hood¶
print("===============================================================\n")
print("Observing vLLM metrics under the hood...\n")
print("===============================================================\n")

def get_vllm_metrics(base_url=VLLM_URL):
    """Scrape vLLM Prometheus /metrics and return {name: value}."""
    r = requests.get(f"{base_url}/metrics")
    metrics = {}
    for line in r.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split()[0]
        try:
            metrics[name] = float(line.split()[-1])
        except (ValueError, IndexError):
            continue
    return metrics

metrics = get_vllm_metrics()
print("Current vLLM Metrics:")
for key in ["vllm:num_requests_running", "vllm:num_requests_waiting",
            "vllm:gpu_cache_usage_perc", "vllm:cpu_cache_usage_perc",
            "vllm:prompt_tokens_total", "vllm:generation_tokens_total"]:
    if key in metrics:
        print(f"  {key.replace('vllm:', '')}: {metrics[key]:g}")

with open("outputs/metrics_snapshot.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nFull metrics saved to outputs/metrics_snapshot.json")


# Continous Batching and Concurrent Requests
print("===============================================================\n")
print("Sending concurrent requests to vLLM...\n")
print("===============================================================\n")

import concurrent.futures

prompts = [
    "What is quantization?",
    "Explain KV caching briefly.",
    "What is continuous batching?",
    "Why is LLM inference memory-bound?",
    "What is PagedAttention?",
]

def _ask(prompt):
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60, temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

before = get_vllm_metrics()
print(f"Sending {len(prompts)} concurrent requests...\n")
start = time.time()

with concurrent.futures.ThreadPoolExecutor(
    max_workers=len(prompts)) as pool:
    futures = {pool.submit(_ask, p): p for p in prompts}
    time.sleep(0.5)
    during = get_vllm_metrics()
    running = during.get("vllm:num_requests_running", "--")
    waiting = during.get("vllm:num_requests_waiting", "--")
    print(f"  [mid-flight]  running: {running}  |  waiting: {waiting}")

    for f in concurrent.futures.as_completed(futures):
        resp = f.result()
        print(f"  done: \"{futures[f][:40]}\" -> {resp.usage.completion_tokens} tokens")

elapsed = time.time() - start
after = get_vllm_metrics()
tokens = after.get("vllm:generation_tokens_total", 0) - before.get(
    "vllm:generation_tokens_total", 0)

print(f"\nAll {len(prompts)} completed in {elapsed:.2f}s")
if tokens > 0:
    print(f"Tokens generated: {tokens:g}  |  ~{tokens / elapsed:.1f} tokens/s")


# Prefix Caching and Shared System Prompts
print("===============================================================\n")
print("Sending multiple requests with the SAME system prompt...\n")
print("===============================================================\n")

SYSTEM_PROMPT = (
    "You are a helpful AI teaching assistant for a course on "
    "LLM optimization. You specialize in explaining concepts like "
    "quantization, inference optimization, and model serving. Keep "
    "answers concise -- one or two sentences."
)

questions = [
    "What is weight quantization?",
    "How does vLLM handle memory?",
    "What is continuous batching?",
    "Why use prefix caching?",
    "What is GPTQ?",
]

before = get_vllm_metrics()
timings = []
tok_counts = []

print("Sending 5 requests with the SAME system prompt...\n")
for i, q in enumerate(questions):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ],
        max_tokens=60, temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dt = time.time() - t0
    timings.append(dt)
    tok_counts.append(resp.usage.completion_tokens)
    tokens = resp.usage.completion_tokens
    ms_per_tok = (dt / tokens * 1000) if tokens > 0 else 0.0
    print(f"  [{i+1}] {q:<35} {dt:.2f}s  ({tokens} tok, {ms_per_tok:.0f} ms/tok)")
    
after = get_vllm_metrics()

prefix_before = before.get("vllm:prefix_cache_queries_total", 0)
prefix_after = after.get("vllm:prefix_cache_queries_total", 0)

print(f"\nPrefix cache queries: {prefix_before:g} -> {prefix_after:g}  (+{prefix_after - prefix_before:g})")

cache_keys = [k for k in after if "prefix" in k.lower() 
              or "cache_hit" in k.lower()]
for k in sorted(cache_keys):
    b, a = before.get(k, 0), after.get(k, 0)
    if a != b and k != "vllm:prefix_cache_queries_total":
        print(f"  {k}: {b:g} -> {a:g}")

print("\n The increasing prefix_cache_queries count confirms vLLM is ")
print("checking and reusing cached KV blocks for the shared system prompt.")

# KV Cache Size Estimation
print("===============================================================\n")
print("Estimating KV cache size for Qwen3-0.6B...\n")
print("===============================================================\n")

num_layers = 28
num_kv_heads = 8     # GQA: 16 Q heads, 8 KV heads
head_dim = 128
dtype_bytes = 2      # BF16

per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

print(f"KV Cache -- Qwen3-0.6B")
print(f"Per token: 2 x {num_layers} x {num_kv_heads} x {head_dim} x {dtype_bytes}"
      f" = {per_token:,} bytes ({per_token // 1024} KB)\n")
print(f"  {'Context':>8}  {'KV Cache':>10}")
print(f"  {'-'*8}  {'-'*10}")
for ctx in [1, 64, 256, 1024, 4096]:
    size = per_token * ctx
    label = f"{size / 1024:.0f} KB" if size < 1024**2 else f"{size / 1024**2:.0f} MB"
    print(f"  {ctx:>7}t  {label:>10}")

print(f"\n  10 concurrent x 4096 ctx = {per_token * 4096 * 10 / 1024**3:.1f} GB")

# Thinking Mode
print("===============================================================\n")
print("Exploring Thinking Mode...\n")
print("===============================================================\n")
prompt = "What makes continuous batching better than static batching?"

for label, thinking, max_tok in [
    ("Thinking OFF", False, 80), ("Thinking ON", True, 200)]:
    print(f"=== {label} ===\n")
    start = time.time()
    tokens = 0
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tok, temperature=0.7, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            sys.stdout.write(chunk.choices[0].delta.content)
            sys.stdout.flush()
            tokens += 1
    elapsed = time.time() - start
    print(f"\n  [{tokens} tokens, {elapsed:.2f}s]\n")