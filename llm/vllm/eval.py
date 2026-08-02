import warnings
warnings.filterwarnings("ignore")
import subprocess
import time, requests, json, os, glob, sys
from openai import OpenAI

VLLM_URL = "http://localhost:8000"

for _ in range(12):
    try:
        r = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
        if r.status_code == 200:
            MODEL = r.json()["data"][0]["id"]
            break
    except requests.ConnectionError:
        time.sleep(5)
else:
    raise RuntimeError("vLLM server not reachable.")

print(f"Connected to {VLLM_URL} — model: {MODEL}")

client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="unused")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", 
               "content": "What is model quantization in one sentence."}],
    max_tokens=30, temperature=0.7,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(f"{MODEL}: {resp.choices[0].message.content.strip()}")

print("================================================================")
print("Benchmarking with GuideLLM...")
print("================================================================")

os.makedirs("outputs", exist_ok=True)


cmd = [
    "guidellm", "run",
    "--backend", f"kind=openai_http,target={VLLM_URL},model={MODEL}",
    "--profile", "kind=synchronous",
    "--constraint", "kind=max_requests,count=10",
    "--tokenizer", "kind=huggingface_auto,model=Qwen/Qwen3-0.6B",
    "--data", "kind=synthetic_text,prompt_tokens=32,output_tokens=16",
    "--output", "kind=console",
    "--output", "kind=json,path=./outputs/benchmarks.json",
]

print(f"Running: {' '.join(cmd)}\n")

result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

if result.returncode == 0:
    print("Benchmark complete!")
    tail = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
    print(tail)
else:
    print(f"GuideLLM exited with code {result.returncode}")
    print(f"STDOUT:\n{result.stdout[-1000:]}")
    print(f"STDERR:\n{result.stderr[-1000:]}")
    sys.exit(result.returncode)

with open("outputs/benchmarks.json") as f:
    report = json.load(f)

bench = report["benchmarks"][0]
metrics = bench["metrics"]
n_requests = metrics["request_totals"]["successful"]

profile_type = bench.get("type") 
print(f"Profile: {profile_type}  |  Requests: {n_requests}\n")

display_metrics = {
    "TTFT (ms)":       "time_to_first_token_ms",
    "ITL (ms)":        "inter_token_latency_ms",
    "E2E Latency (s)": "request_latency",
    "Output tokens":   "output_token_count",
}

print(f"{'Metric':<20} {'Mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
print("-" * 55)
for label, key in display_metrics.items():
    dist = metrics[key]["successful"]
    p = dist["percentiles"]
    print(f"{label:<20} {dist['mean']:>8.2f} {p['p50']:>8.2f} "
          f"{p['p95']:>8.2f} {p['p99']:>8.2f}")

throughput = metrics["output_tokens_per_second"]["successful"]
req_rate = metrics["requests_per_second"]["successful"]
print(f"\nThroughput: {req_rate['mean']:.2f} req/s  |  "
      f"{throughput['mean']:.1f} output tokens/s")



print("================================================================")
## Evaluating Model Quality with lm_eval
print("Evaluating model quality with lm_eval...")
print("================================================================")
import lm_eval

os.environ.setdefault("OPENAI_API_KEY", "unused")

TASK = "gsm8k"
LIMIT = 10
print(f"Running lm_eval on {MODEL} via vLLM server ({TASK}, {LIMIT} examples)...\n")

results = lm_eval.simple_evaluate(
    model="local-chat-completions",
    model_args=(
        f"model={MODEL},"
        f"base_url={VLLM_URL}/v1/chat/completions,"
        "tokenized_requests=False,"
        "tokenizer_backend=huggingface,"
        "tokenizer=Qwen/Qwen3-0.6B,"
        "num_concurrent=1,"
        "batch_size=1"
    ),
    tasks=[TASK],
    limit=LIMIT,
    apply_chat_template=True,
    fewshot_as_multiturn=False,
    gen_kwargs={"temperature": 0.0, "do_sample": False},
)
task_results = results["results"][TASK]

print(f"Model: {MODEL}")
print(f"Task: {TASK}  |  Examples: {LIMIT}\n")
for metric, value in task_results.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.4f}")