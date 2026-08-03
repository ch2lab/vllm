#!/usr/bin/env python3
"""Compare NVFP4 vs FP16 KV-cache output quality."""
import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
import time, requests, subprocess, sys, signal

MODEL = "/data/models/Qwen3.5-4B"
PORT = 8199
PROMPT = "The capital of France is"
MAX_TOKENS = 64

def run_server(kv_dtype, extra_args=None):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--tensor-parallel-size", "1",
        "--kv-cache-dtype", kv_dtype,
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--port", str(PORT),
        "--language-model-only",
    ]
    if extra_args:
        cmd += extra_args
    log = open(f"/data/src/vllm/correctness_{kv_dtype}_log.txt", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                return proc
        except:
            pass
        if proc.poll() is not None:
            break
        time.sleep(2)
    proc.kill()
    return None

def generate(prompt, max_tokens, temperature=0):
    r = requests.post(f"http://localhost:{PORT}/v1/completions", json={
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature,
    }, timeout=120)
    if r.status_code == 200:
        return r.json()["choices"][0]["text"]
    return f"ERROR: {r.status_code}"

def stop_server(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()
        proc.wait()
    time.sleep(3)

# Test prompts
PROMPTS = [
    "The capital of France is",
    "Write a haiku about mountains:",
    "def fibonacci(n):\n    ",
    "The meaning of life is",
]

print("=" * 60)
print("NVFP4 Correctness Test: FP16 vs NVFP4 KV-cache")
print("=" * 60)

# Run FP16 baseline
print("\n[1/2] Starting FP16 server...")
proc = run_server("auto")
if not proc:
    print("FAILED to start FP16 server")
    sys.exit(1)
time.sleep(2)
# warmup
generate("Hello", 8)
fp16_outputs = []
for p in PROMPTS:
    out = generate(p, MAX_TOKENS)
    fp16_outputs.append(out)
    print(f"  FP16: {p!r} -> {out[:80]!r}")
stop_server(proc)

# Run NVFP4
print("\n[2/2] Starting NVFP4 server...")
proc = run_server("nvfp4")
if not proc:
    print("FAILED to start NVFP4 server")
    sys.exit(1)
time.sleep(2)
# warmup
generate("Hello", 8)
nvfp4_outputs = []
for p in PROMPTS:
    out = generate(p, MAX_TOKENS)
    nvfp4_outputs.append(out)
    print(f"  NVFP4: {p!r} -> {out[:80]!r}")
stop_server(proc)

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
match_count = 0
for i, p in enumerate(PROMPTS):
    fp16_first = fp16_outputs[i][:40]
    nvfp4_first = nvfp4_outputs[i][:40]
    match = fp16_first == nvfp4_first
    if match:
        match_count += 1
    status = "MATCH" if match else "DIFF"
    print(f"\n  [{status}] {p!r}")
    if not match:
        print(f"    FP16:  {fp16_outputs[i][:100]!r}")
        print(f"    NVFP4: {nvfp4_outputs[i][:100]!r}")

print(f"\n{match_count}/{len(PROMPTS)} exact prefix matches")
if match_count >= len(PROMPTS) // 2:
    print("PASS: NVFP4 produces reasonable output")
else:
    print("WARN: NVFP4 output differs significantly from FP16")
