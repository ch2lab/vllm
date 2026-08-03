#!/usr/bin/env python3
"""Quick NVFP4 correctness test with --enforce-eager (no CUDA graphs)."""
import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
import time, requests, subprocess, sys, signal

MODEL = "/data/models/Qwen3.5-4B"
PORT = 8199
PROMPT = "The capital of France is"

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
        "--enforce-eager",
    ]
    if extra_args:
        cmd += extra_args
    log = open(f"/data/src/vllm/eager_{kv_dtype}_log.txt", "w")
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

def generate(prompt, max_tokens=64, temperature=0):
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

print("Testing NVFP4 with --enforce-eager (no CUDA graphs)")
print("=" * 60)

proc = run_server("nvfp4")
if not proc:
    print("FAILED to start server")
    sys.exit(1)

time.sleep(2)
# warmup
generate("Hello", 8)

prompts = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
]
for p in prompts:
    out = generate(p)
    print(f"  {p!r} -> {out[:100]!r}")

stop_server(proc)
