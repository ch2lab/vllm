#!/usr/bin/env python3
"""NVFP4 test with fresh Triton cache and long timeout."""
import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_nvfp4_fresh"
import time, requests, subprocess, sys, signal

MODEL = "/data/models/Qwen3.5-4B"
PORT = 8199

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--tensor-parallel-size", "1",
    "--kv-cache-dtype", "nvfp4",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.85",
    "--port", str(PORT),
    "--language-model-only",
    "--enforce-eager",
]
log = open("/data/src/vllm/fresh_cache_log.txt", "w")
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
print(f"Server PID: {proc.pid}, waiting up to 600s...")
deadline = time.time() + 600
while time.time() < deadline:
    try:
        r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
        if r.status_code == 200:
            print(f"Server ready after {600 - (deadline - time.time()):.0f}s")
            break
    except:
        pass
    if proc.poll() is not None:
        print(f"Server DIED (exit={proc.returncode})")
        sys.exit(1)
    time.sleep(5)
else:
    print("TIMEOUT waiting for server")
    proc.kill()
    sys.exit(1)

time.sleep(2)
# warmup
requests.post(f"http://localhost:{PORT}/v1/completions", json={
    "model": MODEL, "prompt": "Hello", "max_tokens": 8, "temperature": 0,
}, timeout=120)

prompts = ["The capital of France is", "def fibonacci(n):\n    "]
for p in prompts:
    r = requests.post(f"http://localhost:{PORT}/v1/completions", json={
        "model": MODEL, "prompt": p, "max_tokens": 64, "temperature": 0,
    }, timeout=120)
    out = r.json()["choices"][0]["text"] if r.status_code == 200 else f"ERR:{r.status_code}"
    print(f"  {p!r} -> {out[:100]!r}")

proc.send_signal(signal.SIGINT)
try:
    proc.wait(timeout=10)
except:
    proc.kill()
    proc.wait()
