import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
import time, requests, subprocess, sys, signal

MODEL = "/data/models/Qwen3.5-4B"
PORT = 8199
cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--tensor-parallel-size", "1",
    "--kv-cache-dtype", "nvfp4",
    "--max-model-len", "32768",
    "--gpu-memory-utilization", "0.92",
    "--port", str(PORT),
    "--language-model-only",
    # NO --enforce-eager: testing CUDA graph compatibility
]
print("Starting NVFP4 KV + CUDA Graph test...")
log_file = open("/data/src/vllm/nvfp4_cg_log.txt", "w")
proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
deadline = time.time() + 180
ready = False
while time.time() < deadline:
    try:
        r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
        if r.status_code == 200:
            ready = True
            break
    except:
        pass
    if proc.poll() is not None:
        break
    time.sleep(2)

if not ready:
    log_file.flush()
    with open("/data/src/vllm/nvfp4_cg_log.txt") as f:
        lines = f.read().strip().split('\n')
    print("FAILED:")
    for l in lines[-20:]:
        print(l)
    proc.kill()
    sys.exit(1)

print("Server ready! Warmup...")
requests.post(f"http://localhost:{PORT}/v1/completions", json={
    "model": MODEL, "prompt": "Hello", "max_tokens": 8, "temperature": 0,
})
time.sleep(1)

print("Testing decode...")
t0 = time.time()
r = requests.post(f"http://localhost:{PORT}/v1/completions", json={
    "model": MODEL, "prompt": "The capital of France is", "max_tokens": 64, "temperature": 0,
})
dt = time.time() - t0
if r.status_code == 200:
    data = r.json()
    text = data["choices"][0]["text"]
    tokens = data["usage"]["completion_tokens"]
    print(f"Output: {text[:80]}")
    print(f"Decode: {tokens/dt:.1f} tok/s ({tokens} tokens in {dt:.2f}s)")
    print("SUCCESS - NVFP4 KV + CUDA Graph works!")
else:
    print(f"Request failed: {r.status_code} {r.text[:200]}")

proc.send_signal(signal.SIGINT)
proc.wait(timeout=10)
