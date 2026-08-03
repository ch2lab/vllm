import os
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"  # simulate worst case
import time, requests, subprocess, sys, signal

MODEL = "/data/models/Qwen3.5-4B"
PORT = 8199
cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--tensor-parallel-size", "1",
    "--kv-cache-dtype", "fp8",
    "--max-model-len", "32768",
    "--gpu-memory-utilization", "0.92",
    "--port", str(PORT),
    "--language-model-only",
]
print(f"Starting server...")
log_file = open("/data/src/vllm/fp8kv_cg_log.txt", "w")
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
    print("FAILED to start server")
    with open("/data/src/vllm/fp8kv_cg_log.txt") as f:
        lines = f.read().strip().split('\n')
    for l in lines[-40:]:
        print(l)
    proc.kill()
    sys.exit(1)

print("Server ready! Testing decode...")
r = requests.post(f"http://localhost:{PORT}/v1/completions", json={
    "model": MODEL,
    "prompt": "The capital of France is",
    "max_tokens": 64,
    "temperature": 0,
})
if r.status_code == 200:
    data = r.json()
    text = data["choices"][0]["text"]
    usage = data["usage"]
    print(f"Output: {text[:100]}")
    print(f"Tokens: {usage['completion_tokens']}")
    print("SUCCESS - FP8 KV + CUDA Graph works on 4B TP1")
else:
    print(f"Request failed: {r.status_code} {r.text[:200]}")

proc.send_signal(signal.SIGINT)
proc.wait(timeout=10)
