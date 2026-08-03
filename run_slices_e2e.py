import subprocess
import sys

log = open("/tmp/e2e_slices20.log", "w")
p = subprocess.run(
    [sys.executable, "/data/src/vllm/test_27b_mtp_long_fp16.py"],
    stdout=log, stderr=subprocess.STDOUT,
)
log.write(f"EXIT={p.returncode}\n")
log.close()
