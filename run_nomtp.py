import subprocess
import sys

log = open("/tmp/nomtp_measure.log", "w")
p = subprocess.run(
    [sys.executable, "/data/src/vllm/test_27b_nomtp.py"],
    stdout=log, stderr=subprocess.STDOUT,
)
log.write(f"EXIT={p.returncode}\n")
log.close()
