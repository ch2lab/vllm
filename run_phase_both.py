import subprocess
import sys

for tag, dtype, mem in (("fp8", "fp8", 0.92), ("nvfp4", "nvfp4", 0.90)):
    log = open(f"/tmp/e2e_phase_{tag}.log", "w")
    p = subprocess.run(
        [sys.executable, "/data/src/vllm/test_27b_phase.py", dtype, str(mem)],
        stdout=log, stderr=subprocess.STDOUT,
    )
    log.write(f"EXIT={p.returncode}\n")
    log.close()
    print(f"{tag}: EXIT={p.returncode}")
