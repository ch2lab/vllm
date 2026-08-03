import subprocess, sys

with open("e2e_fp8_rerun.log", "w") as f:
    r = subprocess.run([sys.executable, "test_27b_mtp_fp8.py"], stdout=f, stderr=subprocess.STDOUT)
    f.write(f"EXIT={r.returncode}\n")
