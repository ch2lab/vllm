import subprocess, sys

with open("nsys_new.log", "w") as f:
    r = subprocess.run(
        ["nsys", "profile", "--cuda-memory-usage=false", "-o", "nsys_eager_new",
         "--force-overwrite", "true", sys.executable, "nsys_eager_short.py"],
        stdout=f, stderr=subprocess.STDOUT,
    )
    f.write(f"EXIT={r.returncode}\n")
