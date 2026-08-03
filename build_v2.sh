#!/bin/bash
# Wrapper: build _C_stable_libtorch (fused AR), log to /tmp/build_ar.log
pkill -f nvcc; pkill -f ninja; sleep 1
cd /data/src/vllm
bash build_sm70.sh _C_stable_libtorch > /tmp/build_ar.log 2>&1
echo "EXIT=$?" > /tmp/build_ar_exit.txt