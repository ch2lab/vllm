#!/bin/bash
# nsys profile 16K decode. Log to /tmp/nsys_decode16k.log
cd /data/src/vllm
nsys profile -o /tmp/decode16k --trace=cuda --cuda-graph-trace=node -s none \
    timeout 900 python3 test_27b_mtp_longctx.py fp8 16384 16000 0.92 \
    > /tmp/nsys_decode16k.log 2>&1
echo "EXIT=$?" > /tmp/nsys_decode16k_exit.txt