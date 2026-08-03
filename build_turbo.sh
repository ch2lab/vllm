#!/bin/bash
# Wrapper: build _sm70_turbomind_C, log to /tmp/build_turbo.log
pkill -f nvcc; pkill -f ninja; sleep 1
cd /data/src/vllm
PATH="/data/vllm-dev/bin:$PATH" TORCH_CUDA_ARCH_LIST="7.0" \
    ninja -C build-sm70 -j32 _sm70_turbomind_C > /tmp/build_turbo.log 2>&1
RC=$?
if [ -f "build-sm70/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so" ]; then
    cp "build-sm70/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so" \
       "vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so"
    echo "=== Installed vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so ===" >> /tmp/build_turbo.log
fi
echo "EXIT=$RC" > /tmp/build_turbo_exit.txt