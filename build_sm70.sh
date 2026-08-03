#!/bin/bash
# SM70 增量编译脚本
# 用法: ./build_sm70.sh [target]
# 默认只编译 _C_stable_libtorch（~30秒 with ccache）
#
# 不会触发 cu130 覆盖（不使用 uv pip install）

set -e
cd /data/src/vllm

TARGET="${1:-_C_stable_libtorch}"
BUILD_DIR="build-sm70"

# 首次运行需要 configure
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    echo "=== First run: configuring cmake ==="
    PATH="/data/vllm-dev/bin:$PATH" TORCH_CUDA_ARCH_LIST="7.0" \
    cmake -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DVLLM_TARGET_DEVICE=cuda \
        -DVLLM_PYTHON_EXECUTABLE=/data/vllm-dev/bin/python \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DFETCHCONTENT_BASE_DIR=/data/src/vllm/.deps \
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON
fi

echo "=== Building target: $TARGET ==="
# 直接用 ninja 避免 cmake --build 触发重新 configure（需要网络）
PATH="/data/vllm-dev/bin:$PATH" TORCH_CUDA_ARCH_LIST="7.0" \
    ninja -C "$BUILD_DIR" -j32 "$TARGET"

# 复制 .so 到 vllm 包目录
if [ -f "$BUILD_DIR/${TARGET}.abi3.so" ]; then
    cp "$BUILD_DIR/${TARGET}.abi3.so" "vllm/${TARGET}.abi3.so"
    echo "=== Installed: vllm/${TARGET}.abi3.so ==="
fi

# 验证 torch 未被覆盖
TORCH_VER=$(/data/vllm-dev/bin/python -c "import torch; print(torch.version.cuda)")
if [ "$TORCH_VER" != "12.8" ]; then
    echo "WARNING: torch CUDA version is $TORCH_VER, restoring cu128..."
    /data/vllm-dev/bin/pip install \
        /data/src/pytorch-src/dist/torch-2.13.0a0+gitcf30153-cp312-cp312-linux_x86_64.whl \
        --force-reinstall --no-deps -q
fi

echo "=== Done ==="
