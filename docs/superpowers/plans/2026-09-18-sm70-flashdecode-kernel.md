# SM70 NVFP4 FlashDecoding 统一 decode/verify kernel — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (subagent-driven 不可用——本会话为无人值守单执行流)。Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SM70/V100 TP2 上实现统一 q≤8 的 NVFP4 paged decode/verify CUDA kernel（split-KV 固定 grid + reduce），长上下文 decode 追平/超越 1Cat FlashDecoding 平曲线，质量过 SM120 token 级 QA。

**Architecture:** 新 CUDA kernel `sm70_flashdecode.cu`（partition kernel: grid=(splits, kv_heads, num_reqs)，GQA 组内 1× KV 读取，in-kernel NVFP4 解码头，online softmax；reduce kernel: 跨 splits 合并 partials）。结构沿用现有 Triton decode 的 split+combine 设计（已证明 cudagraph 安全），差异：CUDA WMMA 张量核 + 无 gather 解码头 codegen。Python 侧 `flashinfer.py` 后端按 env flag 分发。

**Tech Stack:** CUDA 12.x (sm_70), WMMA/HMMA m16n16k16 fp16, torch stable extensions (`STABLE_TORCH_LIBRARY_FRAGMENT`), Triton 仅作参照, ccache 增量构建 (`build_sm70.sh`)。

**Spec:** `docs/superpowers/specs/2026-09-18-sm70-flashdecode-unified-kernel-design.md`

## Global Constraints

- 编译**只能**用 `bash /data/src/vllm/build_sm70.sh`（默认 target `_C_stable_libtorch`）；禁止直接 setup.py/cmake/ninja。
- 禁止 `pkill`/`pkill -f`；清进程用 `pgrep -f <pat>` 拿具体 PID 再 `kill <pid>`。
- 长任务（bench/编译）一律后台 + 写日志 + 轮询（`nohup ... & ` + status 文件），shell 调用 ≤25s。
- 一次运行一个日志；输出重定向 `&> log; echo EXIT=$?`。
- 质量门槛: token 级 A/B vs SM120 参照（192.168.8.208:8000, api-key `0464b22a751c4a5390f0d83c0bbf4fad`），near-tie-only 判据。
- 回滚决策需用户显式确认；forward-fix 优先，不擅自 revert。
- 新 flag `VLLM_SM70_FLASHDECODE` 默认 OFF；QA 通过前不改生产默认（决策点 T8 由用户确认）。
- Python 测试从 `/data/src/vllm` 用 `python3`（已激活 /data/vllm-dev）；1Cat 对比用 `/data/vllm-1cat-1.5/bin/python` 从 /tmp 跑。

---

### Task 0: 收尾基线测量（当前 bench 链，进行中）

**Files:** 无新文件；产出 `/tmp/bench_sm70_varlen.json`、`/tmp/qa_varlen_{0,1}.json`、服务端日志。

- [ ] **Step 1: 等 varlen ladder 完成**（pid 863208，11:20 启动，预计 ~12:00）
  轮询 `grep -E "^\[[0-9]+\]|BENCH_DONE" /tmp/bench_sm70_varlen.log` 直到出现 BENCH_DONE。
  预期四行数据（54K/108K/216K/260K 的 decode ms/step）。
- [ ] **Step 2: 跑 QA A/B（Fix A 正确性）**
  ```bash
  cd /tmp && nohup bash -c '/data/vllm-dev/bin/python /tmp/qa_varlen.py 0; /data/vllm-dev/bin/python /tmp/qa_varlen.py 1' > /tmp/qa_varlen_chain.log 2>&1 &
  ```
  完成后对比 `/tmp/qa_varlen_0.json` vs `/tmp/qa_varlen_1.json`（token 序列，near-tie-only 判据）。
- [ ] **Step 3: 服务端 access log 验证（用户要求的准确口径）**
  起生产服务器: `nohup bash /root/run/run2.sh > /tmp/sm70_server.log 2>&1 &`（port 8002，spec-ON 生产 config）。等 `curl -s localhost:8002/health` OK。
  发 260K 与 54K **唯一内容** prompt（防 prefix-cache 命中；prompt 用随机 seed 生成，保证未缓存）:
  ```bash
  python3 - <<'EOF'  # 生成 260K token 唯一 prompt 并请求
  # ... (请求体构造: 260000 个唯一 token, max_tokens=32, temperature=0)
  EOF
  ```
  从 `/tmp/sm70_server.log` grep `request_access_log` 行: 预期 `input ~260000 tokens(prefill ... ms, ... tok/s, cache hit: 0%), output 32 tokens(decode ... ms, ... tok/s)`。记录两行。
- [ ] **Step 4: 回填 spec §2 基线表 + 更新 `SM70_PHASEB_STATE.md`**
  用 Task 0 实测数据更新 spec 的"待回填基线"节（Fix A 全曲线 + 服务端生产口径数字）。
- [ ] **Step 5: 停服务器，提交 state doc 更新**（`git add SM70_PHASEB_STATE.md docs/superpowers/specs/... && git commit`）

**Interfaces:** 产出数字供 Task 7 对比；QA 结果确认 Fix A 路径可作新 kernel 的正确性参照。

---

### Task 1: Python 侧 op 包装 + 单元测试骨架（TDD 红灯）

**Files:**
- Create: `vllm/v1/attention/ops/sm70_flashdecode.py`（op 包装 + workspace 计算 + 参考实现）
- Test: `test_flashdecode_unit.py`（repo 根，同 test_run.py 惯例）

**Interfaces:**
- Produces: `sm70_flashdecode_op(q, kv_cache, block_table, kv_lens, out, k_scale, v_scale, sm_scale, window_left, splits, q_rows, workspace) -> None`
  - `q`: fp16 `[R, Q, HQ, D]` 连续（Q = q_rows ≤8，**padded** 到 Q；未用行输出忽略）
  - `kv_cache`: uint8 `[B, 2H, N, F]` LBHNC（F=144: 128 data 字节 + 16 scale 字节/token row）
  - `block_table`: int32 `[R, max_blocks]`；`kv_lens`: int32 `[R]` **device**
  - `out`: fp16 `[R, Q, HQ, D]` in-place 写入；`window_left`: int（<0 关窗）
  - `splits`: int（grid 维，固定）；`workspace`: fp32 张量（partial 状态，布局见下）
  - 调用: `torch.ops._C.sm70_flashdecode(...)`（Task 3 提供 C++ 侧；本任务先用 `try/except` 包裹并在未编译时 raise RuntimeError("kernel not built")，保证红灯可跑）

- [ ] **Step 1: 写 torch 参考实现 + 单元测试**

  `vllm/v1/attention/ops/sm70_flashdecode.py` 中:

```python
import torch
from vllm.v1.attention.ops.sm70_nvfp4_dequant import (
    _nvfp4_e2m1_decode, _nvfp4_e4m3_decode)

def sm70_nvfp4_reference_attention(q, kv_cache, block_table, kv_lens_cpu,
                                   k_scale, v_scale, sm_scale, window_left,
                                   H, N, D):
    """fp16 暴力参照: 逐 token 精确 dequant NVFP4 KV 后 softmax(q·K)V。
    kv_lens_cpu: 主机端 int list（测试用）。q: [R, Q, HQ, D] fp16."""
    # dequant: 对每个 (r, token) 经 block_table 取 page, 解 128B data + 16B scale
    # (复用 _nvfp4_e2m1_decode/_nvfp4_e4m3_decode 的 LUT 语义, CPU 实现)
    # K_hat[n, D] = e2m1[nib(d)] * e4m3[sf(d//16)] * k_scale  (even d = low nibble)
    # 滑窗: window_left >= 0 时只取 [kv_len-1-window_left, kv_len)
    # out[r, qq, hq, :] = softmax(K_hat[r, :kv_len, kv_head(hq), :] * sm_scale) @ V_hat
    ...  # 完整实现见执行记录; 关键: nibble 低/高、scale group 16 维、e4m3 subnormal 分支
```

  `test_flashdecode_unit.py`（repo 根）:

```python
import torch
from vllm.v1.attention.ops.sm70_flashdecode import (
    sm70_nvfp4_reference_attention, sm70_flashdecode_op)

def make_case(R=2, Q=2, HQ=24, H=2, D=256, N=16, B=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    kv = torch.randint(0, 255, (B, 2*H, N, 144), dtype=torch.uint8, generator=g)
    bt = torch.randperm(B, generator=g)[:R*N].view(R, -1)[:, :128].to(torch.int32)
    kv_lens = torch.tensor([N*17+3, N*5], dtype=torch.int32)  # 非整页
    q = torch.randn(R, Q, HQ, D, dtype=torch.float16, generator=g)
    return q, kv, bt, kv_lens

def test_reference_matches_naive():
    # 参照自洽: 全页/非整页/滑窗/空 KV 四组, 与独立朴素实现互验 (容差 1e-2 fp16)
    ...

def test_op_q1_matches_reference():
    q, kv, bt, kv_lens = make_case(Q=1)
    out = torch.empty_like(q)
    sm70_flashdecode_op(q, kv, bt, kv_lens.cuda(), out, 1.0, 1.0,
                        1.0/16, -1, splits=4, q_rows=1, workspace=ws)
    ref = sm70_nvfp4_reference_attention(...)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
```

- [ ] **Step 2: 运行确认红灯**
  Run: `python3 test_flashdecode_unit.py -xvs`（repo 根）
  Expected: `test_reference_matches_naive` PASS；`test_op_q1_matches_reference` FAIL 于 `RuntimeError: kernel not built`（或 op 未注册）。
- [ ] **Step 3: 提交**
  `git add vllm/v1/attention/ops/sm70_flashdecode.py test_flashdecode_unit.py && git commit -m "sm70: flashdecode op wrapper + reference test (red)"`

---

### Task 2: CUDA kernel 实现（q=1 优先）

**Files:**
- Create: `csrc/libtorch_stable/attention/sm70_flashdecode.cu`
- Modify: `CMakeLists.txt`（在 line 447 `merge_attn_states.cu` 旁加一行源文件）

**Interfaces:**
- Consumes: Task 1 的 op 签名。
- Produces: `torch.ops._C.sm70_flashdecode(Tensor q, Tensor kv_cache, Tensor block_table, Tensor kv_lens, Tensor out, float k_scale, float v_scale, float sm_scale, int window_left, int splits, int q_rows, Tensor workspace)`

**Kernel 结构（定稿约束）:**

- Partition kernel `sm70_flashdecode_partition_kernel`，`grid=(splits, H, R)`, `block=256` 线程:
  - block = `(sp, h, r)`；`kv_len = kv_lens[r]`（device 读）；`kv_len==0` → 写空 partial 后 return（cudagraph 早退路径）。
  - 滑窗: `lo = max(0, kv_len-1-window_left)`（window_left>=0 时），否则 0。`b_lo=lo//N, nb_hi=ceil(kv_len/N)`; `span=nb_hi-b_lo`; `b0 = b_lo + span*sp/splits`, `b1 = b_lo + span*(sp+1)/splits`（与 Triton 同式，`b0>=b1` 早退）。
  - KV 访问: 页内 row 布局 `[128B data | 16B scale]`/token（`F=144`），data 组: dim d 的 nibble 在 byte `d//2`（偶数 dim=低 nibble）；scale 组: 每 16 dim 一个 e4m3 byte。`cp.async` 16B 双缓冲预取下一页到 shared（页边界 b 处换缓冲）。
  - 解码头（逐 token，寄存器内）: `nib = (even d) ? raw[d//2]&0xF : raw[d//2]>>4`；`val = E2M1_LUT[nib&7]*(nib&8?-1:1)`（LUT: {0,0.5,1,1.5,2,3,4,6}）；`sf = E4M3(scale_byte)`（`e=(b>>3)&0xF; m=b&7; v = (e==0)? m*2^-9 : (1+m*0.125)*2^(e-7); sign=b&0x80`）；`K_hat = val*sf*k_scale`（fp16）。**全程 shift/掩码/LUT，禁止 gather 类索引**（sm_70 Triton codegen 病根）。
  - Attention 主体: q 组（该 kv head 的 12 个 qo heads，q_rows=1）× 256 dim 用 WMMA `m16n16k16` fp16: K-tile 进 shared 后 `mma.sync` 算 scores（256 dim = 16 个 k-chunk）；online softmax（running m/l，flash-attn 标准）；P×V 同样 WMMA 累加。寄存器不够时按 qo head 子组（4 heads/轮，3 轮过 KV——**此时 KV 重读 3×**，编译实测寄存器压力后定：12 heads 一轮 or 子组化）。
  - Partial 写全局 workspace: `ws[(r*H+h)*splits+sp]` 槽位，布局 `[RATIO=12, D+2]` fp32（acc | m | l，与 Triton 同布局；q_rows>1 时外层再包 Q 维，见 Task 3）。空 partial: `m=-inf, l=0, acc=0`。
- Reduce kernel `sm70_flashdecode_reduce_kernel`，`grid=(H, R)`, `block=128`: 与 Triton combine 同逻辑（跨 splits log-sum-exp 合并，空请求 → 零行），q_rows>1 时外层 Q 维。
- Op 绑定: `STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_flashdecode_ops) { .def("sm70_flashdecode(Tensor q, Tensor kv_cache, Tensor block_table, Tensor kv_lens, Tensor out, float k_scale, float v_scale, float sm_scale, int window_left, int splits, int q_rows, Tensor workspace) -> ()") }` + `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ...)`（模板见 `quantization/marlin/sm70_nvfp4_dequant.cu:259-270`）。
- workspace 上界: `R*H*splits*Q*RATIO_p2*(D+2)` fp32（RATIO_p2=next_pow2(HQ//H)=16）。

- [ ] **Step 1: 写 kernel 文件 + CMake 挂源**
  按上述结构实现（解码头/滑窗/早退/部分和布局为正确性关键，逐条对照 Task 1 参考语义）。
- [ ] **Step 2: 编译**
  `bash /data/src/vllm/build_sm70.sh &> /tmp/build_flashdecode.log; echo EXIT=$?`（首次 ~2-5min，后台轮询日志尾）。
- [ ] **Step 3: 跑单元测试**
  `python3 test_flashdecode_unit.py -xvs` → `test_op_q1_matches_reference` PASS（容差 1e-2）。
  失败时: systematic-debugging（先对拍参照 vs kernel 的 dequant 中间量: 小 case 打印 K_hat 前 64 值比对）。
- [ ] **Step 4: 边界 case 测试**（扩 `test_flashdecode_unit.py`）
  `kv_len=0`（零行）、`kv_len=1`、非整页 `kv_len=N*17+3`、`window_left>=0` 滑窗、`splits=1`、`splits` 超过页数（空 split 早退）。全 PASS。
- [ ] **Step 5: 提交**
  `git add csrc/... sm70_flashdecode.cu CMakeLists.txt test_flashdecode_unit.py && git commit -m "sm70: flashdecode CUDA kernel — q=1 split-KV NVFP4 decode"`

---

### Task 3: q≤8 verify 支持（q_rows 维）

**Files:**
- Modify: `csrc/libtorch_stable/attention/sm70_flashdecode.cu`（q 维: scores/acc 变 `[q_rows, ...]`，q tile 进 shared，partial 槽位加 Q 维，reduce 加 Q 维）
- Modify: `vllm/v1/attention/ops/sm70_flashdecode.py`（workspace 公式含 Q）
- Modify: `test_flashdecode_unit.py`（Q=2/4/8 用例）

**Interfaces:**
- Produces: 同 Task 2 签名（`q_rows` 已存在，Q>1 语义生效）。
- 约束: 寄存器/共享内存上界——q_rows×RATIO×D 的 acc 在寄存器放不下时按 `(q_rows 批 2-4) × (qo head 子组 4)` 双层 tile，KV 重读次数 = 批数×子组数（≤3），编译实测后固定默认并注释。

- [ ] **Step 1: 写 Q>1 红灯测试**（Q=2,4,8；参照实现同法）
- [ ] **Step 2: 实现 q 维（shared 中 q tile: `[Q, 12, 256]` fp16 = Q=8 时 48KB——超 96KB? 48KB+KV staging 40KB=88KB 可行；不行则 q 批 4）**
- [ ] **Step 3: 单测全 PASS（Q=1..8）**
- [ ] **Step 4: 提交** `git commit -m "sm70: flashdecode — q<=8 verify support"`

---

### Task 4: 后端 dispatch 集成（`VLLM_SM70_FLASHDECODE` flag）

**Files:**
- Modify: `vllm/envs.py`（加 `VLLM_SM70_FLASHDECODE`，默认 "0"——同 `VLLM_SM70_VARLEN_DECODE` 写法）
- Modify: `vllm/v1/attention/backends/flashinfer.py`:
  - builder `__init__`: `_sm70_flashdecode = envs.VLLM_SM70_FLASHDECODE == "1" and sm70_nvfp4_kv_enabled and ...`
  - decode 分支（line ~3166 `sm70_nvfp4_paged_decode` 调用点）: flag 开 → 调 `sm70_flashdecode_op`（q_rows=1）
  - verify 分支: **先 spike** —— 在 DFlash2 verify 路径加临时 log 打印 query shape/`attn_metadata` 字段，确认 verify 的 per-request q（预期 6 或 7）与 query 打包方式（token-major `[total_q, HQ, D]` 需 reshape 到 `[R, Q, HQ, D]` 并配 per-request block_table/kv_len）；确认后接 flag（reshape 在 dispatch 处做，kernel 输入契约不变）。
- Modify: `vllm/v1/attention/backends/flashinfer.py` metadata: verify 路径需要的 per-request q 与 kv_lens（若现成字段不足，最小化加——参照 `shadow_kv_lens_dev`/`shadow_ws` 的既有 alloc 模式，workspace 用 `shadow_ws` 同款持久 buffer 扩容）。

- [ ] **Step 1: envs.py + builder flag（flag 默认 OFF，不影响现有路径）**
- [ ] **Step 2: decode 分支接 flag（q=1）**；`VLLM_SM70_FLASHDECODE=1` 跑 `python3 test_run.py`（repo 根，MTP4 TP2 decode 基准）→ 无崩溃、数字合理（≈Fix A 水平或更优）。
- [ ] **Step 3: verify 路径 spike + 接入**（log 确认 shape → reshape 接入 → 短上下文 smoke: 5K ctx spec-ON 生成 64 tokens 无错误）
- [ ] **Step 4: 提交** `git commit -m "sm70: flashdecode backend dispatch (VLLM_SM70_FLASHDECODE)"`

---

### Task 5: 质量门槛（token 级 A/B）

**Files:**
- Modify: `/tmp/qa_varlen.py` → 泛化 `/tmp/qa_flashdecode.py <flag: 0=现有|1=varlen|2=flashdecode>`（同 54K/216K, 96 greedy tokens, 同 SM70 envs，从 /tmp 跑）

- [ ] **Step 1: 跑 `qa_flashdecode.py 0` / `1` / `2`（顺序，每轮独立进程）**
- [ ] **Step 2: 三方 token 序列对比**: 2 vs 0（新 kernel vs 现状）、2 vs 1（vs Fix A）——判据: 完全一致或 near-tie-only 分歧（memory 10168 标准）。分歧超界 → systematic-debugging（先对拍单 kernel vs 参照，定位是 dequant/softmax/reduce 哪段）。
- [ ] **Step 3: SM120 参照 A/B**（同 prompt/seed，192.168.8.208:8000）→ 记录分歧统计。
- [ ] **Step 4: 过门槛后提交 QA 证据到 state doc；不过则停下修（不擅自改判据）。**

---

### Task 6: 性能 ladder + 回填

- [ ] **Step 1: 扩展 `/tmp/bench_sm70.py` 支持 env flag → `/tmp/bench_flashdecode.py`（同 54K/108K/216K/260K 双调用测法，spec-OFF q=1）**
  `VLLM_SM70_FLASHDECODE=1 python3 /tmp/bench_flashdecode.py &> /tmp/bench_flashdecode.log &`（~25min，后台轮询）
- [ ] **Step 2: verify 路径性能**: spec-ON server 短/长上下文 tok/s（复用 Task 0 Step 3 的服务端日志法，flag ON/OFF 两轮，54K/108K）。
- [ ] **Step 3: 汇总四表**（现状 Triton / Fix A varlen / 新 kernel / 1Cat）回填 spec §2/§5 + state doc。

---

### Task 7: 报告与决策点

- [ ] **Step 1: 汇总报告**: 基线（Task 0）、质量（Task 5）、性能（Task 6）、与 1Cat 差距闭环情况。
- [ ] **Step 2: 建议 flag 默认值**（生产启用与否）→ **用户确认**（回滚政策强制）。
- [ ] **Step 3: 更新 `SM70_PHASEB_STATE.md` 待办 checklist + memory（设计/决策/结果各一条）。**

---

## Self-Review 记录

- **Spec 覆盖**: spec §3（集成/dispatch）→ T1/T4；§4.1-4.4（grid/解码头/softmax/reduce/cudagraph）→ T2/T3；§5（性能模型）→ T6；§6（QA）→ T5；§7（风险: 寄存器压力→T3 Step2 退路, 对齐边角→T2 Step4, workspace→T3, autotune→T6, graph 稳定性→T4 Step2 验证）；§8 里程碑 M1=T2, M2=T3/T4, M3=T5/T6, 决策点=T7。无孤儿要求。
- **占位符**: "完整实现见执行记录" 出现在参照实现注释中——**修正**: 执行时必须写完整参照实现（T1 红灯测试依赖它自洽互验）；Task 4 Step3 的 spike 是刻意的探索步骤（有明确产出判据: 打印出 q shape 与打包方式），非占位。
- **类型一致**: op 签名 T1→T2→T4 一致（`q [R,Q,HQ,D]`、`kv_lens` device int32、`q_rows` int）；workspace 布局 T2（`[R,H,splits]×[RATIO,D+2]`）与 T3（加 Q 维: `[R,H,splits,Q,RATIO,D+2]`）——T2 的 Q=1 是 T3 的特例，T3 需保持 Q=1 布局兼容（stride 处理: 槽位大小恒按 Q 维分配，Q=1 时退化为同布局）。
