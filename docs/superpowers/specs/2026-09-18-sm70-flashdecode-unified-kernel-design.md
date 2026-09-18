# SM70 NVFP4 FlashDecoding 统一 decode/verify kernel — 设计文档

日期: 2026-09-18 · 状态: 设计定稿（性能目标节待 bench 链数据回填）· 范围: SM70/V100 · TP2

## 1. 背景与问题

Qwen3.8-27B-QUASAR-NVFP4 @ V100/SM70 TP2 的长上下文 decode 衰减（spec-OFF 纯 q=1 测法，干净双调用）：

| 上下文 | 现状 (Triton sm70_nvfp4_paged_decode) | 1Cat (fp8 KV, paged_attention_v2 FlashDecoding) |
|---|---|---|
| 54K | 306.0ms/step (3.27 tok/s) | 121.2ms (8.25 tok/s) |
| 108K | 587.3ms (1.70) | 131.9ms (7.58) |
| 216K | 1150.3ms (0.87) | 131.0ms (7.63) |
| 260K | 1423.1ms (0.70) | 135.5ms (7.38) |

- 现状根因（已定案）: q=1 decode 走 Triton `sm70_nvfp4_paged_decode`（split-KV splits=40, 80 CTA 已填满 SM），gather-based NVFP4 nibble dequant 在 sm_70 的 Triton codegen 极差（有效 KV 带宽 ~1.8GB/s）。
- 1Cat 曲线平坦的原因: FlashDecoding 结构 — `grid=(heads, seqs, max_num_partitions=ceil(max_seq_len/PARTITION_SIZE))` 固定分区 + 独立 reduce kernel，partition 并行度与 seq_len 解耦，cudagraph 下 plan 定死 grid、运行时越界分区早退。
- 生产 spec-ON（DFlash2）长上下文衰减（54K 59.3 → 108K 35.3 tok/s）是另一杠杆：verify/drafter 在 cudagraph 下 flashinfer plan grid 固定 → 自动 split-KV 分区可能未生效（待 profile）。

## 2. 目标与验收

1. **统一 kernel 覆盖 q=1 decode 与 q≤8 verify**（DFlash2 num_speculative_tokens=5 → verify q 待代码确认，上限按 8 设计）。
2. 曲线平坦度：54K→260K 全步衰减 ≤15%（1Cat 实测 11%）。
3. 性能：全步追平或超越 1Cat 长上下文水平（135.5ms@260K）。
   - 注: 全步存在 GEMM+GDN 地板（~110-130ms，1Cat 与我们的模型计算相同），attention 部分的差异化来自 NVFP4 半字节 KV + GQA 感知 1× KV 读取（见 §4.1）。
4. 质量: token 级 A/B vs SM120 参照（192.168.8.208:8000）+ vs Fix A 路径，near-tie-only 判据（沿用 memory 10168 定案标准）。
5. 生产收益: spec-ON 长上下文衰减（59→35 tok/s）由同一 kernel 结构性解决（verify grid 设计定死，不依赖 flashinfer plan 自动分区）。

**实测基线（2026-09-18, `/tmp/bench_sm70_varlen.py`）** — 重大修正:

| 上下文 | 现状 Triton | Fix A (varlen 路由) | 1Cat |
|---|---|---|---|
| 54K | 306.0ms | **20.0ms (49.9 tok/s)** | 121.2ms |
| 108K | 587.3ms | **18.3ms (54.8)** | 131.9ms |
| 216K | 1150.3ms | **20.0ms (50.0)** | 131.0ms |
| 260K | 1423.1ms | 待回填（~12:15） | 135.5ms |

- **Fix A（纯 Python flag, SM70PAGED varlen 路由）已经实现平曲线**（20ms 平台，54K→216K 零衰减），且 **6× 快于 1Cat** —— 之前"SM70PAGED 边际 ~1ms/K"的估计是把 DFlash2 verify 步的 drafter/GDN 开销误记到了 KV 扫描头上。
- **kernel 价值重新定界**: (1) q=1 decode 已由 Fix A 解决，kernel 目标是**不低于 Fix A 路径**（性能上追平即合格）；(2) 真正的差异化 = **结构性解决生产 spec-ON 长上下文衰减**（cudagraph 下 flashinfer plan grid 固定 → 自动 split-KV 分区可能未生效；本 kernel 的 grid 按设计定死、分区数 plan 时选定，不依赖运行时 plan）；(3) 单 .so 自主控制，摆脱 flashinfer Python plan 每步开销。
- 验收相应调整: 质量门槛不变（token 级 A/B）；性能合格线 = q=1 不劣于 Fix A；生产 spec-ON 54K/108K 衰减（59→35 tok/s）是独立验证项（Task 6 Step 2）。

## 3. 范围与集成

- 位置: `vllm/csrc/libtorch_stable/attention/sm70_flashdecode.cu`（新 TU）+ `flashinfer.py` 后端 dispatch。构建仅走 `build_sm70.sh`（规则强制）。
- 新 custom op: `sm70_flashdecode(out, q, k_cache, v_cache, block_tables, seq_lens)`；q 打包 `[seq, q_rows≤8, qo_heads, head_dim]`。
- 独立 env flag `VLLM_SM70_FLASHDECODE`（默认 OFF）。Fix A（`VLLM_SM70_VARLEN_DECODE`）保留为独立备选；两者 QA 通过前都不动生产默认。
- 不动 flashinfer 仓库（其 SM70PAGED/plan 机制保持原样）。
- GDN 层不涉及（独立 kernel，非本设计范围）。

## 4. Kernel 设计

### 4.1 Grid 与 GQA KV 共享（核心差异化）

- block = 一个 `(kv_head, partition)`。TP2 下 num_kv_heads/GPU=2, head_dim=256, qo:kv = 24:2 = 12:1。
- **1Cat 的 grid 以 24 qo_heads 为并行维 → 每个 KV 分区被 12 个 block 重复读取（12× 流量）**；本设计 block 内扇出该 kv head 的 12 个 qo heads，KV 每分区只读 1×。260K 物理 KV 流量 ~2.5GB vs 1Cat ~30GB。
- 扇出子组: 每轮处理 SUBGROUP=4 个 qo heads（寄存器/共享内存压力旋钮），12 heads 分 3 轮。退路: 若子组内 q≤8 行 × 4 heads × 256 dim 寄存器压力过高，SUBGROUP=2。
- `num_partitions = ceil(max_ctx / PARTITION_SIZE)`，PARTITION_SIZE=2048（编译期常量，3 候选 autotune）。max_ctx=262144 → 128 分区 × 2 kv heads = 256 blocks ≈ 3.2 波（V100 80 SM）。
- 每 block 每分区 KV 字节: 每 kv head 2048×256×2(K+V)×(9/16 B) = 576KB；256 线程/block，cp.async（sm_70 支持 16B 异步拷贝）双缓冲预取。

### 4.2 NVFP4 解码头

- 直接移植 SM70PAGED（flashinfer prefill.cuh）已验证实现：nibble(4-bit e2m1) → fp16 LUT × fp8 block scale（每 16 元素 1 个 scale byte），实测 ~8GB/s 有效带宽。
- KV layout: HND paged（`VLLM_KV_CACHE_LAYOUT=HND`），block_tables 寻址；partition 与 page 边界不对齐时按 page 尾截断。

### 4.3 Softmax 与 reduce

- partition kernel: per q 行 online softmax（running m/l，flash-decoding 标准），输出 partial `[seq, q_rows, qo_head, partition, head_dim]` fp16 + (m, l) fp32 到全局。
- reduce kernel: `grid=(qo_heads, seqs, q_rows)`，跨 ≤128 分区做 log-sum-exp 合并，写最终 out（1Cat reduce_kernel 同款结构）。
- partial 总量上界: 256 blocks × 8 × 24 × 256 × 4B ≈ 512MB — **注意**: 需按实际活跃 block 数（= ceil(seq_len/2048)×2）与 q_rows 实际值分配 workspace；q=1 时 partial 仅 ~64MB@260K。workspace 在 plan 时预分配（cudagraph 友好）。

### 4.4 cudagraph 语义

- capture 时: 按 max_ctx 计算 num_partitions 并预分配 partial workspace；grid 定死。
- 运行时: 每 block 读 `seq_lens`，`partition ≥ ceil(seq_len/PARTITION_SIZE)` 则早退（写零 partial 由 reduce 侧 mask，或 reduce 按实际分区数合并）。
- 短上下文（如 8K）: 活跃 block = 4×2=8 → 单波内完成，无 grid 重规划。

## 5. 性能模型

- attention 部分（q=1, 260K, 1× KV 流量）: 2.5GB @ 有效 600-800GB/s ≈ 3-5ms + dequant/softmax 计算 ≈ 全 attention ~5-8ms。
- 全步 ≈ GEMM+GDN 地板（~110-130ms）+ attention → 目标 ~120-150ms@260K。
- verify（q=7-8）: KV 流量同 q=1（分区内 1× 读取），FLOPs ×q → 目标不劣于现状 verify 的快路径成本。
- 短上下文（≤8K）: attention <1ms，全步由地板主导，不劣于现状。

## 6. 正确性与 QA

- 单元: kernel vs torch 参考实现（fp16 暴力 attention over paged KV，NVFP4 精确 dequant）— 数值容差参照现有 SM70 attention 测试基线。
- 集成 QA: token 级 A/B（`/tmp/qa_varlen.py` 同款基建扩展）vs SM120 参照 + vs Fix A 路径；54K/216K 各 96 greedy tokens，near-tie-only 判据。
- 性能: `/tmp/bench_sm70_varlen.py` 同款 ladder（54K/108K/216K/260K），对比: 现状 Triton / Fix A / 新 kernel / 1Cat。
- 生产路径验证: spec-ON server 长上下文 decode tok/s（54K/108K），对比 59.3/35.3 tok/s 基线。

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| 12 qo heads 扇出的寄存器/shared 压力（V100: 96KB shared/block, 128 regs/thread） | SUBGROUP=2 退路；最坏 12× 重读 = 1Cat 持平，仍为现状 5-10× |
| NVFP4 scale 在 partition 尾部的对齐边角 | dequant 头按 16 元素组对齐处理 + 尾部 mask；单测覆盖 partition 不整除 page 的情况 |
| partial workspace 显存（q=8 上界 512MB） | 按实际 q_rows/活跃分区数 plan-time 分配；q≤8 实际典型 q=1-7 |
| 首版带宽仅达理论 60-80% | autotune PARTITION_SIZE∈{1024,2048,4096} × SUBGROUP∈{2,4}；bench 选默认 |
| cudagraph 捕获下 workspace 地址稳定性 | plan 时一次性分配，capture/replay 复用同一指针（沿用 Fix A 的 graph 捕获经验） |

## 8. 里程碑

1. M1: kernel 骨架（q=1, PARTITION=2048, SUBGROUP=4, 单测过）→ bench 54K/260K
2. M2: q≤8 verify 支持 + workspace plan-time 分配 → 集成 QA（SM120 + Fix A 对照）
3. M3: autotune 定默认 + spec-ON 生产路径验证（server 长上下文 tok/s）
4. 决策点: flag 默认值（生产启用）需用户确认（回滚政策）

## 9. 约束（强制规则摘要）

- 编译仅 `bash /data/src/vllm/build_sm70.sh`；禁止 pkill；长任务后台+日志+轮询；shell 调用 ≤25s；一次运行一个日志。
- 回滚决策需用户显式确认；forward-fix 优先。
- 质量门槛: SM120 参照 token 级 A/B（near-tie-only）。
