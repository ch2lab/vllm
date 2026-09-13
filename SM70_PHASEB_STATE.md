
## 2026-08-23 会话 2: 目标明确化 + 格式矩阵验证（进行中）

### 用户明确目标
- input 10k tokens: prefill 2000 tok/s, decode 60 tok/s, MTP=2
- max_model_len=262144 (default), fp8/nvfp4 kv-cache
- model = Qwen3.8-27B FP8/AWQ; 同时修复运行中发现的其它错误
- 1Cat 对标: 64K prefill 1040 tok/s, TPOT 64K 37 tok/s(单流), 并发 64K 100.6

### 当前基线（会话1收尾数据）
- AWQ+nvfp4 TP2 MTP2: prefill 12.3k = 800 tok/s, decode 单流 44.5,
  并发4 ≈119 聚合; hot prefill(prefix命中) 2.7-3.0k
- fp8 KV 写入路径统一按 e4m3 编码/解码（自洽）; e5m2 名义支持

### 本会话动作
1. `VLLM_FLA_CHUNK_SIZE` env 化 (utils.py, 默认 64) — 待 sweep {64,128,256}
2. matrix_smoke.py: 权重×KV 矩阵验证脚本, 后台运行中
   (AWQ×{nvfp4,e4m3,e5m2}, FP8×{e4m3,e5m2,nvfp4}, TP2 MTP2, 0.90)
3. GDN 形状确认: K=V=128, Hk=16, Hv=48, 64层(48 GDN); TP2 后 24 v-heads/卡

### 待办（按预期收益）
- [ ] chunk_o CUDA WMMA (prefill 35-42% 占比, 最大单点)
- [ ] FLA_CHUNK_SIZE sweep
- [ ] decode 步 nsys 新鲜分解 (当前代码)
- [ ] MTP acceptance 提升 (top-k draft sampling 已加入, 待测)

### 矩阵第 1 轮结果 (15:23-15:42, 矩阵1)
- AWQ + nvfp4: SUCCESS — prefill 921 tok/s(6.2k), decode 25.2 tok/s ← 回归!
  DECODE 输出重复("Quantum Quantum um") — 质量回归同步确认
- AWQ + fp8_e4m3: SUCCESS — 同上 (916 tok/s, 25.3 tok/s)
- AWQ + fp8_e5m2: FAILED — SM70 backend supported_kv_cache_dtypes 缺
  "fp8_e5m2" → 落到 Triton backend → SM70 FP8 KV ValueError
- FP8 权重 3 组未跑（被 e5m2 失败后续链跳过）

### 回归定位
- 19:49 改动把 max_query_len<=16 (MTP verify, mq=3) 路由到
  decode_partitioned kernel (xqa=0) — 未经短上下文验证
- 修复: VLLM_SM70_ATT_MQ_DECODE env 门控, 默认 0 (回退旧行为:
  mq>1 走 prefill_paged_batched); AWQ_NVFP4_B 复测中

### e5m2 支持实现 (KV_MODE=4)
- kernel: fp8e5m2_to_half_scaled (__NV_E5M2), paged kernel K/V 双路
  constexpr 分支, LAUNCH_PARTED/LAUNCH_BATCHED 增加 mode 4
- backend: supported_kv_cache_dtypes += fp8_e5m2; 写入路径
  torch.float8_e5m2 + clamp ±57344; forward kv_mode=4

### chunk_o WMMA kernel (prefill 主杠杆, 35-42% 占比)
- fla_chunk_o_sm70: grid(NT,H,V/64), 128 线程, ~42KB smem
  (sQ 复用门控, sB 复用 k/h/v/结果, sA fp16, fp32 acc)
- 两遍 K 循环: pass1 A=q@k^T; 原地门控 q*=exp(g); pass2 o_inter=qg@h^T;
  A 门控+mask→fp16; A@v→acc 复用; 片段级相加; 单次写出
- dispatch: chunk_o.py (VLLM_SM70_FLA_CHUNK_O_WMMA 默认开,
  仅 varlen+BT64+fp16+K≤128+V%64==0+非 capture)
- 编译通过; 正确性对比测试待 GPU 空闲后运行
- VLLM_FLA_CHUNK_SIZE env 化 (默认 64)

### chunk_o WMMA kernel: 两个 bug 已修复 (用户质询"改错了什么"的答复)
- **nvfp4 KV 写入/读取路径从未被修改**, 问题全部出自新 chunk_o kernel:
- Bug1: A@v tiling 混淆 — acc[vt] 只累加 1 个 n-tile(16/64) 且按 vt 写入
  d-tile 列 → identity-v 探测会掩盖(对角对齐), 用 one-hot v(r=5<16) 无法
  暴露, 最终用 v=identity 全量重建 A 才定位; 修复: 外层 d-tile × 内层
  n-tile 双重循环 (4 nt × 4 dt mma)
- Bug2: smem 行 stride +4 half 破坏 WMMA 16B 对齐 → misaligned address
  (kkt/delta_h 均用 +8 即此因); 修复 CO_SQK=+8 (43KB 静态 smem 仍达标)
- 单测: 全部 case PASS (max_abs ≤0.04, fp16 ULP 级; 多序列需按 [1, T_total]
  packed 构造 — 测试夹具也修正了)
- e2e: PREFILL 6.2k = **1016 tok/s** (vs 修复前基线 921, chunk_o WMMA 生效
  +10%); SMOKE 输出正常
- 教训: WMMA kernel 验证必须用能暴露非对角/跨 tile 的探测向量;
  identity 和单点 one-hot 都会与 tiling 错误"对齐"而漏检

### decode 回归调查中 (与 chunk_o 无关, 独立问题)
- 现象: MTP=0 decode 19.0 tok/s (昨日 29.5), MTP=2 24.7 且输出重复;
  MTP=0 输出连贯 → 基础路径正确; eager 7.2 tok/s (host-bound 正常)
- 嫌疑: 19:49 分区数固定 16 (昨日 29.5 时 graph 烧入的是 P=1) →
  VLLM_SM70_ATT_PARTITIONS env 化, A/B {P16,P1} × {rms on,off} 进行中
- FLA decode 确认走 recurrent 路径 (128-token decode 期间 chunk 链 0 次调用)

### decode A/B 结果 (dec_bench, 256t, AWQ+nvfp4 TP2)
| 配置 | tok/s | 输出 |
|------|-------|------|
| P=16 rms=1 MTP2 | 24.7 | 重复("Quantum Quantum") |
| P=1  rms=1 MTP2 | 31.7 | 重复 |
| P=16 rms=0 MTP2 | 32.2 | 重复 |
| P=1  MTP0       | 22.9 | 连贯正常 |
- P=16 固定分区在短上下文损失 ~20% (恢复 1Cat 式 device-side
  active_num_partitions 动态分区为待办)
- **MTP2 贪婪等价性被破坏**: 同 prompt 下 MTP0 连贯、MTP2 从首 token 起
  重复; draft_sample_method 默认 greedy → 属正确性 bug, 非采样设计
  (怀疑点: fla_verify_sm70 融合验证核, E=VLLM_SM70_FLA_FUSED_VERIFY=0
  A/B 进行中)
- MTP 净增益 31.7/22.9 = 1.38x (昨日 1.42x, 结构正常; 绝对值含机器漂移)
- attention.py fp8_e5m2+fp8权重 上游门禁已为 SM70 分支放行
  (自研 backend 自行 e5m2 量化)
