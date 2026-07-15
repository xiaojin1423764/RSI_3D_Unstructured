# SI/RSI Sweep Plan 后续优化记录

本文记录 200k 网格、CUDA Figure 5 高角度运行中 SI/RSI sweep plan 的当前状态、已完成改动和后续优化方向。

## 当前结论

- `S128` 方向数为 `16640`，200k 网格 cell 数为 `194314`。
- 全量常驻 sweep order 仅 `orders` 就约 `16640 * 194314 * sizeof(int) = 12.9 GB`，再加 level offsets、metadata、angular field 和 RSI workspace，会超过 16 GB GPU 可用显存。
- 已把 Figure 5 CUDA 路径改为按方向 chunk 流式构建/上传 sweep plan；默认估算全量 `orders` 超过 8 GiB 时启用 streaming-plan 路径。
- 已实现 chunk plan cache，默认写入 `results/cache/rsi_sweep_plan_chunk_<key>.bin`。
- `S128/8192` 已可完整跑通，但主要耗时仍在 plan 准备而不是 sweep 计算。

## 已完成实现

1. Figure 5 参数可从命令行覆盖：
   - `--figure5-fine-sn`
   - `--figure5-samples`

2. SI fine 角通量流式化：
   - 不再分配完整 `M*C` 的 `angularPsi`。
   - 每个方向 batch sweep 后立即按权重累加到 `currentPhi`。

3. streaming sweep-plan 路径：
   - 静态 GPU 数据 `mesh + ordinates + weights` 上传一次。
   - SI fine 按固定方向 chunk 构建/上传 plan。
   - RSI 按 sample batch 的实际 unique directions 构建 compact plan。

4. chunk plan cache：
   - 复用 `RSI_SWEEP_PLAN_CACHE` 开关。
   - 复用 `RSI_SWEEP_PLAN_CACHE_DIR` 目录设置。
   - 默认只缓存 `K <= 256` 的 chunk，避免 RSI 大 unique-direction chunk 生成大量一次性 cache 文件。

5. streaming-plan 触发阈值：
   - 默认阈值为 8 GiB。
   - 可用 `RSI_CUDA_STREAMING_PLAN_THRESHOLD_MB=0` 强制走 streaming-plan，用于测试。

## 已验证结果

`30k + S40 + 128 samples`：

- 原快路径和 chunk-cache streaming 路径结果逐值一致：
  - `SI_fine max_abs = 0`
  - `RSI max_abs = 0`
  - `RSI_tail max_abs = 0`
- 强制 streaming 后：
  - 无 chunk cache 版本约 `141 s`
  - chunk cache cold 约 `28.6 s`
  - chunk cache warm 约 `25.8 s`

`200k + S128 + 16 samples`：

- streaming-plan 路径完整跑通。
- 4 个 CSV 均为 `194315` 行。
- `phi0` 非有限值数量为 0。

`200k + S128 + 8192 samples`：

- 输出目录：`results/fig5_200k_s128_8192_chunkcache/Cir/`
- wall time：`1574.48 s`
- CUDA internal total：`1674.65 s`
- SI fine：
  - `si_total = 295.296 s`
  - `si_plan = 252.96 s`
  - `si_sweep = 37.5492 s`
- RSI/RSI-tail：
  - `rsi_total = 1379.28 s`
  - `rsi_plan = 1336.36 s`
  - `rsi_sweep = 37.442 s`
- 4 个 CSV 均为 `194315` 行。
- `phi0` 非有限值数量为 0。

## 后续优化方向

### 1. RSI 固定方向 chunk 复用

当前 RSI 每个 sample batch 会收集约 `2800` 个 unique directions，并为这批方向构建 compact plan。不同 batch 的 unique direction 集合不同，因此当前不缓存 `K > 256` 的大 chunk，导致 `rsi_plan` 仍为主要瓶颈。

建议改为固定方向 chunk：

```text
S128 directions -> 130 chunks * 128 directions
```

每个 RSI batch 不再构建一个大 compact plan，而是按固定 chunk 组织 selected directions。这样所有 batch 可复用同一套 chunk cache。

预期收益：

- 避免每个 RSI batch 重建约 2800 个方向的 plan。
- 将 `rsi_plan` 从千秒级降到主要由 chunk 读取/上传和调度决定。

主要代价：

- 需要按 iteration/sample 的 selected directions 重排或分组。
- 每个 iteration 可能跨多个 fixed chunk，kernel launch 数会增加。

### 2. Host 内存 LRU cache

当前 chunk cache 是磁盘缓存；每次使用仍要读取、反序列化、拷贝到 device。

建议增加 host 内存 LRU：

- 缓存最近使用的若干 plan chunks。
- 以字节数限制，如 4-8 GiB。
- SI 的 130 个 fixed chunk 可以按内存预算部分常驻。

预期收益：

- 降低 `si_plan` 中重复磁盘读取和反序列化开销。
- 对 fixed RSI chunk 复用也有帮助。

### 3. 可配置 SI chunk size

当前 SI streaming chunk size 固定为 128 directions。

建议增加环境变量：

```text
RSI_CUDA_SI_PLAN_CHUNK
```

测试 256/512 directions per chunk。

预期收益：

- 减少 SI chunk 数和 kernel 调度次数。
- 减少 chunk cache 文件数量。

风险：

- 单 chunk device memory 增大。
- 大 chunk 上传时间增加。

### 4. Plan cache 压缩

当前 cache 中 `order` 使用 `int32` 原样存储。

可评估：

- 磁盘缓存压缩。
- delta/varint 编码，仅磁盘压缩，加载后展开为 `int32`。

预期收益：

- 降低 cache 占用和磁盘 IO。

风险：

- 增加 CPU 解码时间。
- 对 warm-cache 性能收益不一定稳定。

### 5. Device-side plan 常驻窗口

在显存允许时，保留多个 chunk 的 device plan，避免每轮重复 H2D 上传。

建议先做：

- 固定 chunk size。
- 显存预算控制。
- LRU 或 round-robin device chunk pool。

预期收益：

- 降低 SI 和 RSI 的 repeated upload 成本。

## 当前推荐优先级

1. 固定方向 chunk 复用，先解决 `rsi_plan`。
2. Host LRU cache，降低 SI/RSI 反复读取 chunk cache 的成本。
3. SI chunk size 可配置，测试 256/512。
4. Device chunk pool。
5. 磁盘 cache 压缩。
