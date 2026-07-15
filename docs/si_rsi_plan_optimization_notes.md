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
   - 可用 `RSI_CUDA_MAX_CACHED_PLAN_CHUNK` 调整缓存方向数上限。
   - RSI 大 unique-direction chunk 会按固定 cache chunk 拆分复用；默认 `RSI_CUDA_FIXED_PLAN_CHUNK=1`，即单方向 plan cache。
   - 已增加进程内 host LRU cache，默认 `RSI_CUDA_PLAN_HOST_CACHE_MB=1024`，可设为 `0` 关闭。

5. streaming-plan 触发阈值：
   - 默认阈值为 8 GiB。
   - 可用 `RSI_CUDA_STREAMING_PLAN_THRESHOLD_MB=0` 强制走 streaming-plan，用于测试。

6. SI streaming chunk size：
   - 默认 `RSI_CUDA_SI_PLAN_CHUNK=128`。
   - 可用环境变量覆盖。已测试 512 在 200k/S128 上更慢，因此暂不作为默认值。

7. SI device plan cache：
   - streaming SI 阶段可把一部分已上传的 `DevicePlanChunk` 常驻 GPU。
   - 默认 `RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=8192`。
   - 可设为 `0` 关闭，或设为 `4096` 降低显存占用。

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
- host cache warm，未启用 SI device plan cache：
  - wall time `314.14 s`
  - CUDA internal total `335.934 s`
  - `si_plan = 276.538 s`
  - `si_sweep = 39.6823 s`
- SI device plan cache 4096 MiB：
  - wall time `197.18 s`
  - CUDA internal total `210.049 s`
  - `si_plan = 150.505 s`
  - `si_sweep = 39.6486 s`
- SI device plan cache 8192 MiB：
  - wall time `143.34 s`
  - CUDA internal total `153.483 s`
  - `si_plan = 90.1124 s`
  - `si_sweep = 42.9842 s`
  - 峰值显存约 `13.4 GiB / 16.3 GiB`

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

已实现固定方向 chunk 复用：

```text
RSI_CUDA_FIXED_PLAN_CHUNK=1  # default
```

默认用单方向 cache，是因为 S128/8192 中每个 RSI batch 的随机 unique directions 约为 2800 个，几乎覆盖所有 128-direction fixed chunks。若 fixed chunk 过大，会读取大量当前 batch 未使用的方向 plan。

预期收益：

- 每个方向最多构建一次 plan。
- 后续 batch 复用已缓存的单方向 plan。
- 将 `rsi_plan` 从千秒级降到主要由实际所需方向的 cache 读取/上传决定。

主要代价：

- cache 文件数量会增加，S128 最多约 16640 个单方向 cache 文件。
- 小文件 IO 可能成为新瓶颈，后续可合并为 shard cache。

验证状态：

- `make test-gpu` 已通过。
- 200k/S128/16 已通过输出行数和非有限值检查。

### 2. Host 内存 LRU cache

当前 chunk cache 原本是磁盘缓存；每次使用仍要读取、反序列化、拷贝到 device。

已增加 host 内存 LRU：

- 缓存最近使用的若干 plan chunks。
- 以 `RSI_CUDA_PLAN_HOST_CACHE_MB` 限制，默认 1024 MiB。
- SI fixed chunk 和 RSI 单方向 fixed chunk 都走同一套 host cache。

预期收益：

- 降低 `si_plan` 中重复磁盘读取和反序列化开销。
- 对 fixed RSI chunk 复用也有帮助。

实测结论：

- host LRU 对 200k/S128/16 warm run 有收益，但 SI 顺序扫描 chunk 数量远大于默认 1 GiB host cache 容量，单靠 host LRU 不能解决 `si_plan`。
- SI device plan cache 的收益更直接。

### 3. SI chunk size 参数扫描

当前 SI streaming chunk size 已可配置，默认 128 directions。

建议继续测试：

```text
RSI_CUDA_SI_PLAN_CHUNK=256
RSI_CUDA_SI_PLAN_CHUNK=512
RSI_CUDA_SI_PLAN_CHUNK=1024
```

已测试 512-direction chunk：

- `200k + S128 + 16 samples`
- wall time `440.45 s`
- `si_plan = 409.591 s`
- `si_sweep = 56.2771 s`

该结果慢于 128-direction chunk，因此默认保持 128。

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

已先对 SI streaming 实现：

- 固定 chunk size。
- 显存预算控制。
- 按 chunk index 固定常驻，避免普通 LRU 在顺序扫描下失效。

实测收益：

- `RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=4096` 将 200k/S128/16 的 `si_plan` 降到 `150.505 s`。
- `RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=8192` 将 200k/S128/16 的 `si_plan` 降到 `90.1124 s`。

后续可做：

- 对 RSI fixed-direction chunk 也增加 device 常驻窗口。
- 输出实际 cached chunk 数和 cache bytes，便于自动调参。

## 当前推荐优先级

1. 用 SI device plan cache 跑完整 200k/S128/8192，确认端到端收益。
2. 对 RSI fixed-direction chunk 增加 device/host shard 常驻，继续压低 `rsi_plan`。
3. 输出 cache 命中/常驻统计，便于后续自动选择预算。
4. SI chunk size 可配置，继续测试 64/128/256。
5. 磁盘 cache 压缩。
