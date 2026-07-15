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
   - 默认 `RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=10000`。
   - 可设为 `0` 关闭，或设为 `4096` 降低显存占用。
   - 已输出常驻 chunk 数、device bytes、hit/miss 统计。
   - 默认启用按首轮 plan 成本选择常驻 chunk：`RSI_CUDA_SI_PLAN_COST_ADMISSION=1`。
   - 已增加 SI packed host plan cache：`RSI_CUDA_SI_HOST_PLAN_CACHE_MB`，默认 `4096` MiB，可设为 `0` 关闭。
   - 已预留 SI host plan 异步预取：`RSI_CUDA_SI_PLAN_PREFETCH=1`，但默认关闭。

8. cache 文件目录 shard：
   - 已增加 opt-in 子目录 shard：`RSI_SWEEP_PLAN_CACHE_SHARD_DIRS=1`。
   - 默认关闭，继续使用旧 cache 路径，避免影响已有 warm cache 性能。

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
- 同时打开 `RSI_CUDA_SI_PLAN_COST_ADMISSION=1` 和 `RSI_CUDA_SI_PLAN_PREFETCH=1`：
  - wall time 明显变差，CUDA internal total `320.042 s`
  - `si_plan = 104.569 s`
  - `si_sweep = 173.738 s`
  - 主要原因是后台 host prefetch 与 GPU sweep 争用 CPU/内存带宽，因此 prefetch 默认关闭。
- 默认关闭 prefetch 后的回归：
  - wall time `171.60 s`
  - CUDA internal total `182.778 s`
  - `si_plan = 103.581 s`
  - `si_sweep = 58.6512 s`
  - 4 个 CSV 均为 `194315` 行，`phi0` 非有限值数量为 0。

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
- `RSI_CUDA_SI_PLAN_COST_ADMISSION` 默认开启，但 8 GiB 预算下收益很小。
- 继续保留 `RSI_CUDA_SI_PLAN_PREFETCH` 作为实验开关；当前实测不作为默认。

### 6. 暂不实施的方向

以下方向暂只记录，不进入当前代码路径：

- 进一步提高默认 SI device cache 预算到 10-12 GiB。
- 改变 SI 迭代组织，让一个常驻窗口跑多次局部方向。
- 对 `orders` 做 GPU 端压缩格式或变长解码。
- 将 SI chunk cache 合并为单一二进制 shard/index 文件；当前只实现了 opt-in 目录 shard。

### 7. RSI plan 后续优化方案

当前 RSI streaming 每个 batch 约 `2790` 个 unique directions，完整 `8192 samples` 约 `64` 个 batch。已实现 fixed-direction cache，避免重复构建单方向 sweep plan，但每个 batch 仍会重复：

- 收集并排序 unique directions。
- 从 host/disk cache 读取多个方向 plan。
- 将 selected directions 组装成 compact `DevicePlanChunk`。
- pack 大块 `orders` 和 metadata。
- H2D 上传到 GPU。

优先级：

1. 先拆分 `rsi_plan` 时间占比，确认 cache/load、host pack、H2D upload、sync 各占多少。
2. 将 `maxSamplesPerBatch=128` 改为环境变量，测试 `192/256`，减少 batch 数和重复 plan 组装次数。
3. 做 RSI device direction cache，SI 结束后复用释放出的 GPU 显存，常驻部分方向 plan。
4. 做 RSI 专用 shard cache，减少 per-direction 小文件 open/read 开销。
5. 评估 pinned host buffer 和 async H2D。

已加 timing 输出：

- `rsi_unique`
- `rsi_plan_breakdown_key`
- `rsi_plan_breakdown_cache`
- `rsi_plan_breakdown_build`
- `rsi_plan_breakdown_save`
- `rsi_plan_breakdown_assemble`
- `rsi_plan_breakdown_pack`
- `rsi_plan_breakdown_upload`
- `rsi_plan_breakdown_sync`

初步实测 `200k + S128 + 16 samples`：

- `rsi_plan = 14.2112 s`
- `rsi_plan_breakdown_key = 12.8007 s`
- `rsi_plan_breakdown_cache = 0.786519 s`
- `rsi_plan_breakdown_assemble = 0.0928087 s`
- `rsi_plan_breakdown_pack = 0.0170146 s`
- `rsi_plan_breakdown_upload = 0.0883757 s`
- `rsi_plan_breakdown_build = 0`
- `rsi_plan_breakdown_save = 0`

结论：

- 当前 RSI plan 的主要瓶颈是 cache key 生成，约占 `rsi_plan` 的 90%。
- 直接原因是 fixed-direction reuse 会对许多单方向/小 fixed chunk 重复调用 `sweepPlanChunkCacheKey()`，而该函数每次都会 hash 整个 200k mesh。
- 下一步最优先应把 mesh/ordinates 的 cache key prefix 预计算一次，再对 directions 增量 hash；或为 fixed direction 直接使用预计算 direction cache key。

已实现 cache key mesh prefix 复用：

- 新增 `SweepPlanChunkCacheKeyContext`，按 `directionCount` 缓存已 hash 完的 mesh/cell/face 前缀。
- 保持原 cache key 字节顺序不变，因此已有 `results/cache` 文件仍可命中。
- SI chunk 和 RSI fixed-direction chunk 共用同一个 context。

实测 `200k + S128 + 16 samples`：

- 输出目录：`results/fig5_200k_s128_16_rsi_keyprefix/Cir/`
- CUDA internal total：`108.935 s`
- `si_total = 107.468 s`
- `si_plan = 61.2179 s`
- `si_sweep = 45.3109 s`
- `rsi_total = 1.16265 s`
- `rsi_plan = 0.775026 s`
- `rsi_plan_breakdown_key = 0.108093 s`
- `rsi_plan_breakdown_cache = 0.421822 s`
- `rsi_plan_breakdown_assemble = 0.040589 s`
- `rsi_plan_breakdown_pack = 0.0172718 s`
- `rsi_plan_breakdown_upload = 0.0883532 s`
- 4 个 CSV 均为 `194315` 行，非有限值数量为 0。

对比上一版：

- `rsi_plan_breakdown_key`: `12.8007 s -> 0.108093 s`，约 `118x`。
- `rsi_plan`: `14.2112 s -> 0.775026 s`，约 `18.3x`。
- 当前 RSI plan 的最大剩余项变为 cache load：`0.421822 s`。

### 8. SI plan breakdown

已给 streaming SI 增加 plan breakdown 输出：

- `si_plan_breakdown_key`
- `si_plan_breakdown_cache`
- `si_plan_breakdown_build`
- `si_plan_breakdown_save`
- `si_plan_breakdown_assemble`
- `si_plan_breakdown_pack`
- `si_plan_breakdown_upload`
- `si_plan_breakdown_sync`

实测 `200k + S128 + 16 samples`：

- 输出目录：`results/fig5_200k_s128_16_si_plan_breakdown/Cir/`
- CUDA internal total：`104.618 s`
- `si_total = 102.932 s`
- `si_plan = 56.5946 s`
- `si_sweep = 45.3806 s`
- `si_plan_breakdown_key = 0.0620618 s`
- `si_plan_breakdown_cache = 27.757 s`
- `si_plan_breakdown_pack = 4.38478 s`
- `si_plan_breakdown_upload = 22.1714 s`
- `si_plan_breakdown_sync = 0.34358 s`
- `si_plan_breakdown_build = 0`
- `si_plan_breakdown_save = 0`
- 4 个 CSV 均为 `194315` 行，非有限值数量为 0。

结论：

- SI plan 当前不再是 key 生成瓶颈。
- 最大项是 host cache 读取/反序列化和 H2D 上传，二者合计约 `49.9 s`。
- 下一步优先方向应是减少重复 host cache load 和 H2D upload，例如扩大/调整 device plan cache、做 SI host chunk 常驻全覆盖、或把常用 chunk 固定常驻 GPU。

已继续测试 SI device cache 常驻策略：

- `RSI_CUDA_SI_PLAN_COST_ADMISSION=1` 单独开启，8 GiB 预算：
  - CUDA internal total：`103.318 s`
  - `si_plan = 56.2432 s`
  - `si_sweep = 45.0322 s`
  - cached chunks：`86/130`
  - 对比顺序常驻收益很小，说明各 chunk plan 成本差异不大。
- `RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=12000`：
  - cached chunks：`126/130`
  - `si_plan = 20.6288 s`
  - `si_sweep = 104.725 s`
  - CUDA internal total：`126.795 s`
  - 计划时间下降明显，但显存压力导致 sweep 变慢，不适合作默认。
- `RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=10000`：
  - cached chunks：`105/130`
  - `si_plan = 39.851 s`
  - `si_sweep = 45.3119 s`
  - CUDA internal total：`87.6393 s`
  - 当前比 8 GiB 和 12 GiB 都更合适，后续可进一步扫 `9000/10000/11000`。
- 已把默认 SI device cache 预算定为 10 GiB。

SI packed host plan cache 实测：

- 开关：`RSI_CUDA_SI_HOST_PLAN_CACHE_MB=<MiB>`，默认 `4096`。
- 目的：缓存已经 pack 好的 `HostPlanChunk`，避免未进入 device cache 的 SI chunk 在后续迭代重复 disk/host cache load 和 pack。
- `200k + S128 + 16 samples`，默认 10 GiB device cache，4 GiB host plan cache：
  - 输出目录：`results/fig5_200k_s128_16_si_hostcache_default10gb/Cir/`
  - host cache：`25/130` chunks，`2.49 GB`，hits `309`，misses `146`
  - CUDA internal total：`76.4524 s`
  - `si_plan = 37.9682 s`
  - `si_sweep = 36.0905 s`
  - `si_plan_breakdown_cache = 18.8991 s`
  - `si_plan_breakdown_pack = 1.07792 s`
  - `si_plan_breakdown_upload = 14.6367 s`
  - 4 个 CSV 均为 `194315` 行，非有限值数量为 0。
  - 对比 fused default 输出，SI/RSI/RSI-tail 逐值 `max_abs = 0`。
- 默认关闭 host cache 的一次回归中，`si_plan` 升到 `49.2553 s`，其中 `cache = 29.6315 s`，说明磁盘/OS cache 波动会明显影响 plan。
- 结论：host cache 能稳定降低 cache/pack 抖动，默认启用 4 GiB；若内存紧张，可设 `RSI_CUDA_SI_HOST_PLAN_CACHE_MB=0` 关闭。

上传路径和压缩方向：

- 当前 `upload` 仍约 `14 s`，是 plan 的主要剩余项之一。
- 已实现 temporary `DevicePlanChunk` buffer 复用：
  - `DeviceArray` 记录 capacity，重复上传到同一个 scratch chunk 时复用已有 device allocation。
  - 未进入 device cache 的 SI chunk 上传到 `siTemporaryPlanChunk`，避免每轮反复 `cudaMalloc/cudaFree`。
  - cached device chunk 仍使用独立 `DevicePlanChunk`，保证常驻计划的生命周期不受 scratch 影响。
- `200k + S128 + 16 samples`，默认 10 GiB device cache + 4 GiB host plan cache + temporary buffer reuse：
  - 输出目录：`results/fig5_200k_s128_16_temp_buffer_reuse/Cir/`
  - CUDA internal total：`84.0145 s`
  - `si_plan = 43.6488 s`
  - `si_sweep = 38.5557 s`
  - `si_plan_breakdown_cache = 25.067 s`
  - `si_plan_breakdown_pack = 1.04606 s`
  - `si_plan_breakdown_upload = 13.9563 s`
  - host cache：`25/130` chunks，hits `313`，misses `142`
  - 4 个 CSV 均为 `194315` 行，非有限值数量为 0。
  - 对比 fused default 输出，SI/RSI/RSI-tail 逐值 `max_abs = 0`。
- 结论：temporary buffer reuse 小幅降低 upload（约 `14.64 s -> 13.96 s`），但本轮 cache 波动使端到端没有改善；保留该实现，因为它降低了分配抖动且不改变数值。
- 下一步如果继续压 upload，更可能需要 pinned host memory 和 async H2D；这需要重构 host buffer 生命周期和 CUDA stream 管理，暂不默认做。
- 压缩 `orders` 对磁盘 cache 可能有帮助，但 200k cell 无法用 `uint16`，GPU 端压缩还需要解码 kernel，风险较高，暂只记录。

### 9. SI sweep breakdown

已增加 opt-in sweep 细分开关：

```text
RSI_CUDA_SI_SWEEP_BREAKDOWN=1
```

该开关会在 SI sweep 内部插入额外 CUDA event 同步，仅用于定位，不作为默认性能路径。

实测 `200k + S128 + 16 samples`，`RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=10000`：

- 输出目录：`results/fig5_200k_s128_16_si_sweep_breakdown_10gb/Cir/`
- CUDA internal total：`88.0614 s`
- `si_total = 85.7364 s`
- `si_plan = 39.6558 s`
- `si_sweep = 45.5002 s`
- `si_sweep_breakdown_angular_clear = 0.559341 s`
- `si_sweep_breakdown_kernel = 43.6015 s`
- `si_sweep_breakdown_accumulate = 0.541548 s`
- `si_sweep_breakdown_sync = 0.0254582 s`
- 4 个 CSV 均为 `194315` 行，非有限值数量为 0。

结论：

- SI sweep 主要耗时在 wavefront/level sweep kernel 本身，约占 `si_sweep` 的 96%。
- angular buffer 清零和方向累加各约 `0.55 s`，不是当前瓶颈。
- chunk 末尾同步几乎可以忽略。
- 下一步若优化 sweep，应看 `sweepLevelTiledKernel` 的访存/并行度/level 调度，而不是 clear/reduce。

### 10. SI fused wavefront kernel

已增加 streaming SI fused wavefront 路径：

```text
RSI_CUDA_SI_FUSED_WAVEFRONT=1  # default
```

实现方式：

- 对 acyclic wavefront，每个 direction 使用一个 CUDA block。
- 在 `sweepLevelFusedKernel` 内部按 level 顺序循环。
- level 之间用 block 内 `__syncthreads()` 保证同一 direction 的依赖顺序。
- cyclic direction 仍走原 sweep sample fallback。
- 可用 `RSI_CUDA_SI_FUSED_WAVEFRONT=0` 回退到原 tiled level-by-level launch。

动机：

- `nsys` 显示 200k/S128 smoke 中 `cudaLaunchKernel` 调用约 `742707` 次，API 总耗时约 `7.78 s`。
- 原 tiled wavefront 是 chunk × level × iteration 启动 kernel，launch 数很高。
- fused kernel 用一个 launch 覆盖一个 direction batch 的所有 levels，显著减少 launch 数。

实测 `200k + S128 + 16 samples`，`RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=10000`：

- 原 tiled 10 GiB：
  - CUDA internal total：`87.6393 s`
  - `si_plan = 39.851 s`
  - `si_sweep = 45.3119 s`
- fused wavefront 显式开启：
  - 输出目录：`results/fig5_200k_s128_16_fused_wavefront/Cir/`
  - CUDA internal total：`75.4776 s`
  - `si_plan = 41.1578 s`
  - `si_sweep = 32.4425 s`
- fused wavefront 默认开启后回归：
  - 输出目录：`results/fig5_200k_s128_16_fused_default/Cir/`
  - CUDA internal total：`71.55 s`
  - `si_total = 70.2645 s`
  - `si_plan = 37.7765 s`
  - `si_sweep = 31.887 s`
  - cached chunks：`105/130`
  - 4 个 CSV 均为 `194315` 行，非有限值数量为 0。
  - 对比原 tiled 10 GiB 的 SI/RSI/RSI-tail CSV，逐值 `max_abs = 0`。

小网格补充：

- `30k + S40 + 16 samples` 强制 streaming 下，fused `si_sweep = 26.3354 s`，原 tiled `si_sweep = 23.0909 s`。
- 因此 fused 默认主要面向 200k/S128 这类大网格高角度场景；若小网格强制 streaming，可用 `RSI_CUDA_SI_FUSED_WAVEFRONT=0` 回退。

## 当前推荐优先级

1. 用 fused wavefront + 10 GiB SI device plan cache 跑完整 200k/S128/8192，确认端到端收益。
2. 继续 profile/优化 `sweepLevelFusedKernel`，重点看单 direction block 内的 level 循环和访存。
3. 继续扫 SI device cache 预算，重点测试 `9000/10000/11000`，避免 12 GiB 显存压力导致 sweep 退化。
4. 对 RSI fixed-direction chunk 增加 device/host shard 常驻，继续压低 cache load。
5. 将 `maxSamplesPerBatch=128` 改为环境变量，测试 `192/256`，减少 full 8192 的 batch 数。
6. 基于 cache 命中/常驻统计自动选择预算。
7. SI chunk size 可配置，继续测试 64/128/256。
