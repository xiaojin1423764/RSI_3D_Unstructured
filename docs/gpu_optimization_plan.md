# GPU Figure 5 优化计划

本文记录 3D 非结构网格 RSI/SI Figure 5 CUDA 路径的当前结论、已验证问题和后续优化顺序。目标指标以完整 Figure 5 运行时间为准：`SI_coarse + SI_fine + RSI + RSI_tail + CSV write`，参数固定为 `S4/S32`、`512 samples`、`tailExtra=10`。

## 当前基线与结论

已验证的 30k 网格完整 Figure 5 对比：

| 版本 | wall time | 说明 |
|---|---:|---|
| CPU | 266.32 s | `gmsh_work/data`，36,470 cells |
| GPU direct-global-tail 版本 | 145.23 s | 数值逐 CSV 一致，但比旧 GPU 基线慢 |
| GPU SI cycle-run + 5M direction batch | 75.27 s | `NVCCARCH=sm_120`，快于旧 GPU 基线 89.83 s |
| GPU 分阶段计时版 | 78.86 s | 计时事件有少量同步开销；用于定位瓶颈 |

CPU/GPU 四个 CSV (`SI_coarse`、`SI_fine`、`RSI`、`RSI_tail`) 对比为逐值一致，plain relative L2 和 max abs 均为 0。

direct-global-tail 版本的问题：

- 它把 `RSI_tail` 从 batch-local 累积改成每个 tail iteration 直接 reduce 到全局 cell sum。
- `tailExtra=10` 时，实际需要累计 `N..N+10` 共 11 个 iteration。
- 对 512 samples、128 samples/batch，需要 4 个 batch，即 44 次 per-cell global reduction。
- 30k 下 reduction 开销抵消了 batch 容量提升，完整 GPU 时间变为 145.23 s。

因此当前代码应保留：

- sample batch workspace 一次分配并复用；
- batch capacity 由显存估算和 `128` 上限决定；
- RSI/RSI-tail 共用同一批随机链；
- CPU/GPU 逐 sample 确定性 RNG。

当前代码不应保留：

- 每个 tail iteration 直接 reduce 到 global tail sum 的策略。

## 优先级 1：恢复 batch-local tail accumulation

推荐策略：

```text
每个 sample batch:
  psiA, psiB, sampleTail 一次分配并复用
  每个 tail iteration:
    sampleTail += currentPsi
  batch 结束:
    reduce sampleTail -> globalTailSum
```

这样每个 batch 只做一次 tail reduce，而不是每个 tail iteration 做一次。显存占用回到每 sample 约 `3*C*sizeof(double)`，但 16 GB RTX 5070 Ti 上 30k/200k 都应仍能达到较大的 batch size；原来 200k 的约 25 samples/batch 主要来自人工 `5000000/C` 上限，不是显存硬限制。

验收：

- `make gpu_consistency_test`
- 沙箱外运行 `./gpu_consistency_test`
- 30k 完整 Figure 5 GPU 时间应低于 145.23 s，并尽量接近旧基线 89.83 s。
- CPU/GPU CSV 对比容差：relative L2 `5e-10`，max abs `5e-9`；若保持确定性顺序，优先要求逐值一致。

## 优先级 2：加入分阶段计时

在继续优化 kernel 前，必须增加粗粒度计时，否则难以判断收益来源。

建议记录：

| 阶段 | 内容 |
|---|---|
| problem upload | mesh、ordinates、weights、sweep orders 上传 |
| SI total | 完整 SI fine 时间 |
| SI sweep | 所有 direction batch sweep kernel |
| SI reduce | `angularPsi -> phi` 归约 |
| SI norm | 收敛范数计算和 2-double D2H copy |
| RSI schedule | CPU schedule 生成 |
| RSI direction copy | 每个 batch 的 schedule H2D |
| RSI sweep | sample chain sweep kernels |
| RSI tail accumulate | batch-local tail 累积 |
| RSI reduce | RSI 和 tail 的 batch-end reduction |
| result copy | final `phi` D2H |

输出格式保持机器可读，例如：

```text
CUDA timing: upload=... si_total=... si_sweep=... si_reduce=... rsi_total=...
```

计时完成后的决策规则：

- 如果 `si_clear + si_reduce + si_norm` 占 SI 比例高，优先做 SI direction streaming
  和 reduce/norm 融合。
- 如果 `si_sweep` 仍占绝大部分，优先研究 CUDA Graph、持久 kernel 或 cell-level /
  wavefront 并行 sweep。
- 如果 `rsi_sweep` 占比高，优先优化 sample chain sweep 和 cyclic pass 策略。
- 如果 `rsi_tail_accum + rsi_reduce` 占比高，优先做 RSI batch 自适应和 tail/reduce
  kernel 融合。
- 如果 `rsi_direction_copy` 或 checkpoint 时间异常，优先处理 Host/Device 传输和
  checkpoint 频率。

下一步先跑 30k 完整 Figure 5 确认 75.27 s 版本的阶段占比，再跑 200k 完整
Figure 5 判断大网格瓶颈是否相同。不要在没有阶段占比前继续改 kernel。

30k 分阶段计时结果：

| 阶段 | 时间 |
|---|---:|
| upload | 0.26 s |
| SI total | 37.01 s |
| SI sweep | 35.94 s |
| SI clear | 0.007 s |
| SI reduce | 0.006 s |
| SI norm | 0.004 s |
| RSI total | 41.58 s |
| RSI sweep | 40.30 s |
| RSI tail accumulate | 0.007 s |
| RSI reduce | 0.001 s |
| RSI direction copy | 0.001 s |

30k 结论：剩余瓶颈几乎全部在 sweep kernel 本体，`angularPsi` 清零、方向归约、
收敛范数、tail accumulation 和 Host/Device 拷贝都不是主要瓶颈。若 200k 也呈现
相同占比，下一步应优先做 sweep kernel / pass 结构优化或 CUDA Graph，而不是 SI
streaming。

200k 分阶段计时结果：

| 阶段 | 时间 |
|---|---:|
| upload | 0.60 s |
| SI total | 380.75 s |
| SI sweep | 369.72 s |
| SI clear | 0.033 s |
| SI reduce | 0.031 s |
| SI norm | 0.009 s |
| RSI total | 65.72 s |
| RSI sweep | 63.66 s |
| RSI tail accumulate | 0.035 s |
| RSI reduce | 0.002 s |
| RSI direction copy | 0.005 s |

200k 结论：完整 GPU Figure 5 为 `447.08 s`（`/usr/bin/time` wall 为 `460.03 s`），
相比旧 GPU 基线 `1417.84 s` 已明显下降。瓶颈仍然几乎全部在 sweep kernel：
SI sweep 占总时间约 82.7%，RSI sweep 占约 14.2%。因此下一步不应优先做
SI streaming、reduce/norm 融合或传输优化，而应优先减少 RSI/SI sweep 的无效
pass 和 kernel launch，随后再评估 CUDA Graph 或 sweep kernel 并行粒度重构。

已验证但不保留的 RSI pass 尝试：按每个 RSI batch/iteration 的 selected
directions 判断是否存在 cyclic direction；若没有则只 launch 1 pass。30k 完整
Figure 5 变为 `80.62 s`，慢于分阶段计时基线 `78.86 s`，因此已回退。后续不要
重复这个 batch 级 RSI pass 判断，除非能按 sample 子组更细粒度处理 cyclic 样本。

## 优先级 3：跳过无环方向的空 localPass

当前 sweep kernel 中有：

```cpp
if (localPass > 0 && !hasCycle) return;
```

这避免了无环方向重复计算，但 host 侧仍然启动了 19 次空 kernel。优化方向：

```text
如果 direction batch 全部无环:
  localPassCount = 1
否则:
  localPassCount = 20
```

已验证的简单 batch 级判断会回归：如果一个 batch 中混入少量 cyclic direction，
整个 batch 仍会跑 20 pass，无法真正减少 launch/空转，还会增加 host 侧判断开销。
因此下一版必须按 `hasCycle` 连续区间拆分 SI 方向：

```text
for each contiguous direction run with same hasCycle:
  if acyclic:
    launch localPass=0 only
  else:
    launch localPass=0..19
```

这样保持 `angularPsi[direction*C + cell]` 的原有布局，不需要 scatter，也不会改变
浮点归约顺序。

已实现结果：

- SI direction batch 从 `min(64, 2500000/C)` 调整为 `min(128, 5000000/C)`。
- 每个 direction batch 内再按 `hasCycle` 连续区间拆分；无环区间只 launch
  `localPass=0`，有环区间保持 20 pass。
- 30k 完整 Figure 5 GPU wall time 为 `75.27 s`，低于旧基线 `89.83 s`。
- 默认 `gpu_consistency_test` 通过，Figure5 小规模 CPU/GPU relative L2 约
  `1e-16` 量级。

验收指标：

- SI fine 和 RSI sweep kernel launch 数下降。
- 数值不变。

## tycho2 参考：cell-angle wavefront sweep

`lanl/tycho2` 的 SI/sweep 并行不能直接移植，但它的调度模型值得参考。tycho2
把 sweep 工作单元定义为 `(cell, angle)`，用 upwind 依赖数维护 ready queue；
依赖数归零的 work 进入当前可执行集合，再按 priority 选择执行顺序。相关文件：

- `SweepSchedule.cc`：预生成每个 step 的 work list，每个 step 内可并行。
- `GraphTraverser.cc`：运行时遍历依赖图，并处理 MPI 边界数据。
- `Sweeper.cc`：按 angle group 分给 OpenMP thread，按 schedule 执行。
- `SweeperPBJ.cc` / `SweeperSchur.cc`：偏域分解和边界迭代，暂不作为单 GPU SI 的第一步。

本项目当前 CUDA sweep 的工作单元更粗：一个 CUDA work item 对应一个
direction/sample，进入 `sweepSamplesKernel` 后仍沿 `order[position]` 串行扫完所有
cell。因此 SI 虽然已有角度并行，但单方向内部没有打开 tycho2 式 wavefront 并行。

落地顺序：

1. 扩展 `SweepPlan`，在原有 `order` 外记录 `levelOffsets + levelCells`。
   同一 level 内的 cell upwind 依赖已经满足，可以并行计算。
2. 先只输出统计：acyclic/cyclic 方向数、level 数、平均/最大 wavefront 宽度。
   真实 30k/200k 网格的 level width 足够大时，才值得改 CUDA kernel。
3. 第一版 GPU wavefront 只处理 acyclic 方向；cyclic 方向继续使用当前
   投影排序 + 20 pass Gauss-Seidel fallback。
4. wavefront kernel 必须保持 `angularPsi[direction*C + cell]` 布局，避免同时扰动
   reduction 和 RSI 路径。

验收指标：

- 增加 level 统计后，CPU/GPU 数值不变。
- 若进入 GPU wavefront 实作，先用小规模 `gpu_consistency_test` 验证，再跑 30k
  Figure 5 比较 `si_sweep` 占比。

快速统计命令：

```bash
./rsi_unstructured --only sweep-stats gmsh_work/data/cells.csv gmsh_work/data/faces.csv
```

当前统计结果：

| 网格 | 角度 | acyclic | cyclic | avg levels | max levels | avg width | max width |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30k `gmsh_work/data` | S4 | 24 | 0 | 249.583 | 264 | 146.124 | 265 |
| 30k `gmsh_work/data` | S32 | 1086 | 2 | 230.652 | 287 | 158.117 | 570 |
| 200k `gmsh_work/mesh200k` | S4 | 24 | 0 | 421.75 | 442 | 460.733 | 929 |
| 200k `gmsh_work/mesh200k` | S32 | 1088 | 0 | 372.072 | 446 | 522.249 | 1759 |

结论：两个网格的 S32 绝大多数方向都是 acyclic，且 wavefront 平均宽度在 30k
约 `158`、200k 约 `522`。这说明下一步值得优先做 acyclic 方向的 GPU
level-synchronous sweep；cyclic 方向保留现有 fallback 即可。

已实现的第一版：

- `SweepPlan` 记录 acyclic 方向的 `levelOffsets`，CUDA 端复用已有
  `orders[M*C]` 作为 level cell 列表，不额外上传一份 `levelCells`。
- CUDA SI 中 acyclic 连续方向区间走 `sweepLevelKernel`，每个 level 一个 kernel，
  每个 block 对应一个 direction，block 内线程并行处理该 level 的 cells。
- cyclic 方向仍走旧 `sweepSamplesKernel` 和 20 pass fallback。
- 可用 `RSI_CUDA_SI_WAVEFRONT=0` 临时关闭 wavefront，回退到旧 SI sweep 路径。

注意：新 kernel 中每个 cell 的 face 累加在单线程内完成，旧 kernel 使用 4-lane
warp reduction；二者可能有末位浮点差异，因此验收应使用现有 relative L2/max abs
容差，而不是逐字节比较。

30k GPU 实测，同一机器、输出到 `/tmp`：

| 模式 | total | SI total | SI sweep | RSI total | RSI sweep |
|---|---:|---:|---:|---:|---:|
| wavefront on | 60.1665 s | 23.1602 s | 23.1327 s | 36.7244 s | 36.6884 s |
| `RSI_CUDA_SI_WAVEFRONT=0` | 62.1134 s | 25.0863 s | 25.0592 s | 36.7843 s | 36.7479 s |

开启 wavefront 后 30k 完整 Figure 5 缩短约 `1.95 s`，SI sweep 缩短约
`1.93 s`。`/tmp/rsi_wavefront_on` 与 `/tmp/rsi_wavefront_off` 的四个 CSV
逐值一致：`SI_coarse`、`SI_fine`、`RSI`、`RSI_tail` 的 plain relative L2 和
max abs 均为 0。

RSI combined sweep 也已接入 wavefront，默认开启：

- 对每个 sample batch 的每个 RSI iteration，若该 iteration 内所有 sample 选中
  方向都是 acyclic，则走 `sweepLevelKernel`。
- 若任一 sample 选中 cyclic 方向，则回退旧 `sweepSamplesKernel` 和 20 pass。
- 可用 `RSI_CUDA_RSI_WAVEFRONT=0` 关闭 RSI wavefront。
- `RSI_CUDA_RSI_BATCH_PASS=1` 只跳过全 acyclic iteration 的空 pass，但 30k 收益
  很小，保持实验开关，不默认启用。

30k GPU 实测：

| 模式 | total | SI total | SI sweep | RSI total | RSI sweep |
|---|---:|---:|---:|---:|---:|
| SI+RSI wavefront | 53.0857 s | 23.0132 s | 22.9855 s | 29.7881 s | 29.7573 s |
| SI wavefront only | 60.1665 s | 23.1602 s | 23.1327 s | 36.7244 s | 36.6884 s |

RSI wavefront 使 30k 完整 Figure 5 再缩短约 `7.08 s`，其中 RSI sweep 缩短约
`6.93 s`。`/tmp/rsi_rsi_wavefront_on` 与 `/tmp/rsi_wavefront_on` 的四个 CSV
逐值一致。

## 下一步 profiling 和优化方向

当前 30k 默认路径已降到约 `53.09 s`，主要时间仍在 sweep：

- SI sweep: `22.99 s`
- RSI sweep: `29.76 s`
- 其他阶段合计小于 `1 s`

下一步优化必须先用 Nsight 定位，不再凭猜测改 kernel。profiling 目标：

1. Nsight Systems：确认 wavefront 后的瓶颈是 kernel 执行时间还是大量 level
   kernel launch / host 间隙。重点看 `sweepLevelKernel` 的 launch 密度、GPU
   idle gap、RSI batch 间同步。
2. Nsight Compute：抽样 `sweepLevelKernel`，看 occupancy、memory throughput、
   register pressure、warp stall 原因。当前每个 cell 的 faces 在单线程串行处理；
   如果 occupancy 或 instruction throughput 低，应考虑 warp/4-lane cell 更新。
3. Nsight Compute：抽样旧 `sweepSamplesKernel`，确认 cyclic fallback 和独立
   `runRSIFieldAtNCuda` 路径是否仍值得维护或优化。
4. 200k 新基线：200k 的平均 wavefront width 更大，可能更受益于 level 并行，也
   可能更受 kernel launch 数限制。必须复测后再决定 CUDA Graph / persistent
   kernel 的优先级。

候选优化顺序：

1. 若 Nsight Systems 显示大量 launch gap：优先做 CUDA Graph，先捕获 SI 固定
   level sweep 结构，再评估 RSI batch/iteration 结构。
2. 若 `sweepLevelKernel` 本体占满时间且 SM/warp 效率低：改 kernel mapping，
   恢复 face-lane 并行或 warp-per-cell，同时保留 level wavefront。
3. 若 RSI iteration 里少量 cyclic direction 导致整批回退：把 samples 分为
   acyclic/cyclic 两组，acyclic 组走 wavefront，cyclic 组走旧 fallback。
4. 若 200k 显存或 batch size 成为限制：再做 SI direction streaming；否则
   streaming 仍不是优先项。

Profiling 输出约定：

```bash
results/profiles/wavefront_30k/
```

已完成的 profiling：

- NSYS 30k 默认 wavefront：`fig5_30k_default.nsys-rep`。该环境下 kernel summary
  未导出 CUDA kernel 数据，但 CUDA API summary 显示 `cudaLaunchKernel` 共
  `29182` 次，总 API 时间约 `0.34 s`；主要等待在 `cudaEventSynchronize`。因此
  当前 30k 更像 kernel 本体/同步等待主导，而不是 host launch API 本身主导。
- NCU `sweepLevelKernel` S32 小 run：前几个 level launch 只有 `grid_dim_x=3`，
  `waves_per_sm=0.01`，SM throughput 约 `1%`。这是由 S32 的两个 cyclic 方向把
  acyclic run 切碎造成的。
- NCU `sweepLevelKernel` S32 主 run：跳过前 320 个 launch 后，典型 launch 为
  `grid_dim_x=1083`、`block_size=256`、`registers/thread=40`、`waves_per_sm=2.58`、
  duration 约 `178 us`、SM throughput 约 `32%`。
- NCU detailed 主 run：Memory throughput `91.44%`，L2 throughput `91.44%`，
  L1/TEX throughput `54.11%`，DRAM throughput `30.01%`，L1 hit rate `4.05%`，
  L2 hit rate `88.83%`，achieved occupancy `79.31%`。NCU 报告 global access
  uncoalesced，excessive sectors 约 `77%`。
- NCU `sweepSamplesKernel` fallback：抽样 launch `grid_dim_x=1`，单次 kernel
  `119-174 ms`，SM throughput 约 `0.11-0.12%`。fallback 很差，但 30k S32 只有
  2 个 cyclic 方向，当前不是主路径。

根据 NCU 先做了低风险改动：acyclic wavefront 的每个 level 内按 cell id 排序。
level 内 cell 互不依赖，排序不改变数学结果，但改善部分 cell/geometry 数组访问局部性。

30k 实测：

| 模式 | total | SI sweep | RSI sweep |
|---|---:|---:|---:|
| level 内排序后 | 52.2150 s | 22.1686 s | 29.7138 s |
| 排序前 SI+RSI wavefront | 53.0857 s | 22.9855 s | 29.7573 s |

排序后与排序前四个 CSV 逐值一致。收益主要在 SI sweep，说明 memory locality 有效，
但幅度有限。下一步应优先解决：

1. 小 acyclic run 的 `grid_dim_x=3` 碎片化：不要按方向连续 run 切分 wavefront，
   而是把所有 acyclic directions 打包到同一个 level kernel，并用 per-direction
   level count 跳过已结束方向。
2. 主 run 的 uncoalesced global access：重排 mesh SoA 或在 wavefront kernel 中
   改 cell/face 映射，减少 `refFace/refNeighbor/face*` 随机访问。
3. 旧 cyclic fallback：30k 影响小，200k S32 为 0 个 cyclic，暂不优先。

已实现 acyclic direction packing：SI wavefront 不再按 `hasCycle` 连续 run 切分，
而是把所有 acyclic directions 一起 launch；cyclic directions 单独走旧 fallback。

30k 实测：

| 模式 | total | SI sweep | RSI sweep |
|---|---:|---:|---:|
| level sort + SI packed acyclic | 52.9031 s | 21.3066 s | 31.2854 s |
| level sort only | 52.2150 s | 22.1686 s | 29.7138 s |

SI packed acyclic 使 SI sweep 再降约 `0.86 s`，但该次 RSI sweep 波动偏高，导致
total 未改善。四个 CSV 与 level sort only 逐值一致。该改动对 SI 是正收益，
是否作为长期默认需结合 200k 和重复 30k 运行判断。

200k 当前默认新基线：

```text
/usr/bin/time wall: 30.12 s
CUDA Figure 5 internal total: 5.81625 s
upload: 0.522733 s
SI total: 2.5183 s
SI sweep: 2.41999 s
RSI total: 2.76454 s
RSI sweep: 2.67064 s
```

200k S32 全部 `1088` 个方向都是 acyclic，平均 wavefront width 约 `522`，因此
wavefront 对大网格收益非常大。相比旧 200k 分阶段计时 `447.08 s`，GPU fine
SI+RSI 部分已降到约 `5.82 s`。现在完整命令的主要剩余开销已经转移到 CUDA
计时外。当时的主要剩余项是：

- 粗角度 S4 `SI_coarse` 仍走 CPU `runSIField`。
- S32 sweep plan 构建约 `7.56 s`。
- CSV 写出和其他主机端流程。

因此 200k 下一步不应继续先抠 GPU fine sweep，而应优先：

1. 把 Figure 5 的 coarse S4 SI 也放到 CUDA 路径，避免完整 workflow 仍被 CPU
   coarse SI 卡住。
2. 缓存/复用 sweep plan，或把 S32 plan 构建结果序列化，避免重复运行时每次花
   数秒重建。
3. 如需继续优化 GPU kernel，再看 200k NCU；但当前完整 workflow 已由 host 侧
   阶段主导。

已实现 workflow 优化：`--gpu --only figure5` 下，`SI_coarse` 也走 CUDA 后端。
当前实现复用 `runFigure5GPU(1, 0)` 并只取其中的 SI 场，因此会多算一个很小的
S4/1-sample RSI；200k 下这部分 RSI 约 `0.14 s`，可接受。后续可拆正式
SI-only CUDA API 去掉这点开销。

200k 实测：

| 模式 | `/usr/bin/time` wall | coarse CUDA total | fine CUDA total | fine SI sweep | fine RSI sweep |
|---|---:|---:|---:|---:|---:|
| coarse SI on GPU | 16.60 s | 0.578428 s | 5.76225 s | 2.41558 s | 2.73349 s |
| coarse SI on CPU | 30.12 s | n/a | 5.81625 s | 2.41999 s | 2.67064 s |

coarse SI GPU 化后，200k 完整命令 wall time 降低约 `13.5 s`。四个 CSV 与
coarse CPU 版本逐值一致。

## 当前后续优化计划

200k 当前完整 workflow 已从 GPU sweep 主导转为 host 侧流程主导。后续优先级按
完整命令 wall time 排序，而不是只看 CUDA internal total。

### 已完成：正式 SI-only CUDA API

`runSIFieldCuda(..., int& convergedN)` 已拆出并接入 `RSISolver::runSIField()`。
因此 `--gpu --only figure5` 下的 `SI_coarse` 现在走正式 SI-only CUDA 路径，
不再复用 `runFigure5GPU(1, 0)`，也不再额外执行 S4/1-sample RSI。

```cpp
std::vector<double> runSIFieldCuda(..., int& convergedN)
```

当前状态：

- `make gpu` 通过。
- `make gpu_consistency_test` 通过编译/链接。
- 沙箱内运行 `./gpu_consistency_test` 会在 CUDA 初始化阶段失败：
  `CUDA driver version is insufficient for CUDA runtime version`。
- 沙箱外运行 `./gpu_consistency_test` 通过。小规模 CPU/GPU consistency
  relative L2 为 `2.01657e-16`；Figure 5 的 SI、RSI、RSI-tail relative L2
  分别为 `1.60756e-16`、`1.88684e-16`、`2.21209e-16`。

### 已完成：sweep plan 序列化 / 缓存

200k S32 sweep plan 构建约 `7.56 s`，已经大于 fine CUDA internal total
`5.76 s`。这是当前最有价值的 host 侧优化。

当前实现：

- cache key 基于网格 cell/face 几何和连接、ordinates、`angularN`、
  `sourceShape` 以及 cache 格式版本。
- 二进制 cache 保存每个方向的 `order`、`levelOffsets`、`hasCycle`。
- 默认启用；`RSI_SWEEP_PLAN_CACHE=0` 可关闭。
- `RSI_SWEEP_PLAN_CACHE_DIR=/path` 可指定 cache 目录；未设置时写到
  `results/cache/`。
- cache 缺失、key/version 不匹配时自动重建。
- cache 截断或损坏时输出 warning 并自动重建，避免坏 cache 终止计算。

30k `sweep-stats` 验证，cache 目录为 `/tmp/rsi_sweep_cache_test`：

| 网格/角度 | cold build | warm load | cache size |
|---|---:|---:|---:|
| 30k S4 | 0.0102 s | 0.0029 s | 3.5 MB |
| 30k S32 | 0.4175 s | 0.2135 s | 159.7 MB |

200k Figure 5 cache 验证，cache 目录为 `/tmp/rsi_sweep_cache_200k_0714`：

| 模式 | wall | S4 plan | S32 plan | fine CUDA total | fine SI sweep | fine RSI sweep |
|---|---:|---:|---:|---:|---:|---:|
| cold cache | 14.82 s | build 0.149 s + save 0.015 s | build 6.877 s + save 0.791 s | 5.307 s | 2.270 s | 2.529 s |
| warm cache | 8.28 s | load 0.024 s | load 1.093 s | 5.530 s | 2.271 s | 2.542 s |

Cold/warm 输出目录分别为 `/tmp/rsi_fig5_200k_cold` 和 `/tmp/rsi_fig5_200k_warm`。
四个 Figure 5 CSV (`SI_coarse`、`SI_fine`、`RSI`、`RSI_tail`) 已用 `cmp -s`
确认逐字节一致。warm-cache 完整 wall time 已低于原目标 `9-10 s`。

### 优先级 3：200k NCU，判断是否继续改 GPU sweep

30k NCU 已显示 `sweepLevelKernel` 主要问题是 uncoalesced global access，excessive
sectors 约 `77%`。200k 的 wavefront 更宽，可能暴露不同瓶颈；在 host cache 完成前，
GPU kernel 不是当前最大 wall-time 来源。

若继续做 GPU kernel，应先跑 200k NCU：

```bash
ncu --set basic --kernel-name regex:sweepLevelKernel --launch-skip ... --launch-count ...
```

根据结果决定：

- 若 memory/L2 throughput 仍接近满载：优先做 GPU-specific mesh SoA / face 数据
  重排，减少 `refFace/refNeighbor/face*` 随机访问。
- 若 instruction 或 warp stall 主导：考虑 warp-per-cell 或恢复 face-lane 并行。
- 若 launch gap 明显：再评估 CUDA Graph；目前 30k 的 `cudaLaunchKernel`
  API 总时间只有约 `0.34 s`，不是首要瓶颈。

200k profiling 已完成，输出位于 `results/profiles/wavefront_200k/`：

- NSYS warm-cache：`fig5_200k_warm.nsys-rep` / `.sqlite`。当前环境仍未导出
  CUDA kernel summary，但 CUDA API summary 显示 `cudaLaunchKernel` 共 `54933`
  次，总 API 时间约 `0.532 s`；`cudaEventSynchronize` 约 `4.787 s`。因此主要
  等待仍是 GPU kernel 执行/同步，不是 host launch API 本身。
- NCU fine S32 `sweepLevelKernel`：grid `1088`，duration `252.38 us`，
  memory throughput `78.10%`，L2 throughput `78.10%`，SM throughput `24.40%`，
  achieved occupancy `70.71%`，L1 hit `3.50%`，L2 hit `78.42%`，excessive
  sectors `78%`。
- NCU RSI `sweepLevelKernel`：grid `128`，duration `128.10 us`，waves/SM
  `0.37`，achieved occupancy `34.65%`，excessive sectors `78%`。RSI 除了
  uncoalesced 外还有 grid 太小的问题。

已实现两个基于 profiling 的实验改动：

1. RSI tiled wavefront：对 RSI wavefront 增加 level 内 tile 维度，默认仅在
   `batchSize >= 64` 且 level width 需要多个 tile 时启用；可用
   `RSI_CUDA_RSI_TILED_WAVEFRONT=0` 关闭。200k 下 grid 从 `128` 增至
   `128 x 7 = 896`，achieved occupancy 从 `34.65%` 提高到 `59.47%`。
2. GPU mesh ref-level SoA：上传时把每个 cell-face ref 的 outward normal、area、
   face center、boundary type/value/source fraction 展平成按 `refIndex` 连续的
   SoA，sweep kernel 不再经 `refFace -> face arrays` 二次随机读取。
3. SI tiled wavefront：SI packed acyclic wavefront 也支持 level 内 tile 维度，
   默认开启；可用 `RSI_CUDA_SI_TILED_WAVEFRONT=0` 关闭。30k/200k 均验证为
   不改变 CSV，且对 SI sweep 有小到中等收益。

200k warm-cache 性能对比：

| 模式 | wall | fine CUDA total | fine SI sweep | fine RSI sweep |
|---|---:|---:|---:|---:|
| cache + SI-only CUDA 基线 | 8.28 s | 5.530 s | 2.271 s | 2.542 s |
| tiled off 对照 | 8.57 s | 5.852 s | 2.315 s | 2.601 s |
| RSI tiled on | 9.76 s | 5.716 s | 2.336 s | 2.499 s |
| ref-level SoA + auto tiled | 8.15 s | 5.579 s | 2.358 s | 2.335 s |
| ref-level SoA + SI tiled on | 8.10 s | 5.264 s | 2.081 s | 2.303 s |

`/tmp/rsi_fig5_200k_refsoa` 与 tiled-off 对照的四个 CSV 已用 `cmp -s` 确认逐字节一致。
ref-level SoA 后，fine S32 SI NCU 指标变为：duration `225.63 us`，memory throughput
`68.09%`，L2 throughput `68.09%`，L1 hit `4.26%`，L2 hit `72.90%`，excessive
sectors `75%`。说明二次 face-array 间接访问确有收益，但剩余 uncoalesced 仍然较高。

30k SI tiled 对比：

| 模式 | wall | fine SI sweep | fine RSI sweep |
|---|---:|---:|---:|
| SI tiled off | 38.77 s | 16.398 s | 21.002 s |
| SI tiled on | 38.50 s | 16.356 s | 21.041 s |

`/tmp/rsi_fig5_30k_si_tiled_on` 与 `..._off` 的四个 CSV 已用 `cmp -s` 确认逐字节一致。
因此 `RSI_CUDA_SI_TILED_WAVEFRONT` 已改为默认开启。

下一步 GPU kernel 优化应优先考虑：

1. 对 `currentPsi[direction*C + cell]` / neighbor read 做更适合 wavefront 的布局或
   cell reordering，减少非结构 neighbor 访问造成的 sector waste。
2. 评估 SI 也使用 tiled level kernel 是否能降低 partial wave/tail effect；需要
   防止 block 数变多后 launch 内工作过碎。
3. 若继续改善 memory locality，考虑把 level 内排序从单纯 cell id 排序改为基于
   face/ref 邻接的局部性排序，但必须保持同一 level 内无依赖这一前提。

### 优先级 4：保留但暂不优先的方向

- SI direction streaming：可降低显存和 `angularPsi[M*C]` 常驻空间，但当前
  200k 显存不是瓶颈，且会改变归约顺序，暂不优先。
- SI reduction + convergence norm 融合：能少一次 `C` 规模读写，但现有计时中
  reduce/norm 远小于 sweep 和 host build，不优先。
- RNG schedule 搬到 GPU：会破坏 CPU/GPU 逐 sample 可比性，且 H2D schedule 拷贝
  当前不是瓶颈。
- direct-global-tail：30k 已验证显著回归，不再继续。

## 推荐执行顺序

1. 跑 30k/200k cache on/off 的正式输出目录对比，确认提交前没有临时 `/tmp`
   路径依赖。
2. 若 warm-cache wall time 仍主要在 CUDA sweep，再跑 200k NCU 并决定 mesh SoA / face-lane /
   CUDA Graph 的优先级。
