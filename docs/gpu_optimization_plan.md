# GPU Figure 5 优化计划

本文记录 3D 非结构网格 RSI/SI Figure 5 CUDA 路径的当前结论、已验证问题和后续优化顺序。目标指标以完整 Figure 5 运行时间为准：`SI_coarse + SI_fine + RSI + RSI_tail + CSV write`，参数固定为 `S4/S32`、`512 samples`、`tailExtra=10`。

## 当前基线与结论

已验证的 30k 网格完整 Figure 5 对比：

| 版本 | wall time | 说明 |
|---|---:|---|
| CPU | 266.32 s | `gmsh_work/data`，36,470 cells |
| GPU direct-global-tail 版本 | 145.23 s | 数值逐 CSV 一致，但比旧 GPU 基线慢 |
| GPU SI cycle-run + 5M direction batch | 75.27 s | `NVCCARCH=sm_120`，快于旧 GPU 基线 89.83 s |

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

## 优先级 4：SI 方向 streaming

当前 SI fine 分配完整角通量：

```text
angularPsi[M * C]
```

空间占用：

- 30k: `1088 * 36470 * 8 ≈ 317 MB`
- 200k: `1088 * 194314 * 8 ≈ 1.69 GB`

推荐改为 direction chunk streaming：

```text
currentPhi = 0
for direction chunk:
  angularChunk = sweep(chunk)
  currentPhi += weight * angularChunk
```

收益：

- 删除每轮完整 `M*C` memset。
- 不再常驻完整 `angularPsi`。
- 给 RSI batch workspace 留出更多显存。

风险：

- 浮点加法顺序可能变化，逐字节一致性可能不再保证。
- 应以 relative L2/max abs 容差验收。

## 优先级 5：融合 SI reduction 与 convergence norm

当前 SI 每轮至少两个全局 pass：

```text
reduceDirectionsKernel: angularPsi -> currentPhi
relativeNormKernel: currentPhi/previousPhi -> norm
```

可融合为：

```text
reduceDirectionsAndNormKernel
```

在写 `currentPhi[cell]` 时同步计算该 cell 的 norm 局部贡献。收益是减少一次 `C` 规模全局读写和一次 kernel launch。

## 优先级 6：CUDA Graph 或持久 kernel

在做 CUDA Graph 前，先完成“cyclic/acyclic 连续区间拆分”的 localPass 优化。
上一轮 batch 级跳过无环 pass 的 30k 完整 Figure 5 为 `231.71 s`，未达旧基线，
已回退；`NVCCARCH=sm_120` 当前提交版本为 `185.75 s`，同样未达旧基线。
下一轮只改 SI direction loop，不改 RSI loop，避免把 RSI batch 行为一起扰动。

SI fine 的 kernel 启动结构在 14 轮迭代中基本固定。若分阶段计时显示 launch overhead 明显，可考虑：

- CUDA Graph 捕获每轮 SI 结构；
- 或针对 direction batch 使用持久 kernel。

这项复杂度较高，应在完成计时、tail 修复、localPass 跳过后再做。

## 不建议优先做的方向

暂不建议优先把 RNG schedule 搬到 GPU。原因：

- 当前 CPU/GPU 可做到逐 sample 确定性一致。
- GPU RNG 会改变统计路径，可能失去逐值可比性。
- schedule H2D 数据量相对 sweep/reduction 不是当前首要瓶颈。

暂不建议继续 direct-global-tail。30k 完整 Figure 5 已验证它会显著回归。

## 推荐执行顺序

1. 修复为 batch-local tail accumulation，并保留 workspace reuse。
2. 编译并运行默认 CPU/GPU 一致性测试。
3. 跑 30k 完整 Figure 5 GPU，确认低于 145.23 s。
4. 增加分阶段计时。
5. 跑 200k 完整 Figure 5，定位 SI/RSI/reduction 占比。
6. 实现 host 侧跳过无环方向空 localPass。
7. 实现 SI direction streaming。
8. 视计时结果决定是否做 CUDA Graph。
