代码实现了论文RSI方法在三维非结构网格有限体积(四面体网格)框架下的串行版本，并以论文 Example 1的三维扩展问题验证Figure 2 收敛阶和Figure 5 射线效应，其他参数和边界条件不变。代码中分组G可以任取，但是论文指出G=1是最适合并行的

代码不在求解器内部生成非结构网格，而是从 /gmsh_work/data/cells.csv&faces.csv 读取 cell-face 邻接、面法向、面积、边界类型和材料参数。这两个.csv文件数据由example1.msh转换来,.msh的数据由Gmsh自动生成，可以在
/gmsh_work/example1.geo中修改网格尺寸，以及example1立方体大小，和网格生成算法Mesh.Algorithm3D=1/Mesh.Algorithm3D= 4等

四面体拓扑和TransportSweep.cpp的扫掠算法参考了： 
https://github.com/lanl/tycho2


其他可修改的参数：(都在/src/main.cpp中)
Figure 2 收敛阶实验使用 `angularN : {8,16}`，即 level-symmetric S_N 角向划分，M=N*(N+2)，支持多个角度阶数作比较。

当前角向离散使用 `Quadrature::levelSymmetricSN()` 生成的 level-symmetric S_N 方向集，SI 和 RSI 的所有角方向扫掠都共用这一套方向集
`angularN` 必须是 >=2 的偶数。对 S_N，方向总数为 M=N*(N+2)。
每个八分区使用 i+j+k=N/2+2 的层对称方向，符号反射到全空间。
方向权重统一归一化为 1/[N*(N+2)]，满足 sum_m ω_m=1、sum_m ω_m Ω_m=0、sum_m ω_m Ω_{m,x}^2=sum_m ω_m Ω_{m,y}^2=sum_m ω_m Ω_{m,z}^2=1/3。

cfg.groupCount = 2//分组数G，G<=M

cfg.sampleCounts = {4,8,16,32,64,128,256,512,1024};//样本数量，可以继续加

coarseCfg.angularN = 4;//Figure 5 粗角度 SI，S4 对应 M=24

fineCfg.angularN = 32;//Figure 5 细角度 SI、RSI、RSI_tail 共用这个参数，S32 对应 M=1088

fineCfg.groupCount = 1;//Figure 5 细角度分组。注意：这个参数和 cfg.groupCount 是独立的，用于计算射线效应，cfg.groupCount 则用于计算收敛阶；SI 不用分组，因此 coarseCfg.groupCount = 1 和 fineCfg.groupCount = 1 是默认参数

int S = 512;//计算 Figure 5 RSI 场数据使用的样本数


修改好后需要重新生成网格并编译运行:
make clean
cd gmsh_work
gmsh example1.geo -3 -format msh2 -o example1.msh
python3 msh_to_rsi_csv.py example1.msh data/cells.csv data/faces.csv
cd ..
make run-rec
make plot

入射区域形状不需要重新生成 Gmsh 网格，也不需要切换代码。矩形 Rec 条件和圆形 Cir 条件都已整合在 `TransportSweep.cpp` 的 `boundaryInflow()` 中，运行时用开关选择：

```bash
make run-rec   # rectangle, 输出 Rec_* 图对应的数据
make run-cir   # circle, 输出 Cir_* 图对应的数据
make run-rec-figure5  # rectangle, 只输出 Figure 5 数据
make run-cir-figure5  # circle, 只输出 Figure 5 数据
make plot      # 只生成 Figure 5 图片
make plot-voxel3d  # 只生成 Matplotlib voxel 3D 图
make plot-volume3d  # 只生成 PyVista 半透明体渲染 3D 图
make plot-isosurfaces  # 只生成 PyVista 多层等值面 3D 图
make tecplot  # 导出 Tecplot ASCII DAT 非结构四面体场数据
make plot-all  # 生成 Figure 2、Figure 5、网格和角度划分图片
```

## CUDA 样本并行

CUDA 版本加速 Figure 5 的各向同性 `G=1` RSI、RSI-tail 和 SI。RSI 的独立
样本链并行执行；每个 warp 划分为 8 个 4-lane 子组，同时推进 8 条样本链。
SI 在角度方向上并行，角通量归约和收敛范数也在 GPU 上完成。Figure 2 仍使用
CPU 路径。

```bash
make gpu
./rsi_unstructured_gpu --gpu --source-shape circle --only figure5 \
  --figure5-dir examples/csv_data_gpu \
  gmsh_work/data/cells.csv gmsh_work/data/faces.csv
```

`NVCCARCH` 默认使用本机 GPU 架构，也可以显式指定，例如：

```bash
make gpu NVCCARCH=sm_120
```

运行 CPU/GPU 小规模一致性测试：

```bash
make test-gpu
```

长任务可设置检查点前缀。SI 每轮迭代保存一次，RSI 每个 sample batch 保存一次；
同一命令再次运行会校验网格、角度、sample 数、迭代步数和 tail 长度后继续计算：

```bash
RSI_CUDA_CHECKPOINT=results/figure5_gpu_30k_194k/200k/checkpoint \
./rsi_unstructured_gpu --gpu --source-shape circle --only figure5 \
  --figure5-dir results/figure5_gpu_30k_194k/200k \
  gmsh_work/mesh200k/cells.csv gmsh_work/mesh200k/faces.csv
```

### GPU 基准使用的空间网格

当前空间网格输入如下；目录中的 30k/200k 是近似标记，实际数量以 CSV 为准：

| 图集标记 | 网格输入 | 实际四面体单元数 | 平均单元体积 |
|---|---|---:|---:|
| 30k | `gmsh_work/data/cells.csv`、`faces.csv` | 36,470 | `2.74198e-5` |
| 200k | `gmsh_work/mesh200k/cells.csv`、`faces.csv` | 194,314 | `5.14631e-6` |

这里的粗/细网格指空间四面体网格，不是 Figure 5 中的角度离散：
`SI coarse` 使用 $S_4$（24 个角度），`SI fine` 使用 $S_{32}$（1088 个角度）。
因此 36,470/194,314 是空间 cell 数，24/1088 是角方向数量，两者是独立参数。

当前重新计算的数据位于 `results/figure5_gpu_30k_194k/{30k,200k}/Cir/`。
不能把旧的 164,151-cell 场数据与新的 194,314-cell 网格连接关系混用。

### 已实现的四项 GPU 优化

1. RSI 和 RSI-tail 合并为一次随机链计算。RSI 是第 $N$ 步结果，RSI-tail 是
   第 $N$ 至 $N+10$ 步平均，不再重复计算前 $N$ 步。
2. 网格、角方向和 sweep plan 在一次 Figure 5 调用中上传一次并常驻 GPU，供
   SI、RSI 和 RSI-tail 复用。
3. SI fine 使用 GPU 角度并行 sweep，并在 GPU 上完成角度归约和收敛范数。
   方向按网格规模动态分批，避免 194,314-cell 网格上的单次 kernel 过大。
4. 一个 warp 划分为 8 个 4-lane 子组；四个 lane 对应四面体的四个面，使每个
   warp 同时计算 8 条 sample 链，减少空闲 lane。

为保证合并前后的统计定义明确，CPU/GPU 场计算均使用逐 sample 的确定性 RNG：
`seed + 17*M + 0x9e3779b9*(sample+1)`。因此 RSI 严格对应 RSI-tail 同一批随机链
的前 $N$ 步，而不依赖线程调度或 batch 划分。

36,470-cell、$S_4$ 的完整 Figure 5 CPU/GPU 回归结果为：SI 相对 L2 误差
`1.58222e-16`，RSI 为 `1.77454e-16`，RSI-tail 为 `2.24576e-16`；CPU/GPU
均在第 14 轮收敛。

### 当前进展（2026-07-13）

- 四项 GPU 优化已经实现并通过编译与一致性测试。
- 30k（36,470 cells）和 200k（194,314 cells）的 Figure 5 已重新计算完成。
- 每套结果均包含 `SI_coarse`、`SI_fine`、`RSI`、`RSI_tail`，共 8 个 CSV；
  行数分别为 36,470 和 194,314，`phi0` 全部为有限值。
- 两种网格的 SI fine 均在第 14 轮收敛。Figure 5 使用 24/1088 个角度、
  512 samples 和 `tailExtra=10`。
- CPU/GPU 同参数计时产生的四个 CSV 在 30k 和 200k 上均逐字节一致。
- Figure 5 场图已按要求从 `30k&200k/` 删除，当前只保留两张实际入射区域图。
  `30k_200k_results.pdf` 是删除场图前编译的快照；在 Tecplot 新图完成前不再更新。

### CPU/GPU 完整运行时间

测试包含 sweep plan 构建、SI coarse、SI fine、RSI、RSI-tail 和四个 CSV 写出；
不使用检查点。硬件为 AMD Ryzen 7 9700X（8C/16T）和 RTX 5070 Ti 16 GB。

| 空间网格 | CPU | GPU | 加速比 | 时间降低 |
|---|---:|---:|---:|---:|
| 30k（36,470 cells） | 245.36 s | 89.83 s | 2.73x | 63.39% |
| 200k（194,314 cells） | 1751.69 s | 1417.84 s | 1.24x | 19.06% |

200k 的加速比下降主要来自两类分批开销：SI 的 1088 个方向在该网格上约按
12 directions/batch 执行；RSI 受显存限制约为 25 samples/batch，512 samples
需要约 21 批。当前下一阶段性能重点是减少 SI kernel 的批间同步，以及 RSI
sample batch 的数据管理和同步开销。

### 下一阶段 GPU 优化计划

优化顺序以 200k 完整 Figure 5 墙钟时间为准，同时保留 30k 回归。每项修改后
必须运行 CPU/GPU 一致性测试，并比较四个 CSV；容差继续使用相对 L2 `5e-10`
和最大绝对误差 `5e-9`，正式 30k/200k 输出应保持当前逐字节一致性。

1. 分阶段计时与 Nsight 定位。分别记录 sweep plan、SI 每轮、RSI 每个 batch、
   Host/Device 传输和 CSV 写出时间，并使用 Nsight Systems 确认 200k 中约
   701.59 s system time 的来源。先定位同步或内存问题，再调整 kernel。
2. SI 方向流式化。当前一次分配完整的 `M*C` angular field，并按约 12 个方向
   启动 kernel。改为只保留一个方向 chunk，在同一 CUDA stream 中 sweep 后立即
   累加到 `phi`，避免保存全部 1088 个方向的角通量，并增大可用方向 batch。
3. SI 批次内归约融合。把角通量加权归约、source 更新和局部范数部分融合进批次
   尾部，使用 block reduction 或 CUB，减少每轮 SI 的全局内存往返与同步点。
4. CUDA Graph 或持久 kernel。SI 的 14 轮迭代具有相同启动结构；在 sweep plan
   固定后捕获 CUDA Graph，或者使用持久 kernel 消除大量小方向 batch 的启动开销。
5. RSI 工作区复用。把 sample batch 的临时数组按最大 batch 一次分配，在 21 个
   batch 间复用；避免循环内 `cudaMalloc/cudaFree`、重复清零和不必要的 Host 同步。
6. RSI 分块累计。每个 batch 在 GPU 上直接累加 RSI 与 tail 的 cell sum，只在
   512 samples 完成后拷回两个最终场。检查是否仍有 batch 级结果回传并将其删除。
7. 显存布局压缩。评估 sweep plan 索引使用 32-bit、只读几何数据使用 SoA，及
   不再常驻的中间量；目标是把 200k 的 samples/batch 从约 25 提高到至少 64。
8. 传输与计算重叠。若 batch 间仍需要 Host 交互，使用 pinned memory 和双缓冲
   CUDA streams，使下一批准备与当前批计算重叠；没有真实传输时不引入此复杂度。

阶段验收目标：首先把 200k 从 1417.84 s 降到 900 s 以下（相对当前 GPU 至少
1.58x），随后以 600 s 为目标；30k 不得慢于当前 89.83 s。每次基准都必须无
检查点运行，参数固定为 24/1088 angles、512 samples、`tailExtra=10`。

### 当前 Tecplot 数据

已使用各自正确的 Gmsh 四面体连接关系导出 Tecplot ASCII DAT：

```text
30k&200k/tecplot/30k/Cir/figure5_SI_coarse.dat
30k&200k/tecplot/30k/Cir/figure5_SI_fine.dat
30k&200k/tecplot/30k/Cir/figure5_RSI.dat
30k&200k/tecplot/30k/Cir/figure5_RSI_tail.dat
30k&200k/tecplot/200k/Cir/figure5_SI_coarse.dat
30k&200k/tecplot/200k/Cir/figure5_SI_fine.dat
30k&200k/tecplot/200k/Cir/figure5_RSI.dat
30k&200k/tecplot/200k/Cir/figure5_RSI_tail.dat
```

DAT zone 类型为 `FETETRAHEDRON`、`DATAPACKING=BLOCK`，`phi0` 和 `cell_id`
均为 cell-centered。30k zone 为 6,870 nodes / 36,470 elements；200k zone 为
35,401 nodes / 194,314 elements。下一步由 Tecplot 生成 Iso = 1、0.5、0.05
等值面以及 $y=0$、$y=0.5$ 截面图。



运行结束之后数据保存到:
`examples/csv_data/figure2_data.csv`
`examples/csv_data/Rec/figure5_RSI_tail.csv`
`examples/csv_data/Rec/figure5_RSI.csv`
`examples/csv_data/Rec/figure5_SI_coarse.csv`
`examples/csv_data/Rec/figure5_SI_fine.csv`
`examples/csv_data/Cir/figure5_RSI_tail.csv`
`examples/csv_data/Cir/figure5_RSI.csv`
`examples/csv_data/Cir/figure5_SI_coarse.csv`
`examples/csv_data/Cir/figure5_SI_fine.csv`
对应论文的Figure2 收敛阶和Figure5 射线效应数据

最后运行 /examples/plot_figures.py 生成图片。图片会按入射区域命名：矩形区域前缀为 `Rec`，圆形区域前缀为 `Cir`。
Figure 5 数据按入射区域分开保存，避免 Rec/Cir 相互覆盖：
`examples/csv_data/Rec/figure5_*.csv` 和 `examples/csv_data/Cir/figure5_*.csv`。
`make plot` 会读取已有的 Rec/Cir 数据并分别画图。Figure 5 的 SI coarse、SI fine、RSI、RSI tail 四类场都会输出 y 截面和 layer 堆叠图；voxel 3D 图单独用 `make plot-voxel3d` 生成，PyVista 半透明体渲染图单独用 `make plot-volume3d` 生成，多层等值面图单独用 `make plot-isosurfaces` 生成。例如：
`Cir_SI_coarse_y0.50.png`、`Cir_SI_coarse_layer_stack.png`、`Cir_SI_coarse_voxel3d_iso_back.png`、`Cir_SI_coarse_volume3d_iso_back.png`、`Cir_SI_coarse_isosurface_iso_back.png`。

也可以把 Figure 5 的非结构四面体场数据导出给 Tecplot：

```bash
make tecplot
```

默认输出到 `examples/tecplot/Rec/*.dat` 和 `examples/tecplot/Cir/*.dat`。每个文件都是 Tecplot ASCII `FETETRAHEDRON` zone，`X/Y/Z` 为节点坐标，`phi0` 和 `cell_id` 为 cell-centered 变量。Tecplot 中打开 `.dat` 后可直接对 `phi0` 做 contour、slice、iso-surface 或体渲染。

如果要导出单个文件：

```bash
python3 examples/export_tecplot.py \
  --csv examples/csv_data/Rec/figure5_RSI.csv \
  --out examples/tecplot/Rec/figure5_RSI.dat
```




论文对应关系：

RSI 抽样概率：论文式 (2.3)
随机散射源：论文式 (2.7)
SI 无偏参照：论文式 (1.4)
Figure 2 数据：输出 e_RSI^(N) 随样本数 S 的变化(收敛阶)
Figure 5 数据：输出三个截面的散射情况

##目录

```text
rsi/
  Makefile
  README.md
  rsi_unstructured
  include/
    Types.hpp
    Mesh.hpp
    Quadrature.hpp
    TransportSweep.hpp
    RSI.hpp
  src/
    Mesh.cpp
    Quadrature.cpp
    TransportSweep.cpp
    RSI.cpp
    main.cpp
  gmsh_work/
    data/
      cells.csv
      faces.csv
    example1.geo
    example1.msh
    msh_to_rsi_csv.py
  examples/
    /csv_data
      figure2_data.csv
      Rec/
        figure5_RSI_tail.csv
        figure5_RSI.csv
        figure5_SI_coarse.csv
        figure5_SI_fine.csv
      Cir/
        figure5_RSI_tail.csv
        figure5_RSI.csv
        figure5_SI_coarse.csv
        figure5_SI_fine.csv
    /Figures
      figure2_anisotropic.png
      figure2_isotropic.png
      Rec_SI_coarse_y0.50.png
      Rec_SI_coarse_layer_stack.png
      Cir_SI_coarse_y0.50.png
      Cir_SI_coarse_layer_stack.png
    plot_figures.py
    plot_naming.py
```


## cells.csv

```csv
cell_id,cx,cy,cz,volume,sigma_t,sigma_s,q
0,0.1,0.2,0.3,0.001,1.0,0.5,0.0
```

含义：

cell_id：单元编号，可不连续，但必须唯一
cx,cy,cz：单元中心
volume：单元体积
sigma_t：总截面 ΣT
sigma_s：散射截面 ΣS
q：体源项 Q

### faces.csv

```csv
face_id,left_cell,right_cell,nx,ny,nz,area,fx,fy,fz,bc_type,bc_value
0,3,7,0,1,0,0.01,0.2,0.3,0.4,internal,0
1,3,-1,-1,0,0,0.01,0,0.3,0.4,vacuum,0
```

约定：

`left_cell >= 0` 必须有效。
`right_cell >= 0` 表示内部面。
`right_cell = -1` 表示边界面。
`normal=(nx,ny,nz)` 对内部面表示从 `left_cell` 指向 `right_cell`；对边界面表示 `left_cell` 的外法向。
`area` 是面面积。
`fx,fy,fz` 是面中心，用于 Example 1 入流边界判断。
`bc_type` 支持：
`internal`：内部面。
`vacuum`：真空边界入流为 0。
`inflow`：常数入流，值为 `bc_value`。
`example1`：论文 Example 1 的边界入流三维化。

## 非结构扫掠算法

本程序的空间离散采用三维非结构网格有限体积上风格式。网格由 `cells.csv` 和 `faces.csv` 给出，其中 `Cell` 保存单元中心、体积和材料参数，`Face` 保存左右单元、面法向、面积、面中心和边界条件。


Cell-Face 邻接关系
程序在读取网格后会构造每个 cell 的面邻接表：
```cpp
CellFaceRef {
    int face;      // face 在 mesh.faces 中的下标
    int neighbor;  // 邻居 cell 下标，边界面为 -1
    int sign;      // +1 当前 cell 是 left_cell，-1 当前 cell 是 right_cell
}
这样每个 cell 都能直接知道：
相邻的是哪个 face
face 另一侧是不是内部 cell
当前 cell 看到的外法向是否需要取反

对每个角方向 Ω，程序根据内部面的法向建立上游到下游的依赖关系。
若某个内部面法向 n 从 left_cell 指向 right_cell，则
Ω · n > 0  : left_cell  -> right_cell
Ω · n < 0  : right_cell -> left_cell

一般非结构网格中，由于网格几何、面法向或复杂连接关系，方向依赖图可能存在环。此时严格拓扑排序无法覆盖所有cell，程序会退化为：
按 dot(cell_center, Ω) 从小到大排序
再进行多次上风 Gauss-Seidel 局部迭代

对某个 cell 和方向 Ω，程序使用上风有限体积格式：
(ΣT V + Σ_out |Ω·n_f| A_f) ψ_i
=
V (ΣS φ_i + Q_i)
+ Σ_in |Ω·n_f| A_f ψ_in

其中：
V       : cell 体积
ΣT      : 总截面
ΣS      : 散射截面
Q       : 外源项
A_f     : face 面积
n_f     : 当前 cell 的外法向
ψ_i     : 当前 cell 的角通量
ψ_in    : 入流面上的上游角通量或边界入流

因此当前 cell 的角通量为：
ψ_i =
[
  V (ΣS φ_i + Q_i)
  + Σ_in |Ω·n_f| A_f ψ_in
]
/
[
  ΣT V + Σ_out |Ω·n_f| A_f
]

边界入流
边界面通过 faces.csv 中的 bc_type 指定：
vacuum   : 真空边界，入流为 0
inflow   : 常数入流，值为 bc_value
example1 : 论文 Example 1 的三维化入流边界

example1 边界条件在 y=0 的指定区域给定入流：
ψ_in = 10 exp(-Ω_x^2 - Ω_y^2 - Ω_z^2)，其他边界均视为真空入流

指定区域由程序运行参数控制。`rectangle` 表示 x∈[0.4,0.6], z∈[0.4,0.6] 的矩形区域。
`circle` 表示以 (x,z)=(0.5,0.5) 为圆心的圆形区域。
圆半径为 2*0.1/sqrt(pi) ≈ 0.11284，使圆面积等于原矩形面积 0.04。


与 SI / RSI 的关系
TransportSweep::solveDirection() 只负责给定一个方向 Ω 和当前散射源 φ 时，求出该方向的角通量 ψ
SI 和 RSI 都调用同一个扫掠函数：
sweep_.solveDirection(ordinate, source_phi)
区别在于：
SI  : 每次迭代扫掠所有角方向
RSI : 每次迭代只随机选择部分角方向
