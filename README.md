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
make plot-all  # 生成 Figure 2、Figure 5、网格和角度划分图片
```



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
