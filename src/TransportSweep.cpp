#include "TransportSweep.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <vector>


// TransportSweep.cpp
// 对给定角方向 omega，在三维非结构网格上做一次输运扫掠。



// 根据当前 cell 对某个 face 的方向关系，返回当前 cell 看到的外法向。
//
// 参数：
// f    : face 数据
// sign : CellFaceRef::sign
//        +1 表示当前 cell 是 left_cell，此时 face.normal 已经是当前 cell 外法向
//        -1 表示当前 cell 是 right_cell，此时当前 cell 外法向是 -face.normal
static Vec3 outwardNormalForCell(const Face& f, int sign) {
    if (sign == +1) {
        return f.normal;
    }

    return Vec3{-f.normal.x, -f.normal.y, -f.normal.z};
}


// 判断当前方向 omega 的 cell 依赖图是否存在环。
//
// 如果存在环，严格拓扑扫掠无法覆盖所有 cell，
// 后面会使用“投影排序 + Gauss-Seidel 迭代”作为 fallback。
static bool hasSweepCycle(const Mesh& mesh, const Vec3& omega) {
    const int C = static_cast<int>(mesh.cells.size());

    std::vector<std::vector<int>> graph(C);
    std::vector<int> indeg(C, 0);

    const double eps = 1e-14;

    // 遍历所有内部 face，建立上游 cell -> 下游 cell 的有向边。
    for (const auto& f : mesh.faces) {
        // right_cell < 0 表示边界面，没有 cell-cell 依赖。
        if (f.right_cell < 0) continue;

        // L/R 是 cell 在 mesh.cells 数组中的下标。
        int L = mesh.cellIndex(f.left_cell);
        int R = mesh.cellIndex(f.right_cell);

        // f.normal 约定从 left_cell 指向 right_cell。
        double mu = dot(omega, f.normal);

        // 如果方向几乎平行于该面，则忽略这个面产生的依赖。
        if (std::fabs(mu) <= eps) continue;

        // mu > 0 表示粒子从 L 穿过该面流向 R。
        // mu < 0 表示粒子从 R 穿过该面流向 L。
        int up   = (mu > 0.0) ? L : R;
        int down = (mu > 0.0) ? R : L;

        graph[up].push_back(down);
        indeg[down]++;
    }

    // Kahn 拓扑排序，只用于判断是否存在环。
    std::queue<int> q;

    for (int i = 0; i < C; ++i) {
        if (indeg[i] == 0) q.push(i);
    }

    int visited = 0;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        visited++;

        for (int v : graph[u]) {
            --indeg[v];
            if (indeg[v] == 0) q.push(v);
        }
    }

    // 如果没有访问完所有 cell，说明图中存在环。
    return visited != C;
}


// 根据方向 omega 构造 cell 扫掠顺序。
//
// 无环时：返回严格拓扑 sweep 顺序。
// 有环时：返回按 dot(center, omega) 从小到大排序的 fallback 顺序。
std::vector<int> TransportSweep::buildSweepOrder(const Vec3& omega) const {
    const int C = static_cast<int>(mesh_.cells.size());

    std::vector<std::vector<int>> graph(C);
    std::vector<int> indeg(C, 0);

    const double eps = 1e-14;

    // 对每个内部面建立 upwind -> downwind 依赖边。
    for (const auto& f : mesh_.faces) {
        if (f.right_cell < 0) continue;

        int L = mesh_.cellIndex(f.left_cell);
        int R = mesh_.cellIndex(f.right_cell);

        double mu = dot(omega, f.normal);

        if (std::fabs(mu) <= eps) continue;

        int up   = (mu > 0.0) ? L : R;
        int down = (mu > 0.0) ? R : L;

        graph[up].push_back(down);
        indeg[down]++;
    }

    std::queue<int> q;

    for (int i = 0; i < C; ++i) {
        if (indeg[i] == 0) q.push(i);
    }

    std::vector<int> order;
    order.reserve(C);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        order.push_back(u);

        for (int v : graph[u]) {
            --indeg[v];
            if (indeg[v] == 0) q.push(v);
        }
    }

    // 如果拓扑排序失败，说明方向依赖图有环。
    // 对一般非结构网格这可能出现。
    // 此时按 cell center 在 omega 方向上的投影排序。
    if (static_cast<int>(order.size()) != C) {
        order.resize(C);

        for (int i = 0; i < C; ++i) {
            order[i] = i;
        }

        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return dot(mesh_.cells[a].center, omega)
                 < dot(mesh_.cells[b].center, omega);
        });

        std::cerr
            << "警告: 当前方向 cell 依赖图存在环，"
            << "已改用投影排序 + 上风 Gauss-Seidel 迭代。\n";
    }

    return order;
}


// 计算边界入流值。
//
// 目前支持四类：
//     vacuum   : 真空边界，入流为 0
//     inflow   : 常数入流，值为 face.bc_value
//     example1 : 论文 Example 1 的三维扩展入流，区域形状由 sourceShape_ 控制
double TransportSweep::boundaryInflow(const Face& face, const Vec3& omega) const {
    // 真空边界：无外部粒子进入。
    if (face.bc_type == "vacuum") return 0.0;

    // 常数入流边界：直接使用 faces.csv 里的 bc_value。
    if (face.bc_type == "inflow") return face.bc_value;

    // 论文 Example 1 的三维化边界：y=0 底面上的入流窗口。
    if (face.bc_type == "example1") {
        constexpr double sourceCenterX = 0.5;
        constexpr double sourceCenterZ = 0.5;
        constexpr double rectangleHalfLengthX = 0.15;
        constexpr double rectangleHalfWidthZ = 0.1;
        constexpr double pi = 3.141592653589793238462643383279502884;
        constexpr double equivalentCircleRadius =
            0.2 / std::sqrt(pi);

        const bool onBottomFace = std::fabs(face.center.y) < 1e-10;
        bool inSourceWindow = false;

        if (sourceShape_ == "rectangle") {
            inSourceWindow =
                std::fabs(face.center.x - sourceCenterX) <= rectangleHalfLengthX &&
                std::fabs(face.center.z - sourceCenterZ) <= rectangleHalfWidthZ;
        } else if (sourceShape_ == "circle") {
            const double dx = face.center.x - sourceCenterX;
            const double dz = face.center.z - sourceCenterZ;
            inSourceWindow = dx * dx + dz * dz <=
                equivalentCircleRadius * equivalentCircleRadius;
        } else {
            throw std::runtime_error("TransportSweep: sourceShape 必须是 rectangle 或 circle");
        }

        if (onBottomFace && omega.y > 0.0 && inSourceWindow) {
            return 10.0 *
                   std::exp(-omega.x * omega.x - omega.y * omega.y - omega.z * omega.z);
        }

        return 0.0;
    }

    // 未知边界类型默认按真空处理。
    return 0.0;
}


// 对单个 cell 做一次有限体积上风更新。
//
// 该函数被两种流程共用：
// 1. 无环时的严格拓扑 sweep
// 2. 有环时的 Gauss-Seidel fallback
//
// 离散公式：
//
//   (Sigma_T V + sum_out |Omega·n| A) psi_i
//     = V (Sigma_S phi_i + Q_i)
//       + sum_in |Omega·n| A psi_in
//
// 其中：
//     source_phi[ci] 是传入的散射源 phi_i
//     psiIn 来自上游内部 cell 或边界入流
static double updateOneCellFV(const Mesh& mesh,
                              const Ordinate& ord,
                              const std::vector<double>& source_phi,
                              const std::vector<double>& psi,
                              const std::vector<char>& done,
                              int ci,
                              bool requireDoneUpwind,
                              const TransportSweep& sweep) {
    const Cell& cell = mesh.cells[ci];

    const double eps = 1e-14;

    // 入流贡献：
    //     sum_in |Omega·n| A psi_in
    double inflow = 0.0;

    // 出流贡献系数：
    //     sum_out |Omega·n| A
    //
    // 它会进入左端对角项。
    double outcoef = 0.0;

    // 这里使用新的 cell.faceRefs。
    //
    // 每个 ref 已经包含：
    //     face 下标
    //     neighbor 下标
    //     normal 是否需要反向
    for (const auto& ref : cell.faceRefs) {
        const Face& f = mesh.faces[ref.face];

        // 当前 cell 看到的外法向。
        Vec3 outward = outwardNormalForCell(f, ref.sign);

        // mu = Omega · n_out。
        //
        // mu > 0：出流面
        // mu < 0：入流面
        double mu = dot(ord.omega, outward);

        // 面通量系数。
        double coeff = std::fabs(mu) * f.area;

        // 几乎平行于面的方向忽略。
        if (coeff <= eps) continue;

        if (mu > 0.0) {
            // 出流面使用当前 cell 自己的 psi。
            // 离散后进入左端对角项。
            outcoef += coeff;
        } else {
            // 入流面需要上游 psiIn。
            double psiIn = 0.0;

            if (ref.neighbor >= 0) {
                // 内部入流面：上游是邻居 cell。
                int oi = ref.neighbor;

                if (requireDoneUpwind) {
                    // 严格拓扑 sweep：
                    // 理论上上游邻居应该已经计算完成。
                    // 如果未完成，保守取 0，避免使用未定义值。
                    psiIn = done[oi] ? psi[oi] : 0.0;
                } else {
                    // 有环 fallback：
                    // 使用当前 psi 数组中的值。
                    // 如果邻居本轮已经更新，这是新值；
                    // 如果还没更新，这是上一轮局部迭代的旧值。
                    psiIn = psi[oi];
                }
            } else {
                // 边界入流面：从边界条件获取入流值。
                psiIn = sweep.boundaryInflow(f, ord.omega);
            }

            // 加入右端入流贡献。
            inflow += coeff * psiIn;
        }
    }

    // 右端项：
    //
    //     V (Sigma_S phi + Q) + inflow
    double rhs =
        cell.volume * (cell.sigma_s * source_phi[ci] + cell.q)
        + inflow;

    // 左端对角项：
    //
    //     Sigma_T V + outcoef
    double diag =
        cell.sigma_t * cell.volume
        + outcoef;

    if (diag <= 0.0) {
        throw std::runtime_error("TransportSweep: 非法对角系数 diag <= 0");
    }

    return rhs / diag;
}


// 对一个给定角方向做输运扫掠。
//
// 输入：
//     ord        : 当前角方向，包括 omega 和 weight
//     source_phi : 每个 cell 上的散射源
//
// 输出：
//     psi        : 每个 cell 上该方向的角通量
std::vector<double> TransportSweep::solveDirection(
    const Ordinate& ord,
    const std::vector<double>& source_phi
) const {
    const int C = static_cast<int>(mesh_.cells.size());

    if (static_cast<int>(source_phi.size()) != C) {
        throw std::runtime_error("source_phi 大小与 cell 数不匹配");
    }

    // 当前方向的角通量解。
    std::vector<double> psi(C, 0.0);

    // done[i] 表示 cell i 是否已经在严格 sweep 中算过。
    //
    // 对 fallback Gauss-Seidel，它也会被设置，
    // 但 fallback 实际不依赖 done。
    std::vector<char> done(C, 0);

    // 构造当前方向的 sweep 顺序。
    auto order = buildSweepOrder(ord.omega);

    // 判断是否存在方向依赖环。
    bool cycle = hasSweepCycle(mesh_, ord.omega);

    if (!cycle) {
        // 无环：严格拓扑 sweep，只扫一遍。
        for (int ci : order) {
            psi[ci] = updateOneCellFV(
                mesh_,
                ord,
                source_phi,
                psi,
                done,
                ci,
                true,
                *this
            );

            done[ci] = 1;
        }

        return psi;
    }

    // 有环：使用投影排序 + 上风 Gauss-Seidel 局部迭代。
    //
    // 这样比“未完成邻居直接取 0，只扫一遍”更稳定。
    const int maxLocalIters = 20;
    const double localTol = 1e-12;

    for (int iter = 0; iter < maxLocalIters; ++iter) {
        double maxDiff = 0.0;
        double maxVal = 0.0;

        for (int ci : order) {
            double oldVal = psi[ci];

            psi[ci] = updateOneCellFV(
                mesh_,
                ord,
                source_phi,
                psi,
                done,
                ci,
                false,
                *this
            );

            done[ci] = 1;

            maxDiff = std::max(maxDiff, std::fabs(psi[ci] - oldVal));
            maxVal  = std::max(maxVal,  std::fabs(psi[ci]));
        }

        double rel = maxDiff / std::max(maxVal, 1e-300);

        if (rel < localTol) {
            break;
        }
    }

    return psi;
}
