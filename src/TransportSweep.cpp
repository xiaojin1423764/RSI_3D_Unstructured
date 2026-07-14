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
        double sourceFraction = 0.0;

        if (sourceShape_ == "rectangle") {
            sourceFraction =
                std::fabs(face.center.x - sourceCenterX) <= rectangleHalfLengthX &&
                std::fabs(face.center.z - sourceCenterZ) <= rectangleHalfWidthZ
                    ? 1.0
                    : 0.0;
        } else if (sourceShape_ == "circle") {
            sourceFraction = face.source_fraction;

            // Backward compatibility for old faces.csv files that do not carry a
            // circle overlap fraction: use the previous face-center test.
            if (sourceFraction >= 1.0) {
                const double dx = face.center.x - sourceCenterX;
                const double dz = face.center.z - sourceCenterZ;
                sourceFraction = dx * dx + dz * dz <=
                    equivalentCircleRadius * equivalentCircleRadius ? 1.0 : 0.0;
            }
        } else {
            throw std::runtime_error("TransportSweep: sourceShape 必须是 rectangle 或 circle");
        }

        if (onBottomFace && omega.y > 0.0 && sourceFraction > 0.0) {
            return sourceFraction * 10.0 *
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
    return solveDirectionWithPlan(ord, source_phi, buildSweepPlan(ord.omega));
}


SweepPlan TransportSweep::buildSweepPlan(const Vec3& omega) const {
    const int C = static_cast<int>(mesh_.cells.size());
    constexpr double eps = 1e-14;
    SweepPlan plan;
    plan.order.reserve(C);

    // Count each cell's upwind dependencies directly from its four face refs.
    // This avoids allocating a vector-of-vectors graph for every ordinate.
    std::vector<int> indegree(C, 0);
    for (int cell = 0; cell < C; ++cell) {
        for (const CellFaceRef& ref : mesh_.cells[cell].faceRefs) {
            if (ref.neighbor < 0) continue;
            const Face& face = mesh_.faces[ref.face];
            const Vec3 outward = outwardNormalForCell(face, ref.sign);
            if (dot(omega, outward) < -eps) ++indegree[cell];
        }
    }

    std::vector<int> queue;
    queue.reserve(C);
    for (int cell = 0; cell < C; ++cell) {
        if (indegree[cell] == 0) queue.push_back(cell);
    }

    std::size_t head = 0;
    while (head < queue.size()) {
        plan.levelOffsets.push_back(static_cast<int>(plan.levelCells.size()));
        const std::size_t levelEnd = queue.size();
        std::sort(queue.begin() + head, queue.begin() + levelEnd);
        while (head < levelEnd) {
            const int cell = queue[head++];
            plan.order.push_back(cell);
            plan.levelCells.push_back(cell);
            for (const CellFaceRef& ref : mesh_.cells[cell].faceRefs) {
                if (ref.neighbor < 0) continue;
                const Face& face = mesh_.faces[ref.face];
                const Vec3 outward = outwardNormalForCell(face, ref.sign);
                if (dot(omega, outward) > eps && --indegree[ref.neighbor] == 0) {
                    queue.push_back(ref.neighbor);
                }
            }
        }
    }

    plan.hasCycle = static_cast<int>(plan.order.size()) != C;
    plan.levelOffsets.push_back(static_cast<int>(plan.levelCells.size()));
    if (plan.hasCycle) {
        plan.order.resize(C);
        plan.levelCells.clear();
        plan.levelOffsets.clear();
        for (int cell = 0; cell < C; ++cell) plan.order[cell] = cell;
        std::sort(plan.order.begin(), plan.order.end(), [&](int a, int b) {
            return dot(mesh_.cells[a].center, omega) < dot(mesh_.cells[b].center, omega);
        });
        std::cerr
            << "警告: 当前方向 cell 依赖图存在环，"
            << "已改用投影排序 + 上风 Gauss-Seidel 迭代。\n";
    }
    return plan;
}


std::vector<double> TransportSweep::solveDirectionWithPlan(
    const Ordinate& ord,
    const std::vector<double>& source_phi,
    const SweepPlan& plan
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

    if (!plan.hasCycle) {
        // 无环：严格拓扑 sweep，只扫一遍。
        for (int ci : plan.order) {
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

        for (int ci : plan.order) {
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
