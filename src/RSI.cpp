#include "RSI.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>


RSISolver::RSISolver(const Mesh& mesh, RSIConfig cfg)
    : mesh_(mesh), cfg_(std::move(cfg)), ordinates_(Quadrature::levelSymmetricSN(cfg_.angularN)),
     sweep_(mesh, cfg_.sourceShape) {}
     // sweep_ 是非结构网格输运扫掠器，它负责在给定方向 omega 
     //和散射源 source 的情况下，求解空间离散后的输运方程，得到 psi

// 计算离散散射核 K_{k,m}
// k : 入射方向索引，对应 Omega_k，m : 散射后方向索引，对应 Omega_m
// 返回：K_{k,m}
// 在 SI 中用它计算：phi_m = sum_k w_k K_{k,m} psi_k
// 在 RSI 中用它计算抽样概率和随机源。

double RSISolver::kernel(int k, int m) const {

    // 取出第 k 个离散方向的单位向量 Omega_k
    const auto& ok = ordinates_[k].omega;

    // 取出第 m 个离散方向的单位向量 Omega_m
    const auto& om = ordinates_[m].omega;

    if (cfg_.scattering == "anisotropic") {
        //各向异性散射
        // 这里采用论文 Example 1 中二维各向异性核的三维推广
        // 论文二维形式类似：K = 1 + c*c' + s*s'
        // 其中 c、s 是方向在 x、y 平面的分量
        // 三维中自然推广为：K = 1 + Omega_k · Omega_m
        // dot(ok, om) 表示两个单位方向的点积
        return std::max(0.0, 1.0 + dot(ok, om));
    }
    // 如果不是 anisotropic，就默认使用各向同性散射
    // 由于角权重已经归一化：sum_m w_m = 1
    // 因此各向同性散射核取：K_{k,m} = 1
    // 这样有:sum_k w_k K_{k,m} = sum_k w_k = 1
    return 1.0;
}



// 预计算完整的散射核矩阵 K
// 返回：K[k][m] = K_{k,m}
// 因为 SI 和 RSI 中会频繁使用 K_{k,m}
// 若每次都重新计算点积，会产生重复开销，预计算后可以直接查表
std::vector<std::vector<double>> RSISolver::precomputeKernel() const {
    const int M = static_cast<int>(ordinates_.size());
    std::vector<std::vector<double>> K(M, std::vector<double>(M, 1.0));
    for (int k = 0; k < M; ++k) for (int m = 0; m < M; ++m) K[k][m] = kernel(k, m);
    return K;
}


// 将 M 个角方向划分成 G 个 group
//     groups[g] 是第 g 个方向组。
//     里面保存属于该组的方向编号 m
//     这里是按方向编号顺序做均匀分组
//    如果以后想更严格按角空间区域分组，可以替换这里的分组规则
static std::vector<std::vector<int>> makeGroups(int M, int G) {
    if (G <= 0 || G > M)
        throw std::runtime_error("groupCount 必须满足 1 <= G <= M");

    std::vector<std::vector<int>> groups(G);

    for (int m = 0; m < M; ++m) {
        int g = m * G / M;
        groups[g].push_back(m);
    }

    return groups;
}


// 运行Source Iteration，SI， SI 是论文中 RSI 的基准方法
// 每一步都会对所有角方向 m 求解输运方程
// 然后用所有方向的 psi 更新散射源 phi
// 参数：
// convergedN 是输出参数。
// 函数结束时，convergedN 会被设置为 SI 收敛时的迭代步数
// 返回：
// phi0History
// phi0History[n-1][i] 表示第 n 次 SI 迭代后，
// 第 i 个 cell 上的零阶矩 phi0
std::vector<std::vector<double>> RSISolver::runSI(int& convergedN) const {
    //非结构网格中的 cell 数量
    const int C = static_cast<int>(mesh_.cells.size());

     //离散角方向数量
    const int M = static_cast<int>(ordinates_.size());

    // 预计算散射核矩阵 K[k][m]
    auto K = precomputeKernel();
    std::vector<SweepPlan> sweepPlans;
    sweepPlans.reserve(M);
    for (const auto& ord : ordinates_) {
        sweepPlans.push_back(sweep_.buildSweepPlan(ord.omega));
    }


    // phi[m][i] 表示：
    // 第 m 个方向、第 i 个 cell 上的散射源 phi_m，初始设为 0
    std::vector<std::vector<double>> phi;
    std::vector<double> isotropicPhi(C, 0.0);
    if (cfg_.scattering != "isotropic") {
        phi.assign(M, std::vector<double>(C, 0.0));
    }

    // psi[m][i] 表示：第 m 个方向、第 i 个 cell 上的角通量 psi_m。
    // 各向同性散射只需要当前迭代的 phi0，不需要同时保存所有方向的 psi。
    std::vector<std::vector<double>> psi;
    if (cfg_.scattering != "isotropic") {
        psi.assign(M, std::vector<double>(C, 0.0));
    }

    // 保存每次 SI 迭代得到的零阶矩 phi0
    // phi0_i = sum_m w_m psi_{m,i}
    // 它是论文 Figure 2 中误差比较使用的宏观量。
    std::vector<std::vector<double>> phi0History;

    for (int it = 1; it <= cfg_.maxSIters; ++it) {
        std::vector<double> phi0(C, 0.0);
        if (cfg_.scattering == "isotropic") {
            for (int m = 0; m < M; ++m) {
                std::vector<double> psiM =
                    sweep_.solveDirectionWithPlan(ordinates_[m], isotropicPhi, sweepPlans[m]);
                const double weight = ordinates_[m].weight;
                for (int i = 0; i < C; ++i) {
                    phi0[i] += weight * psiM[i];
                }
            }
            isotropicPhi = phi0;
        } else {
            // 对所有角方向做非结构网格输运扫掠
            // 对每个方向 m，求解：
            //     Omega_m · grad psi_m + Sigma_T psi_m
            //       = Sigma_S phi_m_old + Q
            // 这里 phi[m] 是上一轮得到的散射源
            // solveDirection 返回该方向在所有 cell 上的 psi
            for (int m = 0; m < M; ++m) {
                psi[m] = sweep_.solveDirectionWithPlan(ordinates_[m], phi[m], sweepPlans[m]);
            }

            std::vector<std::vector<double>> newPhi(M, std::vector<double>(C, 0.0));
            for (int i = 0; i < C; ++i) {
                for (int m = 0; m < M; ++m) {
                    double v = 0.0;
                    // 根据 DOM 散射项离散公式：
                    // phi_m = sum_k w_k K_{k,m} psi_k
                    // 这里 k 是入射方向索引
                    for (int k = 0; k < M; ++k) v += ordinates_[k].weight * K[k][m] * psi[k][i];
                    newPhi[m][i] = v;
                    phi0[i] += ordinates_[m].weight * psi[m][i];
                }
            }
            // 把本轮更新得到的newPhi作为下一轮的旧散射源phi
            phi = std::move(newPhi);
        }
        phi0History.push_back(phi0);


        // 从第二次迭代开始，可以比较相邻两次的 phi0 是否收敛
        if (it > 1) {
            // 计算：
            //
            //     ||phi0^{it} - phi0^{it-1}||_2
            //     --------------------------------
            //          ||phi0^{it-1}||_2
            //
            // 这里使用非结构网格体积作为积分权重
            double rel = relativeL2(phi0History[it - 1], phi0History[it - 2], mesh_);
            if (rel < cfg_.siTolerance) {
                convergedN = it;
                return phi0History;
            }
        }
    }
    convergedN = cfg_.maxSIters;
    return phi0History;
}



// 在指定迭代步 N 处运行 RSI，并计算误差。
// 输入：
// siPhi0History : SI 的零阶矩历史，用作参考
// N             : 比较第 N 次迭代
// sampleCount   : RSI 样本数 S

// 返回：
// e_RSI^(N) = || avg_phi0_RSI^(N) - phi0_SI^(N) ||_2
//             / || phi0_SI^(N) ||_2
//
// 这是论文 Figure2中的误差

double RSISolver::runRSIErrorAtN(const std::vector<std::vector<double>>& siPhi0History,
                                 int N,
                                 int sampleCount) const {
    const int C = static_cast<int>(mesh_.cells.size());
    const int M = static_cast<int>(ordinates_.size());
    const int G = cfg_.groupCount;

    auto K = precomputeKernel();
    std::vector<SweepPlan> sweepPlans;
    sweepPlans.reserve(M);
    for (const auto& ord : ordinates_) {
        sweepPlans.push_back(sweep_.buildSweepPlan(ord.omega));
    }
    auto groups = makeGroups(M, G);
    
    //创建随机数
    std::mt19937 rng(cfg_.seed + 17u * static_cast<unsigned>(M));

    std::vector<double> avgPhi0(C, 0.0);

    for (int s = 0; s < sampleCount; ++s) {
        std::vector<int> prevSet;
        std::vector<double> prevProb(M, 0.0);
        std::vector<std::vector<double>> prevPsi;

        // 初始 V^(0)：每个 group 均匀选一个方向
        for (const auto& group : groups) {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(group.size()) - 1);
            int k = group[dist(rng)];
            prevSet.push_back(k);
            prevProb[k] = 1.0 / static_cast<double>(group.size());
            prevPsi.push_back(std::vector<double>(C, 0.0));
        }

        for (int it = 1; it <= N; ++it) {
            // 论文式 (2.2): c_m = sum_{k in V^(n-1)} w_k K_{k,m}
            std::vector<double> c(M, 0.0);
            for (int m = 0; m < M; ++m) {
                for (int k : prevSet) {
                    c[m] += ordinates_[k].weight * K[k][m];
                }
            }

            // 论文式 (2.3): 每组独立抽一个方向
            std::vector<int> curSet;
            std::vector<double> curProb(M, 0.0);

            for (const auto& group : groups) {
                std::vector<double> probs;
                probs.reserve(group.size());

                double psum = 0.0;
                for (int m : group) {
                    double p = ordinates_[m].weight * c[m];
                    probs.push_back(p);
                    psum += p;
                }

                if (psum <= 0.0)
                    throw std::runtime_error("某个 group 的 RSI 抽样概率全为零");

                for (double& p : probs) p /= psum;

                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int local = dist(rng);
                int cur = group[local];

                curSet.push_back(cur);
                curProb[cur] = probs[local];
            }

            // 对当前选中的 G 个方向分别扫掠
            std::vector<std::vector<double>> curPsi;
            curPsi.reserve(curSet.size());

            for (int m : curSet) {
                std::vector<double> source(C, 0.0);

                // 论文式 (2.7)
                if (it > 1) {
                    for (int a = 0; a < static_cast<int>(prevSet.size()); ++a) {
                        int k = prevSet[a];
                        double pk = prevProb[k];

                        double coef = ordinates_[k].weight * K[k][m] / pk;

                        for (int i = 0; i < C; ++i) {
                            source[i] += coef * prevPsi[a][i];
                        }
                    }
                }

                curPsi.push_back(sweep_.solveDirectionWithPlan(ordinates_[m], source, sweepPlans[m]));
            }

            // 目标步 N：计算零阶矩估计
            if (it == N) {
                for (int a = 0; a < static_cast<int>(curSet.size()); ++a) {
                    int k = curSet[a];
                    double pk = curProb[k];

                    for (int i = 0; i < C; ++i) {
                        avgPhi0[i] += ordinates_[k].weight * (1.0 / pk) * curPsi[a][i];
                    }
                }
            }

            prevSet = std::move(curSet);
            prevProb = std::move(curProb);
            prevPsi = std::move(curPsi);
        }
    }

    for (double& v : avgPhi0)
        v /= static_cast<double>(sampleCount);

    return relativeL2(avgPhi0, siPhi0History[N - 1], mesh_);
}


// 运行论文 Figure 2 类型的实验。
//
// Figure 2 的横轴是样本数 S，纵轴是 RSI 误差 e_RSI^(N)。
// 所以这个函数做的事情是：
// 1. 先运行 SI，确定收敛步数 N
// 2. 对多个样本数 S 分别运行 RSI
// 3. 计算每个 S 对应的误差 e
// 4. 返回所有结果，用于写入 figure2_data.csv
std::vector<Figure2Row> RSISolver::runFigure2Experiment() {
    int N = 0;
    auto siPhi0History = runSI(N);
    std::vector<Figure2Row> rows;
    for (int S : cfg_.sampleCounts) {
        double e = runRSIErrorAtN(siPhi0History, N, S);
        rows.push_back({cfg_.scattering, static_cast<int>(ordinates_.size()), S, N, e});
        std::cout << cfg_.scattering << ", M=" << ordinates_.size() << ", S=" << S << ", N=" << N << ", e=" << e << "\n";
    }
    return rows;
}



// 计算两个 cell 标量场 a 和 b 的相对 L2 误差
//
// a: 待比较结果，例如 RSI 样本平均结果
// b: 参考结果，例如 SI 结果
// mesh: 非结构网格，用于提供每个 cell 的体积
//
// 非结构网格上的离散 L2 范数使用体积加权：
//
//     ||a-b||_2^2 ≈ sum_i V_i (a_i - b_i)^2
//
// 相对误差为：
//
//     relativeL2 = sqrt(
//         sum_i V_i (a_i - b_i)^2
//         /
//         sum_i V_i b_i^2
//     )
double RSISolver::relativeL2(const std::vector<double>& a,
                             const std::vector<double>& b,
                             const Mesh& mesh) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        num += mesh.cells[i].volume * d * d;
        den += mesh.cells[i].volume * b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}



// 运行一次SI，并返回 SI 收敛后的零阶矩空间场 phi0。
std::vector<double> RSISolver::runSIField(int& convergedN) const {
    auto hist = runSI(convergedN);
    return hist.back();
}


// 运行 RSI，并返回用于 Figure 5 绘图的空间零阶矩场 phi0。
// 这个函数和 runRSIErrorAtN() 的区别是：
//     runRSIErrorAtN() 只返回一个误差 e
//     runRSIFieldAtN() 返回每个 cell 上的 phi0 值
std::vector<double> RSISolver::runRSIFieldAtN(
    int N,
    int sampleCount,
    int tailExtra
) const {
    const int C = static_cast<int>(mesh_.cells.size());
    const int M = static_cast<int>(ordinates_.size());
    // G>1 时：
    //     把 M 个方向分成 G 组，
    //     每组独立抽一个方向，
    //     因此每步一共选 G 个方向
    const int G = cfg_.groupCount;

    auto K = precomputeKernel();
    std::vector<SweepPlan> sweepPlans;
    sweepPlans.reserve(M);
    for (const auto& ord : ordinates_) {
        sweepPlans.push_back(sweep_.buildSweepPlan(ord.omega));
    }


    // 将 M 个方向划分成 G 个 group。
    // groups[g] 是第 g 个方向组，里面存放方向编号。
    // RSI 每一步会从每个 group 中随机选择一个方向
    auto groups = makeGroups(M, G);
    std::vector<double> groupUniformProb;
    groupUniformProb.reserve(groups.size());
    for (const auto& group : groups) {
        groupUniformProb.push_back(1.0 / static_cast<double>(group.size()));
    }

    std::mt19937 rng(cfg_.seed + 17u * static_cast<unsigned>(M));


    // avgPhi0 用于累加 RSI 的零阶矩估计。
    // avgPhi0[i] 表示第 i 个 cell 上累加的 phi0。
    // 最后会除以 usedCount 得到平均
    std::vector<double> avgPhi0(C, 0.0);

     // usedCount 记录实际累加了多少个“样本层”。
    // 如果：
    //     sampleCount = S
    //     tailExtra = T
    // 那么理论上：
    //     usedCount = S * (T + 1)
    // 因为每条 RSI 链会贡献第 N 到第 N+T 的 T+1 个结果
    int usedCount = 0;

    const int lastIter = N + tailExtra;

    for (int s = 0; s < sampleCount; ++s) {
        std::vector<int> prevSet;
        std::vector<double> prevProb(M, 0.0);
        std::vector<std::vector<double>> prevPsi;

        for (const auto& group : groups) {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(group.size()) - 1);
            int k = group[dist(rng)];
            prevSet.push_back(k);
            prevProb[k] = 1.0 / static_cast<double>(group.size());
            prevPsi.push_back(std::vector<double>(C, 0.0));
        }

        for (int it = 1; it <= lastIter; ++it) {
            // c[m] 对应论文式 (2.2)：
            //      c_m^(n-1) = sum_{k in V^(n-1)} omega_k K_{k,m}
            // 它描述上一轮选中的方向集合对当前候选方向 m 的散射贡献强度
            std::vector<double> c;
            if (cfg_.scattering != "isotropic") {
                c.assign(M, 0.0);
                // 对每个候选方向m计算 c_m
                for (int m = 0; m < M; ++m) {
                    for (int k : prevSet) {
                        c[m] += ordinates_[k].weight * K[k][m];
                    }
                }
            }

            std::vector<int> curSet;
            std::vector<double> curProb(M, 0.0);


            // 对每个 group 独立抽一个方向。
            // 这对应论文式 (2.3)：
            //     p_m^(n)
            //       = omega_m c_m^(n-1)
            //         /
            //         sum_{m' in V(m)} omega_m' c_m'^(n-1)
            // 其中 V(m) 是方向 m 所在的 group
            for (size_t gi = 0; gi < groups.size(); ++gi) {
                const auto& group = groups[gi];
                if (cfg_.scattering == "isotropic") {
                    std::vector<double> probs(group.size(), groupUniformProb[gi]);
                    std::discrete_distribution<int> dist(probs.begin(), probs.end());
                    int cur = group[dist(rng)];
                    curSet.push_back(cur);
                    curProb[cur] = groupUniformProb[gi];
                    continue;
                }

                std::vector<double> probs;
                probs.reserve(group.size());

                double psum = 0.0;


                // 计算当前 group 内每个方向的未归一化概率：
                //     p_m ∝ omega_m c_m
                for (int m : group) {
                    double p = ordinates_[m].weight * c[m];
                    probs.push_back(p);
                    psum += p;
                }


                // 如果当前 group 的概率总和为 0
                // 说明这个 group 里所有方向都不可能被选中
                // 这种情况通常意味着散射核或方向分组有问题
                if (psum <= 0.0)
                    throw std::runtime_error("某个 group 的 RSI 抽样概率全为零");

                // 归一化当前 group 内的概率，使其和为 1
                for (double& p : probs) p /= psum;

                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int local = dist(rng);
                int cur = group[local];

                // 当前方向加入 V^(n)
                curSet.push_back(cur);
                curProb[cur] = probs[local];
            }


            // curPsi[a][i] 保存当前轮第 a 个选中方向在 cell i 上的角通量。
            std::vector<std::vector<double>> curPsi;
            curPsi.reserve(curSet.size());

            // 对当前轮选中的每个方向 m 分别做一次输运扫掠
            for (int m : curSet) {
                std::vector<double> source(C, 0.0);
                // it=1 时，初始 prevPsi 全为 0，
                // 因此散射源保持为 0
                // it>1 时，根据论文式 (2.7) 构造随机源：
                //     phi_tilde_m^(n-1)
                //       = sum_{k in V^(n-1)}
                //           omega_k K_{k,m} q_k^(n-1) psi_tilde_k^(n-1)
                // 其中：
                //     q_k = 1 / p_k

                if (it > 1) {
                    for (int a = 0; a < static_cast<int>(prevSet.size()); ++a) {
                        int k = prevSet[a];
                        double pk = prevProb[k];
                        // 对应论文中的：
                        //     omega_k K_{k,m} q_k
                        // 其中：
                        //     q_k = 1 / pk
                        // 所以：
                        //     coef = omega_k K_{k,m} / pk

                        double coef = (cfg_.scattering == "isotropic")
                            ? ordinates_[k].weight / pk
                            : ordinates_[k].weight * K[k][m] / pk;

                        for (int i = 0; i < C; ++i) {
                            source[i] += coef * prevPsi[a][i];
                        }
                    }
                }



                // 对当前方向 m 做非结构网格输运扫掠
                // solveDirection 求解：
                //
                //     Omega_m · grad psi_m + Sigma_T psi_m
                //       = Sigma_S source + Q
                // 返回当前方向在所有 cell 上的 psi
                curPsi.push_back(sweep_.solveDirectionWithPlan(ordinates_[m], source, sweepPlans[m]));
            }



            // 如果当前迭代步 it 已经进入统计区间，
            // 就把当前轮的零阶矩估计累加到 avgPhi0。
            //
            // tailExtra = 0:
            //     只在 it == N 时进入。
            //
            // tailExtra = 10:
            //     在 it = N, N+1, ..., N+10 时都进入
            if (it >= N) {
                for (int a = 0; a < static_cast<int>(curSet.size()); ++a) {
                    int k = curSet[a];
                    double pk = curProb[k];

                    for (int i = 0; i < C; ++i) {
                        avgPhi0[i] += ordinates_[k].weight * (1.0 / pk) * curPsi[a][i];
                    }
                }

                usedCount++;
            }

            prevSet = std::move(curSet);
            prevProb = std::move(curProb);
            prevPsi = std::move(curPsi);
        }
    }

    for (double& v : avgPhi0)
        v /= static_cast<double>(usedCount);;

    return avgPhi0;
}

void RSISolver::writeFieldCSV(const std::string& file,
                              const Mesh& mesh,
                              const std::vector<double>& phi0) {
    std::ofstream fout(file);
    fout << "cell_id,x,y,z,phi0\n";

    for (size_t i = 0; i < mesh.cells.size(); ++i) {
        const auto& c = mesh.cells[i];
        fout << c.id << ','
             << c.center.x << ','
             << c.center.y << ','
             << c.center.z << ','
             << phi0[i] << '\n';
    }
}
