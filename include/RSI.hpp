#pragma once
#include "Mesh.hpp"
#include "Quadrature.hpp"
#include "TransportSweep.hpp"
#include <random>
#include <string>
#include <vector>

struct FieldRow {
    int cell_id;
    double x, y, z;
    double phi0;
};

struct RSIConfig {
    int angularN = 4;              // level-symmetric S_N 阶数，M = N * (N + 2)
    int maxSIters = 100;
    double siTolerance = 1e-10;
    std::vector<int> sampleCounts; // Figure 2 的横坐标 S
    unsigned int seed = 12345;
    std::string scattering = "isotropic"; // isotropic / anisotropic
    std::string sourceShape = "rectangle"; // rectangle / circle
    int groupCount = 1;  // RSI 每步选择的方向数 G
    bool useGPU = false; // CUDA sample-parallel backend for isotropic G=1 field runs
};

struct Figure2Row {
    std::string scattering;
    int M;
    int S;
    int iterationN;
    double eRSI;
};

struct Figure5Fields {
    std::vector<double> siFine;
    std::vector<double> rsi;
    std::vector<double> rsiTail;
    int convergedN = 0;
};

class RSISolver {
public:
    RSISolver(const Mesh& mesh, RSIConfig cfg);

    std::vector<Figure2Row> runFigure2Experiment();
    std::vector<Figure2Row> runFigure2GPUConvergence();
    std::vector<double> runSIField(int& convergedN) const;
    std::vector<double> runRSIFieldAtN(int N, int sampleCount, int tailExtra) const;
    Figure5Fields runFigure5GPU(int sampleCount, int tailExtra) const;
    void printSweepPlanStats() const;
    static void writeFieldCSV(const std::string& file,
                              const Mesh& mesh,
                              const std::vector<double>& phi0);
private:
    const Mesh& mesh_;
    RSIConfig cfg_;
    std::vector<Ordinate> ordinates_;
    TransportSweep sweep_;
    mutable std::vector<SweepPlan> sweepPlansCache_;

    const std::vector<SweepPlan>& cachedSweepPlans() const;

    double kernel(int k, int m) const;
    std::vector<std::vector<double>> precomputeKernel() const;

    // SI 作为无偏目标：返回每次迭代的零阶矩 phi0；同时输出收敛迭代 N。
    std::vector<std::vector<double>> runSI(int& convergedN) const;

    // 对给定 S 运行 RSI 样本，和 SI 第 N 步比较，返回 e_RSI^(N)。
    double runRSIErrorAtN(const std::vector<std::vector<double>>& siPhi0History,
                          int N, int sampleCount) const;

    static double relativeL2(const std::vector<double>& a,
                             const std::vector<double>& b,
                             const Mesh& mesh);
};
