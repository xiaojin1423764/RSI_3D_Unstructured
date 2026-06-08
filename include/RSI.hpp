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
    int angularN = 4;              // M = angularN^2
    int maxSIters = 100;
    double siTolerance = 1e-10;
    std::vector<int> sampleCounts; // Figure 2 的横坐标 S
    unsigned int seed = 12345;
    std::string scattering = "isotropic"; // isotropic / anisotropic
    std::string sourceShape = "rectangle"; // rectangle / circle
    int groupCount = 1;  // RSI 每步选择的方向数 G
};

struct Figure2Row {
    std::string scattering;
    int M;
    int S;
    int iterationN;
    double eRSI;
};

class RSISolver {
public:
    RSISolver(const Mesh& mesh, RSIConfig cfg);

    std::vector<Figure2Row> runFigure2Experiment();
    std::vector<double> runSIField(int& convergedN) const;
    std::vector<double> runRSIFieldAtN(int N, int sampleCount, int tailExtra) const;
    static void writeFieldCSV(const std::string& file,
                              const Mesh& mesh,
                              const std::vector<double>& phi0);
private:
    const Mesh& mesh_;
    RSIConfig cfg_;
    std::vector<Ordinate> ordinates_;
    TransportSweep sweep_;

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
