#include "Mesh.hpp"
#include "RSI.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string cells = argc > 1 ? argv[1] : "Data/gmsh/cells.csv";
    const std::string faces = argc > 2 ? argv[2] : "Data/gmsh/faces.csv";
    const int angularN = argc > 3 ? std::stoi(argv[3]) : 4;
    const int samples = argc > 4 ? std::stoi(argv[4]) : 8;
    const int N = argc > 5 ? std::stoi(argv[5]) : 3;
    const int tailExtra = argc > 6 ? std::stoi(argv[6]) : 2;
    const bool gpuOnly = argc > 7 && std::string(argv[7]) == "gpu-only";
    const bool cpuOnly = argc > 7 && std::string(argv[7]) == "cpu-only";
    Mesh mesh = Mesh::readCSV(cells, faces);

    RSIConfig config;
    config.angularN = angularN;
    config.groupCount = 1;
    config.scattering = "isotropic";
    config.sourceShape = "circle";
    config.seed = 20260513u;

    if (cpuOnly) {
        const auto start = std::chrono::steady_clock::now();
        config.useGPU = false;
        RSISolver cpuSolver(mesh, config);
        const std::vector<double> cpu = cpuSolver.runRSIFieldAtN(N, samples, tailExtra);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();
        const double checksum = std::accumulate(cpu.begin(), cpu.end(), 0.0);
        std::cout << "CPU-only timing=" << seconds << " s, checksum=" << checksum << "\n";
        return 0;
    }

    config.useGPU = true;
    RSISolver gpuSolver(mesh, config);
    const std::vector<double> gpu = gpuSolver.runRSIFieldAtN(N, samples, tailExtra);
    if (gpuOnly) {
        const double checksum = std::accumulate(gpu.begin(), gpu.end(), 0.0);
        std::cout << "GPU-only checksum=" << checksum << "\n";
        return 0;
    }

    config.useGPU = false;
    RSISolver cpuSolver(mesh, config);
    const std::vector<double> cpu = cpuSolver.runRSIFieldAtN(N, samples, tailExtra);

    double numerator = 0.0;
    double denominator = 0.0;
    double maxAbsolute = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        const double difference = gpu[i] - cpu[i];
        numerator += mesh.cells[i].volume * difference * difference;
        denominator += mesh.cells[i].volume * cpu[i] * cpu[i];
        maxAbsolute = std::max(maxAbsolute, std::fabs(difference));
    }
    const double relativeL2 = std::sqrt(numerator / std::max(denominator, 1.0e-300));
    std::cout << "CPU-GPU consistency: relative_L2=" << relativeL2
              << ", max_abs=" << maxAbsolute << "\n";
    if (relativeL2 > 5.0e-11 || maxAbsolute > 5.0e-10) {
        throw std::runtime_error("CUDA RSI result differs from CPU reference");
    }

    if (argc == 1) {
        config.maxSIters = 30;
        config.siTolerance = 1.0e-10;
        config.useGPU = false;
        RSISolver cpuFigure5Solver(mesh, config);
        int cpuN = 0;
        const std::vector<double> cpuSI = cpuFigure5Solver.runSIField(cpuN);
        const std::vector<double> cpuRSI =
            cpuFigure5Solver.runRSIFieldAtN(cpuN, samples, 0);
        const std::vector<double> cpuTail =
            cpuFigure5Solver.runRSIFieldAtN(cpuN, samples, tailExtra);

        config.useGPU = true;
        RSISolver gpuFigure5Solver(mesh, config);
        const Figure5Fields fields = gpuFigure5Solver.runFigure5GPU(samples, tailExtra);
        if (fields.convergedN != cpuN) {
            throw std::runtime_error("CUDA SI converged at a different iteration");
        }

        auto compareField = [&](const char* name, const std::vector<double>& reference,
                                const std::vector<double>& candidate) {
            double fieldNumerator = 0.0;
            double fieldDenominator = 0.0;
            double fieldMax = 0.0;
            for (std::size_t i = 0; i < reference.size(); ++i) {
                const double difference = candidate[i] - reference[i];
                fieldNumerator += mesh.cells[i].volume * difference * difference;
                fieldDenominator += mesh.cells[i].volume * reference[i] * reference[i];
                fieldMax = std::max(fieldMax, std::fabs(difference));
            }
            const double fieldRelative =
                std::sqrt(fieldNumerator / std::max(fieldDenominator, 1.0e-300));
            std::cout << "Figure5 CPU-GPU " << name << ": relative_L2=" << fieldRelative
                      << ", max_abs=" << fieldMax << "\n";
            if (fieldRelative > 5.0e-10 || fieldMax > 5.0e-9) {
                throw std::runtime_error(std::string("CUDA Figure 5 mismatch: ") + name);
            }
        };
        compareField("SI", cpuSI, fields.siFine);
        compareField("RSI", cpuRSI, fields.rsi);
        compareField("RSI-tail", cpuTail, fields.rsiTail);
    }
    return 0;
}
