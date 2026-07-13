#pragma once

#include "Mesh.hpp"
#include "Quadrature.hpp"
#include "TransportSweep.hpp"
#include <string>
#include <vector>

bool cudaRSIAvailable(std::string* reason = nullptr);

struct CudaFigure5Result {
    std::vector<double> siFine;
    std::vector<double> rsi;
    std::vector<double> rsiTail;
    int convergedN = 0;
};

CudaFigure5Result runFigure5Cuda(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const std::vector<SweepPlan>& sweepPlans,
    const std::string& sourceShape,
    unsigned int seed,
    int maxSIters,
    double siTolerance,
    int sampleCount,
    int tailExtra
);

std::vector<double> runRSIFieldAtNCuda(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const std::vector<SweepPlan>& sweepPlans,
    const std::string& sourceShape,
    unsigned int seed,
    int N,
    int sampleCount,
    int tailExtra
);
