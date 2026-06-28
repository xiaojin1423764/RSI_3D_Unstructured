#pragma once
#include "Mesh.hpp"
#include "Quadrature.hpp"
#include <string>
#include <vector>

struct SweepPlan {
    std::vector<int> order;
    bool hasCycle = false;
};

class TransportSweep {
public:
    explicit TransportSweep(const Mesh& mesh, const std::string& sourceShape = "rectangle")
        : mesh_(mesh), sourceShape_(sourceShape) {}


    double boundaryInflow(const Face& face, const Vec3& omega) const;

    // 对一个方向做一次非结构有限体积上风扫掠。
    // source_phi[cell] 是本方向散射源 phi_m 在各 cell 上的值。
    std::vector<double> solveDirection(const Ordinate& ord,
                                       const std::vector<double>& source_phi) const;

    // Same sweep as solveDirection, using a precomputed order/cycle for repeated
    // solves with the same ordinate.
    std::vector<double> solveDirectionWithPlan(const Ordinate& ord,
                                               const std::vector<double>& source_phi,
                                               const SweepPlan& plan) const;

    SweepPlan buildSweepPlan(const Vec3& omega) const;

private:
    const Mesh& mesh_;
    std::string sourceShape_;

    std::vector<int> buildSweepOrder(const Vec3& omega) const;
};
