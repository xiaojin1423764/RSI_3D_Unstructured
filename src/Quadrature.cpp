#include "Quadrature.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

std::vector<Ordinate> Quadrature::levelSymmetricSN(int snOrder) {
    if (snOrder < 2 || snOrder % 2 != 0) {
        throw std::runtime_error("S_N 阶数必须是 >= 2 的偶数");
    }

    const int levels = snOrder / 2;
    const int levelSum = levels + 2;
    const double directionWeight =
        1.0 / static_cast<double>(snOrder * (snOrder + 2));

    std::vector<Ordinate> ordinates;
    ordinates.reserve(snOrder * (snOrder + 2));

    for (int i = 1; i <= levels; ++i) {
        for (int j = 1; j <= levels; ++j) {
            const int k = levelSum - i - j;
            if (k < 1 || k > levels) continue;

            const double norm = std::sqrt(
                static_cast<double>(i * i + j * j + k * k)
            );
            const double omegaXBase = static_cast<double>(i) / norm;
            const double omegaYBase = static_cast<double>(j) / norm;
            const double omegaZBase = static_cast<double>(k) / norm;

            for (int sx : {-1, 1}) {
                for (int sy : {-1, 1}) {
                    for (int sz : {-1, 1}) {
                        ordinates.push_back({
                            {sx * omegaXBase, sy * omegaYBase, sz * omegaZBase},
                            directionWeight
                        });
                    }
                }
            }
        }
    }

    if (static_cast<int>(ordinates.size()) != snOrder * (snOrder + 2)) {
        throw std::runtime_error("level-symmetric S_N 方向数生成错误");
    }

    double weightSum = 0.0;
    Vec3 firstMoment;
    for (const auto& ord : ordinates) {
        weightSum += ord.weight;
        firstMoment.x += ord.weight * ord.omega.x;
        firstMoment.y += ord.weight * ord.omega.y;
        firstMoment.z += ord.weight * ord.omega.z;
    }

    const double tol = 1e-12;
    if (std::fabs(weightSum - 1.0) > tol ||
        std::fabs(firstMoment.x) > tol ||
        std::fabs(firstMoment.y) > tol ||
        std::fabs(firstMoment.z) > tol) {
        throw std::runtime_error("level-symmetric S_N 归一化检查失败");
    }

    return ordinates;
}
