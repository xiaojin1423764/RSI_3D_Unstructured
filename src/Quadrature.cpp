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

    // This is a validation-only reduction.  S316 has 100,488 directions, so
    // double sequential summation can exceed the diagnostic tolerance even
    // though the symmetric direction set itself is correct.
    long double weightSum = 0.0L;
    long double firstMomentX = 0.0L;
    long double firstMomentY = 0.0L;
    long double firstMomentZ = 0.0L;
    for (const auto& ord : ordinates) {
        weightSum += ord.weight;
        firstMomentX += static_cast<long double>(ord.weight) * ord.omega.x;
        firstMomentY += static_cast<long double>(ord.weight) * ord.omega.y;
        firstMomentZ += static_cast<long double>(ord.weight) * ord.omega.z;
    }

    constexpr long double tol = 1e-12L;
    if (std::fabs(weightSum - 1.0L) > tol ||
        std::fabs(firstMomentX) > tol ||
        std::fabs(firstMomentY) > tol ||
        std::fabs(firstMomentZ) > tol) {
        throw std::runtime_error("level-symmetric S_N 归一化检查失败");
    }

    return ordinates;
}
