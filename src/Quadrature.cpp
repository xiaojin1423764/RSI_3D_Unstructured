#include "Quadrature.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

static std::vector<std::pair<double, double>> gaussLegendre(int n) {
    const int maxIter = 100;
    const double tol = 1e-14;
    const double pi = std::acos(-1.0);
    std::vector<double> x(n, 0.0);
    std::vector<double> w(n, 0.0);

    const int m = (n + 1) / 2;
    for (int i = 0; i < m; ++i) {
        double z = std::cos(pi * (static_cast<double>(i) + 0.75) /
                            (static_cast<double>(n) + 0.5));
        double pp = 0.0;
        for (int iter = 0; iter < maxIter; ++iter) {
            double p1 = 1.0;
            double p2 = 0.0;
            for (int j = 1; j <= n; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) /
                     static_cast<double>(j);
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            const double zOld = z;
            z = zOld - p1 / pp;
            if (std::fabs(z - zOld) < tol) break;
            if (iter == maxIter - 1) {
                throw std::runtime_error("Gauss-Legendre 节点迭代未收敛");
            }
        }

        x[i] = -z;
        x[n - 1 - i] = z;
        const double wi = 2.0 / ((1.0 - z * z) * pp * pp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }

    std::vector<std::pair<double, double>> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back({x[i], w[i]});
    return out;
}

std::vector<Ordinate> Quadrature::uniformSphere(int angularN) {
    if (angularN < 2) throw std::runtime_error("angularN 至少为 2");
    std::vector<Ordinate> ords;
    const double pi = std::acos(-1.0);

    const int N = angularN;
    const auto muQuad = gaussLegendre(N);

    for (int l = 0; l < N; ++l) {
        const double mu = muQuad[l].first;
        const double weightMu = muQuad[l].second;
        double r = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        for (int k = 0; k < N; ++k) {
            double theta = 2.0 * pi * (static_cast<double>(k) + 0.5) /
                           static_cast<double>(N);
            double weight = weightMu / (2.0 * static_cast<double>(N));
            ords.push_back({{r * std::cos(theta), r * std::sin(theta), mu}, weight});
        }
    }
    return ords;
}
