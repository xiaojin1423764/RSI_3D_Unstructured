#pragma once
#include "Types.hpp"
#include <vector>

struct Ordinate {
    Vec3 omega;   // 单位方向 Omega=(c,s,mu)
    double weight; // 权重，满足 sum weight = 1
};

class Quadrature {
public:
    static std::vector<Ordinate> levelSymmetricSN(int snOrder);
};
