#pragma once
#include "Types.hpp"
#include <vector>

struct Ordinate {
    Vec3 omega;   // 单位方向 Omega=(c,s,mu)
    double weight; // 权重，满足 sum weight = 1
};

// 论文 Example 1 使用 (xi,theta) 均匀角网格；这里扩展到 S2：mu in [-1,1], theta in [0,2pi)。
class Quadrature {
public:
    static std::vector<Ordinate> uniformSphere(int angularN);
};
