#pragma once
#include <array>
#include <string>
#include <vector>

// 三维向量，所有几何量和角方向都使用它。
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
