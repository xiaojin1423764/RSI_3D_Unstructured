#pragma once
#include "Types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct CellFaceRef {
    int face;       // face 在 mesh.faces 中的下标
    int neighbor;   // 邻居 cell 在 mesh.cells 中的下标；边界为 -1
    int sign;       // +1 当前 cell 是 left_cell；-1 当前 cell 是 right_cell
};

// 单元数据：非结构网格中每个控制体只需要中心、体积和材料系数。
struct Cell {
    int id = -1;
    Vec3 center;
    double volume = 0.0;
    double sigma_t = 1.0; // 总截面 Sigma_T
    double sigma_s = 0.5; // 散射截面 Sigma_S
    double q = 0.0;       // 体源项 Q
    
    std::vector<CellFaceRef> faceRefs;
};

// 面数据：normal 约定为从 left_cell 指向 right_cell；边界面 right_cell=-1，normal 为 left_cell 外法向。
struct Face {
    int id = -1;
    int left_cell = -1;
    int right_cell = -1;
    Vec3 normal;
    double area = 0.0;
    Vec3 center;
    std::string bc_type = "vacuum"; // internal/vacuum/example1/inflow
    double bc_value = 0.0;
};

class Mesh {
public:
    std::vector<Cell> cells;
    std::vector<Face> faces;
    std::unordered_map<int, int> cellIdToIndex;
    std::unordered_map<int, int> faceIdToIndex;

    static Mesh readCSV(const std::string& cellsFile, const std::string& facesFile);

    int cellIndex(int id) const;
    int faceIndex(int id) const;
    void buildAdjacency();
    void validate() const;
};
