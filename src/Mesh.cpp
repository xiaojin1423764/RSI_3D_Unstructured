#include "Mesh.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
// 将一行 CSV 文本按英文逗号 ',' 拆分成字符串数组
// 例如："0,0.5,0.5,0.5,1.0"
// 会被拆成：["0", "0.5", "0.5", "0.5", "1.0"]
std::vector<std::string> splitCSVLine(const std::string& line) {
    std::vector<std::string> out;// 保存拆分后的每一列字符串
    std::string cur; // 临时变量，用来保存当前读到
    std::stringstream ss(line);
    while (std::getline(ss, cur, ',')) out.push_back(cur);
    return out;
}

// 去掉字符串首尾空白字符
// 例如： "  vacuum  " 会变成 "vacuum"
std::string trim(std::string s) {
    auto notSpace = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}
}

// 从 cells.csv 和 faces.csv 中读取三维非结构网格
// cellsFile 是单元文件路径
// facesFile 是面文件路径
// 返回值是构造好的 Mesh 对象
Mesh Mesh::readCSV(const std::string& cellsFile, const std::string& facesFile) {
    Mesh mesh;

    // cells.csv 格式：cell_id,cx,cy,cz,volume,sigma_t,sigma_s,q
    // cell_id,cx,cy,cz,volume,sigma_t,sigma_s,q
    // cell_id  : 单元编号，可以不连续，但必须唯一
    // cx,cy,cz : 单元中心坐标
    // volume   : 单元体积
    // sigma_t  : 总截面 ΣT
    // sigma_s  : 散射截面 ΣS
    // q        : 外源项 Q
    std::ifstream cinFile(cellsFile);
    if (!cinFile) throw std::runtime_error("无法打开 cells.csv: " + cellsFile);
    std::string line;
    std::getline(cinFile, line); // header
    while (std::getline(cinFile, line)) {
        if (trim(line).empty()) continue;
        auto t = splitCSVLine(line);
        // cells.csv 至少需要 8 列，如果不足 8 列，说明格式错误
        if (t.size() < 8) throw std::runtime_error("cells.csv 列数不足: " + line);
        Cell c;
        c.id = std::stoi(t[0]);
        c.center = {std::stod(t[1]), std::stod(t[2]), std::stod(t[3])};
        c.volume = std::stod(t[4]);
        c.sigma_t = std::stod(t[5]);
        c.sigma_s = std::stod(t[6]);
        c.q = std::stod(t[7]);

        // 建立 cell_id 到数组下标的映射
        // mesh.cells.size() 是当前单元将要插入的位置
        // 例如 cell_id=100 可能存放在 cells[0]，这里就记录 100 -> 0
        mesh.cellIdToIndex[c.id] = static_cast<int>(mesh.cells.size());
        mesh.cells.push_back(c);
    }

    // faces.csv 格式：face_id,left_cell,right_cell,nx,ny,nz,area,fx,fy,fz,bc_type,bc_value
    //face_id     : 面编号，可以不连续，但必须唯一
    // left_cell  : 面左侧单元编号，必须有效
    // right_cell : 面右侧单元编号；如果为 -1，表示这是边界面
    // nx,ny,nz   : 面法向量，约定从 left_cell 指向 right_cell
    // area       : 面面积
    // fx,fy,fz   : 面中心坐标
    // bc_type    : 边界类型，例如 vacuum、inflow、example1_inflow、internal
    // bc_value   : 边界给定值

    std::ifstream finFile(facesFile);
    if (!finFile) throw std::runtime_error("无法打开 faces.csv: " + facesFile);
    std::getline(finFile, line); // header
    while (std::getline(finFile, line)) {
        if (trim(line).empty()) continue;
        auto t = splitCSVLine(line);

        // faces.csv 至少需要 12 列
        if (t.size() < 12) throw std::runtime_error("faces.csv 列数不足: " + line);
        Face f;
        f.id = std::stoi(t[0]);
        f.left_cell = std::stoi(t[1]);
        f.right_cell = std::stoi(t[2]);
        f.normal = {std::stod(t[3]), std::stod(t[4]), std::stod(t[5])};
        f.area = std::stod(t[6]);
        f.center = {std::stod(t[7]), std::stod(t[8]), std::stod(t[9])};
        f.bc_type = trim(t[10]);
        f.bc_value = std::stod(t[11]);

        // 建立 face_id 到 faces 数组下标的映射
        mesh.faceIdToIndex[f.id] = static_cast<int>(mesh.faces.size());
        mesh.faces.push_back(f);
    }

    mesh.buildAdjacency();
    mesh.validate();
    return mesh;
}

int Mesh::cellIndex(int id) const {
    auto it = cellIdToIndex.find(id);
    if (it == cellIdToIndex.end()) throw std::runtime_error("未知 cell_id: " + std::to_string(id));
    return it->second;
}

int Mesh::faceIndex(int id) const {
    auto it = faceIdToIndex.find(id);
    if (it == faceIdToIndex.end()) throw std::runtime_error("未知 face_id: " + std::to_string(id));
    return it->second;
}

void Mesh::buildAdjacency() {
    for (auto& c : cells) {
        c.faceRefs.clear();
    }

    for (int fi = 0; fi < static_cast<int>(faces.size()); ++fi) {
        const Face& f = faces[fi];

        if (f.left_cell < 0) {
            throw std::runtime_error("face left_cell 必须有效");
        }

        int L = cellIndex(f.left_cell);

        cells[L].faceRefs.push_back({
            fi,
            f.right_cell >= 0 ? cellIndex(f.right_cell) : -1,
            +1
        });

        if (f.right_cell >= 0) {
            int R = cellIndex(f.right_cell);

            cells[R].faceRefs.push_back({
                fi,
                L,
                -1
            });
        }
    }
}

void Mesh::validate() const {
    if (cells.empty()) throw std::runtime_error("网格没有 cell");
    if (faces.empty()) throw std::runtime_error("网格没有 face");
    for (const auto& c : cells) {
        if (c.volume <= 0) throw std::runtime_error("cell volume 必须为正");
        if (c.sigma_t <= 0) throw std::runtime_error("sigma_t 必须为正");
        if (c.sigma_s < 0 || c.sigma_s >= c.sigma_t)
            std::cerr << "警告: cell " << c.id << " 不满足 0 <= sigma_s < sigma_t，SI/RSI 可能不收敛。\n";
    }
    for (const auto& f : faces) {
        if (f.left_cell < 0) throw std::runtime_error("face left_cell 必须有效");
        (void)cellIndex(f.left_cell);
        if (f.right_cell >= 0) (void)cellIndex(f.right_cell);
        if (f.area <= 0) throw std::runtime_error("face area 必须为正");
    }
}
