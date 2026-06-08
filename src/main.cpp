#include "Mesh.hpp"
#include "RSI.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static bool isSourceShape(const std::string& value) {
    return value == "rectangle" || value == "circle";
}

static std::string sourcePrefix(const std::string& sourceShape) {
    if (sourceShape == "rectangle") return "Rec";
    if (sourceShape == "circle") return "Cir";
    throw std::runtime_error("sourceShape 必须是 rectangle 或 circle");
}

static void writeSourceShapeFile(const std::string& sourceShape) {
    std::ofstream fout("examples/csv_data/source_shape.txt");
    fout << sourceShape << '\n';
}

int main(int argc, char** argv) {
    std::string cellsFile = "data/cells.csv";
    std::string facesFile = "data/faces.csv";
    std::string outFile = "examples/figure2_data.csv";
    std::string sourceShape = "rectangle";

    if (argc >= 3) {
        cellsFile = argv[1];
        facesFile = argv[2];
    }
    if (argc >= 4) {
        std::string arg3 = argv[3];
        if (isSourceShape(arg3)) {
            sourceShape = arg3;
        } else {
            outFile = arg3;
        }
    }
    if (argc >= 5) sourceShape = argv[4];

    try {
        if (!isSourceShape(sourceShape)) {
            throw std::runtime_error("入射区域形状必须是 rectangle 或 circle");
        }

        Mesh mesh = Mesh::readCSV(cellsFile, facesFile);
        writeSourceShapeFile(sourceShape);
        const std::string figurePrefix = sourcePrefix(sourceShape);
        std::cout << "入射区域形状: " << sourceShape << " (" << figurePrefix << ")\n";
        std::vector<Figure2Row> allRows;


        for (std::string scat : {"isotropic", "anisotropic"}) {//各向同性，各向异性
            for (int angularN : {8,16}) {//M=N*N
                RSIConfig cfg;
                cfg.groupCount = 2;
                cfg.angularN = angularN;
                cfg.maxSIters = 80;
                cfg.siTolerance = 1e-10;
                cfg.sampleCounts = {4,8,16,32,64,128,256,512,1024};
                cfg.scattering = scat;
                cfg.sourceShape = sourceShape;
                cfg.seed = 20260514u;//设置随机种子
                RSISolver solver(mesh, cfg);
                auto rows = solver.runFigure2Experiment();
                allRows.insert(allRows.end(), rows.begin(), rows.end());
            }
        }

        std::ofstream fout(outFile);
        fout << "scattering,M,S,iterationN,e_RSI_N\n";
        for (const auto& r : allRows) {
            fout << r.scattering << ',' << r.M << ',' << r.S << ',' << r.iterationN << ',' << r.eRSI << '\n';
        }
        std::cout << "已输出: " << outFile << "\n";


        {
            std::cout << "开始输出 Figure 5 空间场数据...\n";

            // 粗角度SI
            //M=16(4*4)
            RSIConfig coarseCfg;
            coarseCfg.groupCount = 1;
            coarseCfg.angularN = 4;
            coarseCfg.maxSIters = 80;
            coarseCfg.siTolerance = 1e-10;
            coarseCfg.sampleCounts = {256};
            coarseCfg.scattering = "isotropic";
            coarseCfg.sourceShape = sourceShape;
            coarseCfg.seed = 20260513u;

            RSISolver coarseSolver(mesh, coarseCfg);
            int Ncoarse = 0;
            auto phiSIcoarse = coarseSolver.runSIField(Ncoarse);
            RSISolver::writeFieldCSV("examples/csv_data/figure5_SI_coarse.csv", mesh, phiSIcoarse);

            // 细角度SI
            RSIConfig fineCfg;
            fineCfg.groupCount = 1;
            fineCfg.angularN =16;
            fineCfg.maxSIters = 80;
            fineCfg.siTolerance = 1e-10;
            fineCfg.sampleCounts = {256};
            fineCfg.scattering = "isotropic";
            fineCfg.sourceShape = sourceShape;
            fineCfg.seed = 20260513u;

            RSISolver fineSolver(mesh, fineCfg);
            int Nfine = 0;
            auto phiSIfine = fineSolver.runSIField(Nfine);
            RSISolver::writeFieldCSV("examples/csv_data/figure5_SI_fine.csv", mesh, phiSIfine);

            // RSI，使用256个样本
            int S = 256;
            auto phiRSI = fineSolver.runRSIFieldAtN(Nfine, S, 0);
            RSISolver::writeFieldCSV("examples/csv_data/figure5_RSI.csv", mesh, phiRSI);

            // RSI + 尾部平均。
            // tailExtra=10 表示平均 Nfine 到 Nfine+10 的数据。
            int tailExtra = 10;
            auto phiRSITail = fineSolver.runRSIFieldAtN(Nfine, S, tailExtra);
            RSISolver::writeFieldCSV("examples/csv_data/figure5_RSI_tail.csv", mesh, phiRSITail);

            std::cout << "Figure 5 数据已输出:\n";
            std::cout << "  examples/csv_data/figure5_SI_coarse.csv\n";
            std::cout << "  examples/csv_data/figure5_SI_fine.csv\n";
            std::cout << "  examples/csv_data/figure5_RSI.csv\n";
            std::cout << "  examples/csv_data/figure5_RSI_tail.csv\n";
        }
    } 
    
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
