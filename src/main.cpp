#include "Mesh.hpp"
#include "RSI.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
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

static std::string figure5OutputPath(const std::string& figurePrefix, const std::string& fileName) {
    const std::string dir = "examples/csv_data/" + figurePrefix;
    std::filesystem::create_directories(dir);
    return dir + "/" + fileName;
}

static void printUsage(const char* prog) {
    std::cerr
        << "用法:\n"
        << "  " << prog << " [cells.csv faces.csv [figure2_data.csv] [rectangle|circle]]\n"
        << "  " << prog << " --source-shape rectangle|circle [--out figure2_data.csv] [--only all|figure2|figure5] [cells.csv faces.csv]\n";
}

int main(int argc, char** argv) {
    std::string cellsFile = "data/cells.csv";
    std::string facesFile = "data/faces.csv";
    std::string outFile = "examples/figure2_data.csv";
    std::string sourceShape = "rectangle";
    std::string only = "all";

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--source-shape") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            sourceShape = argv[++i];
        } else if (arg == "--out") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            outFile = argv[++i];
        } else if (arg == "--only") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            only = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "未知参数: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() >= 2) {
        cellsFile = positional[0];
        facesFile = positional[1];
    }
    if (positional.size() >= 3) {
        if (isSourceShape(positional[2])) {
            sourceShape = positional[2];
        } else {
            outFile = positional[2];
        }
    }
    if (positional.size() >= 4) sourceShape = positional[3];
    if (positional.size() > 4) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        if (!isSourceShape(sourceShape)) {
            throw std::runtime_error("入射区域形状必须是 rectangle 或 circle");
        }
        if (only != "all" && only != "figure2" && only != "figure5") {
            throw std::runtime_error("--only 必须是 all、figure2 或 figure5");
        }

        Mesh mesh = Mesh::readCSV(cellsFile, facesFile);
        writeSourceShapeFile(sourceShape);
        const std::string figurePrefix = sourcePrefix(sourceShape);
        std::cout << "入射区域形状: " << sourceShape << " (" << figurePrefix << ")\n";

        if (only == "all" || only == "figure2") {
            std::vector<Figure2Row> allRows;
            for (std::string scat : {"isotropic", "anisotropic"}) {//各向同性，各向异性
                for (int angularN : {8,16}) {//level-symmetric S_N, M=N*(N+2)
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
        }

        if (only == "all" || only == "figure5") {
            std::cout << "开始输出 Figure 5 空间场数据...\n";

            // 粗角度SI
            // S4: M=24
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
            const std::string siCoarseFile = figure5OutputPath(figurePrefix, "figure5_SI_coarse.csv");
            const std::string siFineFile = figure5OutputPath(figurePrefix, "figure5_SI_fine.csv");
            const std::string rsiFile = figure5OutputPath(figurePrefix, "figure5_RSI.csv");
            const std::string rsiTailFile = figure5OutputPath(figurePrefix, "figure5_RSI_tail.csv");
            RSISolver::writeFieldCSV(siCoarseFile, mesh, phiSIcoarse);

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
            RSISolver::writeFieldCSV(siFineFile, mesh, phiSIfine);

            // RSI，使用256个样本
            int S = 256;
            auto phiRSI = fineSolver.runRSIFieldAtN(Nfine, S, 0);
            RSISolver::writeFieldCSV(rsiFile, mesh, phiRSI);

            // RSI + 尾部平均。
            // tailExtra=10 表示平均 Nfine 到 Nfine+10 的数据。
            int tailExtra = 10;
            auto phiRSITail = fineSolver.runRSIFieldAtN(Nfine, S, tailExtra);
            RSISolver::writeFieldCSV(rsiTailFile, mesh, phiRSITail);

            std::cout << "Figure 5 数据已输出:\n";
            std::cout << "  " << siCoarseFile << "\n";
            std::cout << "  " << siFineFile << "\n";
            std::cout << "  " << rsiFile << "\n";
            std::cout << "  " << rsiTailFile << "\n";
        }
    } 
    
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
