#include "Mesh.hpp"
#include "RSI.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sstream>
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

static void writeSourceShapeFile(const std::string& sourceShape, const std::string& figure5Dir) {
    std::filesystem::create_directories(figure5Dir);
    std::ofstream fout(figure5Dir + "/source_shape.txt");
    fout << sourceShape << '\n';
}

static std::string figure5OutputPath(
    const std::string& figure5Dir,
    const std::string& figurePrefix,
    const std::string& fileName
) {
    const std::string dir = figure5Dir + "/" + figurePrefix;
    std::filesystem::create_directories(dir);
    return dir + "/" + fileName;
}

static void printUsage(const char* prog) {
    std::cerr
        << "用法:\n"
        << "  " << prog << " [cells.csv faces.csv [figure2_data.csv] [rectangle|circle]]\n"
        << "  " << prog << " --source-shape rectangle|circle [--gpu] [--out figure2_data.csv] [--figure5-dir Data/csv_data] [--figure5-fine-sn 32] [--figure5-samples 512] [--figure2-angular-list 16,128] [--figure2-samples-list 8,16,32,64,128,256,512,1024,2048,4096,8192] [--figure2-scattering-list isotropic,anisotropic] [--only all|figure2|figure2-gpu-convergence|figure5|sweep-stats] [cells.csv faces.csv]\n";
}

static std::vector<int> parseIntList(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("整数列表中存在空项: " + text);
        }
        values.push_back(std::stoi(item));
    }
    if (values.empty()) {
        throw std::runtime_error("整数列表不能为空");
    }
    return values;
}

static std::vector<std::string> parseStringList(const std::string& text) {
    std::vector<std::string> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("字符串列表中存在空项: " + text);
        }
        values.push_back(item);
    }
    if (values.empty()) {
        throw std::runtime_error("字符串列表不能为空");
    }
    return values;
}

int main(int argc, char** argv) {
    std::string cellsFile = "Data/gmsh/cells.csv";
    std::string facesFile = "Data/gmsh/faces.csv";
    std::string outFile = "Data/csv_data/figure2_data.csv";
    std::string figure5Dir = "Data/csv_data";
    std::string sourceShape = "rectangle";
    std::string only = "all";
    int figure5FineSN = 32;
    int figure5Samples = 512;
    std::vector<int> figure2AngularList = {16, 128};
    std::vector<int> figure2SamplesList = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    std::vector<std::string> figure2ScatteringList = {"isotropic", "anisotropic"};
    bool useGPU = false;

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
        } else if (arg == "--figure5-dir") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            figure5Dir = argv[++i];
        } else if (arg == "--only") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            only = argv[++i];
        } else if (arg == "--figure5-fine-sn") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            figure5FineSN = std::stoi(argv[++i]);
        } else if (arg == "--figure5-samples") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            figure5Samples = std::stoi(argv[++i]);
        } else if (arg == "--figure2-angular-list") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            figure2AngularList = parseIntList(argv[++i]);
        } else if (arg == "--figure2-samples-list") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            figure2SamplesList = parseIntList(argv[++i]);
        } else if (arg == "--figure2-scattering-list") {
            if (i + 1 >= argc) {
                printUsage(argv[0]);
                return 1;
            }
            figure2ScatteringList = parseStringList(argv[++i]);
        } else if (arg == "--gpu") {
            useGPU = true;
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
        if (only != "all" && only != "figure2" && only != "figure2-gpu-convergence" &&
            only != "figure5" && only != "sweep-stats") {
            throw std::runtime_error("--only 必须是 all、figure2、figure2-gpu-convergence、figure5 或 sweep-stats");
        }
        if (figure5FineSN < 2 || figure5FineSN % 2 != 0) {
            throw std::runtime_error("--figure5-fine-sn 必须是 >=2 的偶数");
        }
        if (figure5Samples <= 0) {
            throw std::runtime_error("--figure5-samples 必须为正整数");
        }
        for (int angularN : figure2AngularList) {
            if (angularN < 2 || angularN % 2 != 0) {
                throw std::runtime_error("--figure2-angular-list 中的每个 S_N 阶数必须是 >=2 的偶数");
            }
        }
        for (int samples : figure2SamplesList) {
            if (samples <= 0) {
                throw std::runtime_error("--figure2-samples-list 中的每个样本数必须为正整数");
            }
        }
        for (const std::string& scattering : figure2ScatteringList) {
            if (scattering != "isotropic" && scattering != "anisotropic") {
                throw std::runtime_error("--figure2-scattering-list 只能包含 isotropic 或 anisotropic");
            }
        }

        Mesh mesh = Mesh::readCSV(cellsFile, facesFile);
        writeSourceShapeFile(sourceShape, figure5Dir);
        const std::string figurePrefix = sourcePrefix(sourceShape);
        std::cout << "入射区域形状: " << sourceShape << " (" << figurePrefix << ")\n";
        if (useGPU) {
            std::cout << "GPU模式: RSI/RSI-tail使用CUDA样本并行；SI使用CUDA角度并行；Figure 2保持CPU路径\n";
        }

        if (only == "sweep-stats") {
            for (int angularN : {4, 32}) {
                RSIConfig cfg;
                cfg.groupCount = 1;
                cfg.angularN = angularN;
                cfg.maxSIters = 1;
                cfg.scattering = "isotropic";
                cfg.sourceShape = sourceShape;
                std::cout << "Sweep stats for S" << angularN << ":\n";
                RSISolver solver(mesh, cfg);
                solver.printSweepPlanStats();
            }
            return 0;
        }

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

            const std::filesystem::path outPath(outFile);
            if (outPath.has_parent_path()) {
                std::filesystem::create_directories(outPath.parent_path());
            }
            std::ofstream fout(outFile);
            fout << "scattering,M,S,iterationN,e_RSI_N\n";
            for (const auto& r : allRows) {
                fout << r.scattering << ',' << r.M << ',' << r.S << ',' << r.iterationN << ',' << r.eRSI << '\n';
            }
            std::cout << "已输出: " << outFile << "\n";
        }

        if (only == "figure2-gpu-convergence") {
            const std::filesystem::path outPath(outFile);
            if (outPath.has_parent_path()) {
                std::filesystem::create_directories(outPath.parent_path());
            }
            std::ofstream fout(outFile);
            fout << "scattering,M,S,iterationN,e_RSI_N\n";
            fout.flush();

            for (const std::string& scattering : figure2ScatteringList) {
            for (int angularN : figure2AngularList) {
                RSIConfig cfg;
                cfg.groupCount = 1;
                cfg.angularN = angularN;
                cfg.maxSIters = 80;
                cfg.siTolerance = 1e-10;
                cfg.sampleCounts = figure2SamplesList;
                cfg.scattering = scattering;
                cfg.sourceShape = sourceShape;
                cfg.seed = 20260514u;
                cfg.useGPU = true;
                RSISolver solver(mesh, cfg);
                auto rows = solver.runFigure2GPUConvergence();
                for (const auto& r : rows) {
                    fout << r.scattering << ',' << r.M << ',' << r.S << ','
                         << r.iterationN << ',' << r.eRSI << '\n';
                }
                fout.flush();
            }
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

            int Ncoarse = 0;
            std::vector<double> phiSIcoarse;
            if (useGPU) {
                coarseCfg.useGPU = true;
                RSISolver coarseGpuSolver(mesh, coarseCfg);
                phiSIcoarse = coarseGpuSolver.runSIField(Ncoarse);
                std::cout << "GPU Figure 5 coarse SI converged at N=" << Ncoarse << "\n";
            } else {
                RSISolver coarseSolver(mesh, coarseCfg);
                phiSIcoarse = coarseSolver.runSIField(Ncoarse);
            }
            const std::string siCoarseFile = figure5OutputPath(figure5Dir, figurePrefix, "figure5_SI_coarse.csv");
            const std::string siFineFile = figure5OutputPath(figure5Dir, figurePrefix, "figure5_SI_fine.csv");
            const std::string rsiFile = figure5OutputPath(figure5Dir, figurePrefix, "figure5_RSI.csv");
            const std::string rsiTailFile = figure5OutputPath(figure5Dir, figurePrefix, "figure5_RSI_tail.csv");
            RSISolver::writeFieldCSV(siCoarseFile, mesh, phiSIcoarse);

            // 细角度SI
            RSIConfig fineCfg;
            fineCfg.groupCount = 1;
            fineCfg.angularN = figure5FineSN;
            fineCfg.maxSIters = 80;
            fineCfg.siTolerance = 1e-10;
            fineCfg.sampleCounts = {figure5Samples};
            fineCfg.scattering = "isotropic";
            fineCfg.sourceShape = sourceShape;
            fineCfg.seed = 20260513u;
            fineCfg.useGPU = useGPU;

            RSISolver fineSolver(mesh, fineCfg);
            int S = figure5Samples;
            int tailExtra = 10;
            if (useGPU) {
                const Figure5Fields fields = fineSolver.runFigure5GPU(S, tailExtra);
                RSISolver::writeFieldCSV(siFineFile, mesh, fields.siFine);
                RSISolver::writeFieldCSV(rsiFile, mesh, fields.rsi);
                RSISolver::writeFieldCSV(rsiTailFile, mesh, fields.rsiTail);
                std::cout << "GPU Figure 5 SI converged at N=" << fields.convergedN << "\n";
            } else {
                int Nfine = 0;
                auto phiSIfine = fineSolver.runSIField(Nfine);
                RSISolver::writeFieldCSV(siFineFile, mesh, phiSIfine);

                // RSI，使用512个样本
                auto phiRSI = fineSolver.runRSIFieldAtN(Nfine, S, 0);
                RSISolver::writeFieldCSV(rsiFile, mesh, phiRSI);

                // RSI + 尾部平均：平均 Nfine 到 Nfine+tailExtra。
                auto phiRSITail = fineSolver.runRSIFieldAtN(Nfine, S, tailExtra);
                RSISolver::writeFieldCSV(rsiTailFile, mesh, phiRSITail);
            }

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
