#include "CudaRSI.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream out;
    out << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(out.str());
}

template <typename T>
struct DeviceArray {
    T* ptr = nullptr;

    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this == &other) return *this;
        if (ptr) cudaFree(ptr);
        ptr = other.ptr;
        other.ptr = nullptr;
        return *this;
    }

    ~DeviceArray() {
        if (ptr) cudaFree(ptr);
    }

    void allocate(std::size_t count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
    }

    void copyFrom(const std::vector<T>& values) {
        allocate(values.size());
        checkCuda(
            cudaMemcpy(ptr, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy host to device"
        );
    }
};

enum BoundaryType : int {
    BC_INTERNAL = 0,
    BC_VACUUM = 1,
    BC_INFLOW = 2,
    BC_EXAMPLE1 = 3,
};

struct DeviceMeshView {
    int cellCount;
    const double* volume;
    const double* sigmaT;
    const double* sigmaS;
    const double* cellQ;
    const int* cellFaceOffsets;
    const int* refFace;
    const int* refNeighbor;
    const int* refSign;
    const double* faceNx;
    const double* faceNy;
    const double* faceNz;
    const double* faceArea;
    const double* faceFx;
    const double* faceFy;
    const double* faceFz;
    const int* faceBcType;
    const double* faceBcValue;
    const double* faceSourceFraction;
};

struct DeviceMeshStorage {
    DeviceArray<double> volume;
    DeviceArray<double> sigmaT;
    DeviceArray<double> sigmaS;
    DeviceArray<double> cellQ;
    DeviceArray<int> cellFaceOffsets;
    DeviceArray<int> refFace;
    DeviceArray<int> refNeighbor;
    DeviceArray<int> refSign;
    DeviceArray<double> faceNx;
    DeviceArray<double> faceNy;
    DeviceArray<double> faceNz;
    DeviceArray<double> faceArea;
    DeviceArray<double> faceFx;
    DeviceArray<double> faceFy;
    DeviceArray<double> faceFz;
    DeviceArray<int> faceBcType;
    DeviceArray<double> faceBcValue;
    DeviceArray<double> faceSourceFraction;

    DeviceMeshView view(int cellCount) const {
        return {
            cellCount,
            volume.ptr,
            sigmaT.ptr,
            sigmaS.ptr,
            cellQ.ptr,
            cellFaceOffsets.ptr,
            refFace.ptr,
            refNeighbor.ptr,
            refSign.ptr,
            faceNx.ptr,
            faceNy.ptr,
            faceNz.ptr,
            faceArea.ptr,
            faceFx.ptr,
            faceFy.ptr,
            faceFz.ptr,
            faceBcType.ptr,
            faceBcValue.ptr,
            faceSourceFraction.ptr,
        };
    }
};

int boundaryType(const std::string& type) {
    if (type == "internal") return BC_INTERNAL;
    if (type == "vacuum") return BC_VACUUM;
    if (type == "inflow") return BC_INFLOW;
    if (type == "example1") return BC_EXAMPLE1;
    return BC_VACUUM;
}

DeviceMeshStorage uploadMesh(const Mesh& mesh) {
    DeviceMeshStorage storage;
    const int C = static_cast<int>(mesh.cells.size());
    const int F = static_cast<int>(mesh.faces.size());

    std::vector<double> volume(C), sigmaT(C), sigmaS(C), cellQ(C);
    std::vector<int> offsets(C + 1, 0), refFace, refNeighbor, refSign;
    refFace.reserve(static_cast<std::size_t>(C) * 4);
    refNeighbor.reserve(static_cast<std::size_t>(C) * 4);
    refSign.reserve(static_cast<std::size_t>(C) * 4);

    for (int i = 0; i < C; ++i) {
        const Cell& cell = mesh.cells[i];
        volume[i] = cell.volume;
        sigmaT[i] = cell.sigma_t;
        sigmaS[i] = cell.sigma_s;
        cellQ[i] = cell.q;
        offsets[i] = static_cast<int>(refFace.size());
        for (const CellFaceRef& ref : cell.faceRefs) {
            refFace.push_back(ref.face);
            refNeighbor.push_back(ref.neighbor);
            refSign.push_back(ref.sign);
        }
    }
    offsets[C] = static_cast<int>(refFace.size());

    std::vector<double> nx(F), ny(F), nz(F), area(F), fx(F), fy(F), fz(F);
    std::vector<double> bcValue(F), sourceFraction(F);
    std::vector<int> bcType(F);
    for (int i = 0; i < F; ++i) {
        const Face& face = mesh.faces[i];
        nx[i] = face.normal.x;
        ny[i] = face.normal.y;
        nz[i] = face.normal.z;
        area[i] = face.area;
        fx[i] = face.center.x;
        fy[i] = face.center.y;
        fz[i] = face.center.z;
        bcType[i] = boundaryType(face.bc_type);
        bcValue[i] = face.bc_value;
        sourceFraction[i] = face.source_fraction;
    }

    storage.volume.copyFrom(volume);
    storage.sigmaT.copyFrom(sigmaT);
    storage.sigmaS.copyFrom(sigmaS);
    storage.cellQ.copyFrom(cellQ);
    storage.cellFaceOffsets.copyFrom(offsets);
    storage.refFace.copyFrom(refFace);
    storage.refNeighbor.copyFrom(refNeighbor);
    storage.refSign.copyFrom(refSign);
    storage.faceNx.copyFrom(nx);
    storage.faceNy.copyFrom(ny);
    storage.faceNz.copyFrom(nz);
    storage.faceArea.copyFrom(area);
    storage.faceFx.copyFrom(fx);
    storage.faceFy.copyFrom(fy);
    storage.faceFz.copyFrom(fz);
    storage.faceBcType.copyFrom(bcType);
    storage.faceBcValue.copyFrom(bcValue);
    storage.faceSourceFraction.copyFrom(sourceFraction);
    return storage;
}

__device__ double boundaryInflow(
    const DeviceMeshView& mesh,
    int face,
    double ox,
    double oy,
    double oz,
    int sourceShape
) {
    const int type = mesh.faceBcType[face];
    if (type == BC_VACUUM || type == BC_INTERNAL) return 0.0;
    if (type == BC_INFLOW) return mesh.faceBcValue[face];
    if (type != BC_EXAMPLE1) return 0.0;

    constexpr double centerX = 0.5;
    constexpr double centerZ = 0.5;
    constexpr double halfLengthX = 0.15;
    constexpr double halfWidthZ = 0.1;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double radius = 0.2 / sqrt(pi);

    if (fabs(mesh.faceFy[face]) >= 1.0e-10 || oy <= 0.0) return 0.0;

    double fraction = 0.0;
    if (sourceShape == 0) {
        fraction =
            fabs(mesh.faceFx[face] - centerX) <= halfLengthX &&
            fabs(mesh.faceFz[face] - centerZ) <= halfWidthZ
                ? 1.0
                : 0.0;
    } else {
        fraction = mesh.faceSourceFraction[face];
        if (fraction >= 1.0) {
            const double dx = mesh.faceFx[face] - centerX;
            const double dz = mesh.faceFz[face] - centerZ;
            fraction = dx * dx + dz * dz <= radius * radius ? 1.0 : 0.0;
        }
    }

    if (fraction <= 0.0) return 0.0;
    return fraction * 10.0 * exp(-(ox * ox + oy * oy + oz * oz));
}

__global__ void sweepSamplesKernel(
    DeviceMeshView mesh,
    const double* ordinateX,
    const double* ordinateY,
    const double* ordinateZ,
    const int* sweepOrders,
    const unsigned char* sweepHasCycle,
    const int* selectedDirection,
    int sampleCount,
    int sourceShape,
    const double* previousPsi,
    double* currentPsi,
    bool sourceShared,
    int localPass
) {
    constexpr int faceLanes = 4;
    constexpr int samplesPerBlock = 32;
    const int sample = blockIdx.x * samplesPerBlock + threadIdx.x / faceLanes;
    const int faceLane = threadIdx.x % faceLanes;
    if (sample >= sampleCount) return;

    const int C = mesh.cellCount;
    const int direction = selectedDirection[sample];
    const double ox = ordinateX[direction];
    const double oy = ordinateY[direction];
    const double oz = ordinateZ[direction];
    const int* order = sweepOrders + static_cast<std::size_t>(direction) * C;
    const bool hasCycle = sweepHasCycle[direction] != 0;
    if (localPass > 0 && !hasCycle) return;
    const int laneInWarp = threadIdx.x & 31;
    const int subgroupBase = laneInWarp & ~(faceLanes - 1);
    const unsigned mask = 0xFu << subgroupBase;

    for (int position = 0; position < C; ++position) {
        const int cell = order[position];
        const std::size_t cellIndex = static_cast<std::size_t>(sample) * C + cell;
        double inflow = 0.0;
        double outflow = 0.0;

        const int begin = mesh.cellFaceOffsets[cell];
        const int end = mesh.cellFaceOffsets[cell + 1];
        for (int refIndex = begin + faceLane; refIndex < end; refIndex += faceLanes) {
            const int face = mesh.refFace[refIndex];
            const int sign = mesh.refSign[refIndex];
            const double nx = sign > 0 ? mesh.faceNx[face] : -mesh.faceNx[face];
            const double ny = sign > 0 ? mesh.faceNy[face] : -mesh.faceNy[face];
            const double nz = sign > 0 ? mesh.faceNz[face] : -mesh.faceNz[face];
            const double mu = ox * nx + oy * ny + oz * nz;
            const double coefficient = fabs(mu) * mesh.faceArea[face];
            if (coefficient <= 1.0e-14) continue;

            if (mu > 0.0) {
                outflow += coefficient;
            } else {
                const int neighbor = mesh.refNeighbor[refIndex];
                const double psiIn = neighbor >= 0
                    ? currentPsi[static_cast<std::size_t>(sample) * C + neighbor]
                    : boundaryInflow(mesh, face, ox, oy, oz, sourceShape);
                inflow += coefficient * psiIn;
            }
        }

        for (int offset = 2; offset > 0; offset /= 2) {
            inflow += __shfl_down_sync(mask, inflow, offset, faceLanes);
            outflow += __shfl_down_sync(mask, outflow, offset, faceLanes);
        }

        if (faceLane == 0) {
            const double source = sourceShared ? previousPsi[cell] : previousPsi[cellIndex];
            const double rhs =
                mesh.volume[cell] * (mesh.sigmaS[cell] * source + mesh.cellQ[cell]) + inflow;
            const double diagonal = mesh.sigmaT[cell] * mesh.volume[cell] + outflow;
            currentPsi[cellIndex] = rhs / diagonal;
        }
        __syncwarp(mask);
    }
}

__global__ void reduceSamplesKernel(
    const double* sampleAccum,
    int sampleCount,
    int cellCount,
    double* globalSum
) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;
    double sum = 0.0;
    for (int sample = 0; sample < sampleCount; ++sample) {
        sum += sampleAccum[static_cast<std::size_t>(sample) * cellCount + cell];
    }
    globalSum[cell] += sum;
}

__global__ void accumulateValuesKernel(
    const double* values,
    std::size_t valueCount,
    double* sum
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < valueCount) sum[index] += values[index];
}

__global__ void reduceDirectionsKernel(
    const double* angularPsi,
    const double* weights,
    int directionCount,
    int cellCount,
    double* phi0
) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;
    double sum = 0.0;
    for (int direction = 0; direction < directionCount; ++direction) {
        sum += weights[direction] *
               angularPsi[static_cast<std::size_t>(direction) * cellCount + cell];
    }
    phi0[cell] = sum;
}

__global__ void relativeNormKernel(
    const double* current,
    const double* previous,
    const double* volume,
    int cellCount,
    double* numerator,
    double* denominator
) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;
    const double difference = current[cell] - previous[cell];
    atomicAdd(numerator, volume[cell] * difference * difference);
    atomicAdd(denominator, volume[cell] * previous[cell] * previous[cell]);
}

std::vector<int> generateDirectionSchedule(
    unsigned int seed,
    int directionCount,
    int sampleCount,
    int iterationCount
) {
    std::vector<double> probabilities(directionCount, 1.0 / static_cast<double>(directionCount));
    std::vector<int> schedule(static_cast<std::size_t>(sampleCount) * iterationCount);
    for (int sample = 0; sample < sampleCount; ++sample) {
        std::mt19937 rng(
            seed + 17u * static_cast<unsigned>(directionCount) +
            0x9e3779b9u * static_cast<unsigned>(sample + 1)
        );
        std::uniform_int_distribution<int> initialDistribution(0, directionCount - 1);
        std::discrete_distribution<int> iterationDistribution(
            probabilities.begin(), probabilities.end()
        );
        (void)initialDistribution(rng);
        for (int iteration = 0; iteration < iterationCount; ++iteration) {
            schedule[static_cast<std::size_t>(sample) * iterationCount + iteration] =
                iterationDistribution(rng);
        }
    }
    return schedule;
}

struct DeviceProblem {
    int cellCount = 0;
    int directionCount = 0;
    std::vector<unsigned char> hostHasCycle;
    DeviceMeshStorage mesh;
    DeviceArray<double> ordinateX;
    DeviceArray<double> ordinateY;
    DeviceArray<double> ordinateZ;
    DeviceArray<double> weights;
    DeviceArray<unsigned char> hasCycle;
    DeviceArray<int> orders;
    DeviceArray<int> allDirections;
};

int sameCycleRunEnd(
    const std::vector<unsigned char>& hasCycle,
    int begin,
    int end
) {
    const unsigned char value = hasCycle[begin];
    int current = begin + 1;
    while (current < end && hasCycle[current] == value) ++current;
    return current;
}

std::unique_ptr<DeviceProblem> uploadProblem(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const std::vector<SweepPlan>& sweepPlans
) {
    auto problem = std::make_unique<DeviceProblem>();
    problem->cellCount = static_cast<int>(mesh.cells.size());
    problem->directionCount = static_cast<int>(ordinates.size());
    const int C = problem->cellCount;
    const int M = problem->directionCount;
    if (static_cast<int>(sweepPlans.size()) != M) {
        throw std::runtime_error("CUDA sweep plan count mismatch");
    }

    problem->mesh = uploadMesh(mesh);
    std::vector<double> ox(M), oy(M), oz(M), weights(M);
    std::vector<unsigned char> hasCycle(M);
    std::vector<int> directions(M);
    for (int m = 0; m < M; ++m) {
        if (static_cast<int>(sweepPlans[m].order.size()) != C) {
            throw std::runtime_error("CUDA sweep order size mismatch");
        }
        ox[m] = ordinates[m].omega.x;
        oy[m] = ordinates[m].omega.y;
        oz[m] = ordinates[m].omega.z;
        weights[m] = ordinates[m].weight;
        hasCycle[m] = sweepPlans[m].hasCycle ? 1 : 0;
        directions[m] = m;
    }
    problem->ordinateX.copyFrom(ox);
    problem->ordinateY.copyFrom(oy);
    problem->ordinateZ.copyFrom(oz);
    problem->weights.copyFrom(weights);
    problem->hostHasCycle = hasCycle;
    problem->hasCycle.copyFrom(hasCycle);
    problem->allDirections.copyFrom(directions);
    problem->orders.allocate(static_cast<std::size_t>(M) * C);
    for (int m = 0; m < M; ++m) {
        checkCuda(
            cudaMemcpy(
                problem->orders.ptr + static_cast<std::size_t>(m) * C,
                sweepPlans[m].order.data(),
                static_cast<std::size_t>(C) * sizeof(int),
                cudaMemcpyHostToDevice
            ),
            "copy CUDA sweep orders"
        );
    }
    return problem;
}

constexpr std::uint64_t siCheckpointMagic = 0x5253495349435031ULL;
constexpr std::uint64_t rsiCheckpointMagic = 0x5253495253494350ULL;

struct SICheckpointHeader {
    std::uint64_t magic;
    int cellCount;
    int directionCount;
    int iteration;
    int converged;
};

struct RSICheckpointHeader {
    std::uint64_t magic;
    int cellCount;
    int directionCount;
    int sampleCount;
    int convergedN;
    int tailExtra;
    int nextSample;
};

bool loadSICheckpoint(
    const std::string& path,
    int cellCount,
    int directionCount,
    SICheckpointHeader& header,
    std::vector<double>& phi
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.magic != siCheckpointMagic ||
        header.cellCount != cellCount || header.directionCount != directionCount) {
        throw std::runtime_error("CUDA SI checkpoint does not match the current mesh/quadrature");
    }
    phi.resize(cellCount);
    input.read(reinterpret_cast<char*>(phi.data()),
               static_cast<std::streamsize>(phi.size() * sizeof(double)));
    if (!input) throw std::runtime_error("truncated CUDA SI checkpoint");
    return true;
}

void saveSICheckpoint(
    const std::string& path,
    int cellCount,
    int directionCount,
    int iteration,
    bool converged,
    const std::vector<double>& phi
) {
    const SICheckpointHeader header{
        siCheckpointMagic, cellCount, directionCount, iteration, converged ? 1 : 0
    };
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write CUDA SI checkpoint: " + path);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(phi.data()),
                 static_cast<std::streamsize>(phi.size() * sizeof(double)));
}

bool loadRSICheckpoint(
    const std::string& path,
    int cellCount,
    int directionCount,
    int sampleCount,
    int convergedN,
    int tailExtra,
    RSICheckpointHeader& header,
    std::vector<double>& rsiSum,
    std::vector<double>& tailSum
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.magic != rsiCheckpointMagic ||
        header.cellCount != cellCount || header.directionCount != directionCount ||
        header.sampleCount != sampleCount || header.convergedN != convergedN ||
        header.tailExtra != tailExtra) {
        throw std::runtime_error("CUDA RSI checkpoint does not match the current run");
    }
    rsiSum.resize(cellCount);
    tailSum.resize(cellCount);
    input.read(reinterpret_cast<char*>(rsiSum.data()),
               static_cast<std::streamsize>(rsiSum.size() * sizeof(double)));
    input.read(reinterpret_cast<char*>(tailSum.data()),
               static_cast<std::streamsize>(tailSum.size() * sizeof(double)));
    if (!input) throw std::runtime_error("truncated CUDA RSI checkpoint");
    return true;
}

void saveRSICheckpoint(
    const std::string& path,
    int cellCount,
    int directionCount,
    int sampleCount,
    int convergedN,
    int tailExtra,
    int nextSample,
    const std::vector<double>& rsiSum,
    const std::vector<double>& tailSum
) {
    const RSICheckpointHeader header{
        rsiCheckpointMagic, cellCount, directionCount, sampleCount,
        convergedN, tailExtra, nextSample
    };
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write CUDA RSI checkpoint: " + path);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(rsiSum.data()),
                 static_cast<std::streamsize>(rsiSum.size() * sizeof(double)));
    output.write(reinterpret_cast<const char*>(tailSum.data()),
                 static_cast<std::streamsize>(tailSum.size() * sizeof(double)));
}

} // namespace

bool cudaRSIAvailable(std::string* reason) {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        if (reason) *reason = cudaGetErrorString(status);
        cudaGetLastError();
        return false;
    }
    if (count == 0) {
        if (reason) *reason = "no CUDA device found";
        return false;
    }
    return true;
}

CudaFigure5Result runFigure5Cuda(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const std::vector<SweepPlan>& sweepPlans,
    const std::string& sourceShape,
    unsigned int seed,
    int maxSIters,
    double siTolerance,
    int sampleCount,
    int tailExtra
) {
    if (maxSIters <= 0 || sampleCount <= 0 || tailExtra < 0) {
        throw std::runtime_error("invalid CUDA Figure 5 iteration/sample configuration");
    }
    std::string unavailableReason;
    if (!cudaRSIAvailable(&unavailableReason)) {
        throw std::runtime_error("CUDA unavailable: " + unavailableReason);
    }

    const auto totalStart = std::chrono::steady_clock::now();
    std::unique_ptr<DeviceProblem> problem = uploadProblem(mesh, ordinates, sweepPlans);
    const int C = problem->cellCount;
    const int M = problem->directionCount;
    const int sourceShapeCode = sourceShape == "rectangle" ? 0 : 1;
    constexpr int sweepThreads = 128;
    constexpr int samplesPerSweepBlock = 32;
    constexpr int reductionThreads = 256;
    const int cellBlocks = (C + reductionThreads - 1) / reductionThreads;
    const char* checkpointEnvironment = std::getenv("RSI_CUDA_CHECKPOINT");
    const std::string checkpointPrefix =
        checkpointEnvironment ? checkpointEnvironment : "";
    const std::string siCheckpointPath = checkpointPrefix + ".si.bin";
    const std::string rsiCheckpointPath = checkpointPrefix + ".rsi.bin";

    CudaFigure5Result result;

    // SI fine: all angular directions are independent within one source iteration.
    {
        const std::size_t angularValueCount = static_cast<std::size_t>(M) * C;
        DeviceArray<double> angularPsi, phiA, phiB, normValues;
        angularPsi.allocate(angularValueCount);
        phiA.allocate(C);
        phiB.allocate(C);
        normValues.allocate(2);
        int firstIteration = 1;
        bool siAlreadyConverged = false;
        std::vector<double> checkpointPhi;
        SICheckpointHeader siHeader{};
        if (!checkpointPrefix.empty() &&
            loadSICheckpoint(siCheckpointPath, C, M, siHeader, checkpointPhi)) {
            checkCuda(
                cudaMemcpy(phiA.ptr, checkpointPhi.data(),
                           static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyHostToDevice),
                "restore CUDA SI checkpoint"
            );
            result.convergedN = siHeader.iteration;
            firstIteration = siHeader.iteration + 1;
            siAlreadyConverged = siHeader.converged != 0;
            std::cerr << "Resuming CUDA SI checkpoint at iteration="
                      << siHeader.iteration << ", converged=" << siHeader.converged << "\n";
        } else {
            checkCuda(cudaMemset(phiA.ptr, 0, static_cast<std::size_t>(C) * sizeof(double)),
                      "clear CUDA SI initial source");
        }

        double* previousPhi = phiA.ptr;
        double* currentPhi = phiB.ptr;
        for (int iteration = firstIteration;
             iteration <= maxSIters && !siAlreadyConverged; ++iteration) {
            checkCuda(cudaMemset(angularPsi.ptr, 0, angularValueCount * sizeof(double)),
                      "clear CUDA SI angular field");
            const int directionsPerBatch = std::max(4, std::min(128, 5000000 / C));
            for (int directionStart = 0; directionStart < M;) {
                const int batchEnd = std::min(directionStart + directionsPerBatch, M);
                const int runEnd = sameCycleRunEnd(
                    problem->hostHasCycle, directionStart, batchEnd
                );
                const int directionBatch = runEnd - directionStart;
                const int directionSweepBlocks =
                    (directionBatch + samplesPerSweepBlock - 1) /
                    samplesPerSweepBlock;
                const int localPassCount =
                    problem->hostHasCycle[directionStart] != 0 ? 20 : 1;
                for (int localPass = 0; localPass < localPassCount; ++localPass) {
                    sweepSamplesKernel<<<directionSweepBlocks, sweepThreads>>>(
                        problem->mesh.view(C),
                        problem->ordinateX.ptr,
                        problem->ordinateY.ptr,
                        problem->ordinateZ.ptr,
                        problem->orders.ptr,
                        problem->hasCycle.ptr,
                        problem->allDirections.ptr + directionStart,
                        directionBatch,
                        sourceShapeCode,
                        previousPhi,
                        angularPsi.ptr + static_cast<std::size_t>(directionStart) * C,
                        true,
                        localPass
                    );
                    checkCuda(cudaGetLastError(), "launch CUDA SI angular sweep batch");
                }
                directionStart = runEnd;
            }
            reduceDirectionsKernel<<<cellBlocks, reductionThreads>>>(
                angularPsi.ptr, problem->weights.ptr, M, C, currentPhi
            );
            checkCuda(cudaGetLastError(), "launch CUDA SI angular reduction");

            bool converged = false;
            double relative = 0.0;
            if (iteration > 1) {
                checkCuda(cudaMemset(normValues.ptr, 0, 2 * sizeof(double)),
                          "clear CUDA SI norm values");
                relativeNormKernel<<<cellBlocks, reductionThreads>>>(
                    currentPhi, previousPhi, problem->mesh.volume.ptr, C,
                    normValues.ptr, normValues.ptr + 1
                );
                checkCuda(cudaGetLastError(), "launch CUDA SI convergence norm");
                double hostNorms[2] = {0.0, 0.0};
                checkCuda(
                    cudaMemcpy(hostNorms, normValues.ptr, 2 * sizeof(double),
                               cudaMemcpyDeviceToHost),
                    "copy CUDA SI convergence norm"
                );
                relative = std::sqrt(
                    hostNorms[0] / std::max(hostNorms[1], 1.0e-300)
                );
                converged = relative < siTolerance;
            } else {
                checkCuda(cudaDeviceSynchronize(), "synchronize first CUDA SI iteration");
            }
            std::cerr << "CUDA SI iteration=" << iteration
                      << ", relative=" << relative << "\n";
            std::swap(previousPhi, currentPhi);
            result.convergedN = iteration;
            if (!checkpointPrefix.empty()) {
                checkpointPhi.resize(C);
                checkCuda(
                    cudaMemcpy(checkpointPhi.data(), previousPhi,
                               static_cast<std::size_t>(C) * sizeof(double),
                               cudaMemcpyDeviceToHost),
                    "copy CUDA SI checkpoint field"
                );
                saveSICheckpoint(
                    siCheckpointPath, C, M, iteration, converged, checkpointPhi
                );
            }
            if (converged) break;
        }

        result.siFine.resize(C);
        checkCuda(
            cudaMemcpy(result.siFine.data(), previousPhi,
                       static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
            "copy CUDA SI fine result"
        );
    }

    // RSI and RSI-tail share the same random chains and all iterations up to N+T.
    const int iterationCount = result.convergedN + tailExtra;
    const std::vector<int> schedule =
        generateDirectionSchedule(seed, M, sampleCount, iterationCount);
    DeviceArray<double> globalRSISum, globalTailSum;
    globalRSISum.allocate(C);
    globalTailSum.allocate(C);
    int firstSample = 0;
    std::vector<double> checkpointRSISum, checkpointTailSum;
    RSICheckpointHeader rsiHeader{};
    if (!checkpointPrefix.empty() &&
        loadRSICheckpoint(
            rsiCheckpointPath, C, M, sampleCount, result.convergedN, tailExtra,
            rsiHeader, checkpointRSISum, checkpointTailSum
        )) {
        firstSample = rsiHeader.nextSample;
        checkCuda(cudaMemcpy(globalRSISum.ptr, checkpointRSISum.data(),
                             static_cast<std::size_t>(C) * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "restore CUDA RSI sum");
        checkCuda(cudaMemcpy(globalTailSum.ptr, checkpointTailSum.data(),
                             static_cast<std::size_t>(C) * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "restore CUDA RSI-tail sum");
        std::cerr << "Resuming CUDA RSI checkpoint at sample=" << firstSample << "\n";
    } else {
        checkCuda(cudaMemset(globalRSISum.ptr, 0, static_cast<std::size_t>(C) * sizeof(double)),
                  "clear CUDA RSI sum");
        checkCuda(cudaMemset(globalTailSum.ptr, 0, static_cast<std::size_t>(C) * sizeof(double)),
                  "clear CUDA RSI-tail sum");
    }

    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");
    const std::size_t bytesPerSample =
        static_cast<std::size_t>(3) * C * sizeof(double) +
        static_cast<std::size_t>(iterationCount) * sizeof(int);
    constexpr int maxSamplesPerBatch = 128;
    const int batchCapacity = static_cast<int>(std::min<std::size_t>(
        std::min(sampleCount, maxSamplesPerBatch),
        std::max<std::size_t>(1, static_cast<std::size_t>(freeBytes * 0.75) /
                                std::max<std::size_t>(bytesPerSample, 1))
    ));

    const std::size_t workspaceCellCount = static_cast<std::size_t>(batchCapacity) * C;
    DeviceArray<double> psiA, psiB, sampleTail;
    DeviceArray<int> selectedDirections;
    psiA.allocate(workspaceCellCount);
    psiB.allocate(workspaceCellCount);
    sampleTail.allocate(workspaceCellCount);
    selectedDirections.allocate(static_cast<std::size_t>(batchCapacity) * iterationCount);
    std::vector<int> batchDirections(static_cast<std::size_t>(batchCapacity) * iterationCount);

    for (int batchStart = firstSample; batchStart < sampleCount;
         batchStart += batchCapacity) {
        const int batchSize = std::min(batchCapacity, sampleCount - batchStart);
        const std::size_t batchCellCount = static_cast<std::size_t>(batchSize) * C;
        checkCuda(cudaMemset(psiA.ptr, 0, batchCellCount * sizeof(double)),
                  "clear CUDA RSI previous layer");
        checkCuda(cudaMemset(sampleTail.ptr, 0, batchCellCount * sizeof(double)),
                  "clear CUDA RSI-tail batch accumulation");

        for (int iteration = 0; iteration < iterationCount; ++iteration) {
            for (int localSample = 0; localSample < batchSize; ++localSample) {
                batchDirections[static_cast<std::size_t>(iteration) * batchSize + localSample] =
                    schedule[static_cast<std::size_t>(batchStart + localSample) * iterationCount +
                             iteration];
            }
        }
        checkCuda(
            cudaMemcpy(selectedDirections.ptr, batchDirections.data(),
                       static_cast<std::size_t>(batchSize) * iterationCount * sizeof(int),
                       cudaMemcpyHostToDevice),
            "copy CUDA RSI direction schedule"
        );

        double* previousPsi = psiA.ptr;
        double* currentPsi = psiB.ptr;
        const int sampleSweepBlocks =
            (batchSize + samplesPerSweepBlock - 1) / samplesPerSweepBlock;
        for (int iteration = 1; iteration <= iterationCount; ++iteration) {
            checkCuda(cudaMemset(currentPsi, 0, batchCellCount * sizeof(double)),
                      "clear CUDA RSI current layer");
            for (int localPass = 0; localPass < 20; ++localPass) {
                sweepSamplesKernel<<<sampleSweepBlocks, sweepThreads>>>(
                    problem->mesh.view(C),
                    problem->ordinateX.ptr,
                    problem->ordinateY.ptr,
                    problem->ordinateZ.ptr,
                    problem->orders.ptr,
                    problem->hasCycle.ptr,
                    selectedDirections.ptr +
                        static_cast<std::size_t>(iteration - 1) * batchSize,
                    batchSize,
                    sourceShapeCode,
                    previousPsi,
                    currentPsi,
                    false,
                    localPass
                );
                checkCuda(cudaGetLastError(), "launch combined CUDA RSI sweep");
            }
            if (iteration == result.convergedN) {
                reduceSamplesKernel<<<cellBlocks, reductionThreads>>>(
                    currentPsi, batchSize, C, globalRSISum.ptr
                );
                checkCuda(cudaGetLastError(), "reduce CUDA RSI field");
            }
            if (iteration >= result.convergedN) {
                const int accumulationBlocks = static_cast<int>(
                    (batchCellCount + reductionThreads - 1) / reductionThreads
                );
                accumulateValuesKernel<<<accumulationBlocks, reductionThreads>>>(
                    currentPsi, batchCellCount, sampleTail.ptr
                );
                checkCuda(cudaGetLastError(), "accumulate CUDA RSI-tail batch field");
            }
            std::swap(previousPsi, currentPsi);
        }
        reduceSamplesKernel<<<cellBlocks, reductionThreads>>>(
            sampleTail.ptr, batchSize, C, globalTailSum.ptr
        );
        checkCuda(cudaGetLastError(), "reduce CUDA RSI-tail field");
        checkCuda(cudaDeviceSynchronize(), "synchronize combined CUDA Figure 5 batch");
        if (!checkpointPrefix.empty()) {
            checkpointRSISum.resize(C);
            checkpointTailSum.resize(C);
            checkCuda(cudaMemcpy(checkpointRSISum.data(), globalRSISum.ptr,
                                 static_cast<std::size_t>(C) * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                      "copy CUDA RSI checkpoint sum");
            checkCuda(cudaMemcpy(checkpointTailSum.data(), globalTailSum.ptr,
                                 static_cast<std::size_t>(C) * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                      "copy CUDA RSI-tail checkpoint sum");
            saveRSICheckpoint(
                rsiCheckpointPath, C, M, sampleCount, result.convergedN, tailExtra,
                batchStart + batchSize, checkpointRSISum, checkpointTailSum
            );
        }
        std::cerr << "CUDA RSI combined batch complete: samples="
                  << batchStart + batchSize << "/" << sampleCount << "\n";
    }

    result.rsi.resize(C);
    result.rsiTail.resize(C);
    checkCuda(cudaMemcpy(result.rsi.data(), globalRSISum.ptr,
                         static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
              "copy combined CUDA RSI result");
    checkCuda(cudaMemcpy(result.rsiTail.data(), globalTailSum.ptr,
                         static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
              "copy combined CUDA RSI-tail result");
    const double rsiDenominator = static_cast<double>(sampleCount);
    const double tailDenominator =
        static_cast<double>(sampleCount) * static_cast<double>(tailExtra + 1);
    for (double& value : result.rsi) value /= rsiDenominator;
    for (double& value : result.rsiTail) value /= tailDenominator;

    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - totalStart
    ).count();
    std::cout << "CUDA Figure 5 complete: cells=" << C << ", directions=" << M
              << ", SI_iterations=" << result.convergedN
              << ", samples=" << sampleCount << ", seconds=" << seconds << "\n";
    return result;
}

std::vector<double> runRSIFieldAtNCuda(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const std::vector<SweepPlan>& sweepPlans,
    const std::string& sourceShape,
    unsigned int seed,
    int N,
    int sampleCount,
    int tailExtra
) {
    const auto totalStart = std::chrono::steady_clock::now();
    if (N <= 0 || sampleCount <= 0 || tailExtra < 0) {
        throw std::runtime_error("CUDA RSI requires N > 0, sampleCount > 0, tailExtra >= 0");
    }
    std::string unavailableReason;
    if (!cudaRSIAvailable(&unavailableReason)) {
        throw std::runtime_error("CUDA RSI unavailable: " + unavailableReason);
    }

    const int C = static_cast<int>(mesh.cells.size());
    const int M = static_cast<int>(ordinates.size());
    const int iterationCount = N + tailExtra;
    if (static_cast<int>(sweepPlans.size()) != M) {
        throw std::runtime_error("CUDA RSI sweep plan count mismatch");
    }

    DeviceMeshStorage deviceMesh = uploadMesh(mesh);
    std::vector<double> ox(M), oy(M), oz(M);
    std::vector<unsigned char> hasCycle(M);
    for (int m = 0; m < M; ++m) {
        if (static_cast<int>(sweepPlans[m].order.size()) != C) {
            throw std::runtime_error("CUDA RSI sweep order size mismatch");
        }
        ox[m] = ordinates[m].omega.x;
        oy[m] = ordinates[m].omega.y;
        oz[m] = ordinates[m].omega.z;
        hasCycle[m] = sweepPlans[m].hasCycle ? 1 : 0;
    }

    DeviceArray<double> deviceOx, deviceOy, deviceOz;
    DeviceArray<unsigned char> deviceHasCycle;
    deviceOx.copyFrom(ox);
    deviceOy.copyFrom(oy);
    deviceOz.copyFrom(oz);
    deviceHasCycle.copyFrom(hasCycle);

    DeviceArray<int> deviceOrders;
    deviceOrders.allocate(static_cast<std::size_t>(M) * C);
    for (int m = 0; m < M; ++m) {
        checkCuda(
            cudaMemcpy(
                deviceOrders.ptr + static_cast<std::size_t>(m) * C,
                sweepPlans[m].order.data(),
                static_cast<std::size_t>(C) * sizeof(int),
                cudaMemcpyHostToDevice
            ),
            "copy CUDA RSI sweep orders"
        );
    }

    const std::vector<int> schedule =
        generateDirectionSchedule(seed, M, sampleCount, iterationCount);
    const int cyclicSelections = static_cast<int>(std::count_if(
        schedule.begin(), schedule.end(),
        [&](int direction) { return hasCycle[direction] != 0; }
    ));
    DeviceArray<double> globalSum;
    globalSum.allocate(C);
    checkCuda(cudaMemset(globalSum.ptr, 0, static_cast<std::size_t>(C) * sizeof(double)),
              "clear CUDA RSI global sum");

    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");
    const std::size_t bytesPerSample =
        static_cast<std::size_t>(3) * C * sizeof(double) +
        static_cast<std::size_t>(iterationCount) * sizeof(int);
    const std::size_t usableBytes = static_cast<std::size_t>(freeBytes * 0.75);
    constexpr int maxSamplesPerBatch = 128;
    const int batchCapacity = static_cast<int>(std::min<std::size_t>(
        std::min(sampleCount, maxSamplesPerBatch),
        std::max<std::size_t>(1, usableBytes / std::max<std::size_t>(bytesPerSample, 1))
    ));
    const int sourceShapeCode = sourceShape == "rectangle" ? 0 : 1;

    const auto setupEnd = std::chrono::steady_clock::now();

    std::cout << "CUDA RSI sample parallel: samples=" << sampleCount
              << ", batch_capacity=" << batchCapacity
              << ", cells=" << C << ", directions=" << M
              << ", cyclic_selections=" << cyclicSelections << "\n";

    const std::size_t workspaceCellCount = static_cast<std::size_t>(batchCapacity) * C;
    DeviceArray<double> psiA, psiB, sampleAccum;
    DeviceArray<int> deviceDirections;
    psiA.allocate(workspaceCellCount);
    psiB.allocate(workspaceCellCount);
    sampleAccum.allocate(workspaceCellCount);
    deviceDirections.allocate(static_cast<std::size_t>(batchCapacity) * iterationCount);
    std::vector<int> batchDirections(static_cast<std::size_t>(batchCapacity) * iterationCount);

    for (int batchStart = 0; batchStart < sampleCount; batchStart += batchCapacity) {
        const int batchSize = std::min(batchCapacity, sampleCount - batchStart);
        std::cerr << "CUDA RSI batch: start=" << batchStart
                  << ", size=" << batchSize << "\n";
        const std::size_t batchCellCount = static_cast<std::size_t>(batchSize) * C;
        checkCuda(cudaMemset(psiA.ptr, 0, batchCellCount * sizeof(double)), "clear CUDA RSI psi A");
        checkCuda(cudaMemset(psiB.ptr, 0, batchCellCount * sizeof(double)), "clear CUDA RSI psi B");
        checkCuda(cudaMemset(sampleAccum.ptr, 0, batchCellCount * sizeof(double)),
                  "clear CUDA RSI batch accumulation");

        for (int iteration = 0; iteration < iterationCount; ++iteration) {
            for (int localSample = 0; localSample < batchSize; ++localSample) {
                const int globalSample = batchStart + localSample;
                batchDirections[static_cast<std::size_t>(iteration) * batchSize + localSample] =
                    schedule[static_cast<std::size_t>(globalSample) * iterationCount + iteration];
            }
        }
        checkCuda(
            cudaMemcpy(
                deviceDirections.ptr,
                batchDirections.data(),
                static_cast<std::size_t>(batchSize) * iterationCount * sizeof(int),
                cudaMemcpyHostToDevice
            ),
            "copy CUDA RSI direction schedule"
        );

        double* previous = psiA.ptr;
        double* current = psiB.ptr;
        for (int iteration = 0; iteration < iterationCount; ++iteration) {
            checkCuda(cudaMemset(current, 0, batchCellCount * sizeof(double)),
                      "clear CUDA RSI current layer");
            for (int localPass = 0; localPass < 20; ++localPass) {
                constexpr int sweepThreads = 128;
                constexpr int samplesPerSweepBlock = 32;
                const int sweepBlocks =
                    (batchSize + samplesPerSweepBlock - 1) / samplesPerSweepBlock;
                sweepSamplesKernel<<<sweepBlocks, sweepThreads>>>(
                    deviceMesh.view(C),
                    deviceOx.ptr,
                    deviceOy.ptr,
                    deviceOz.ptr,
                    deviceOrders.ptr,
                    deviceHasCycle.ptr,
                    deviceDirections.ptr + static_cast<std::size_t>(iteration) * batchSize,
                    batchSize,
                    sourceShapeCode,
                    previous,
                    current,
                    false,
                    localPass
                );
                checkCuda(cudaGetLastError(), "launch CUDA RSI sample sweep");
            }
            if (iteration + 1 >= N) {
                constexpr int accumulationThreads = 256;
                const int accumulationBlocks = static_cast<int>(
                    (batchCellCount + accumulationThreads - 1) / accumulationThreads
                );
                accumulateValuesKernel<<<accumulationBlocks, accumulationThreads>>>(
                    current, batchCellCount, sampleAccum.ptr
                );
                checkCuda(cudaGetLastError(), "launch CUDA RSI tail accumulation");
            }
            std::swap(previous, current);
        }
        constexpr int threads = 256;
        const int blocks = (C + threads - 1) / threads;
        reduceSamplesKernel<<<blocks, threads>>>(
            sampleAccum.ptr, batchSize, C, globalSum.ptr
        );
        checkCuda(cudaGetLastError(), "launch CUDA RSI sample reduction");
        checkCuda(cudaDeviceSynchronize(), "synchronize CUDA RSI batch");
        std::cerr << "CUDA RSI batch complete: start=" << batchStart << "\n";
    }

    const auto sweepEnd = std::chrono::steady_clock::now();

    std::vector<double> result(C);
    checkCuda(
        cudaMemcpy(result.data(), globalSum.ptr, static_cast<std::size_t>(C) * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "copy CUDA RSI result"
    );
    const double denominator = static_cast<double>(sampleCount) * (tailExtra + 1);
    for (double& value : result) value /= denominator;
    const double setupSeconds = std::chrono::duration<double>(setupEnd - totalStart).count();
    const double sweepSeconds = std::chrono::duration<double>(sweepEnd - setupEnd).count();
    const double totalSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - totalStart
    ).count();
    std::cout << "CUDA RSI timing: setup=" << setupSeconds
              << " s, sweeps=" << sweepSeconds
              << " s, total=" << totalSeconds << " s\n";
    return result;
}
