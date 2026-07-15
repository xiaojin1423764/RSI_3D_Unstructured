#include "CudaRSI.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream out;
    out << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(out.str());
}

double secondsBetween(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end
) {
    return std::chrono::duration<double>(end - begin).count();
}

struct CudaEventTimer {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;

    CudaEventTimer() {
        checkCuda(cudaEventCreate(&begin), "create CUDA timing begin event");
        checkCuda(cudaEventCreate(&end), "create CUDA timing end event");
    }
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    ~CudaEventTimer() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }

    void start() {
        checkCuda(cudaEventRecord(begin), "record CUDA timing begin event");
    }

    double stop(const char* operation) {
        checkCuda(cudaEventRecord(end), operation);
        checkCuda(cudaEventSynchronize(end), operation);
        float milliseconds = 0.0f;
        checkCuda(cudaEventElapsedTime(&milliseconds, begin, end), operation);
        return static_cast<double>(milliseconds) * 1.0e-3;
    }
};

struct CudaStream {
    cudaStream_t stream = nullptr;

    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking) {
        checkCuda(cudaStreamCreateWithFlags(&stream, flags), "create CUDA stream");
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    ~CudaStream() {
        if (stream) cudaStreamDestroy(stream);
    }
};

template <typename T>
struct PinnedHostBuffer {
    T* ptr = nullptr;
    std::size_t capacity = 0;

    PinnedHostBuffer() = default;
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    ~PinnedHostBuffer() {
        if (ptr) cudaFreeHost(ptr);
    }

    void allocate(std::size_t count) {
        if (count == 0) return;
        if (ptr && capacity >= count) return;
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
            capacity = 0;
        }
        checkCuda(
            cudaMallocHost(reinterpret_cast<void**>(&ptr), count * sizeof(T)),
            "cudaMallocHost"
        );
        capacity = count;
    }
};

template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    std::size_t capacity = 0;

    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept : ptr(other.ptr), capacity(other.capacity) {
        other.ptr = nullptr;
        other.capacity = 0;
    }
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this == &other) return *this;
        if (ptr) cudaFree(ptr);
        ptr = other.ptr;
        capacity = other.capacity;
        other.ptr = nullptr;
        other.capacity = 0;
        return *this;
    }

    ~DeviceArray() {
        if (ptr) cudaFree(ptr);
    }

    void allocate(std::size_t count) {
        if (count == 0) return;
        if (ptr && capacity >= count) return;
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            capacity = 0;
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
        capacity = count;
    }

    void copyFrom(const std::vector<T>& values) {
        if (values.empty()) return;
        allocate(values.size());
        checkCuda(
            cudaMemcpy(ptr, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy host to device"
        );
    }

    void copyFromAsync(const std::vector<T>& values, cudaStream_t stream) {
        if (values.empty()) return;
        allocate(values.size());
        checkCuda(
            cudaMemcpyAsync(
                ptr, values.data(), values.size() * sizeof(T),
                cudaMemcpyHostToDevice, stream
            ),
            "cudaMemcpyAsync host to device"
        );
    }

    void copyFromPinnedAsync(
        const std::vector<T>& values,
        PinnedHostBuffer<T>& staging,
        cudaStream_t stream
    ) {
        if (values.empty()) return;
        allocate(values.size());
        staging.allocate(values.size());
        std::copy(values.begin(), values.end(), staging.ptr);
        checkCuda(
            cudaMemcpyAsync(
                ptr, staging.ptr, values.size() * sizeof(T),
                cudaMemcpyHostToDevice, stream
            ),
            "cudaMemcpyAsync pinned host to device"
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
    const double* refNx;
    const double* refNy;
    const double* refNz;
    const double* refArea;
    const double* refFx;
    const double* refFy;
    const double* refFz;
    const int* refBcType;
    const double* refBcValue;
    const double* refSourceFraction;
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
    DeviceArray<double> refNx;
    DeviceArray<double> refNy;
    DeviceArray<double> refNz;
    DeviceArray<double> refArea;
    DeviceArray<double> refFx;
    DeviceArray<double> refFy;
    DeviceArray<double> refFz;
    DeviceArray<int> refBcType;
    DeviceArray<double> refBcValue;
    DeviceArray<double> refSourceFraction;

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
            refNx.ptr,
            refNy.ptr,
            refNz.ptr,
            refArea.ptr,
            refFx.ptr,
            refFy.ptr,
            refFz.ptr,
            refBcType.ptr,
            refBcValue.ptr,
            refSourceFraction.ptr,
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

    std::vector<double> volume(C), sigmaT(C), sigmaS(C), cellQ(C);
    std::vector<int> offsets(C + 1, 0), refFace, refNeighbor, refSign;
    std::vector<double> refNx, refNy, refNz, refArea, refFx, refFy, refFz;
    std::vector<double> refBcValue, refSourceFraction;
    std::vector<int> refBcType;
    refFace.reserve(static_cast<std::size_t>(C) * 4);
    refNeighbor.reserve(static_cast<std::size_t>(C) * 4);
    refSign.reserve(static_cast<std::size_t>(C) * 4);
    refNx.reserve(static_cast<std::size_t>(C) * 4);
    refNy.reserve(static_cast<std::size_t>(C) * 4);
    refNz.reserve(static_cast<std::size_t>(C) * 4);
    refArea.reserve(static_cast<std::size_t>(C) * 4);
    refFx.reserve(static_cast<std::size_t>(C) * 4);
    refFy.reserve(static_cast<std::size_t>(C) * 4);
    refFz.reserve(static_cast<std::size_t>(C) * 4);
    refBcType.reserve(static_cast<std::size_t>(C) * 4);
    refBcValue.reserve(static_cast<std::size_t>(C) * 4);
    refSourceFraction.reserve(static_cast<std::size_t>(C) * 4);

    for (int i = 0; i < C; ++i) {
        const Cell& cell = mesh.cells[i];
        volume[i] = cell.volume;
        sigmaT[i] = cell.sigma_t;
        sigmaS[i] = cell.sigma_s;
        cellQ[i] = cell.q;
        offsets[i] = static_cast<int>(refFace.size());
        for (const CellFaceRef& ref : cell.faceRefs) {
            const Face& face = mesh.faces[ref.face];
            refFace.push_back(ref.face);
            refNeighbor.push_back(ref.neighbor);
            refSign.push_back(ref.sign);
            refNx.push_back(ref.sign > 0 ? face.normal.x : -face.normal.x);
            refNy.push_back(ref.sign > 0 ? face.normal.y : -face.normal.y);
            refNz.push_back(ref.sign > 0 ? face.normal.z : -face.normal.z);
            refArea.push_back(face.area);
            refFx.push_back(face.center.x);
            refFy.push_back(face.center.y);
            refFz.push_back(face.center.z);
            refBcType.push_back(boundaryType(face.bc_type));
            refBcValue.push_back(face.bc_value);
            refSourceFraction.push_back(face.source_fraction);
        }
    }
    offsets[C] = static_cast<int>(refFace.size());

    storage.volume.copyFrom(volume);
    storage.sigmaT.copyFrom(sigmaT);
    storage.sigmaS.copyFrom(sigmaS);
    storage.cellQ.copyFrom(cellQ);
    storage.cellFaceOffsets.copyFrom(offsets);
    storage.refFace.copyFrom(refFace);
    storage.refNeighbor.copyFrom(refNeighbor);
    storage.refSign.copyFrom(refSign);
    storage.refNx.copyFrom(refNx);
    storage.refNy.copyFrom(refNy);
    storage.refNz.copyFrom(refNz);
    storage.refArea.copyFrom(refArea);
    storage.refFx.copyFrom(refFx);
    storage.refFy.copyFrom(refFy);
    storage.refFz.copyFrom(refFz);
    storage.refBcType.copyFrom(refBcType);
    storage.refBcValue.copyFrom(refBcValue);
    storage.refSourceFraction.copyFrom(refSourceFraction);
    return storage;
}

__device__ double boundaryInflowRef(
    const DeviceMeshView& mesh,
    int refIndex,
    double ox,
    double oy,
    double oz,
    int sourceShape
) {
    const int type = mesh.refBcType[refIndex];
    if (type == BC_VACUUM || type == BC_INTERNAL) return 0.0;
    if (type == BC_INFLOW) return mesh.refBcValue[refIndex];
    if (type != BC_EXAMPLE1) return 0.0;

    constexpr double centerX = 0.5;
    constexpr double centerZ = 0.5;
    constexpr double halfLengthX = 0.15;
    constexpr double halfWidthZ = 0.1;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double radius = 0.2 / sqrt(pi);

    if (fabs(mesh.refFy[refIndex]) >= 1.0e-10 || oy <= 0.0) return 0.0;

    double fraction = 0.0;
    if (sourceShape == 0) {
        fraction =
            fabs(mesh.refFx[refIndex] - centerX) <= halfLengthX &&
            fabs(mesh.refFz[refIndex] - centerZ) <= halfWidthZ
                ? 1.0
                : 0.0;
    } else {
        fraction = mesh.refSourceFraction[refIndex];
        if (fraction >= 1.0) {
            const double dx = mesh.refFx[refIndex] - centerX;
            const double dz = mesh.refFz[refIndex] - centerZ;
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
            const double nx = mesh.refNx[refIndex];
            const double ny = mesh.refNy[refIndex];
            const double nz = mesh.refNz[refIndex];
            const double mu = ox * nx + oy * ny + oz * nz;
            const double coefficient = fabs(mu) * mesh.refArea[refIndex];
            if (coefficient <= 1.0e-14) continue;

            if (mu > 0.0) {
                outflow += coefficient;
            } else {
                const int neighbor = mesh.refNeighbor[refIndex];
                const double psiIn = neighbor >= 0
                    ? currentPsi[static_cast<std::size_t>(sample) * C + neighbor]
                    : boundaryInflowRef(mesh, refIndex, ox, oy, oz, sourceShape);
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

__global__ void sweepLevelKernel(
    DeviceMeshView mesh,
    const double* ordinateX,
    const double* ordinateY,
    const double* ordinateZ,
    const int* sweepOrders,
    const int* sweepLevelOffsetBase,
    const int* sweepLevelCount,
    const int* sweepLevelOffsets,
    const int* selectedDirection,
    int directionBatch,
    int sourceShape,
    const double* previousPsi,
    double* currentPsi,
    bool sourceShared,
    int level
) {
    const int localDirection = blockIdx.x;
    if (localDirection >= directionBatch) return;

    const int C = mesh.cellCount;
    const int direction = selectedDirection[localDirection];
    const int levelCount = sweepLevelCount[direction];
    if (level >= levelCount) return;

    const double ox = ordinateX[direction];
    const double oy = ordinateY[direction];
    const double oz = ordinateZ[direction];
    const int* order = sweepOrders + static_cast<std::size_t>(direction) * C;
    const int levelBase = sweepLevelOffsetBase[direction];
    const int begin = sweepLevelOffsets[levelBase + level];
    const int end = sweepLevelOffsets[levelBase + level + 1];

    for (int index = begin + threadIdx.x; index < end; index += blockDim.x) {
        const int cell = order[index];
        const std::size_t outputIndex = static_cast<std::size_t>(localDirection);
        const std::size_t cellIndex = outputIndex * C + cell;
        double inflow = 0.0;
        double outflow = 0.0;

        const int faceBegin = mesh.cellFaceOffsets[cell];
        const int faceEnd = mesh.cellFaceOffsets[cell + 1];
        for (int refIndex = faceBegin; refIndex < faceEnd; ++refIndex) {
            const double nx = mesh.refNx[refIndex];
            const double ny = mesh.refNy[refIndex];
            const double nz = mesh.refNz[refIndex];
            const double mu = ox * nx + oy * ny + oz * nz;
            const double coefficient = fabs(mu) * mesh.refArea[refIndex];
            if (coefficient <= 1.0e-14) continue;

            if (mu > 0.0) {
                outflow += coefficient;
            } else {
                const int neighbor = mesh.refNeighbor[refIndex];
                const double psiIn = neighbor >= 0
                    ? currentPsi[outputIndex * C + neighbor]
                    : boundaryInflowRef(mesh, refIndex, ox, oy, oz, sourceShape);
                inflow += coefficient * psiIn;
            }
        }

        const double source = sourceShared ? previousPsi[cell] : previousPsi[cellIndex];
        const double rhs =
            mesh.volume[cell] * (mesh.sigmaS[cell] * source + mesh.cellQ[cell]) + inflow;
        const double diagonal = mesh.sigmaT[cell] * mesh.volume[cell] + outflow;
        currentPsi[cellIndex] = rhs / diagonal;
    }
}

__global__ void sweepLevelTiledKernel(
    DeviceMeshView mesh,
    const double* ordinateX,
    const double* ordinateY,
    const double* ordinateZ,
    const int* sweepOrders,
    const int* sweepLevelOffsetBase,
    const int* sweepLevelCount,
    const int* sweepLevelOffsets,
    const int* selectedDirection,
    int directionBatch,
    int sourceShape,
    const double* previousPsi,
    double* currentPsi,
    bool sourceShared,
    int level
) {
    const int localDirection = blockIdx.x;
    if (localDirection >= directionBatch) return;

    const int C = mesh.cellCount;
    const int direction = selectedDirection[localDirection];
    const int levelCount = sweepLevelCount[direction];
    if (level >= levelCount) return;

    const double ox = ordinateX[direction];
    const double oy = ordinateY[direction];
    const double oz = ordinateZ[direction];
    const int* order = sweepOrders + static_cast<std::size_t>(direction) * C;
    const int levelBase = sweepLevelOffsetBase[direction];
    const int begin = sweepLevelOffsets[levelBase + level];
    const int end = sweepLevelOffsets[levelBase + level + 1];
    const int index = begin + static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (index >= end) return;

    const int cell = order[index];
    const std::size_t outputIndex = static_cast<std::size_t>(localDirection);
    const std::size_t cellIndex = outputIndex * C + cell;
    double inflow = 0.0;
    double outflow = 0.0;

    const int faceBegin = mesh.cellFaceOffsets[cell];
    const int faceEnd = mesh.cellFaceOffsets[cell + 1];
    for (int refIndex = faceBegin; refIndex < faceEnd; ++refIndex) {
        const double nx = mesh.refNx[refIndex];
        const double ny = mesh.refNy[refIndex];
        const double nz = mesh.refNz[refIndex];
        const double mu = ox * nx + oy * ny + oz * nz;
        const double coefficient = fabs(mu) * mesh.refArea[refIndex];
        if (coefficient <= 1.0e-14) continue;

        if (mu > 0.0) {
            outflow += coefficient;
        } else {
            const int neighbor = mesh.refNeighbor[refIndex];
            const double psiIn = neighbor >= 0
                ? currentPsi[outputIndex * C + neighbor]
                : boundaryInflowRef(mesh, refIndex, ox, oy, oz, sourceShape);
            inflow += coefficient * psiIn;
        }
    }

    const double source = sourceShared ? previousPsi[cell] : previousPsi[cellIndex];
    const double rhs =
        mesh.volume[cell] * (mesh.sigmaS[cell] * source + mesh.cellQ[cell]) + inflow;
    const double diagonal = mesh.sigmaT[cell] * mesh.volume[cell] + outflow;
    currentPsi[cellIndex] = rhs / diagonal;
}

__global__ void sweepLevelFusedKernel(
    DeviceMeshView mesh,
    const double* ordinateX,
    const double* ordinateY,
    const double* ordinateZ,
    const int* sweepOrders,
    const int* sweepLevelOffsetBase,
    const int* sweepLevelCount,
    const int* sweepLevelOffsets,
    const int* selectedDirection,
    int directionBatch,
    int sourceShape,
    const double* previousPsi,
    double* currentPsi,
    bool sourceShared
) {
    const int localDirection = blockIdx.x;
    if (localDirection >= directionBatch) return;

    const int C = mesh.cellCount;
    const int direction = selectedDirection[localDirection];
    const int levelCount = sweepLevelCount[direction];
    const double ox = ordinateX[direction];
    const double oy = ordinateY[direction];
    const double oz = ordinateZ[direction];
    const int* order = sweepOrders + static_cast<std::size_t>(direction) * C;
    const int levelBase = sweepLevelOffsetBase[direction];
    const std::size_t outputIndex = static_cast<std::size_t>(localDirection);

    for (int level = 0; level < levelCount; ++level) {
        const int begin = sweepLevelOffsets[levelBase + level];
        const int end = sweepLevelOffsets[levelBase + level + 1];
        for (int index = begin + threadIdx.x; index < end; index += blockDim.x) {
            const int cell = order[index];
            const std::size_t cellIndex = outputIndex * C + cell;
            double inflow = 0.0;
            double outflow = 0.0;

            const int faceBegin = mesh.cellFaceOffsets[cell];
            const int faceEnd = mesh.cellFaceOffsets[cell + 1];
            for (int refIndex = faceBegin; refIndex < faceEnd; ++refIndex) {
                const double nx = mesh.refNx[refIndex];
                const double ny = mesh.refNy[refIndex];
                const double nz = mesh.refNz[refIndex];
                const double mu = ox * nx + oy * ny + oz * nz;
                const double coefficient = fabs(mu) * mesh.refArea[refIndex];
                if (coefficient <= 1.0e-14) continue;

                if (mu > 0.0) {
                    outflow += coefficient;
                } else {
                    const int neighbor = mesh.refNeighbor[refIndex];
                    const double psiIn = neighbor >= 0
                        ? currentPsi[outputIndex * C + neighbor]
                        : boundaryInflowRef(mesh, refIndex, ox, oy, oz, sourceShape);
                    inflow += coefficient * psiIn;
                }
            }

            const double source = sourceShared ? previousPsi[cell] : previousPsi[cellIndex];
            const double rhs =
                mesh.volume[cell] * (mesh.sigmaS[cell] * source + mesh.cellQ[cell]) + inflow;
            const double diagonal = mesh.sigmaT[cell] * mesh.volume[cell] + outflow;
            currentPsi[cellIndex] = rhs / diagonal;
        }
        __syncthreads();
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

__global__ void accumulateDirectionBatchKernel(
    const double* angularPsi,
    const double* weights,
    const int* selectedDirection,
    int directionBatch,
    int cellCount,
    double* phi0
) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;
    double sum = 0.0;
    for (int localDirection = 0; localDirection < directionBatch; ++localDirection) {
        const int direction = selectedDirection[localDirection];
        sum += weights[direction] *
               angularPsi[static_cast<std::size_t>(localDirection) * cellCount + cell];
    }
    phi0[cell] += sum;
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
    int maxSweepLevelCount = 0;
    int maxSweepLevelWidth = 0;
    std::vector<unsigned char> hostHasCycle;
    std::vector<int> hostLevelCount;
    std::vector<int> hostAcyclicDirections;
    DeviceMeshStorage mesh;
    DeviceArray<double> ordinateX;
    DeviceArray<double> ordinateY;
    DeviceArray<double> ordinateZ;
    DeviceArray<double> weights;
    DeviceArray<unsigned char> hasCycle;
    DeviceArray<int> orders;
    DeviceArray<int> levelOffsetBase;
    DeviceArray<int> levelCount;
    DeviceArray<int> levelOffsets;
    DeviceArray<int> allDirections;
    DeviceArray<int> acyclicDirections;
};

struct DeviceStaticProblem {
    int cellCount = 0;
    int directionCount = 0;
    DeviceMeshStorage mesh;
    DeviceArray<double> ordinateX;
    DeviceArray<double> ordinateY;
    DeviceArray<double> ordinateZ;
    DeviceArray<double> weights;
};

struct DevicePlanChunk {
    int directionCount = 0;
    int maxSweepLevelCount = 0;
    int maxSweepLevelWidth = 0;
    std::vector<unsigned char> hostHasCycle;
    std::vector<int> hostLevelCount;
    std::vector<int> globalDirections;
    DeviceArray<double> ordinateX;
    DeviceArray<double> ordinateY;
    DeviceArray<double> ordinateZ;
    DeviceArray<double> weights;
    DeviceArray<unsigned char> hasCycle;
    DeviceArray<int> orders;
    DeviceArray<int> levelOffsetBase;
    DeviceArray<int> levelCount;
    DeviceArray<int> levelOffsets;
    DeviceArray<int> directions;
};

struct HostPlanChunk {
    int directionCount = 0;
    int maxSweepLevelCount = 0;
    int maxSweepLevelWidth = 0;
    std::vector<unsigned char> hasCycle;
    std::vector<int> levelOffsetBase;
    std::vector<int> levelCount;
    std::vector<int> levelOffsets;
    std::vector<int> directions;
    std::vector<int> globalDirections;
    std::vector<double> ordinateX;
    std::vector<double> ordinateY;
    std::vector<double> ordinateZ;
    std::vector<double> weights;
    std::vector<int> orders;
};

struct PlanUploadPinnedStaging {
    PinnedHostBuffer<unsigned char> hasCycle;
    PinnedHostBuffer<int> levelOffsetBase;
    PinnedHostBuffer<int> levelCount;
    PinnedHostBuffer<int> levelOffsets;
    PinnedHostBuffer<int> directions;
    PinnedHostBuffer<double> ordinateX;
    PinnedHostBuffer<double> ordinateY;
    PinnedHostBuffer<double> ordinateZ;
    PinnedHostBuffer<double> weights;
    PinnedHostBuffer<int> orders;
};

struct PlanChunkTiming {
    double keySeconds = 0.0;
    double cacheSeconds = 0.0;
    double buildSeconds = 0.0;
    double saveSeconds = 0.0;
    double assembleSeconds = 0.0;
    double packSeconds = 0.0;
    double uploadSeconds = 0.0;
    double syncSeconds = 0.0;
};

std::size_t hostPlanChunkBytes(const HostPlanChunk& chunk) {
    return sizeof(HostPlanChunk) +
           chunk.hasCycle.size() * sizeof(unsigned char) +
           chunk.levelOffsetBase.size() * sizeof(int) +
           chunk.levelCount.size() * sizeof(int) +
           chunk.levelOffsets.size() * sizeof(int) +
           chunk.directions.size() * sizeof(int) +
           chunk.globalDirections.size() * sizeof(int) +
           chunk.ordinateX.size() * sizeof(double) +
           chunk.ordinateY.size() * sizeof(double) +
           chunk.ordinateZ.size() * sizeof(double) +
           chunk.weights.size() * sizeof(double) +
           chunk.orders.size() * sizeof(int);
}

std::size_t estimateDevicePlanChunkBytes(const HostPlanChunk& chunk, int cellCount) {
    const std::size_t K = static_cast<std::size_t>(chunk.directionCount);
    const std::size_t C = static_cast<std::size_t>(cellCount);
    std::size_t bytes = K * C * sizeof(int);
    bytes += K * (3 * sizeof(double) + sizeof(double));
    bytes += K * (sizeof(unsigned char) + 3 * sizeof(int));
    bytes += static_cast<std::size_t>(chunk.maxSweepLevelCount + 1) * K * sizeof(int);
    return bytes;
}

constexpr std::uint64_t planChunkCacheMagic = 0x5253495357434831ULL;
constexpr std::uint32_t planChunkCacheVersion = 1;

struct PlanChunkCacheHeader {
    std::uint64_t magic;
    std::uint32_t version;
    std::uint32_t reserved;
    std::uint64_t key;
    std::uint64_t directionCount;
    std::uint64_t cellCount;
};

std::uint64_t fnv1aBytesCuda(std::uint64_t hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

template <typename T>
std::uint64_t fnv1aValueCuda(std::uint64_t hash, const T& value) {
    return fnv1aBytesCuda(hash, &value, sizeof(T));
}

bool sweepPlanChunkCacheEnabled() {
    const char* value = std::getenv("RSI_SWEEP_PLAN_CACHE");
    return !value || std::atoi(value) != 0;
}

std::size_t sweepPlanBytes(const SweepPlan& plan) {
    return plan.order.size() * sizeof(int) +
           plan.levelOffsets.size() * sizeof(int) +
           sizeof(SweepPlan);
}

std::size_t sweepPlansBytes(const std::vector<SweepPlan>& plans) {
    std::size_t bytes = sizeof(std::vector<SweepPlan>);
    for (const SweepPlan& plan : plans) bytes += sweepPlanBytes(plan);
    return bytes;
}

std::size_t sweepPlanHostCacheBudgetBytes() {
    const char* value = std::getenv("RSI_CUDA_PLAN_HOST_CACHE_MB");
    const long mb = value ? std::atol(value) : 1024;
    if (mb <= 0) return 0;
    constexpr long maxMb = 65536;
    return static_cast<std::size_t>(std::min(mb, maxMb)) * 1024ull * 1024ull;
}

struct HostPlanChunkCacheEntry {
    std::vector<SweepPlan> plans;
    std::size_t bytes = 0;
    std::list<std::uint64_t>::iterator lruIt;
};

struct HostPlanChunkCache {
    std::mutex mutex;
    std::size_t bytes = 0;
    std::list<std::uint64_t> lru;
    std::unordered_map<std::uint64_t, HostPlanChunkCacheEntry> entries;
};

HostPlanChunkCache& hostPlanChunkCache() {
    static HostPlanChunkCache cache;
    return cache;
}

bool loadHostPlanChunkCache(
    std::uint64_t key,
    std::size_t directionCount,
    std::size_t cellCount,
    std::vector<SweepPlan>& plans
) {
    if (sweepPlanHostCacheBudgetBytes() == 0) return false;
    HostPlanChunkCache& cache = hostPlanChunkCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto found = cache.entries.find(key);
    if (found == cache.entries.end()) return false;
    const std::vector<SweepPlan>& cachedPlans = found->second.plans;
    if (cachedPlans.size() != directionCount) return false;
    for (const SweepPlan& plan : cachedPlans) {
        if (plan.order.size() != cellCount) return false;
    }
    cache.lru.splice(cache.lru.begin(), cache.lru, found->second.lruIt);
    found->second.lruIt = cache.lru.begin();
    plans = cachedPlans;
    return true;
}

void storeHostPlanChunkCache(std::uint64_t key, const std::vector<SweepPlan>& plans) {
    const std::size_t budget = sweepPlanHostCacheBudgetBytes();
    if (budget == 0) return;
    const std::size_t bytes = sweepPlansBytes(plans);
    if (bytes > budget) return;
    HostPlanChunkCache& cache = hostPlanChunkCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto existing = cache.entries.find(key);
    if (existing != cache.entries.end()) {
        cache.bytes -= existing->second.bytes;
        cache.lru.erase(existing->second.lruIt);
        cache.entries.erase(existing);
    }
    cache.lru.push_front(key);
    HostPlanChunkCacheEntry entry;
    entry.plans = plans;
    entry.bytes = bytes;
    entry.lruIt = cache.lru.begin();
    cache.bytes += bytes;
    cache.entries.emplace(key, std::move(entry));
    while (cache.bytes > budget && !cache.lru.empty()) {
        const std::uint64_t victim = cache.lru.back();
        auto victimIt = cache.entries.find(victim);
        cache.lru.pop_back();
        if (victimIt == cache.entries.end()) continue;
        cache.bytes -= victimIt->second.bytes;
        cache.entries.erase(victimIt);
    }
}

std::string sweepPlanChunkCachePath(std::uint64_t key) {
    std::ostringstream hexKey;
    hexKey << std::hex << key;
    const std::string keyString = hexKey.str();
    std::ostringstream base;
    const char* dir = std::getenv("RSI_SWEEP_PLAN_CACHE_DIR");
    if (dir && *dir) {
        base << dir;
        const std::string dirString(dir);
        if (!dirString.empty() && dirString.back() != '/') base << '/';
    } else {
        base << "results/cache/";
    }
    const std::string baseDir = base.str();
    const std::string legacyPath =
        baseDir + "rsi_sweep_plan_chunk_" + keyString + ".bin";
    const char* shardValue = std::getenv("RSI_SWEEP_PLAN_CACHE_SHARD_DIRS");
    if (!shardValue || std::atoi(shardValue) == 0) {
        return legacyPath;
    }
    const std::string shard = keyString.substr(0, std::min<std::size_t>(2, keyString.size()));
    const std::string shardedPath =
        baseDir + shard + "/rsi_sweep_plan_chunk_" + keyString + ".bin";
    if (!std::filesystem::exists(shardedPath) && std::filesystem::exists(legacyPath)) {
        return legacyPath;
    }
    return shardedPath;
}

std::uint64_t sweepPlanChunkCacheMeshPrefix(
    const Mesh& mesh,
    std::uint64_t directionCount
) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = fnv1aValueCuda(hash, planChunkCacheVersion);
    const std::uint64_t cellCount = static_cast<std::uint64_t>(mesh.cells.size());
    const std::uint64_t faceCount = static_cast<std::uint64_t>(mesh.faces.size());
    hash = fnv1aValueCuda(hash, cellCount);
    hash = fnv1aValueCuda(hash, faceCount);
    hash = fnv1aValueCuda(hash, directionCount);
    for (const Cell& cell : mesh.cells) {
        hash = fnv1aValueCuda(hash, cell.id);
        hash = fnv1aValueCuda(hash, cell.center.x);
        hash = fnv1aValueCuda(hash, cell.center.y);
        hash = fnv1aValueCuda(hash, cell.center.z);
        hash = fnv1aValueCuda(hash, cell.volume);
        const std::uint64_t refCount = static_cast<std::uint64_t>(cell.faceRefs.size());
        hash = fnv1aValueCuda(hash, refCount);
        for (const CellFaceRef& ref : cell.faceRefs) {
            hash = fnv1aValueCuda(hash, ref.face);
            hash = fnv1aValueCuda(hash, ref.neighbor);
            hash = fnv1aValueCuda(hash, ref.sign);
        }
    }
    for (const Face& face : mesh.faces) {
        hash = fnv1aValueCuda(hash, face.id);
        hash = fnv1aValueCuda(hash, face.left_cell);
        hash = fnv1aValueCuda(hash, face.right_cell);
        hash = fnv1aValueCuda(hash, face.normal.x);
        hash = fnv1aValueCuda(hash, face.normal.y);
        hash = fnv1aValueCuda(hash, face.normal.z);
        hash = fnv1aValueCuda(hash, face.area);
    }
    return hash;
}

struct SweepPlanChunkCacheKeyContext {
    const Mesh* mesh = nullptr;
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::uint64_t> meshPrefixByDirectionCount;

    explicit SweepPlanChunkCacheKeyContext(const Mesh& meshRef) : mesh(&meshRef) {}

    std::uint64_t meshPrefix(std::uint64_t directionCount) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = meshPrefixByDirectionCount.find(directionCount);
        if (found != meshPrefixByDirectionCount.end()) return found->second;
        const std::uint64_t prefix =
            sweepPlanChunkCacheMeshPrefix(*mesh, directionCount);
        meshPrefixByDirectionCount.emplace(directionCount, prefix);
        return prefix;
    }
};

std::uint64_t sweepPlanChunkCacheKey(
    SweepPlanChunkCacheKeyContext* context,
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const std::vector<int>& globalDirections
) {
    const std::uint64_t directionCount =
        static_cast<std::uint64_t>(globalDirections.size());
    std::uint64_t hash = context && context->mesh == &mesh
        ? context->meshPrefix(directionCount)
        : sweepPlanChunkCacheMeshPrefix(mesh, directionCount);
    for (int direction : globalDirections) {
        hash = fnv1aValueCuda(hash, direction);
        const Ordinate& ordinate = ordinates[direction];
        hash = fnv1aValueCuda(hash, ordinate.omega.x);
        hash = fnv1aValueCuda(hash, ordinate.omega.y);
        hash = fnv1aValueCuda(hash, ordinate.omega.z);
    }
    return hash;
}

void readExactCuda(std::ifstream& input, void* data, std::size_t size, const char* label) {
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!input) throw std::runtime_error(std::string("truncated sweep plan chunk cache: ") + label);
}

void writeExactCuda(std::ofstream& output, const void* data, std::size_t size) {
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!output) throw std::runtime_error("failed while writing sweep plan chunk cache");
}

bool loadSweepPlanChunkCache(
    const std::string& path,
    std::uint64_t key,
    std::size_t directionCount,
    std::size_t cellCount,
    std::vector<SweepPlan>& plans
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    PlanChunkCacheHeader header{};
    readExactCuda(input, &header, sizeof(header), "header");
    if (header.magic != planChunkCacheMagic ||
        header.version != planChunkCacheVersion ||
        header.key != key ||
        header.directionCount != directionCount ||
        header.cellCount != cellCount) {
        return false;
    }
    plans.clear();
    plans.resize(directionCount);
    for (std::size_t direction = 0; direction < directionCount; ++direction) {
        std::uint32_t hasCycle = 0;
        std::uint64_t orderSize = 0;
        std::uint64_t levelOffsetSize = 0;
        readExactCuda(input, &hasCycle, sizeof(hasCycle), "hasCycle");
        readExactCuda(input, &orderSize, sizeof(orderSize), "order size");
        readExactCuda(input, &levelOffsetSize, sizeof(levelOffsetSize), "level offset size");
        if (orderSize != cellCount) {
            throw std::runtime_error("sweep plan chunk cache has invalid order size");
        }
        SweepPlan& plan = plans[direction];
        plan.hasCycle = hasCycle != 0;
        plan.order.resize(static_cast<std::size_t>(orderSize));
        readExactCuda(
            input,
            plan.order.data(),
            static_cast<std::size_t>(orderSize) * sizeof(int),
            "order"
        );
        plan.levelOffsets.resize(static_cast<std::size_t>(levelOffsetSize));
        if (levelOffsetSize > 0) {
            readExactCuda(
                input,
                plan.levelOffsets.data(),
                static_cast<std::size_t>(levelOffsetSize) * sizeof(int),
                "level offsets"
            );
        }
        if (!plan.hasCycle) {
            if (plan.levelOffsets.empty() ||
                plan.levelOffsets.front() != 0 ||
                plan.levelOffsets.back() != static_cast<int>(cellCount)) {
                throw std::runtime_error("sweep plan chunk cache has invalid level offsets");
            }
            plan.levelCells = plan.order;
        }
    }
    return true;
}

bool tryLoadSweepPlanChunkCache(
    const std::string& path,
    std::uint64_t key,
    std::size_t directionCount,
    std::size_t cellCount,
    std::vector<SweepPlan>& plans
) {
    try {
        if (loadHostPlanChunkCache(key, directionCount, cellCount, plans)) return true;
        const bool loaded =
            loadSweepPlanChunkCache(path, key, directionCount, cellCount, plans);
        if (loaded) storeHostPlanChunkCache(key, plans);
        return loaded;
    } catch (const std::exception& e) {
        plans.clear();
        std::cerr << "Warning: ignoring sweep plan chunk cache " << path
                  << ": " << e.what() << "\n";
        return false;
    }
}

void saveSweepPlanChunkCache(
    const std::string& path,
    std::uint64_t key,
    std::size_t directionCount,
    std::size_t cellCount,
    const std::vector<SweepPlan>& plans
) {
    const std::string tempPath = path + ".tmp";
    const std::filesystem::path cachePath(path);
    const std::filesystem::path parent = cachePath.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            std::cerr << "Warning: cannot create sweep plan chunk cache directory: "
                      << parent.string() << ": " << ec.message() << "\n";
            return;
        }
    }
    std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "Warning: cannot write sweep plan chunk cache: " << tempPath << "\n";
        return;
    }
    const PlanChunkCacheHeader header{
        planChunkCacheMagic,
        planChunkCacheVersion,
        0,
        key,
        static_cast<std::uint64_t>(directionCount),
        static_cast<std::uint64_t>(cellCount)
    };
    writeExactCuda(output, &header, sizeof(header));
    for (const SweepPlan& plan : plans) {
        const std::uint32_t hasCycle = plan.hasCycle ? 1u : 0u;
        const std::uint64_t orderSize = static_cast<std::uint64_t>(plan.order.size());
        const std::uint64_t levelOffsetSize =
            static_cast<std::uint64_t>(plan.levelOffsets.size());
        writeExactCuda(output, &hasCycle, sizeof(hasCycle));
        writeExactCuda(output, &orderSize, sizeof(orderSize));
        writeExactCuda(output, &levelOffsetSize, sizeof(levelOffsetSize));
        writeExactCuda(output, plan.order.data(), plan.order.size() * sizeof(int));
        if (!plan.levelOffsets.empty()) {
            writeExactCuda(
                output,
                plan.levelOffsets.data(),
                plan.levelOffsets.size() * sizeof(int)
            );
        }
    }
    output.close();
    if (!output) {
        std::cerr << "Warning: failed to flush sweep plan chunk cache: " << tempPath << "\n";
        return;
    }
    if (std::rename(tempPath.c_str(), path.c_str()) != 0) {
        std::cerr << "Warning: cannot install sweep plan chunk cache: " << path << "\n";
    }
    storeHostPlanChunkCache(key, plans);
}

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

bool envFlagEnabled(const char* name, bool defaultValue) {
    const char* value = std::getenv(name);
    if (!value) return defaultValue;
    return std::atoi(value) != 0;
}

int envIntValue(const char* name, int defaultValue, int minValue, int maxValue) {
    const char* value = std::getenv(name);
    if (!value) return defaultValue;
    const int parsed = std::atoi(value);
    return std::max(minValue, std::min(maxValue, parsed));
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
    std::vector<int> levelOffsetBase(M, 0);
    std::vector<int> levelCount(M, 0);
    std::vector<int> levelOffsets;
    std::vector<int> directions(M);
    std::vector<int> acyclicDirections;
    for (int m = 0; m < M; ++m) {
        if (static_cast<int>(sweepPlans[m].order.size()) != C) {
            throw std::runtime_error("CUDA sweep order size mismatch");
        }
        ox[m] = ordinates[m].omega.x;
        oy[m] = ordinates[m].omega.y;
        oz[m] = ordinates[m].omega.z;
        weights[m] = ordinates[m].weight;
        hasCycle[m] = sweepPlans[m].hasCycle ? 1 : 0;
        if (!sweepPlans[m].hasCycle) {
            if (sweepPlans[m].levelOffsets.empty() ||
                sweepPlans[m].levelOffsets.front() != 0 ||
                sweepPlans[m].levelOffsets.back() != C) {
                throw std::runtime_error("CUDA sweep level offsets are inconsistent");
            }
            levelOffsetBase[m] = static_cast<int>(levelOffsets.size());
            levelOffsets.insert(
                levelOffsets.end(),
                sweepPlans[m].levelOffsets.begin(),
                sweepPlans[m].levelOffsets.end()
            );
            levelCount[m] = static_cast<int>(sweepPlans[m].levelOffsets.size()) - 1;
            problem->maxSweepLevelCount =
                std::max(problem->maxSweepLevelCount, levelCount[m]);
            for (int level = 0; level < levelCount[m]; ++level) {
                const int width =
                    sweepPlans[m].levelOffsets[static_cast<std::size_t>(level + 1)] -
                    sweepPlans[m].levelOffsets[static_cast<std::size_t>(level)];
                problem->maxSweepLevelWidth =
                    std::max(problem->maxSweepLevelWidth, width);
            }
            acyclicDirections.push_back(m);
        }
        directions[m] = m;
    }
    problem->ordinateX.copyFrom(ox);
    problem->ordinateY.copyFrom(oy);
    problem->ordinateZ.copyFrom(oz);
    problem->weights.copyFrom(weights);
    problem->hostHasCycle = hasCycle;
    problem->hostLevelCount = levelCount;
    problem->hostAcyclicDirections = acyclicDirections;
    problem->hasCycle.copyFrom(hasCycle);
    problem->levelOffsetBase.copyFrom(levelOffsetBase);
    problem->levelCount.copyFrom(levelCount);
    problem->levelOffsets.copyFrom(levelOffsets);
    problem->allDirections.copyFrom(directions);
    problem->acyclicDirections.copyFrom(acyclicDirections);
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

std::unique_ptr<DeviceStaticProblem> uploadStaticProblem(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates
) {
    auto problem = std::make_unique<DeviceStaticProblem>();
    problem->cellCount = static_cast<int>(mesh.cells.size());
    problem->directionCount = static_cast<int>(ordinates.size());
    const int M = problem->directionCount;
    problem->mesh = uploadMesh(mesh);
    std::vector<double> ox(M), oy(M), oz(M), weights(M);
    for (int m = 0; m < M; ++m) {
        ox[m] = ordinates[m].omega.x;
        oy[m] = ordinates[m].omega.y;
        oz[m] = ordinates[m].omega.z;
        weights[m] = ordinates[m].weight;
    }
    problem->ordinateX.copyFrom(ox);
    problem->ordinateY.copyFrom(oy);
    problem->ordinateZ.copyFrom(oz);
    problem->weights.copyFrom(weights);
    return problem;
}

std::vector<SweepPlan> buildSweepPlansParallel(
    const std::vector<Ordinate>& ordinates,
    const TransportSweep& sweep,
    const std::vector<int>& globalDirections
) {
    const int K = static_cast<int>(globalDirections.size());
    std::vector<SweepPlan> plans(K);
    std::atomic<int> next{0};
    std::exception_ptr workerError;
    std::mutex errorMutex;
    const unsigned workerCount = std::max(
        1u,
        std::min<unsigned>(
            static_cast<unsigned>(K),
            std::thread::hardware_concurrency()
        )
    );
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (unsigned worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&] {
            try {
                while (true) {
                    const int local = next.fetch_add(1);
                    if (local >= K) break;
                    const int global = globalDirections[local];
                    plans[local] = sweep.buildSweepPlan(ordinates[global].omega);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!workerError) workerError = std::current_exception();
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    if (workerError) std::rethrow_exception(workerError);
    return plans;
}

HostPlanChunk preparePlanChunkHost(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const TransportSweep& sweep,
    const std::vector<int>& globalDirections,
    SweepPlanChunkCacheKeyContext* cacheKeyContext = nullptr,
    PlanChunkTiming* timing = nullptr
) {
    HostPlanChunk host;
    const int C = static_cast<int>(mesh.cells.size());
    const int K = static_cast<int>(globalDirections.size());
    if (K <= 0) throw std::runtime_error("empty CUDA sweep plan chunk");
    host.directionCount = K;
    host.globalDirections = globalDirections;

    host.hasCycle.resize(K);
    host.ordinateX.resize(K);
    host.ordinateY.resize(K);
    host.ordinateZ.resize(K);
    host.weights.resize(K);
    host.levelOffsetBase.assign(K, 0);
    host.levelCount.assign(K, 0);
    host.orders.resize(static_cast<std::size_t>(K) * C);
    std::vector<SweepPlan> localPlans;
    const int maxCachedChunkDirections =
        envIntValue("RSI_CUDA_MAX_CACHED_PLAN_CHUNK", 256, 1, 4096);
    const bool useCache = sweepPlanChunkCacheEnabled() && K <= maxCachedChunkDirections;
    const auto keyStart = std::chrono::steady_clock::now();
    const std::uint64_t cacheKey =
        sweepPlanChunkCacheKey(cacheKeyContext, mesh, ordinates, globalDirections);
    const std::string cachePath = sweepPlanChunkCachePath(cacheKey);
    if (timing) {
        timing->keySeconds += secondsBetween(keyStart, std::chrono::steady_clock::now());
    }
    bool loadedFromCache = false;
    if (useCache) {
        const auto cacheStart = std::chrono::steady_clock::now();
        loadedFromCache = tryLoadSweepPlanChunkCache(
            cachePath, cacheKey, globalDirections.size(), mesh.cells.size(), localPlans
        );
        if (timing) {
            timing->cacheSeconds += secondsBetween(
                cacheStart, std::chrono::steady_clock::now()
            );
        }
    }
    if (!loadedFromCache) {
        const bool useFixedChunkReuse =
            sweepPlanChunkCacheEnabled() &&
            envFlagEnabled("RSI_CUDA_FIXED_PLAN_CHUNK_REUSE", true) &&
            K > maxCachedChunkDirections;
        if (useFixedChunkReuse) {
            const int fixedChunkSize =
                envIntValue("RSI_CUDA_FIXED_PLAN_CHUNK", 1, 1, 4096);
            const int M = static_cast<int>(ordinates.size());
            std::vector<int> requestedLocal(M, -1);
            for (int local = 0; local < K; ++local) {
                requestedLocal[globalDirections[local]] = local;
            }
            localPlans.resize(K);
            for (int chunkStart = 0; chunkStart < M; chunkStart += fixedChunkSize) {
                const int chunkEnd = std::min(chunkStart + fixedChunkSize, M);
                bool needed = false;
                for (int global = chunkStart; global < chunkEnd; ++global) {
                    if (requestedLocal[global] >= 0) {
                        needed = true;
                        break;
                    }
                }
                if (!needed) continue;

                std::vector<int> fixedDirections;
                fixedDirections.reserve(chunkEnd - chunkStart);
                for (int global = chunkStart; global < chunkEnd; ++global) {
                    fixedDirections.push_back(global);
                }
                const auto fixedKeyStart = std::chrono::steady_clock::now();
                const std::uint64_t fixedKey =
                    sweepPlanChunkCacheKey(cacheKeyContext, mesh, ordinates, fixedDirections);
                const std::string fixedPath = sweepPlanChunkCachePath(fixedKey);
                if (timing) {
                    timing->keySeconds += secondsBetween(
                        fixedKeyStart, std::chrono::steady_clock::now()
                    );
                }
                std::vector<SweepPlan> fixedPlans;
                const auto cacheStart = std::chrono::steady_clock::now();
                bool fixedLoaded = tryLoadSweepPlanChunkCache(
                    fixedPath, fixedKey, fixedDirections.size(), mesh.cells.size(), fixedPlans
                );
                if (timing) {
                    timing->cacheSeconds += secondsBetween(
                        cacheStart, std::chrono::steady_clock::now()
                    );
                }
                if (!fixedLoaded) {
                    const auto buildStart = std::chrono::steady_clock::now();
                    fixedPlans = buildSweepPlansParallel(ordinates, sweep, fixedDirections);
                    if (timing) {
                        timing->buildSeconds += secondsBetween(
                            buildStart, std::chrono::steady_clock::now()
                        );
                    }
                    if (static_cast<int>(fixedDirections.size()) <= maxCachedChunkDirections) {
                        const auto saveStart = std::chrono::steady_clock::now();
                        saveSweepPlanChunkCache(
                            fixedPath, fixedKey, fixedDirections.size(),
                            mesh.cells.size(), fixedPlans
                        );
                        if (timing) {
                            timing->saveSeconds += secondsBetween(
                                saveStart, std::chrono::steady_clock::now()
                            );
                        }
                    }
                }
                const auto assembleStart = std::chrono::steady_clock::now();
                for (int global = chunkStart; global < chunkEnd; ++global) {
                    const int local = requestedLocal[global];
                    if (local < 0) continue;
                    localPlans[local] = fixedPlans[global - chunkStart];
                }
                if (timing) {
                    timing->assembleSeconds += secondsBetween(
                        assembleStart, std::chrono::steady_clock::now()
                    );
                }
            }
        } else {
            const auto buildStart = std::chrono::steady_clock::now();
            localPlans = buildSweepPlansParallel(ordinates, sweep, globalDirections);
            if (timing) {
                timing->buildSeconds += secondsBetween(
                    buildStart, std::chrono::steady_clock::now()
                );
            }
        }
    }
    const auto packStart = std::chrono::steady_clock::now();
    for (int local = 0; local < K; ++local) {
        const int global = globalDirections[local];
        if (global < 0 || global >= static_cast<int>(ordinates.size())) {
            throw std::runtime_error("CUDA sweep plan chunk direction out of range");
        }
        host.ordinateX[local] = ordinates[global].omega.x;
        host.ordinateY[local] = ordinates[global].omega.y;
        host.ordinateZ[local] = ordinates[global].omega.z;
        host.weights[local] = ordinates[global].weight;
        const SweepPlan& plan = localPlans[local];
        if (static_cast<int>(plan.order.size()) != C) {
            throw std::runtime_error("CUDA sweep order size mismatch");
        }
        std::copy(
            plan.order.begin(), plan.order.end(),
            host.orders.begin() + static_cast<std::size_t>(local) * C
        );
        host.hasCycle[local] = plan.hasCycle ? 1 : 0;
        if (!plan.hasCycle) {
            if (plan.levelOffsets.empty() ||
                plan.levelOffsets.front() != 0 ||
                plan.levelOffsets.back() != C) {
                throw std::runtime_error("CUDA sweep level offsets are inconsistent");
            }
            host.levelOffsetBase[local] = static_cast<int>(host.levelOffsets.size());
            host.levelOffsets.insert(
                host.levelOffsets.end(), plan.levelOffsets.begin(), plan.levelOffsets.end()
            );
            host.levelCount[local] = static_cast<int>(plan.levelOffsets.size()) - 1;
            host.maxSweepLevelCount =
                std::max(host.maxSweepLevelCount, host.levelCount[local]);
            for (int level = 0; level < host.levelCount[local]; ++level) {
                const int width =
                    plan.levelOffsets[static_cast<std::size_t>(level + 1)] -
                    plan.levelOffsets[static_cast<std::size_t>(level)];
                host.maxSweepLevelWidth = std::max(host.maxSweepLevelWidth, width);
            }
        }
    }
    if (useCache && !loadedFromCache) {
        const auto saveStart = std::chrono::steady_clock::now();
        saveSweepPlanChunkCache(
            cachePath, cacheKey, globalDirections.size(), mesh.cells.size(), localPlans
        );
        if (timing) {
            timing->saveSeconds += secondsBetween(
                saveStart, std::chrono::steady_clock::now()
            );
        }
    }

    host.directions.resize(K);
    for (int local = 0; local < K; ++local) host.directions[local] = local;
    if (timing) {
        timing->packSeconds += secondsBetween(packStart, std::chrono::steady_clock::now());
    }
    return host;
}

void uploadPreparedPlanChunkInto(
    HostPlanChunk&& host,
    DevicePlanChunk& chunk,
    PlanChunkTiming* timing = nullptr,
    cudaStream_t uploadStream = nullptr,
    PlanUploadPinnedStaging* pinnedStaging = nullptr
) {
    const auto uploadStart = std::chrono::steady_clock::now();
    chunk.directionCount = host.directionCount;
    chunk.maxSweepLevelCount = host.maxSweepLevelCount;
    chunk.maxSweepLevelWidth = host.maxSweepLevelWidth;
    if (uploadStream && pinnedStaging) {
        chunk.ordinateX.copyFromPinnedAsync(
            host.ordinateX, pinnedStaging->ordinateX, uploadStream
        );
        chunk.ordinateY.copyFromPinnedAsync(
            host.ordinateY, pinnedStaging->ordinateY, uploadStream
        );
        chunk.ordinateZ.copyFromPinnedAsync(
            host.ordinateZ, pinnedStaging->ordinateZ, uploadStream
        );
        chunk.weights.copyFromPinnedAsync(host.weights, pinnedStaging->weights, uploadStream);
        chunk.hasCycle.copyFromPinnedAsync(
            host.hasCycle, pinnedStaging->hasCycle, uploadStream
        );
        chunk.levelOffsetBase.copyFromPinnedAsync(
            host.levelOffsetBase, pinnedStaging->levelOffsetBase, uploadStream
        );
        chunk.levelCount.copyFromPinnedAsync(
            host.levelCount, pinnedStaging->levelCount, uploadStream
        );
        chunk.levelOffsets.copyFromPinnedAsync(
            host.levelOffsets, pinnedStaging->levelOffsets, uploadStream
        );
        chunk.directions.copyFromPinnedAsync(
            host.directions, pinnedStaging->directions, uploadStream
        );
        chunk.orders.copyFromPinnedAsync(host.orders, pinnedStaging->orders, uploadStream);
    } else if (uploadStream) {
        chunk.ordinateX.copyFromAsync(host.ordinateX, uploadStream);
        chunk.ordinateY.copyFromAsync(host.ordinateY, uploadStream);
        chunk.ordinateZ.copyFromAsync(host.ordinateZ, uploadStream);
        chunk.weights.copyFromAsync(host.weights, uploadStream);
        chunk.hasCycle.copyFromAsync(host.hasCycle, uploadStream);
        chunk.levelOffsetBase.copyFromAsync(host.levelOffsetBase, uploadStream);
        chunk.levelCount.copyFromAsync(host.levelCount, uploadStream);
        chunk.levelOffsets.copyFromAsync(host.levelOffsets, uploadStream);
        chunk.directions.copyFromAsync(host.directions, uploadStream);
        chunk.orders.copyFromAsync(host.orders, uploadStream);
    } else {
        chunk.ordinateX.copyFrom(host.ordinateX);
        chunk.ordinateY.copyFrom(host.ordinateY);
        chunk.ordinateZ.copyFrom(host.ordinateZ);
        chunk.weights.copyFrom(host.weights);
        chunk.hasCycle.copyFrom(host.hasCycle);
        chunk.levelOffsetBase.copyFrom(host.levelOffsetBase);
        chunk.levelCount.copyFrom(host.levelCount);
        chunk.levelOffsets.copyFrom(host.levelOffsets);
        chunk.directions.copyFrom(host.directions);
        chunk.orders.copyFrom(host.orders);
    }
    chunk.hostHasCycle = std::move(host.hasCycle);
    chunk.hostLevelCount = std::move(host.levelCount);
    chunk.globalDirections = std::move(host.globalDirections);
    if (timing) {
        timing->uploadSeconds += secondsBetween(uploadStart, std::chrono::steady_clock::now());
    }
}

std::unique_ptr<DevicePlanChunk> uploadPreparedPlanChunk(
    HostPlanChunk&& host,
    PlanChunkTiming* timing = nullptr
) {
    auto chunk = std::make_unique<DevicePlanChunk>();
    uploadPreparedPlanChunkInto(std::move(host), *chunk, timing);
    return chunk;
}

std::unique_ptr<DevicePlanChunk> uploadPlanChunk(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const TransportSweep& sweep,
    const std::vector<int>& globalDirections,
    SweepPlanChunkCacheKeyContext* cacheKeyContext = nullptr,
    PlanChunkTiming* timing = nullptr
) {
    HostPlanChunk host = preparePlanChunkHost(
        mesh, ordinates, sweep, globalDirections, cacheKeyContext, timing
    );
    std::unique_ptr<DevicePlanChunk> chunk = uploadPreparedPlanChunk(
        std::move(host), timing
    );
    return chunk;
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
    const auto uploadStart = std::chrono::steady_clock::now();
    std::unique_ptr<DeviceProblem> problem = uploadProblem(mesh, ordinates, sweepPlans);
    checkCuda(cudaDeviceSynchronize(), "synchronize CUDA problem upload");
    const double uploadSeconds =
        secondsBetween(uploadStart, std::chrono::steady_clock::now());
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
    CudaEventTimer gpuTimer;
    double siTotalSeconds = 0.0;
    double siClearSeconds = 0.0;
    double siSweepSeconds = 0.0;
    double siReduceSeconds = 0.0;
    double siNormSeconds = 0.0;
    double siCopySeconds = 0.0;
    double siCheckpointSeconds = 0.0;

    // SI fine: all angular directions are independent within one source iteration.
    {
        const auto siStart = std::chrono::steady_clock::now();
        const int directionsPerBatch = std::max(4, std::min(128, 5000000 / C));
        const std::size_t angularValueCount =
            static_cast<std::size_t>(directionsPerBatch) * C;
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
            gpuTimer.start();
            checkCuda(cudaMemset(currentPhi, 0, static_cast<std::size_t>(C) * sizeof(double)),
                      "clear CUDA SI scalar field");
            siClearSeconds += gpuTimer.stop("time CUDA SI angular clear");

            gpuTimer.start();
            constexpr int levelSweepThreads = 256;
            const bool useSIWavefront = envFlagEnabled("RSI_CUDA_SI_WAVEFRONT", true);
            const bool useSITiledWavefront =
                envFlagEnabled("RSI_CUDA_SI_TILED_WAVEFRONT", true);
            const int siLevelTileCount = std::max(
                1,
                (problem->maxSweepLevelWidth + levelSweepThreads - 1) / levelSweepThreads
            );
            for (int directionStart = 0; directionStart < M;) {
                const int batchEnd = std::min(directionStart + directionsPerBatch, M);
                const int runEnd = sameCycleRunEnd(
                    problem->hostHasCycle, directionStart, batchEnd
                );
                const int directionBatch = runEnd - directionStart;
                checkCuda(cudaMemset(
                              angularPsi.ptr, 0,
                              static_cast<std::size_t>(directionBatch) * C * sizeof(double)
                          ),
                          "clear CUDA SI angular batch");

                if (useSIWavefront && problem->hostHasCycle[directionStart] == 0) {
                    int maxLevelCount = 0;
                    for (int direction = directionStart; direction < runEnd; ++direction) {
                        maxLevelCount = std::max(
                            maxLevelCount,
                            problem->hostLevelCount[direction]
                        );
                    }
                    for (int level = 0; level < maxLevelCount; ++level) {
                        sweepLevelKernel<<<directionBatch, levelSweepThreads>>>(
                            problem->mesh.view(C),
                            problem->ordinateX.ptr,
                            problem->ordinateY.ptr,
                            problem->ordinateZ.ptr,
                            problem->orders.ptr,
                            problem->levelOffsetBase.ptr,
                            problem->levelCount.ptr,
                            problem->levelOffsets.ptr,
                            problem->allDirections.ptr + directionStart,
                            directionBatch,
                            sourceShapeCode,
                            previousPhi,
                            angularPsi.ptr,
                            true,
                            level
                        );
                        checkCuda(cudaGetLastError(), "launch CUDA SI wavefront sweep level");
                    }
                } else {
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
                            angularPsi.ptr,
                            true,
                            localPass
                        );
                        checkCuda(cudaGetLastError(), "launch CUDA SI angular sweep batch");
                    }
                }
                accumulateDirectionBatchKernel<<<cellBlocks, reductionThreads>>>(
                    angularPsi.ptr,
                    problem->weights.ptr,
                    problem->allDirections.ptr + directionStart,
                    directionBatch,
                    C,
                    currentPhi
                );
                checkCuda(cudaGetLastError(), "launch CUDA SI angular batch accumulation");
                directionStart = runEnd;
            }
            siSweepSeconds += gpuTimer.stop("time CUDA SI angular sweeps");

            gpuTimer.start();
            checkCuda(cudaDeviceSynchronize(), "synchronize CUDA SI angular reduction");
            siReduceSeconds += gpuTimer.stop("time CUDA SI angular reduction");

            bool converged = false;
            double relative = 0.0;
            gpuTimer.start();
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
            siNormSeconds += gpuTimer.stop("time CUDA SI convergence norm");
            std::cerr << "CUDA SI iteration=" << iteration
                      << ", relative=" << relative << "\n";
            std::swap(previousPhi, currentPhi);
            result.convergedN = iteration;
            if (!checkpointPrefix.empty()) {
                const auto checkpointStart = std::chrono::steady_clock::now();
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
                siCheckpointSeconds += secondsBetween(
                    checkpointStart, std::chrono::steady_clock::now()
                );
            }
            if (converged) break;
        }

        result.siFine.resize(C);
        const auto siCopyStart = std::chrono::steady_clock::now();
        checkCuda(
            cudaMemcpy(result.siFine.data(), previousPhi,
                       static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
            "copy CUDA SI fine result"
        );
        siCopySeconds += secondsBetween(siCopyStart, std::chrono::steady_clock::now());
        siTotalSeconds = secondsBetween(siStart, std::chrono::steady_clock::now());
    }

    // RSI and RSI-tail share the same random chains and all iterations up to N+T.
    const auto rsiStart = std::chrono::steady_clock::now();
    const int iterationCount = result.convergedN + tailExtra;
    const auto scheduleStart = std::chrono::steady_clock::now();
    const std::vector<int> schedule =
        generateDirectionSchedule(seed, M, sampleCount, iterationCount);
    const double rsiScheduleSeconds =
        secondsBetween(scheduleStart, std::chrono::steady_clock::now());
    double rsiDirectionCopySeconds = 0.0;
    double rsiSweepSeconds = 0.0;
    double rsiTailAccumSeconds = 0.0;
    double rsiReduceSeconds = 0.0;
    double rsiCopySeconds = 0.0;
    double rsiCheckpointSeconds = 0.0;
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
    std::vector<unsigned char> batchIterationHasCycle(iterationCount, 0);
    std::vector<int> batchIterationMaxLevel(iterationCount, 0);
    const bool useRSIBatchPass = envFlagEnabled("RSI_CUDA_RSI_BATCH_PASS", false);
    const bool useRSIWavefront = envFlagEnabled("RSI_CUDA_RSI_WAVEFRONT", true);
    const bool useRSITiledWavefront =
        envFlagEnabled("RSI_CUDA_RSI_TILED_WAVEFRONT", true);
    constexpr int levelSweepThreads = 256;
    const int levelTileCount = std::max(
        1,
        (problem->maxSweepLevelWidth + levelSweepThreads - 1) / levelSweepThreads
    );

    for (int batchStart = firstSample; batchStart < sampleCount;
         batchStart += batchCapacity) {
        const int batchSize = std::min(batchCapacity, sampleCount - batchStart);
        const std::size_t batchCellCount = static_cast<std::size_t>(batchSize) * C;
        checkCuda(cudaMemset(psiA.ptr, 0, batchCellCount * sizeof(double)),
                  "clear CUDA RSI previous layer");
        checkCuda(cudaMemset(sampleTail.ptr, 0, batchCellCount * sizeof(double)),
                  "clear CUDA RSI-tail batch accumulation");

        std::fill(batchIterationHasCycle.begin(), batchIterationHasCycle.end(), 0);
        std::fill(batchIterationMaxLevel.begin(), batchIterationMaxLevel.end(), 0);
        for (int iteration = 0; iteration < iterationCount; ++iteration) {
            for (int localSample = 0; localSample < batchSize; ++localSample) {
                const int direction =
                    schedule[static_cast<std::size_t>(batchStart + localSample) * iterationCount +
                             iteration];
                batchDirections[static_cast<std::size_t>(iteration) * batchSize + localSample] =
                    direction;
                if (problem->hostHasCycle[direction] != 0) {
                    batchIterationHasCycle[iteration] = 1;
                }
                batchIterationMaxLevel[iteration] = std::max(
                    batchIterationMaxLevel[iteration],
                    problem->hostLevelCount[direction]
                );
            }
        }
        const auto directionCopyStart = std::chrono::steady_clock::now();
        checkCuda(
            cudaMemcpy(selectedDirections.ptr, batchDirections.data(),
                       static_cast<std::size_t>(batchSize) * iterationCount * sizeof(int),
                       cudaMemcpyHostToDevice),
            "copy CUDA RSI direction schedule"
        );
        rsiDirectionCopySeconds += secondsBetween(
            directionCopyStart, std::chrono::steady_clock::now()
        );

        double* previousPsi = psiA.ptr;
        double* currentPsi = psiB.ptr;
        const int sampleSweepBlocks =
            (batchSize + samplesPerSweepBlock - 1) / samplesPerSweepBlock;
        for (int iteration = 1; iteration <= iterationCount; ++iteration) {
            checkCuda(cudaMemset(currentPsi, 0, batchCellCount * sizeof(double)),
                      "clear CUDA RSI current layer");
            gpuTimer.start();
            const int directionOffset = iteration - 1;
            const bool iterationHasCycle = batchIterationHasCycle[directionOffset] != 0;
            const int* iterationDirections =
                selectedDirections.ptr + static_cast<std::size_t>(directionOffset) * batchSize;
            const bool useTiledForBatch =
                useRSITiledWavefront && batchSize >= 64 && levelTileCount > 1;
            if (useRSIWavefront && !iterationHasCycle) {
                for (int level = 0; level < batchIterationMaxLevel[directionOffset]; ++level) {
                    if (useTiledForBatch) {
                        const dim3 grid(batchSize, levelTileCount);
                        sweepLevelTiledKernel<<<grid, levelSweepThreads>>>(
                            problem->mesh.view(C),
                            problem->ordinateX.ptr,
                            problem->ordinateY.ptr,
                            problem->ordinateZ.ptr,
                            problem->orders.ptr,
                            problem->levelOffsetBase.ptr,
                            problem->levelCount.ptr,
                            problem->levelOffsets.ptr,
                            iterationDirections,
                            batchSize,
                            sourceShapeCode,
                            previousPsi,
                            currentPsi,
                            false,
                            level
                        );
                    } else {
                        sweepLevelKernel<<<batchSize, levelSweepThreads>>>(
                            problem->mesh.view(C),
                            problem->ordinateX.ptr,
                            problem->ordinateY.ptr,
                            problem->ordinateZ.ptr,
                            problem->orders.ptr,
                            problem->levelOffsetBase.ptr,
                            problem->levelCount.ptr,
                            problem->levelOffsets.ptr,
                            iterationDirections,
                            batchSize,
                            sourceShapeCode,
                            previousPsi,
                            currentPsi,
                            false,
                            level
                        );
                    }
                    checkCuda(cudaGetLastError(), "launch combined CUDA RSI wavefront level");
                }
            } else {
                const int localPassCount =
                    useRSIBatchPass && !iterationHasCycle ? 1 : 20;
                for (int localPass = 0; localPass < localPassCount; ++localPass) {
                    sweepSamplesKernel<<<sampleSweepBlocks, sweepThreads>>>(
                        problem->mesh.view(C),
                        problem->ordinateX.ptr,
                        problem->ordinateY.ptr,
                        problem->ordinateZ.ptr,
                        problem->orders.ptr,
                        problem->hasCycle.ptr,
                        iterationDirections,
                        batchSize,
                        sourceShapeCode,
                        previousPsi,
                        currentPsi,
                        false,
                        localPass
                    );
                    checkCuda(cudaGetLastError(), "launch combined CUDA RSI sweep");
                }
            }
            rsiSweepSeconds += gpuTimer.stop("time CUDA RSI sample sweeps");
            if (iteration == result.convergedN) {
                gpuTimer.start();
                reduceSamplesKernel<<<cellBlocks, reductionThreads>>>(
                    currentPsi, batchSize, C, globalRSISum.ptr
                );
                checkCuda(cudaGetLastError(), "reduce CUDA RSI field");
                rsiReduceSeconds += gpuTimer.stop("time CUDA RSI field reduction");
            }
            if (iteration >= result.convergedN) {
                const int accumulationBlocks = static_cast<int>(
                    (batchCellCount + reductionThreads - 1) / reductionThreads
                );
                gpuTimer.start();
                accumulateValuesKernel<<<accumulationBlocks, reductionThreads>>>(
                    currentPsi, batchCellCount, sampleTail.ptr
                );
                checkCuda(cudaGetLastError(), "accumulate CUDA RSI-tail batch field");
                rsiTailAccumSeconds += gpuTimer.stop("time CUDA RSI-tail accumulation");
            }
            std::swap(previousPsi, currentPsi);
        }
        gpuTimer.start();
        reduceSamplesKernel<<<cellBlocks, reductionThreads>>>(
            sampleTail.ptr, batchSize, C, globalTailSum.ptr
        );
        checkCuda(cudaGetLastError(), "reduce CUDA RSI-tail field");
        rsiReduceSeconds += gpuTimer.stop("time CUDA RSI-tail reduction");
        checkCuda(cudaDeviceSynchronize(), "synchronize combined CUDA Figure 5 batch");
        if (!checkpointPrefix.empty()) {
            const auto checkpointStart = std::chrono::steady_clock::now();
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
            rsiCheckpointSeconds += secondsBetween(
                checkpointStart, std::chrono::steady_clock::now()
            );
        }
        std::cerr << "CUDA RSI combined batch complete: samples="
                  << batchStart + batchSize << "/" << sampleCount << "\n";
    }

    result.rsi.resize(C);
    result.rsiTail.resize(C);
    const auto rsiCopyStart = std::chrono::steady_clock::now();
    checkCuda(cudaMemcpy(result.rsi.data(), globalRSISum.ptr,
                         static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
              "copy combined CUDA RSI result");
    checkCuda(cudaMemcpy(result.rsiTail.data(), globalTailSum.ptr,
                         static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
              "copy combined CUDA RSI-tail result");
    rsiCopySeconds += secondsBetween(rsiCopyStart, std::chrono::steady_clock::now());
    const double rsiDenominator = static_cast<double>(sampleCount);
    const double tailDenominator =
        static_cast<double>(sampleCount) * static_cast<double>(tailExtra + 1);
    for (double& value : result.rsi) value /= rsiDenominator;
    for (double& value : result.rsiTail) value /= tailDenominator;

    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - totalStart
    ).count();
    const double rsiTotalSeconds = secondsBetween(rsiStart, std::chrono::steady_clock::now());
    std::cout << "CUDA timing: upload=" << uploadSeconds
              << ", total=" << seconds << "\n";
    std::cout << "CUDA timing: si_total=" << siTotalSeconds
              << ", si_clear=" << siClearSeconds
              << ", si_sweep=" << siSweepSeconds
              << ", si_reduce=" << siReduceSeconds
              << ", si_norm=" << siNormSeconds
              << ", si_copy=" << siCopySeconds
              << ", si_checkpoint=" << siCheckpointSeconds << "\n";
    std::cout << "CUDA timing: rsi_total=" << rsiTotalSeconds
              << ", rsi_schedule=" << rsiScheduleSeconds
              << ", rsi_direction_copy=" << rsiDirectionCopySeconds
              << ", rsi_sweep=" << rsiSweepSeconds
              << ", rsi_tail_accum=" << rsiTailAccumSeconds
              << ", rsi_reduce=" << rsiReduceSeconds
              << ", rsi_copy=" << rsiCopySeconds
              << ", rsi_checkpoint=" << rsiCheckpointSeconds << "\n";
    std::cout << "CUDA Figure 5 complete: cells=" << C << ", directions=" << M
              << ", SI_iterations=" << result.convergedN
              << ", samples=" << sampleCount << ", seconds=" << seconds << "\n";
    return result;
}

CudaFigure5Result runFigure5CudaStreamingPlans(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const TransportSweep& sweep,
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
    const auto uploadStart = std::chrono::steady_clock::now();
    std::unique_ptr<DeviceStaticProblem> staticProblem =
        uploadStaticProblem(mesh, ordinates);
    checkCuda(cudaDeviceSynchronize(), "synchronize CUDA static problem upload");
    const double uploadSeconds =
        secondsBetween(uploadStart, std::chrono::steady_clock::now());
    const int C = staticProblem->cellCount;
    const int M = staticProblem->directionCount;
    const int sourceShapeCode = sourceShape == "rectangle" ? 0 : 1;
    constexpr int sweepThreads = 128;
    constexpr int samplesPerSweepBlock = 32;
    constexpr int reductionThreads = 256;
    constexpr int levelSweepThreads = 256;
    const int cellBlocks = (C + reductionThreads - 1) / reductionThreads;
    const int siDirectionsPerChunk =
        envIntValue("RSI_CUDA_SI_PLAN_CHUNK", 128, 4, 2048);
    SweepPlanChunkCacheKeyContext cacheKeyContext(mesh);

    CudaFigure5Result result;
    CudaEventTimer gpuTimer;
    double siTotalSeconds = 0.0;
    double siPlanSeconds = 0.0;
    double siClearSeconds = 0.0;
    double siSweepSeconds = 0.0;
    double siSweepAngularClearSeconds = 0.0;
    double siSweepKernelSeconds = 0.0;
    double siSweepAccumulateSeconds = 0.0;
    double siSweepSyncSeconds = 0.0;
    double siReduceSeconds = 0.0;
    double siNormSeconds = 0.0;
    double siCopySeconds = 0.0;
    PlanChunkTiming siPlanBreakdown;

    {
        const auto siStart = std::chrono::steady_clock::now();
        const std::size_t angularValueCount =
            static_cast<std::size_t>(siDirectionsPerChunk) * C;
        std::vector<std::vector<int>> siDirectionChunks;
        for (int directionStart = 0; directionStart < M;
             directionStart += siDirectionsPerChunk) {
            const int directionEnd = std::min(directionStart + siDirectionsPerChunk, M);
            std::vector<int> globalDirections;
            globalDirections.reserve(directionEnd - directionStart);
            for (int direction = directionStart; direction < directionEnd; ++direction) {
                globalDirections.push_back(direction);
            }
            siDirectionChunks.push_back(std::move(globalDirections));
        }
        const std::size_t siDevicePlanCacheBudget =
            static_cast<std::size_t>(
                envIntValue("RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB", 10000, 0, 14000)
            ) * 1024ull * 1024ull;
        const std::size_t siHostPlanCacheBudget =
            static_cast<std::size_t>(
                envIntValue("RSI_CUDA_SI_HOST_PLAN_CACHE_MB", 4096, 0, 65536)
            ) * 1024ull * 1024ull;
        std::size_t siDevicePlanCacheBytes = 0;
        std::size_t siHostPlanCacheBytes = 0;
        std::vector<std::unique_ptr<DevicePlanChunk>> siDevicePlanCache(
            siDirectionChunks.size()
        );
        std::vector<std::unique_ptr<HostPlanChunk>> siHostPlanCache(
            siDirectionChunks.size()
        );
        DevicePlanChunk siTemporaryPlanChunk;
        std::vector<std::size_t> siDevicePlanCacheChunkBytes(siDirectionChunks.size(), 0);
        std::vector<std::size_t> siHostPlanCacheChunkBytes(siDirectionChunks.size(), 0);
        std::vector<double> siDevicePlanCacheScore(siDirectionChunks.size(), 0.0);
        std::size_t siDevicePlanCacheHits = 0;
        std::size_t siDevicePlanCacheMisses = 0;
        std::size_t siHostPlanCacheHits = 0;
        std::size_t siHostPlanCacheMisses = 0;
        const bool useSIPlanCostAdmission =
            envFlagEnabled("RSI_CUDA_SI_PLAN_COST_ADMISSION", true);
        const bool useSIPlanPrefetch =
            envFlagEnabled("RSI_CUDA_SI_PLAN_PREFETCH", false);
        const bool useSIPinnedPlanUpload =
            envFlagEnabled("RSI_CUDA_SI_PINNED_PLAN_UPLOAD", false);
        const bool useSIAsyncPlanUpload =
            useSIPinnedPlanUpload ||
            envFlagEnabled("RSI_CUDA_SI_ASYNC_PLAN_UPLOAD", false);
        std::unique_ptr<CudaStream> siPlanUploadStream;
        if (useSIAsyncPlanUpload) {
            siPlanUploadStream = std::make_unique<CudaStream>();
        }
        PlanUploadPinnedStaging siPlanUploadStaging;
        std::future<HostPlanChunk> siPrefetchFuture;
        std::size_t siPrefetchIndex = siDirectionChunks.size();
        auto scheduleSIPrefetch = [&](std::size_t beginIndex) {
            if (!useSIPlanPrefetch || siPrefetchFuture.valid()) return;
            for (std::size_t index = beginIndex; index < siDirectionChunks.size(); ++index) {
                if (siDevicePlanCache[index]) continue;
                siPrefetchIndex = index;
                siPrefetchFuture = std::async(
                    std::launch::async,
                    [&mesh, &ordinates, &sweep, &siDirectionChunks,
                     &cacheKeyContext, index] {
                        return preparePlanChunkHost(
                            mesh, ordinates, sweep, siDirectionChunks[index],
                            &cacheKeyContext
                        );
                    }
                );
                return;
            }
        };
        DeviceArray<double> angularPsi, phiA, phiB, normValues;
        angularPsi.allocate(angularValueCount);
        phiA.allocate(C);
        phiB.allocate(C);
        normValues.allocate(2);
        checkCuda(cudaMemset(phiA.ptr, 0, static_cast<std::size_t>(C) * sizeof(double)),
                  "clear streaming CUDA SI initial source");

        double* previousPhi = phiA.ptr;
        double* currentPhi = phiB.ptr;
        const bool useSIWavefront = envFlagEnabled("RSI_CUDA_SI_WAVEFRONT", true);
        const bool useSIFusedWavefront =
            envFlagEnabled("RSI_CUDA_SI_FUSED_WAVEFRONT", true);
        const bool useSITiledWavefront =
            envFlagEnabled("RSI_CUDA_SI_TILED_WAVEFRONT", true);
        const bool useSISweepBreakdown =
            envFlagEnabled("RSI_CUDA_SI_SWEEP_BREAKDOWN", false);
        CudaEventTimer siSweepBreakdownTimer;
        for (int iteration = 1; iteration <= maxSIters; ++iteration) {
            gpuTimer.start();
            checkCuda(cudaMemset(currentPhi, 0, static_cast<std::size_t>(C) * sizeof(double)),
                      "clear streaming CUDA SI scalar field");
            siClearSeconds += gpuTimer.stop("time streaming CUDA SI scalar clear");

            for (std::size_t chunkIndex = 0; chunkIndex < siDirectionChunks.size();
                 ++chunkIndex) {
                const auto planStart = std::chrono::steady_clock::now();
                std::unique_ptr<DevicePlanChunk> temporaryChunk;
                DevicePlanChunk* chunk = siDevicePlanCache[chunkIndex].get();
                if (chunk) {
                    ++siDevicePlanCacheHits;
                } else {
                    ++siDevicePlanCacheMisses;
                    PlanChunkTiming chunkPlanTiming;
                    HostPlanChunk hostChunk;
                    if (siHostPlanCache[chunkIndex]) {
                        ++siHostPlanCacheHits;
                        hostChunk = std::move(*siHostPlanCache[chunkIndex]);
                        siHostPlanCache[chunkIndex].reset();
                        siHostPlanCacheBytes -= siHostPlanCacheChunkBytes[chunkIndex];
                        siHostPlanCacheChunkBytes[chunkIndex] = 0;
                    } else if (siPrefetchFuture.valid() && siPrefetchIndex == chunkIndex) {
                        ++siHostPlanCacheMisses;
                        hostChunk = siPrefetchFuture.get();
                        siPrefetchIndex = siDirectionChunks.size();
                    } else {
                        ++siHostPlanCacheMisses;
                        hostChunk = preparePlanChunkHost(
                            mesh, ordinates, sweep, siDirectionChunks[chunkIndex],
                            &cacheKeyContext, &chunkPlanTiming
                        );
                    }
                    const std::size_t hostChunkBytes = hostPlanChunkBytes(hostChunk);
                    const std::size_t projectedDeviceChunkBytes =
                        estimateDevicePlanChunkBytes(hostChunk, C);
                    std::unique_ptr<HostPlanChunk> hostChunkForCache;
                    if (siHostPlanCacheBudget > 0 &&
                        !siHostPlanCache[chunkIndex] &&
                        hostChunkBytes <= siHostPlanCacheBudget &&
                        (iteration > 1 ||
                         siDevicePlanCacheBudget == 0 ||
                         siDevicePlanCacheBytes + projectedDeviceChunkBytes >
                             siDevicePlanCacheBudget)) {
                        hostChunkForCache = std::make_unique<HostPlanChunk>(hostChunk);
                    }
                    const std::size_t chunkBytes = projectedDeviceChunkBytes;
                    const double preUploadAdmissionScore =
                        chunkPlanTiming.cacheSeconds +
                        chunkPlanTiming.buildSeconds +
                        chunkPlanTiming.packSeconds;
                    bool admitDeviceCache = false;
                    if (siDevicePlanCacheBudget > 0 &&
                        chunkBytes <= siDevicePlanCacheBudget &&
                        iteration == 1 &&
                        useSIPlanCostAdmission) {
                        while (siDevicePlanCacheBytes + chunkBytes >
                               siDevicePlanCacheBudget) {
                            std::size_t victim = siDevicePlanCache.size();
                            double victimScore = preUploadAdmissionScore;
                            for (std::size_t candidate = 0;
                                 candidate < siDevicePlanCache.size(); ++candidate) {
                                if (!siDevicePlanCache[candidate]) continue;
                                if (siDevicePlanCacheScore[candidate] < victimScore) {
                                    victim = candidate;
                                    victimScore = siDevicePlanCacheScore[candidate];
                                }
                            }
                            if (victim == siDevicePlanCache.size()) break;
                            siDevicePlanCache[victim].reset();
                            siDevicePlanCacheBytes -= siDevicePlanCacheChunkBytes[victim];
                            siDevicePlanCacheChunkBytes[victim] = 0;
                            siDevicePlanCacheScore[victim] = 0.0;
                        }
                        if (siDevicePlanCacheBytes + chunkBytes <=
                            siDevicePlanCacheBudget) {
                            admitDeviceCache = true;
                        }
                    } else if (siDevicePlanCacheBudget > 0 &&
                               siDevicePlanCacheBytes + chunkBytes <=
                               siDevicePlanCacheBudget) {
                        admitDeviceCache = true;
                    }

                    if (admitDeviceCache) {
                        temporaryChunk = std::make_unique<DevicePlanChunk>();
                        uploadPreparedPlanChunkInto(
                            std::move(hostChunk), *temporaryChunk, &chunkPlanTiming,
                            siPlanUploadStream ? siPlanUploadStream->stream : nullptr,
                            useSIPinnedPlanUpload ? &siPlanUploadStaging : nullptr
                        );
                    } else {
                        uploadPreparedPlanChunkInto(
                            std::move(hostChunk), siTemporaryPlanChunk, &chunkPlanTiming,
                            siPlanUploadStream ? siPlanUploadStream->stream : nullptr,
                            useSIPinnedPlanUpload ? &siPlanUploadStaging : nullptr
                        );
                    }
                    const auto planSyncStart = std::chrono::steady_clock::now();
                    if (siPlanUploadStream) {
                        checkCuda(
                            cudaStreamSynchronize(siPlanUploadStream->stream),
                            "synchronize streaming CUDA SI plan upload stream"
                        );
                    } else {
                        checkCuda(
                            cudaDeviceSynchronize(),
                            "synchronize streaming CUDA SI plan upload"
                        );
                    }
                    chunkPlanTiming.syncSeconds += secondsBetween(
                        planSyncStart, std::chrono::steady_clock::now()
                    );
                    siPlanBreakdown.keySeconds += chunkPlanTiming.keySeconds;
                    siPlanBreakdown.cacheSeconds += chunkPlanTiming.cacheSeconds;
                    siPlanBreakdown.buildSeconds += chunkPlanTiming.buildSeconds;
                    siPlanBreakdown.saveSeconds += chunkPlanTiming.saveSeconds;
                    siPlanBreakdown.assembleSeconds += chunkPlanTiming.assembleSeconds;
                    siPlanBreakdown.packSeconds += chunkPlanTiming.packSeconds;
                    siPlanBreakdown.uploadSeconds += chunkPlanTiming.uploadSeconds;
                    siPlanBreakdown.syncSeconds += chunkPlanTiming.syncSeconds;
                    const double planAdmissionScore =
                        preUploadAdmissionScore +
                        chunkPlanTiming.uploadSeconds +
                        chunkPlanTiming.syncSeconds;
                    if (admitDeviceCache) {
                        siDevicePlanCacheBytes += chunkBytes;
                        siDevicePlanCacheChunkBytes[chunkIndex] = chunkBytes;
                        siDevicePlanCacheScore[chunkIndex] = planAdmissionScore;
                        siDevicePlanCache[chunkIndex] = std::move(temporaryChunk);
                        chunk = siDevicePlanCache[chunkIndex].get();
                    } else {
                        chunk = &siTemporaryPlanChunk;
                    }
                    if (!admitDeviceCache && hostChunkForCache &&
                        siHostPlanCacheBytes + hostChunkBytes <= siHostPlanCacheBudget) {
                        siHostPlanCacheBytes += hostChunkBytes;
                        siHostPlanCacheChunkBytes[chunkIndex] = hostChunkBytes;
                        siHostPlanCache[chunkIndex] = std::move(hostChunkForCache);
                    }
                }
                siPlanSeconds += secondsBetween(planStart, std::chrono::steady_clock::now());
                scheduleSIPrefetch(chunkIndex + 1);

                const int K = chunk->directionCount;
                const int siLevelTileCount = std::max(
                    1,
                    (chunk->maxSweepLevelWidth + levelSweepThreads - 1) / levelSweepThreads
                );
                gpuTimer.start();
                for (int localStart = 0; localStart < K;) {
                    const int localEnd = sameCycleRunEnd(
                        chunk->hostHasCycle, localStart, K
                    );
                    const int directionBatch = localEnd - localStart;
                    if (useSISweepBreakdown) siSweepBreakdownTimer.start();
                    checkCuda(cudaMemset(
                                  angularPsi.ptr, 0,
                                  static_cast<std::size_t>(directionBatch) * C * sizeof(double)
                              ),
                              "clear streaming CUDA SI angular batch");
                    if (useSISweepBreakdown) {
                        siSweepAngularClearSeconds += siSweepBreakdownTimer.stop(
                            "time streaming CUDA SI angular clear"
                        );
                        siSweepBreakdownTimer.start();
                    }
                    if (useSIWavefront && chunk->hostHasCycle[localStart] == 0) {
                        int maxLevelCount = 0;
                        for (int local = localStart; local < localEnd; ++local) {
                            maxLevelCount = std::max(
                                maxLevelCount, chunk->hostLevelCount[local]
                            );
                        }
                        if (useSIFusedWavefront) {
                            sweepLevelFusedKernel<<<directionBatch, levelSweepThreads>>>(
                                staticProblem->mesh.view(C),
                                chunk->ordinateX.ptr,
                                chunk->ordinateY.ptr,
                                chunk->ordinateZ.ptr,
                                chunk->orders.ptr,
                                chunk->levelOffsetBase.ptr,
                                chunk->levelCount.ptr,
                                chunk->levelOffsets.ptr,
                                chunk->directions.ptr + localStart,
                                directionBatch,
                                sourceShapeCode,
                                previousPhi,
                                angularPsi.ptr,
                                true
                            );
                            checkCuda(
                                cudaGetLastError(),
                                "launch streaming CUDA SI fused levels"
                            );
                        } else {
                            for (int level = 0; level < maxLevelCount; ++level) {
                                if (useSITiledWavefront && siLevelTileCount > 1) {
                                    const dim3 grid(directionBatch, siLevelTileCount);
                                    sweepLevelTiledKernel<<<grid, levelSweepThreads>>>(
                                        staticProblem->mesh.view(C),
                                        chunk->ordinateX.ptr,
                                        chunk->ordinateY.ptr,
                                        chunk->ordinateZ.ptr,
                                        chunk->orders.ptr,
                                        chunk->levelOffsetBase.ptr,
                                        chunk->levelCount.ptr,
                                        chunk->levelOffsets.ptr,
                                        chunk->directions.ptr + localStart,
                                        directionBatch,
                                        sourceShapeCode,
                                        previousPhi,
                                        angularPsi.ptr,
                                        true,
                                        level
                                    );
                                } else {
                                    sweepLevelKernel<<<directionBatch, levelSweepThreads>>>(
                                        staticProblem->mesh.view(C),
                                        chunk->ordinateX.ptr,
                                        chunk->ordinateY.ptr,
                                        chunk->ordinateZ.ptr,
                                        chunk->orders.ptr,
                                        chunk->levelOffsetBase.ptr,
                                        chunk->levelCount.ptr,
                                        chunk->levelOffsets.ptr,
                                        chunk->directions.ptr + localStart,
                                        directionBatch,
                                        sourceShapeCode,
                                        previousPhi,
                                        angularPsi.ptr,
                                        true,
                                        level
                                    );
                                }
                                checkCuda(cudaGetLastError(), "launch streaming CUDA SI level");
                            }
                        }
                    } else {
                        const int directionSweepBlocks =
                            (directionBatch + samplesPerSweepBlock - 1) /
                            samplesPerSweepBlock;
                        const int localPassCount =
                            chunk->hostHasCycle[localStart] != 0 ? 20 : 1;
                        for (int localPass = 0; localPass < localPassCount; ++localPass) {
                            sweepSamplesKernel<<<directionSweepBlocks, sweepThreads>>>(
                                staticProblem->mesh.view(C),
                                chunk->ordinateX.ptr,
                                chunk->ordinateY.ptr,
                                chunk->ordinateZ.ptr,
                                chunk->orders.ptr,
                                chunk->hasCycle.ptr,
                                chunk->directions.ptr + localStart,
                                directionBatch,
                                sourceShapeCode,
                                previousPhi,
                                angularPsi.ptr,
                                true,
                                localPass
                            );
                            checkCuda(cudaGetLastError(), "launch streaming CUDA SI sweep");
                        }
                    }
                    if (useSISweepBreakdown) {
                        siSweepKernelSeconds += siSweepBreakdownTimer.stop(
                            "time streaming CUDA SI sweep kernels"
                        );
                        siSweepBreakdownTimer.start();
                    }
                    accumulateDirectionBatchKernel<<<cellBlocks, reductionThreads>>>(
                        angularPsi.ptr,
                        chunk->weights.ptr,
                        chunk->directions.ptr + localStart,
                        directionBatch,
                        C,
                        currentPhi
                    );
                    checkCuda(cudaGetLastError(), "accumulate streaming CUDA SI batch");
                    if (useSISweepBreakdown) {
                        siSweepAccumulateSeconds += siSweepBreakdownTimer.stop(
                            "time streaming CUDA SI accumulate"
                        );
                    }
                    localStart = localEnd;
                }
                const auto siSweepSyncStart = std::chrono::steady_clock::now();
                checkCuda(cudaDeviceSynchronize(), "synchronize streaming CUDA SI chunk");
                if (useSISweepBreakdown) {
                    siSweepSyncSeconds += secondsBetween(
                        siSweepSyncStart, std::chrono::steady_clock::now()
                    );
                }
                siSweepSeconds += gpuTimer.stop("time streaming CUDA SI chunk sweeps");
            }

            gpuTimer.start();
            checkCuda(cudaDeviceSynchronize(), "synchronize streaming CUDA SI reduction");
            siReduceSeconds += gpuTimer.stop("time streaming CUDA SI reduction");

            bool converged = false;
            double relative = 0.0;
            gpuTimer.start();
            if (iteration > 1) {
                checkCuda(cudaMemset(normValues.ptr, 0, 2 * sizeof(double)),
                          "clear streaming CUDA SI norm values");
                relativeNormKernel<<<cellBlocks, reductionThreads>>>(
                    currentPhi, previousPhi, staticProblem->mesh.volume.ptr, C,
                    normValues.ptr, normValues.ptr + 1
                );
                checkCuda(cudaGetLastError(), "launch streaming CUDA SI norm");
                double hostNorms[2] = {0.0, 0.0};
                checkCuda(cudaMemcpy(hostNorms, normValues.ptr, 2 * sizeof(double),
                                     cudaMemcpyDeviceToHost),
                          "copy streaming CUDA SI norm");
                relative = std::sqrt(
                    hostNorms[0] / std::max(hostNorms[1], 1.0e-300)
                );
                converged = relative < siTolerance;
            } else {
                checkCuda(cudaDeviceSynchronize(), "synchronize first streaming CUDA SI iteration");
            }
            siNormSeconds += gpuTimer.stop("time streaming CUDA SI norm");
            std::cerr << "CUDA streaming SI iteration=" << iteration
                      << ", relative=" << relative << "\n";
            std::swap(previousPhi, currentPhi);
            result.convergedN = iteration;
            if (converged) break;
        }

        result.siFine.resize(C);
        const auto siCopyStart = std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(result.siFine.data(), previousPhi,
                             static_cast<std::size_t>(C) * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "copy streaming CUDA SI result");
        siCopySeconds += secondsBetween(siCopyStart, std::chrono::steady_clock::now());
        siTotalSeconds = secondsBetween(siStart, std::chrono::steady_clock::now());
        std::size_t siDevicePlanCachedChunks = 0;
        for (const auto& cachedChunk : siDevicePlanCache) {
            if (cachedChunk) ++siDevicePlanCachedChunks;
        }
        std::cerr << "CUDA streaming SI device plan cache: chunks="
                  << siDevicePlanCachedChunks << "/" << siDirectionChunks.size()
                  << ", bytes=" << siDevicePlanCacheBytes
                  << ", hits=" << siDevicePlanCacheHits
                  << ", misses=" << siDevicePlanCacheMisses << "\n";
        std::size_t siHostPlanCachedChunks = 0;
        for (const auto& cachedChunk : siHostPlanCache) {
            if (cachedChunk) ++siHostPlanCachedChunks;
        }
        std::cerr << "CUDA streaming SI host plan cache: chunks="
                  << siHostPlanCachedChunks << "/" << siDirectionChunks.size()
                  << ", bytes=" << siHostPlanCacheBytes
                  << ", hits=" << siHostPlanCacheHits
                  << ", misses=" << siHostPlanCacheMisses << "\n";
    }

    const auto rsiStart = std::chrono::steady_clock::now();
    const int iterationCount = result.convergedN + tailExtra;
    const auto scheduleStart = std::chrono::steady_clock::now();
    const std::vector<int> schedule =
        generateDirectionSchedule(seed, M, sampleCount, iterationCount);
    const double rsiScheduleSeconds =
        secondsBetween(scheduleStart, std::chrono::steady_clock::now());
    double rsiPlanSeconds = 0.0;
    double rsiUniqueSeconds = 0.0;
    double rsiDirectionCopySeconds = 0.0;
    double rsiSweepSeconds = 0.0;
    double rsiTailAccumSeconds = 0.0;
    double rsiReduceSeconds = 0.0;
    double rsiCopySeconds = 0.0;
    PlanChunkTiming rsiPlanBreakdown;

    DeviceArray<double> globalRSISum, globalTailSum;
    globalRSISum.allocate(C);
    globalTailSum.allocate(C);
    checkCuda(cudaMemset(globalRSISum.ptr, 0, static_cast<std::size_t>(C) * sizeof(double)),
              "clear streaming CUDA RSI sum");
    checkCuda(cudaMemset(globalTailSum.ptr, 0, static_cast<std::size_t>(C) * sizeof(double)),
              "clear streaming CUDA RSI-tail sum");

    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "streaming cudaMemGetInfo");
    const std::size_t bytesPerSample =
        static_cast<std::size_t>(3) * C * sizeof(double) +
        static_cast<std::size_t>(iterationCount) * sizeof(int);
    constexpr int maxSamplesPerBatch = 128;
    const int batchCapacity = static_cast<int>(std::min<std::size_t>(
        std::min(sampleCount, maxSamplesPerBatch),
        std::max<std::size_t>(1, static_cast<std::size_t>(freeBytes * 0.50) /
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
    std::vector<unsigned char> batchIterationHasCycle(iterationCount, 0);
    std::vector<int> batchIterationMaxLevel(iterationCount, 0);
    std::vector<int> globalToLocal(M, -1);
    const bool useRSIBatchPass = envFlagEnabled("RSI_CUDA_RSI_BATCH_PASS", false);
    const bool useRSIWavefront = envFlagEnabled("RSI_CUDA_RSI_WAVEFRONT", true);
    const bool useRSITiledWavefront =
        envFlagEnabled("RSI_CUDA_RSI_TILED_WAVEFRONT", true);

    for (int batchStart = 0; batchStart < sampleCount; batchStart += batchCapacity) {
        const int batchSize = std::min(batchCapacity, sampleCount - batchStart);
        const std::size_t batchCellCount = static_cast<std::size_t>(batchSize) * C;

        const auto uniqueStart = std::chrono::steady_clock::now();
        std::vector<int> uniqueDirections;
        uniqueDirections.reserve(static_cast<std::size_t>(batchSize) * iterationCount);
        for (int localSample = 0; localSample < batchSize; ++localSample) {
            const int sample = batchStart + localSample;
            for (int iteration = 0; iteration < iterationCount; ++iteration) {
                uniqueDirections.push_back(
                    schedule[static_cast<std::size_t>(sample) * iterationCount + iteration]
                );
            }
        }
        std::sort(uniqueDirections.begin(), uniqueDirections.end());
        uniqueDirections.erase(
            std::unique(uniqueDirections.begin(), uniqueDirections.end()),
            uniqueDirections.end()
        );
        rsiUniqueSeconds += secondsBetween(uniqueStart, std::chrono::steady_clock::now());
        const auto planStart = std::chrono::steady_clock::now();
        PlanChunkTiming batchPlanTiming;
        std::unique_ptr<DevicePlanChunk> chunk =
            uploadPlanChunk(
                mesh, ordinates, sweep, uniqueDirections,
                &cacheKeyContext, &batchPlanTiming
            );
        const auto planSyncStart = std::chrono::steady_clock::now();
        checkCuda(cudaDeviceSynchronize(), "synchronize streaming CUDA RSI plan upload");
        batchPlanTiming.syncSeconds += secondsBetween(
            planSyncStart, std::chrono::steady_clock::now()
        );
        rsiPlanBreakdown.cacheSeconds += batchPlanTiming.cacheSeconds;
        rsiPlanBreakdown.keySeconds += batchPlanTiming.keySeconds;
        rsiPlanBreakdown.buildSeconds += batchPlanTiming.buildSeconds;
        rsiPlanBreakdown.saveSeconds += batchPlanTiming.saveSeconds;
        rsiPlanBreakdown.assembleSeconds += batchPlanTiming.assembleSeconds;
        rsiPlanBreakdown.packSeconds += batchPlanTiming.packSeconds;
        rsiPlanBreakdown.uploadSeconds += batchPlanTiming.uploadSeconds;
        rsiPlanBreakdown.syncSeconds += batchPlanTiming.syncSeconds;
        rsiPlanSeconds += secondsBetween(planStart, std::chrono::steady_clock::now());

        for (int local = 0; local < static_cast<int>(uniqueDirections.size()); ++local) {
            globalToLocal[uniqueDirections[local]] = local;
        }
        std::fill(batchIterationHasCycle.begin(), batchIterationHasCycle.end(), 0);
        std::fill(batchIterationMaxLevel.begin(), batchIterationMaxLevel.end(), 0);
        for (int iteration = 0; iteration < iterationCount; ++iteration) {
            for (int localSample = 0; localSample < batchSize; ++localSample) {
                const int globalDirection =
                    schedule[static_cast<std::size_t>(batchStart + localSample) * iterationCount +
                             iteration];
                const int localDirection = globalToLocal[globalDirection];
                batchDirections[static_cast<std::size_t>(iteration) * batchSize + localSample] =
                    localDirection;
                if (chunk->hostHasCycle[localDirection] != 0) {
                    batchIterationHasCycle[iteration] = 1;
                }
                batchIterationMaxLevel[iteration] = std::max(
                    batchIterationMaxLevel[iteration],
                    chunk->hostLevelCount[localDirection]
                );
            }
        }
        for (int globalDirection : uniqueDirections) globalToLocal[globalDirection] = -1;

        const auto directionCopyStart = std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(selectedDirections.ptr, batchDirections.data(),
                             static_cast<std::size_t>(batchSize) * iterationCount * sizeof(int),
                             cudaMemcpyHostToDevice),
                  "copy streaming CUDA RSI directions");
        rsiDirectionCopySeconds += secondsBetween(
            directionCopyStart, std::chrono::steady_clock::now()
        );

        checkCuda(cudaMemset(psiA.ptr, 0, batchCellCount * sizeof(double)),
                  "clear streaming CUDA RSI previous layer");
        checkCuda(cudaMemset(sampleTail.ptr, 0, batchCellCount * sizeof(double)),
                  "clear streaming CUDA RSI-tail accumulation");
        double* previousPsi = psiA.ptr;
        double* currentPsi = psiB.ptr;
        const int sampleSweepBlocks =
            (batchSize + samplesPerSweepBlock - 1) / samplesPerSweepBlock;
        const int levelTileCount = std::max(
            1,
            (chunk->maxSweepLevelWidth + levelSweepThreads - 1) / levelSweepThreads
        );
        for (int iteration = 1; iteration <= iterationCount; ++iteration) {
            checkCuda(cudaMemset(currentPsi, 0, batchCellCount * sizeof(double)),
                      "clear streaming CUDA RSI current layer");
            gpuTimer.start();
            const int directionOffset = iteration - 1;
            const bool iterationHasCycle = batchIterationHasCycle[directionOffset] != 0;
            const int* iterationDirections =
                selectedDirections.ptr + static_cast<std::size_t>(directionOffset) * batchSize;
            const bool useTiledForBatch =
                useRSITiledWavefront && batchSize >= 64 && levelTileCount > 1;
            if (useRSIWavefront && !iterationHasCycle) {
                for (int level = 0; level < batchIterationMaxLevel[directionOffset]; ++level) {
                    if (useTiledForBatch) {
                        const dim3 grid(batchSize, levelTileCount);
                        sweepLevelTiledKernel<<<grid, levelSweepThreads>>>(
                            staticProblem->mesh.view(C),
                            chunk->ordinateX.ptr,
                            chunk->ordinateY.ptr,
                            chunk->ordinateZ.ptr,
                            chunk->orders.ptr,
                            chunk->levelOffsetBase.ptr,
                            chunk->levelCount.ptr,
                            chunk->levelOffsets.ptr,
                            iterationDirections,
                            batchSize,
                            sourceShapeCode,
                            previousPsi,
                            currentPsi,
                            false,
                            level
                        );
                    } else {
                        sweepLevelKernel<<<batchSize, levelSweepThreads>>>(
                            staticProblem->mesh.view(C),
                            chunk->ordinateX.ptr,
                            chunk->ordinateY.ptr,
                            chunk->ordinateZ.ptr,
                            chunk->orders.ptr,
                            chunk->levelOffsetBase.ptr,
                            chunk->levelCount.ptr,
                            chunk->levelOffsets.ptr,
                            iterationDirections,
                            batchSize,
                            sourceShapeCode,
                            previousPsi,
                            currentPsi,
                            false,
                            level
                        );
                    }
                    checkCuda(cudaGetLastError(), "launch streaming CUDA RSI level");
                }
            } else {
                const int localPassCount =
                    useRSIBatchPass && !iterationHasCycle ? 1 : 20;
                for (int localPass = 0; localPass < localPassCount; ++localPass) {
                    sweepSamplesKernel<<<sampleSweepBlocks, sweepThreads>>>(
                        staticProblem->mesh.view(C),
                        chunk->ordinateX.ptr,
                        chunk->ordinateY.ptr,
                        chunk->ordinateZ.ptr,
                        chunk->orders.ptr,
                        chunk->hasCycle.ptr,
                        iterationDirections,
                        batchSize,
                        sourceShapeCode,
                        previousPsi,
                        currentPsi,
                        false,
                        localPass
                    );
                    checkCuda(cudaGetLastError(), "launch streaming CUDA RSI sweep");
                }
            }
            rsiSweepSeconds += gpuTimer.stop("time streaming CUDA RSI sweeps");
            if (iteration == result.convergedN) {
                gpuTimer.start();
                reduceSamplesKernel<<<cellBlocks, reductionThreads>>>(
                    currentPsi, batchSize, C, globalRSISum.ptr
                );
                checkCuda(cudaGetLastError(), "reduce streaming CUDA RSI field");
                rsiReduceSeconds += gpuTimer.stop("time streaming CUDA RSI field reduction");
            }
            if (iteration >= result.convergedN) {
                const int accumulationBlocks = static_cast<int>(
                    (batchCellCount + reductionThreads - 1) / reductionThreads
                );
                gpuTimer.start();
                accumulateValuesKernel<<<accumulationBlocks, reductionThreads>>>(
                    currentPsi, batchCellCount, sampleTail.ptr
                );
                checkCuda(cudaGetLastError(), "accumulate streaming CUDA RSI-tail");
                rsiTailAccumSeconds += gpuTimer.stop("time streaming CUDA RSI-tail accumulation");
            }
            std::swap(previousPsi, currentPsi);
        }
        gpuTimer.start();
        reduceSamplesKernel<<<cellBlocks, reductionThreads>>>(
            sampleTail.ptr, batchSize, C, globalTailSum.ptr
        );
        checkCuda(cudaGetLastError(), "reduce streaming CUDA RSI-tail field");
        rsiReduceSeconds += gpuTimer.stop("time streaming CUDA RSI-tail reduction");
        checkCuda(cudaDeviceSynchronize(), "synchronize streaming CUDA RSI batch");
        std::cerr << "CUDA streaming RSI batch complete: samples="
                  << batchStart + batchSize << "/" << sampleCount
                  << ", unique_directions=" << uniqueDirections.size() << "\n";
    }

    result.rsi.resize(C);
    result.rsiTail.resize(C);
    const auto rsiCopyStart = std::chrono::steady_clock::now();
    checkCuda(cudaMemcpy(result.rsi.data(), globalRSISum.ptr,
                         static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
              "copy streaming CUDA RSI result");
    checkCuda(cudaMemcpy(result.rsiTail.data(), globalTailSum.ptr,
                         static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
              "copy streaming CUDA RSI-tail result");
    rsiCopySeconds += secondsBetween(rsiCopyStart, std::chrono::steady_clock::now());
    const double rsiDenominator = static_cast<double>(sampleCount);
    const double tailDenominator =
        static_cast<double>(sampleCount) * static_cast<double>(tailExtra + 1);
    for (double& value : result.rsi) value /= rsiDenominator;
    for (double& value : result.rsiTail) value /= tailDenominator;

    const double seconds = secondsBetween(totalStart, std::chrono::steady_clock::now());
    const double rsiTotalSeconds = secondsBetween(rsiStart, std::chrono::steady_clock::now());
    std::cout << "CUDA streaming timing: upload=" << uploadSeconds
              << ", total=" << seconds << "\n";
    std::cout << "CUDA streaming timing: si_total=" << siTotalSeconds
              << ", si_plan=" << siPlanSeconds
              << ", si_clear=" << siClearSeconds
              << ", si_sweep=" << siSweepSeconds
              << ", si_reduce=" << siReduceSeconds
              << ", si_norm=" << siNormSeconds
              << ", si_copy=" << siCopySeconds << "\n";
    std::cout << "CUDA streaming timing: si_sweep_breakdown_angular_clear="
              << siSweepAngularClearSeconds
              << ", si_sweep_breakdown_kernel=" << siSweepKernelSeconds
              << ", si_sweep_breakdown_accumulate=" << siSweepAccumulateSeconds
              << ", si_sweep_breakdown_sync=" << siSweepSyncSeconds
              << "\n";
    std::cout << "CUDA streaming timing: si_plan_breakdown_key="
              << siPlanBreakdown.keySeconds
              << ", si_plan_breakdown_cache=" << siPlanBreakdown.cacheSeconds
              << ", si_plan_breakdown_build=" << siPlanBreakdown.buildSeconds
              << ", si_plan_breakdown_save=" << siPlanBreakdown.saveSeconds
              << ", si_plan_breakdown_assemble=" << siPlanBreakdown.assembleSeconds
              << ", si_plan_breakdown_pack=" << siPlanBreakdown.packSeconds
              << ", si_plan_breakdown_upload=" << siPlanBreakdown.uploadSeconds
              << ", si_plan_breakdown_sync=" << siPlanBreakdown.syncSeconds
              << "\n";
    std::cout << "CUDA streaming timing: rsi_total=" << rsiTotalSeconds
              << ", rsi_plan=" << rsiPlanSeconds
              << ", rsi_schedule=" << rsiScheduleSeconds
              << ", rsi_unique=" << rsiUniqueSeconds
              << ", rsi_direction_copy=" << rsiDirectionCopySeconds
              << ", rsi_sweep=" << rsiSweepSeconds
              << ", rsi_tail_accum=" << rsiTailAccumSeconds
              << ", rsi_reduce=" << rsiReduceSeconds
              << ", rsi_copy=" << rsiCopySeconds << "\n";
    std::cout << "CUDA streaming timing: rsi_plan_breakdown_key="
              << rsiPlanBreakdown.keySeconds
              << ", rsi_plan_breakdown_cache=" << rsiPlanBreakdown.cacheSeconds
              << ", rsi_plan_breakdown_build=" << rsiPlanBreakdown.buildSeconds
              << ", rsi_plan_breakdown_save=" << rsiPlanBreakdown.saveSeconds
              << ", rsi_plan_breakdown_assemble=" << rsiPlanBreakdown.assembleSeconds
              << ", rsi_plan_breakdown_pack=" << rsiPlanBreakdown.packSeconds
              << ", rsi_plan_breakdown_upload=" << rsiPlanBreakdown.uploadSeconds
              << ", rsi_plan_breakdown_sync=" << rsiPlanBreakdown.syncSeconds
              << "\n";
    std::cout << "CUDA streaming Figure 5 complete: cells=" << C
              << ", directions=" << M
              << ", SI_iterations=" << result.convergedN
              << ", samples=" << sampleCount
              << ", seconds=" << seconds << "\n";
    return result;
}

std::vector<double> runSIFieldCuda(
    const Mesh& mesh,
    const std::vector<Ordinate>& ordinates,
    const std::vector<SweepPlan>& sweepPlans,
    const std::string& sourceShape,
    int maxSIters,
    double siTolerance,
    int& convergedN
) {
    if (maxSIters <= 0) {
        throw std::runtime_error("invalid CUDA SI iteration configuration");
    }
    std::string unavailableReason;
    if (!cudaRSIAvailable(&unavailableReason)) {
        throw std::runtime_error("CUDA unavailable: " + unavailableReason);
    }

    const auto totalStart = std::chrono::steady_clock::now();
    const auto uploadStart = std::chrono::steady_clock::now();
    std::unique_ptr<DeviceProblem> problem = uploadProblem(mesh, ordinates, sweepPlans);
    checkCuda(cudaDeviceSynchronize(), "synchronize CUDA SI problem upload");
    const double uploadSeconds =
        secondsBetween(uploadStart, std::chrono::steady_clock::now());

    const int C = problem->cellCount;
    const int M = problem->directionCount;
    const int sourceShapeCode = sourceShape == "rectangle" ? 0 : 1;
    constexpr int sweepThreads = 128;
    constexpr int samplesPerSweepBlock = 32;
    constexpr int reductionThreads = 256;
    constexpr int levelSweepThreads = 256;
    const int cellBlocks = (C + reductionThreads - 1) / reductionThreads;

    CudaEventTimer gpuTimer;
    double siClearSeconds = 0.0;
    double siSweepSeconds = 0.0;
    double siReduceSeconds = 0.0;
    double siNormSeconds = 0.0;
    double siCopySeconds = 0.0;

    const int directionsPerBatch = std::max(4, std::min(128, 5000000 / C));
    const std::size_t angularValueCount =
        static_cast<std::size_t>(directionsPerBatch) * C;
    DeviceArray<double> angularPsi, phiA, phiB, normValues;
    angularPsi.allocate(angularValueCount);
    phiA.allocate(C);
    phiB.allocate(C);
    normValues.allocate(2);
    checkCuda(cudaMemset(phiA.ptr, 0, static_cast<std::size_t>(C) * sizeof(double)),
              "clear CUDA SI-only initial source");

    double* previousPhi = phiA.ptr;
    double* currentPhi = phiB.ptr;
    convergedN = maxSIters;
    const bool useSIWavefront = envFlagEnabled("RSI_CUDA_SI_WAVEFRONT", true);
    const bool useSITiledWavefront =
        envFlagEnabled("RSI_CUDA_SI_TILED_WAVEFRONT", true);
    const int siLevelTileCount = std::max(
        1,
        (problem->maxSweepLevelWidth + levelSweepThreads - 1) / levelSweepThreads
    );

    for (int iteration = 1; iteration <= maxSIters; ++iteration) {
        gpuTimer.start();
        checkCuda(cudaMemset(currentPhi, 0, static_cast<std::size_t>(C) * sizeof(double)),
                  "clear CUDA SI-only scalar field");
        siClearSeconds += gpuTimer.stop("time CUDA SI-only angular clear");

        gpuTimer.start();
        for (int directionStart = 0; directionStart < M;) {
            const int batchEnd = std::min(directionStart + directionsPerBatch, M);
            const int runEnd = sameCycleRunEnd(
                problem->hostHasCycle, directionStart, batchEnd
            );
            const int directionBatch = runEnd - directionStart;
            checkCuda(cudaMemset(
                          angularPsi.ptr, 0,
                          static_cast<std::size_t>(directionBatch) * C * sizeof(double)
                      ),
                      "clear CUDA SI-only angular batch");

            if (useSIWavefront && problem->hostHasCycle[directionStart] == 0) {
                int maxLevelCount = 0;
                for (int direction = directionStart; direction < runEnd; ++direction) {
                    maxLevelCount = std::max(
                        maxLevelCount,
                        problem->hostLevelCount[direction]
                    );
                }
                for (int level = 0; level < maxLevelCount; ++level) {
                    if (useSITiledWavefront && siLevelTileCount > 1) {
                        const dim3 grid(directionBatch, siLevelTileCount);
                        sweepLevelTiledKernel<<<grid, levelSweepThreads>>>(
                            problem->mesh.view(C),
                            problem->ordinateX.ptr,
                            problem->ordinateY.ptr,
                            problem->ordinateZ.ptr,
                            problem->orders.ptr,
                            problem->levelOffsetBase.ptr,
                            problem->levelCount.ptr,
                            problem->levelOffsets.ptr,
                            problem->allDirections.ptr + directionStart,
                            directionBatch,
                            sourceShapeCode,
                            previousPhi,
                            angularPsi.ptr,
                            true,
                            level
                        );
                    } else {
                        sweepLevelKernel<<<directionBatch, levelSweepThreads>>>(
                            problem->mesh.view(C),
                            problem->ordinateX.ptr,
                            problem->ordinateY.ptr,
                            problem->ordinateZ.ptr,
                            problem->orders.ptr,
                            problem->levelOffsetBase.ptr,
                            problem->levelCount.ptr,
                            problem->levelOffsets.ptr,
                            problem->allDirections.ptr + directionStart,
                            directionBatch,
                            sourceShapeCode,
                            previousPhi,
                            angularPsi.ptr,
                            true,
                            level
                        );
                    }
                    checkCuda(cudaGetLastError(), "launch CUDA SI-only wavefront sweep level");
                }
            } else {
                const int directionSweepBlocks =
                    (directionBatch + samplesPerSweepBlock - 1) / samplesPerSweepBlock;
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
                        angularPsi.ptr,
                        true,
                        localPass
                    );
                    checkCuda(cudaGetLastError(), "launch CUDA SI-only angular sweep batch");
                }
            }
            accumulateDirectionBatchKernel<<<cellBlocks, reductionThreads>>>(
                angularPsi.ptr,
                problem->weights.ptr,
                problem->allDirections.ptr + directionStart,
                directionBatch,
                C,
                currentPhi
            );
            checkCuda(cudaGetLastError(), "launch CUDA SI-only angular batch accumulation");
            directionStart = runEnd;
        }
        siSweepSeconds += gpuTimer.stop("time CUDA SI-only angular sweeps");

        gpuTimer.start();
        checkCuda(cudaDeviceSynchronize(), "synchronize CUDA SI-only angular reduction");
        siReduceSeconds += gpuTimer.stop("time CUDA SI-only angular reduction");

        bool converged = false;
        double relative = 0.0;
        gpuTimer.start();
        if (iteration > 1) {
            checkCuda(cudaMemset(normValues.ptr, 0, 2 * sizeof(double)),
                      "clear CUDA SI-only norm values");
            relativeNormKernel<<<cellBlocks, reductionThreads>>>(
                currentPhi, previousPhi, problem->mesh.volume.ptr, C,
                normValues.ptr, normValues.ptr + 1
            );
            checkCuda(cudaGetLastError(), "launch CUDA SI-only convergence norm");
            double hostNorms[2] = {0.0, 0.0};
            checkCuda(
                cudaMemcpy(hostNorms, normValues.ptr, 2 * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "copy CUDA SI-only convergence norm"
            );
            relative = std::sqrt(hostNorms[0] / std::max(hostNorms[1], 1.0e-300));
            converged = relative < siTolerance;
        } else {
            checkCuda(cudaDeviceSynchronize(), "synchronize first CUDA SI-only iteration");
        }
        siNormSeconds += gpuTimer.stop("time CUDA SI-only convergence norm");
        std::cerr << "CUDA SI-only iteration=" << iteration
                  << ", relative=" << relative << "\n";

        std::swap(previousPhi, currentPhi);
        convergedN = iteration;
        if (converged) break;
    }

    std::vector<double> phi(C);
    const auto siCopyStart = std::chrono::steady_clock::now();
    checkCuda(
        cudaMemcpy(phi.data(), previousPhi,
                   static_cast<std::size_t>(C) * sizeof(double), cudaMemcpyDeviceToHost),
        "copy CUDA SI-only result"
    );
    siCopySeconds += secondsBetween(siCopyStart, std::chrono::steady_clock::now());

    const double totalSeconds = secondsBetween(totalStart, std::chrono::steady_clock::now());
    std::cout << "CUDA SI-only timing: upload=" << uploadSeconds
              << ", total=" << totalSeconds
              << ", si_clear=" << siClearSeconds
              << ", si_sweep=" << siSweepSeconds
              << ", si_reduce=" << siReduceSeconds
              << ", si_norm=" << siNormSeconds
              << ", si_copy=" << siCopySeconds << "\n";
    std::cout << "CUDA SI-only complete: cells=" << C << ", directions=" << M
              << ", SI_iterations=" << convergedN
              << ", seconds=" << totalSeconds << "\n";
    return phi;
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
    std::vector<unsigned char> batchIterationHasCycle(iterationCount, 0);
    const bool useRSIBatchPass = envFlagEnabled("RSI_CUDA_RSI_BATCH_PASS", false);

    for (int batchStart = 0; batchStart < sampleCount; batchStart += batchCapacity) {
        const int batchSize = std::min(batchCapacity, sampleCount - batchStart);
        std::cerr << "CUDA RSI batch: start=" << batchStart
                  << ", size=" << batchSize << "\n";
        const std::size_t batchCellCount = static_cast<std::size_t>(batchSize) * C;
        checkCuda(cudaMemset(psiA.ptr, 0, batchCellCount * sizeof(double)), "clear CUDA RSI psi A");
        checkCuda(cudaMemset(psiB.ptr, 0, batchCellCount * sizeof(double)), "clear CUDA RSI psi B");
        checkCuda(cudaMemset(sampleAccum.ptr, 0, batchCellCount * sizeof(double)),
                  "clear CUDA RSI batch accumulation");

        std::fill(batchIterationHasCycle.begin(), batchIterationHasCycle.end(), 0);
        for (int iteration = 0; iteration < iterationCount; ++iteration) {
            for (int localSample = 0; localSample < batchSize; ++localSample) {
                const int globalSample = batchStart + localSample;
                const int direction =
                    schedule[static_cast<std::size_t>(globalSample) * iterationCount + iteration];
                batchDirections[static_cast<std::size_t>(iteration) * batchSize + localSample] =
                    direction;
                if (hasCycle[direction] != 0) {
                    batchIterationHasCycle[iteration] = 1;
                }
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
            const int localPassCount =
                useRSIBatchPass && batchIterationHasCycle[iteration] == 0 ? 1 : 20;
            for (int localPass = 0; localPass < localPassCount; ++localPass) {
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
