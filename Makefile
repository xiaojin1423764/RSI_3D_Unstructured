CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pthread -Iinclude
SRC = src/Mesh.cpp src/Quadrature.cpp src/TransportSweep.cpp src/RSI.cpp src/main.cpp
TARGET = rsi_unstructured
NVCC ?= nvcc
NVCCARCH ?= native
NVCCFLAGS ?= -std=c++17 -O3 -Iinclude -DRSI_ENABLE_CUDA -arch=$(NVCCARCH) -Xcompiler=-pthread
GPU_TARGET = rsi_unstructured_gpu
GPU_BUILD_DIR = build/gpu
GPU_CORE_OBJ = $(GPU_BUILD_DIR)/Mesh.o $(GPU_BUILD_DIR)/Quadrature.o \
	$(GPU_BUILD_DIR)/TransportSweep.o $(GPU_BUILD_DIR)/RSI.o $(GPU_BUILD_DIR)/CudaRSI.o
SOURCE_SHAPE ?= rectangle

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

gpu: $(GPU_TARGET)

$(GPU_BUILD_DIR):
	mkdir -p $(GPU_BUILD_DIR)

$(GPU_BUILD_DIR)/%.o: src/%.cpp | $(GPU_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/RSI.o: include/RSI.hpp include/TransportSweep.hpp include/Mesh.hpp
$(GPU_BUILD_DIR)/TransportSweep.o: include/TransportSweep.hpp include/Mesh.hpp
$(GPU_BUILD_DIR)/Mesh.o: include/Mesh.hpp include/Types.hpp
$(GPU_BUILD_DIR)/Quadrature.o: include/Quadrature.hpp include/Types.hpp

$(GPU_BUILD_DIR)/CudaRSI.o: src/CudaRSI.cu include/CudaRSI.hpp include/RSI.hpp include/TransportSweep.hpp include/Mesh.hpp | $(GPU_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/main.o: src/main.cpp include/RSI.hpp include/Mesh.hpp | $(GPU_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_TARGET): $(GPU_CORE_OBJ) $(GPU_BUILD_DIR)/main.o
	$(NVCC) $(NVCCFLAGS) $^ -o $(GPU_TARGET)

$(GPU_BUILD_DIR)/gpu_consistency.o: tests/gpu_consistency.cpp include/RSI.hpp include/Mesh.hpp | $(GPU_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

gpu_consistency_test: $(GPU_CORE_OBJ) $(GPU_BUILD_DIR)/gpu_consistency.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test-gpu: gpu_consistency_test
	./gpu_consistency_test

run: $(TARGET)
	mkdir -p Data/csv_data
	./$(TARGET) --source-shape $(SOURCE_SHAPE) --out Data/csv_data/figure2_data.csv Data/gmsh/cells.csv Data/gmsh/faces.csv

run-rec: SOURCE_SHAPE = rectangle
run-rec: run

run-cir: SOURCE_SHAPE = circle
run-cir: run

run-figure5: $(TARGET)
	mkdir -p Data/csv_data
	./$(TARGET) --source-shape $(SOURCE_SHAPE) --figure5-dir Data/csv_data --only figure5 Data/gmsh/cells.csv Data/gmsh/faces.csv

run-rec-figure5: SOURCE_SHAPE = rectangle
run-rec-figure5: run-figure5

run-cir-figure5: SOURCE_SHAPE = circle
run-cir-figure5: run-figure5

plot: plot-figure2

plot-figure2:
	python3 scripts/plot_figures.py --only figure2

tecplot:
	python3 scripts/export_tecplot.py
clean:
	rm -f $(TARGET) $(GPU_TARGET) gpu_consistency_test
	rm -rf $(GPU_BUILD_DIR)
