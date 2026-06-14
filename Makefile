CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
SRC = src/Mesh.cpp src/Quadrature.cpp src/TransportSweep.cpp src/RSI.cpp src/main.cpp
TARGET = rsi_unstructured
SOURCE_SHAPE ?= rectangle

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	mkdir -p examples/csv_data
	./$(TARGET) --source-shape $(SOURCE_SHAPE) --out examples/csv_data/figure2_data.csv gmsh_work/data/cells.csv gmsh_work/data/faces.csv

run-rec: SOURCE_SHAPE = rectangle
run-rec: run

run-cir: SOURCE_SHAPE = circle
run-cir: run

run-figure5: $(TARGET)
	mkdir -p examples/csv_data
	./$(TARGET) --source-shape $(SOURCE_SHAPE) --only figure5 gmsh_work/data/cells.csv gmsh_work/data/faces.csv

run-rec-figure5: SOURCE_SHAPE = rectangle
run-rec-figure5: run-figure5

run-cir-figure5: SOURCE_SHAPE = circle
run-cir-figure5: run-figure5

plot:
	python3 examples/plot_figures.py --only figure5

plot-voxel3d:
	python3 examples/plot_figures.py --only voxel3d

plot-all:
	python3 examples/plot_figures.py
clean:
	rm -f $(TARGET)
