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
	./$(TARGET) gmsh_work/data/cells.csv gmsh_work/data/faces.csv examples/csv_data/figure2_data.csv $(SOURCE_SHAPE)
clean:
	rm -f $(TARGET)
