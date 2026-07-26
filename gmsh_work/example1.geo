SetFactory("OpenCASCADE");

// 单位立方体 [0,1]^3
Box(1) = {0, 0, 0, 1, 1, 1};

// 网格尺寸，越小网格越密。可用 gmsh -setnumber meshSize <h> 覆盖，
// 以生成性能基准所需的独立网格而不改动默认 30k 输入。
DefineConstant[
  meshSize = {0.061, Name "Benchmark/Mesh size"}
];
Mesh.CharacteristicLengthMin = meshSize;
Mesh.CharacteristicLengthMax = meshSize;

Mesh.Algorithm3D = 4;

// 生成三维四面体网格
Mesh 3;
