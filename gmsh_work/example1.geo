SetFactory("OpenCASCADE");

// 单位立方体 [0,1]^3
Box(1) = {0, 0, 0, 1, 1, 1};

// 网格尺寸，越小网格越密
Mesh.CharacteristicLengthMin = 0.061;
Mesh.CharacteristicLengthMax = 0.061;

Mesh.Algorithm3D = 4;

// 生成三维四面体网格
Mesh 3;
