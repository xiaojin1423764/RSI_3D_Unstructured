SetFactory("OpenCASCADE");

// Unit cube [0,1]^3.
Box(1) = {0, 0, 0, 1, 1, 1};

// Target about 200k tetrahedra.
Mesh.CharacteristicLengthMin = 0.033;
Mesh.CharacteristicLengthMax = 0.033;

Mesh.Algorithm3D = 4;

Mesh 3;
