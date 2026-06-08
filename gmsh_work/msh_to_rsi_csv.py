import sys
import numpy as np
import meshio
from collections import defaultdict

if len(sys.argv) != 4:
    print("用法: python3 msh_to_rsi_csv.py input.msh cells.csv faces.csv")
    sys.exit(1)

msh_file, cells_csv, faces_csv = sys.argv[1], sys.argv[2], sys.argv[3]

mesh = meshio.read(msh_file)
points = mesh.points[:, :3]

tets = None
for block in mesh.cells:
    if block.type == "tetra":
        tets = block.data
        break

if tets is None:
    raise RuntimeError("没有找到 tetra 四面体单元，请确认 gmsh 用了 -3")

def tet_volume(p0, p1, p2, p3):
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

def tri_area_normal(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    area = np.linalg.norm(n) / 2.0
    if area == 0:
        raise RuntimeError("发现零面积三角面")
    return area, n / np.linalg.norm(n)

cell_centers = []
cell_volumes = []

for tet in tets:
    ps = points[tet]
    cell_centers.append(ps.mean(axis=0))
    cell_volumes.append(tet_volume(ps[0], ps[1], ps[2], ps[3]))

cell_centers = np.array(cell_centers)
cell_volumes = np.array(cell_volumes)

# 写 cells.csv
# Example 1: sigma_t=1, sigma_s=0.5, q=0
with open(cells_csv, "w") as f:
    f.write("cell_id,cx,cy,cz,volume,sigma_t,sigma_s,q\n")
    for i, c in enumerate(cell_centers):
        f.write(f"{i},{c[0]},{c[1]},{c[2]},{cell_volumes[i]},1.0,0.5,0.0\n")

# 四面体的四个面，按顶点编号组合
tet_faces = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
]

face_map = defaultdict(list)

for ci, tet in enumerate(tets):
    for loc in tet_faces:
        face_nodes = tuple(tet[i] for i in loc)
        key = tuple(sorted(face_nodes))
        face_map[key].append((ci, face_nodes))

with open(faces_csv, "w") as f:
    f.write("face_id,left_cell,right_cell,nx,ny,nz,area,fx,fy,fz,bc_type,bc_value\n")

    fid = 0
    for key, owners in face_map.items():
        face_nodes = owners[0][1]
        p0, p1, p2 = points[list(face_nodes)]
        area, n = tri_area_normal(p0, p1, p2)
        center = (p0 + p1 + p2) / 3.0

        left = owners[0][0]
        right = owners[1][0] if len(owners) == 2 else -1

        # 让法向量从 left 指向 right；边界面则从 left 指向外部
        left_center = cell_centers[left]
        if right >= 0:
            target = cell_centers[right] - left_center
            bc_type = "internal"
            bc_value = 0.0
        else:
            target = center - left_center
            bc_type = "example1"
            bc_value = 0.0

        if np.dot(n, target) < 0:
            n = -n

        f.write(
            f"{fid},{left},{right},"
            f"{n[0]},{n[1]},{n[2]},"
            f"{area},{center[0]},{center[1]},{center[2]},"
            f"{bc_type},{bc_value}\n"
        )
        fid += 1

print(f"写出 {cells_csv}")
print(f"写出 {faces_csv}")
print(f"cells = {len(tets)}, faces = {len(face_map)}")
