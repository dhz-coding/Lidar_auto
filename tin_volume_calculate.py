# 三角网格计算体积（法向高度）
import argparse
import json
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
import open3d as o3d


DEFAULT_PC_PATH = Path(r"G:\sharef\boat2_after_2026-01-19\boat2_3\output\boat2_voxel_0.02.pcd")
DEFAULT_PLANE = [-0.0022, 0.0298, 0.9996, 8.4943]
MAX_EDGE = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="通过泥沙表面 TIN 到船舱底面平面的高度积分计算体积",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_PC_PATH), help="泥沙表面点云 PCD 文件路径")
    parser.add_argument(
        "--plane",
        type=float,
        nargs=4,
        default=DEFAULT_PLANE,
        metavar=("A", "B", "C", "D"),
        help="底面平面方程 ax + by + cz + d = 0",
    )
    parser.add_argument("--plane_json", type=str, default="", help="从 JSON 文件读取底面平面参数")
    parser.add_argument("--max_edge", type=float, default=MAX_EDGE, help="TIN 三角形最大边长")
    parser.add_argument("--mesh_output", type=str, default="", help="输出 TIN 网格 PLY 路径")
    parser.add_argument("--report_output", type=str, default="", help="输出体积报告 JSON 路径")
    parser.add_argument(
        "--vis_mode",
        choices=["none", "mesh"],
        default="mesh",
        help="可视化模式",
    )
    return parser.parse_args()


def load_plane_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "plane_model" in data:
            return data["plane_model"]
        if "plane" in data:
            return data["plane"]
        if all(key in data for key in ("a", "b", "c", "d")):
            return [data["a"], data["b"], data["c"], data["d"]]
    if isinstance(data, list) and len(data) == 4:
        return data
    raise ValueError(f"无法从平面 JSON 读取 [a, b, c, d]: {path}")


def calculate_tin_volume(
    pcd: o3d.geometry.PointCloud,
    plane_model,
    max_edge: float = MAX_EDGE,
):
    """计算点云 TIN 到底面平面的体积，返回体积、网格和统计报告。"""
    pts = np.asarray(pcd.points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        raise ValueError("体积计算至少需要 3 个有效点。")

    plane = np.asarray(plane_model, dtype=np.float64)
    if plane.shape != (4,):
        raise ValueError("plane_model 必须是 [a, b, c, d]。")

    normal = plane[:3]
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 0:
        raise ValueError("平面法向量非法。")

    heights = (pts @ normal + plane[3]) / normal_norm
    if np.mean(heights) < 0:
        heights = -heights

    valid = heights > 0
    pts = pts[valid]
    heights = heights[valid]
    if pts.shape[0] < 3:
        raise ValueError("底面上方有效点少于 3 个，无法计算体积。")

    print("Loaded points:", pts.shape)
    print("Points above bottom plane:", pts.shape[0])

    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
    triangles = tri.triangles
    print("Initial triangles:", triangles.shape[0])

    p0 = pts[triangles[:, 0]]
    p1 = pts[triangles[:, 1]]
    p2 = pts[triangles[:, 2]]

    e01 = np.linalg.norm(p1[:, :2] - p0[:, :2], axis=1)
    e12 = np.linalg.norm(p2[:, :2] - p1[:, :2], axis=1)
    e20 = np.linalg.norm(p0[:, :2] - p2[:, :2], axis=1)

    good = (e01 < max_edge) & (e12 < max_edge) & (e20 < max_edge)
    triangles = triangles[good]
    p0 = p0[good]
    p1 = p1[good]
    p2 = p2[good]
    print("Triangles kept:", triangles.shape[0])

    v1 = p1[:, :2] - p0[:, :2]
    v2 = p2[:, :2] - p0[:, :2]
    area_xy = 0.5 * np.abs(
        v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    )

    h0 = heights[triangles[:, 0]]
    h1 = heights[triangles[:, 1]]
    h2 = heights[triangles[:, 2]]
    avg_h = (h0 + h1 + h2) / 3.0
    volume = float(np.sum(area_xy * avg_h))

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(pts),
        triangles=o3d.utility.Vector3iVector(triangles.astype(np.int32)),
    )
    mesh.compute_vertex_normals()

    report = {
        "volume_m3": volume,
        "points_used": int(pts.shape[0]),
        "initial_triangles": int(tri.triangles.shape[0]),
        "triangles_kept": int(triangles.shape[0]),
        "max_edge": float(max_edge),
        "plane_model": plane.tolist(),
    }
    return volume, mesh, report


def calculate_tin_volume_file(
    input_path,
    plane_model,
    max_edge: float = MAX_EDGE,
    mesh_output=None,
    report_output=None,
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    pcd = o3d.io.read_point_cloud(str(input_path))
    volume, mesh, report = calculate_tin_volume(pcd, plane_model, max_edge=max_edge)

    if mesh_output is None or str(mesh_output) == "":
        mesh_output = input_path.with_suffix(".tin_plane_corrected.ply")
    mesh_output = Path(mesh_output)
    mesh_output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(mesh_output), mesh)
    print("Saved mesh:", mesh_output)

    if report_output is not None and str(report_output) != "":
        report_output = Path(report_output)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        with open(report_output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("Saved report:", report_output)

    print(f"\nEstimated Volume (TIN + plane-normal) = {volume:.3f} m^3\n")
    return volume, mesh, report


def main():
    args = parse_args()
    plane_model = load_plane_from_json(args.plane_json) if args.plane_json else args.plane
    _, mesh, _ = calculate_tin_volume_file(
        args.input,
        plane_model=plane_model,
        max_edge=args.max_edge,
        mesh_output=args.mesh_output,
        report_output=args.report_output,
    )
    if args.vis_mode == "mesh":
        o3d.visualization.draw_geometries([mesh], window_name="TIN (Plane-corrected)")
    else:
        print("跳过可视化显示")


if __name__ == "__main__":
    main()
