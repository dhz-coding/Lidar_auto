#点云补全（底部）
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


# =========================
# 参数配置
# =========================
def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对船舱的底面进行补全",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 默认输入路径：可以按需修改
    default_base = Path(__file__).resolve().parent
    default_input = r"D:\LiDAR\boat2_data\test_all\boat1b_denoised_40_5.0.pcd"

    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="输入点云 PCD 文件路径",
    )

    # 直通滤波范围（按你的要求给默认值）
    parser.add_argument("--RANSAC_DIST", type=float, default=0.02, help="距离阈值")
    parser.add_argument("--GRID_RES", type=float, default=0.02, help="补点密度")
    parser.add_argument("--BOTTOM_RATIO", type=float, default=0.08, help="取Z最小的x%点作为底面候选")
    parser.add_argument("--output", type=str, default="", help="底面补全点云输出路径")
    parser.add_argument("--plane_output", type=str, default="", help="底面平面方程 JSON 输出路径")
    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=["none", "bottom", "repaired", "all"],
        default="all",
        help="可视化模式",
    )

    return parser.parse_args()


def extract_bottom_points(pcd, ratio, visualize=False):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("输入点云为空。")
    z_sorted = np.sort(pts[:, 2])
    z_thresh = z_sorted[int(len(z_sorted) * ratio)]
    mask = pts[:, 2] <= z_thresh
    bottom_pts = pts[mask]
    bottom_pcd = o3d.geometry.PointCloud()
    bottom_pcd.points = o3d.utility.Vector3dVector(bottom_pts)
    if visualize:
        o3d.visualization.draw_geometries([bottom_pcd], window_name="Bottom candidate points")

    return bottom_pcd


def fit_plane_and_fill(bottom_pcd, RANSAC_DIST, grid_res):
    # --- RANSAC 拟合平面 ---
    plane_model, inliers = bottom_pcd.segment_plane(
        distance_threshold=RANSAC_DIST,
        ransac_n=3,
        num_iterations=2000
    )
    a, b, c, d = plane_model
    pts = np.asarray(bottom_pcd.points)
    if pts.size == 0:
        raise ValueError("底面候选点云为空。")

    # --- 在 XZ 平面生成规则网格 ---
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_res),
        np.arange(y_min, y_max, grid_res)
    )
    # --- 根据平面方程反算 z ---
    # aX + bY + cZ + d = 0  =>  Z = (-aX - bY - d) / c
    zz = (-a * xx - b * yy - d) / c

    filled_pts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    filled_pcd = o3d.geometry.PointCloud()
    filled_pcd.points = o3d.utility.Vector3dVector(filled_pts)
    return filled_pcd, plane_model, inliers


def save_plane_json(plane_model, output_path, inliers=None, bottom_point_count=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "plane_model": [float(v) for v in plane_model],
        "a": float(plane_model[0]),
        "b": float(plane_model[1]),
        "c": float(plane_model[2]),
        "d": float(plane_model[3]),
    }
    if inliers is not None:
        report["inlier_count"] = int(len(inliers))
    if bottom_point_count is not None:
        report["bottom_point_count"] = int(bottom_point_count)
        if bottom_point_count > 0 and inliers is not None:
            report["inlier_ratio"] = float(len(inliers) / bottom_point_count)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"底面平面方程已保存到: {output_path}")
    return report


def fit_bottom_plane_file(
    input_path,
    output_path=None,
    plane_output=None,
    ransac_dist=0.02,
    grid_res=0.02,
    bottom_ratio=0.08,
    visualize=False,
):
    input_path = Path(input_path)
    pcd = o3d.io.read_point_cloud(str(input_path))
    print("提取底部点...")
    bottom_pcd = extract_bottom_points(pcd, bottom_ratio, visualize=visualize)
    print("RANSAC拟合底面并补全...")
    filled_pcd, plane_model, inliers = fit_plane_and_fill(bottom_pcd, ransac_dist, grid_res)
    print(f"底面平面方程: {plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")

    repaired = pcd + filled_pcd
    if output_path is None or str(output_path) == "":
        output_path = input_path.parent / f"{input_path.stem}_Bottomed.pcd"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(output_path), repaired, write_ascii=True)
    if not ok:
        raise RuntimeError(f"点云保存失败: {output_path}")
    print(f"结果已保存到: {output_path}")

    plane_report = None
    if plane_output is not None and str(plane_output) != "":
        plane_report = save_plane_json(
            plane_model,
            plane_output,
            inliers=inliers,
            bottom_point_count=len(bottom_pcd.points),
        )
    return repaired, plane_model, plane_report, output_path


def main():
    args = parse_args()
    repaired, _, _, _ = fit_bottom_plane_file(
        args.input,
        output_path=args.output,
        plane_output=args.plane_output,
        ransac_dist=args.RANSAC_DIST,
        grid_res=args.GRID_RES,
        bottom_ratio=args.BOTTOM_RATIO,
        visualize=args.vis_mode in {"bottom", "all"},
    )
    if args.vis_mode in {"repaired", "all"}:
        o3d.visualization.draw_geometries([repaired], window_name="Bottom plane filled point cloud")
    else:
        print("跳过补全点云可视化")

if __name__ == "__main__":
    main()
