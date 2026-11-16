# 可视化 PCD 点云文件
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_PCD_PATH = r"D:\LiDAR\boat2_data\test_all\boat1a_voxel_0.02_grid_idw_filled_level.pcd"


def parse_args():
    parser = argparse.ArgumentParser(
        description="显示 PCD 格式点云文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=DEFAULT_PCD_PATH, help="输入 PCD 文件路径")
    parser.add_argument("--inputs", type=str, nargs="*", default=None, help="同时显示的多个 PCD 文件路径")
    parser.add_argument("--window_name", type=str, default="", help="显示窗口标题")
    parser.add_argument("--width", type=int, default=960, help="窗口宽度")
    parser.add_argument("--height", type=int, default=720, help="窗口高度")
    parser.add_argument("--left", type=int, default=100, help="窗口左上角 X")
    parser.add_argument("--top", type=int, default=100, help="窗口左上角 Y")
    parser.add_argument("--clear_colors", action="store_true", help="显示前清除点云颜色")
    parser.add_argument("--show_normals", action="store_true", help="显示法向量")
    return parser.parse_args()


def print_point_cloud_info(pcd: o3d.geometry.PointCloud, name: str = "point cloud"):
    """输出点云基本信息。"""
    points = np.asarray(pcd.points)
    print(f"点云: {name}")
    print(pcd)
    print(f"点数: {points.shape[0]}")
    if points.size == 0:
        return

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    print(f"范围 X: {mins[0]:.2f} ~ {maxs[0]:.2f}")
    print(f"范围 Y: {mins[1]:.2f} ~ {maxs[1]:.2f}")
    print(f"范围 Z: {mins[2]:.2f} ~ {maxs[2]:.2f}")


def show_point_cloud(
    pcd: o3d.geometry.PointCloud,
    window_name: str = "PCD Viewer",
    width: int = 960,
    height: int = 720,
    left: int = 100,
    top: int = 100,
    clear_colors: bool = False,
    show_normals: bool = False,
    print_info: bool = True,
):
    """显示内存中的 Open3D 点云对象。"""
    if len(pcd.points) == 0:
        raise ValueError("输入点云为空，无法显示。")

    pcd_vis = o3d.geometry.PointCloud(pcd)
    if clear_colors:
        pcd_vis.colors = o3d.utility.Vector3dVector()

    if print_info:
        print_point_cloud_info(pcd_vis, window_name)

    print("打开点云可视化窗口（可旋转、缩放、平移）...")
    o3d.visualization.draw_geometries(
        [pcd_vis],
        window_name=window_name,
        width=width,
        height=height,
        left=left,
        top=top,
        point_show_normal=show_normals,
    )


def show_point_clouds(
    point_clouds,
    names=None,
    colors=None,
    window_name: str = "PCD Viewer",
    width: int = 960,
    height: int = 720,
    left: int = 100,
    top: int = 100,
    clear_colors: bool = False,
    show_normals: bool = False,
    print_info: bool = True,
):
    """在同一窗口显示多个 Open3D 点云对象。"""
    if not point_clouds:
        raise ValueError("至少需要一个点云用于显示。")

    default_colors = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.4, 1.0],
        [1.0, 0.7, 0.0],
        [0.8, 0.0, 1.0],
        [0.0, 0.8, 0.8],
    ]
    names = names or [f"point cloud {idx + 1}" for idx in range(len(point_clouds))]
    colors = colors or default_colors

    geometries = []
    for idx, pcd in enumerate(point_clouds):
        if len(pcd.points) == 0:
            raise ValueError(f"第 {idx + 1} 个点云为空，无法显示。")

        pcd_vis = o3d.geometry.PointCloud(pcd)
        if clear_colors or not pcd_vis.has_colors():
            pcd_vis.paint_uniform_color(colors[idx % len(colors)])

        if print_info:
            print_point_cloud_info(pcd_vis, names[idx])
        geometries.append(pcd_vis)

    print("打开多点云可视化窗口（可旋转、缩放、平移）...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=width,
        height=height,
        left=left,
        top=top,
        point_show_normal=show_normals,
    )


def show_point_cloud_file(
    input_path,
    window_name: str = "",
    width: int = 960,
    height: int = 720,
    left: int = 100,
    top: int = 100,
    clear_colors: bool = False,
    show_normals: bool = False,
    print_info: bool = True,
):
    """读取并显示 PCD 点云文件，返回读取到的点云对象。"""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    pcd = o3d.io.read_point_cloud(str(input_path))
    if len(pcd.points) == 0:
        raise ValueError(f"点云为空或读取失败: {input_path}")

    title = window_name or f"{input_path.name} PCD Viewer"
    show_point_cloud(
        pcd,
        window_name=title,
        width=width,
        height=height,
        left=left,
        top=top,
        clear_colors=clear_colors,
        show_normals=show_normals,
        print_info=print_info,
    )
    return pcd


def show_point_cloud_files(
    input_paths,
    window_name: str = "",
    colors=None,
    width: int = 960,
    height: int = 720,
    left: int = 100,
    top: int = 100,
    clear_colors: bool = False,
    show_normals: bool = False,
    print_info: bool = True,
):
    """读取多个 PCD 文件并显示到同一个窗口，返回点云对象列表。"""
    input_paths = [Path(path) for path in input_paths]
    if not input_paths:
        raise ValueError("至少需要一个 PCD 文件用于显示。")

    point_clouds = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")
        pcd = o3d.io.read_point_cloud(str(input_path))
        if len(pcd.points) == 0:
            raise ValueError(f"点云为空或读取失败: {input_path}")
        point_clouds.append(pcd)

    title = window_name or "Multiple PCD Viewer"
    show_point_clouds(
        point_clouds,
        names=[path.name for path in input_paths],
        colors=colors,
        window_name=title,
        width=width,
        height=height,
        left=left,
        top=top,
        clear_colors=clear_colors,
        show_normals=show_normals,
        print_info=print_info,
    )
    return point_clouds


def main():
    args = parse_args()
    if args.inputs:
        show_point_cloud_files(
            args.inputs,
            window_name=args.window_name,
            width=args.width,
            height=args.height,
            left=args.left,
            top=args.top,
            clear_colors=args.clear_colors,
            show_normals=args.show_normals,
        )
        return

    show_point_cloud_file(
        args.input,
        window_name=args.window_name,
        width=args.width,
        height=args.height,
        left=args.left,
        top=args.top,
        clear_colors=args.clear_colors,
        show_normals=args.show_normals,
    )


if __name__ == "__main__":
    main()
