#体素质心下采样
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对去噪后的点云进行体素质心下采样",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    base_dir = Path(__file__).resolve().parent
    default_input = r"D:\LiDAR\boat2_data\test_all\boat1a_denoised_60_3.5.pcd"

    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="输入点云PCD文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出点云 PCD 文件路径",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="体素大小，单位与点云坐标单位一致(例如米)",
    )
    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=["none", "before_after", "after_only"],
        default="after_only",
        help=(
            "可视化模式: 'none' 不显示; 'before_after' 显示下采样前后对比; "
            "'after_only' 只显示下采样后的点云"
        ),
    )

    return parser.parse_args()


def voxel_downsample_centroid(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> o3d.geometry.PointCloud:
    """对点云进行体素下采样（体素质心）。

    Open3D 的 voxel_down_sample 会将落入同一体素中的点合并为一个点，
    该点位置为该体素内所有点的质心。
    """
    print("开始体素质心下采样...")
    print(f"  voxel_size = {voxel_size}")

    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)

    n_before = np.asarray(pcd.points).shape[0]
    n_after = np.asarray(pcd_ds.points).shape[0]
    n_removed = n_before - n_after
    print(f"原始点数: {n_before}")
    print(f"保留后点数: {n_after}")
    print(f"移除点数: {n_removed}")
    if n_before > 0:
        ratio = n_after / n_before
        print(f"保留比例: {ratio:.4f} ( {ratio*100:.2f}% )")

    return pcd_ds


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str) -> None:
    """保存下采样后的点云到指定路径。"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=True)
    if not ok:
        raise RuntimeError(f"点云保存失败: {out_path}")
    print(f"体素下采样结果已保存到: {out_path}")


def build_default_output_path(input_path: Path, voxel_size: float) -> Path:
    stem = input_path.stem
    prefix = stem.split("_")[0] if "_" in stem else stem
    voxel_str = str(voxel_size)
    return input_path.parent / f"{prefix}_voxel_{voxel_str}.pcd"


def voxel_downsample_file(input_path, output_path=None, voxel_size: float = 0.02):
    input_path = Path(input_path)
    if output_path is None or str(output_path) == "":
        output_path = build_default_output_path(input_path, voxel_size)
    output_path = Path(output_path)

    pcd = o3d.io.read_point_cloud(str(input_path))
    if len(pcd.points) == 0:
        raise ValueError(f"输入点云为空或读取失败: {input_path}")
    pcd_ds = voxel_downsample_centroid(pcd, voxel_size=voxel_size)
    save_point_cloud(pcd_ds, str(output_path))
    return pcd_ds, output_path


def visualize_before_after(
    pcd_original: o3d.geometry.PointCloud,
    pcd_down: o3d.geometry.PointCloud,
    window_name: str,
) -> None:
    """可视化体素下采样前后对比：原始为红色，下采样结果为绿色。"""
    if len(pcd_original.points) == 0:
        print("原始点云为空，跳过可视化。")
        return
    if len(pcd_down.points) == 0:
        print("下采样后点云为空，跳过可视化。")
        return

    pcd_orig_vis = o3d.geometry.PointCloud(pcd_original)
    pcd_down_vis = o3d.geometry.PointCloud(pcd_down)

    pcd_orig_vis.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    pcd_down_vis.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色

    print("打开可视化窗口：红色=下采样前, 绿色=下采样后")
    o3d.visualization.draw_geometries(
        [pcd_orig_vis, pcd_down_vis],
        window_name=window_name,
    )


def main():
    args = parse_args()

    input_path = Path(args.input)
    out_path = Path(args.output) if args.output else build_default_output_path(
        input_path, args.voxel_size
    )

    # 1. 读取点云
    pcd = o3d.io.read_point_cloud(str(input_path))

    # 2. 体素下采样
    pcd_ds, _ = voxel_downsample_file(
        input_path,
        output_path=out_path,
        voxel_size=args.voxel_size,
    )

    # 4. 可视化
    if args.vis_mode == "before_after":
        win_title = f"{input_path.name} Voxel Downsample Viewer"
        visualize_before_after(pcd, pcd_ds, window_name=win_title)
    elif args.vis_mode == "after_only":
        print("只显示体素下采样后的点云（绿色）...")
        pcd_ds_vis = o3d.geometry.PointCloud(pcd_ds)
        pcd_ds_vis.paint_uniform_color([0.0, 1.0, 0.0])
        win_title = f"{input_path.name} Voxel Downsample (After Only)"
        o3d.visualization.draw_geometries([pcd_ds_vis], window_name=win_title)
    else:
        print("跳过可视化显示")


if __name__ == "__main__":
    main()
