#统计滤波去除噪声
import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用统计离群点去除法(Statistical Outlier Removal)对点云进行去噪",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 默认输入路径：可以按需修改
    default_base = Path(__file__).resolve().parent
    default_input = r"D:\LiDAR\boat2_data\test_all\boat1a_denoised_40_5.0_autoptf2_crop.pcd"

    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="输入点云 PCD 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出点云 PCD 文件路径",
    )
    parser.add_argument(
        "--nb_neighbors",
        type=int,
        default=60,
        help="统计滤波中用于计算均值距离的邻域点数 k，数值越大统计越稳定但计算更慢",
    )
    parser.add_argument(
        "--std_ratio",
        type=float,
        default=3.5,
        help="标准差倍数阈值，值越小去噪越严格(移除更多点)",
    )
    # 可视化模式：none(不显示), before_after(前后对比), after_only(只显示去噪后)
    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=["none", "before_after", "after_only"],
        default="after_only",
        help=(
            "可视化模式: 'none' 不显示; 'before_after' 显示去噪前后对比; "
            "'after_only' 只显示去噪后的点云"
        ),
    )

    return parser.parse_args()


def statistical_denoise(
    pcd: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """使用 Open3D 的统计离群点去除算法进行去噪。

    - 对每个点，计算其 k 近邻平均距离；
    - 统计所有点的平均距离分布，计算全局均值和标准差；
    - 若某点的平均距离 > 均值 + std_ratio * 标准差，则认为是离群点并移除。
    """
    print("开始统计滤波去噪...")
    print(f"  nb_neighbors = {nb_neighbors}, std_ratio = {std_ratio}")

    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    num_original = np.asarray(pcd.points).shape[0]
    num_clean = np.asarray(cl.points).shape[0]
    num_removed = num_original - num_clean

    print(f"原始点数: {num_original}")
    print(f"保留点数: {num_clean}")
    print(f"移除点数: {num_removed}")
    if num_original > 0:
        ratio = num_clean / num_original
        print(f"保留比例: {ratio:.4f} ( {ratio*100:.2f}% )")

    return cl, np.asarray(ind, dtype=int)


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str) -> None:
    """保存去噪后的点云到指定路径。"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 ASCII 方便检查，也可以改为二进制
    success = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=True)
    if not success:
        raise RuntimeError(f"点云保存失败: {out_path}")
    print(f"去噪结果已保存到: {out_path}")


def build_default_output_path(input_path: Path, nb_neighbors: int, std_ratio: float) -> Path:
    stem = input_path.stem
    prefix = stem.split("_")[0] if "_" in stem else stem
    std_str = str(std_ratio)
    return input_path.parent / f"{prefix}_denoised_{nb_neighbors}_{std_str}.pcd"


def statistical_denoise_file(
    input_path,
    output_path=None,
    nb_neighbors: int = 60,
    std_ratio: float = 3.5,
):
    input_path = Path(input_path)
    if output_path is None or str(output_path) == "":
        output_path = build_default_output_path(input_path, nb_neighbors, std_ratio)
    output_path = Path(output_path)

    pcd = o3d.io.read_point_cloud(str(input_path))
    if len(pcd.points) == 0:
        raise ValueError(f"输入点云为空或读取失败: {input_path}")
    print("成功读取点云文件")
    print(f"点数: {np.asarray(pcd.points).shape[0]}")

    pcd_clean, ind = statistical_denoise(
        pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    save_point_cloud(pcd_clean, str(output_path))
    return pcd_clean, ind, output_path


def visualize_before_after(
    pcd_original: o3d.geometry.PointCloud, pcd_clean: o3d.geometry.PointCloud
) -> None:
    """简单可视化去噪前后的点云对比。

    - 原始点云显示为红色
    - 去噪后点云显示为绿色
    """
    if len(pcd_original.points) == 0:
        print("原始点云为空，跳过可视化。")
        return
    if len(pcd_clean.points) == 0:
        print("去噪后点云为空，跳过可视化。")
        return

    pcd_orig_vis = o3d.geometry.PointCloud(pcd_original)
    pcd_clean_vis = o3d.geometry.PointCloud(pcd_clean)

    # 给原始点云上色为红色
    pcd_orig_vis.paint_uniform_color([1.0, 0.0, 0.0])
    # 去噪点云上色为绿色
    pcd_clean_vis.paint_uniform_color([0.0, 1.0, 0.0])

    print("打开可视化窗口：红色=去噪前, 绿色=去噪后")
    o3d.visualization.draw_geometries(
        [pcd_orig_vis, pcd_clean_vis],
        window_name="Statistical Denoising: Before (Red) vs After (Green)",
    )


def main():
    args = parse_args()

    # 根据输入文件名自动生成输出文件名：
    # <input_stem>_denoised_<nb_neighbors>_<std_ratio>.pcd
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else build_default_output_path(
        input_path, args.nb_neighbors, args.std_ratio
    )
    pcd = o3d.io.read_point_cloud(str(input_path))
    pcd_clean, _ = statistical_denoise_file(
        input_path,
        output_path=output_path,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )

    # 4. 可视化（根据 vis_mode 选择不同方式）
    if args.vis_mode == "before_after":
        print("显示去噪前（红色）后（绿色）的点云...")
        visualize_before_after(pcd, pcd_clean)
    elif args.vis_mode == "after_only":
        print("只显示去噪后的点云（绿色）...")
        pcd_clean_vis = o3d.geometry.PointCloud(pcd_clean)
        pcd_clean_vis.paint_uniform_color([0.0, 1.0, 0.0])
        o3d.visualization.draw_geometries(
            [pcd_clean_vis],
            window_name="After Statistical Denoising (Green): "
        )
    else:  # "none"
        print("跳过可视化显示")


if __name__ == "__main__":
    main()
