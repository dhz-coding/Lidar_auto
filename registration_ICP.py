#ICP配准
import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 ICP 对源点云和目标点云进行配准",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    base_dir = Path(__file__).resolve().parent
    sdir = r"C:\Users\xm\Desktop\boatdata\boat2_noise\boat2_voxel_0.02_o5_fpfh_registered_source.pcd"
    tdir = r"C:\Users\xm\Desktop\boatdata\boat2_noise\boat1_voxel_0.02.pcd"
    parser.add_argument(
        "--source",
        type=str,
        default=str(sdir),
        help="源点云路径",

    )
    parser.add_argument(
        "--target",
        type=str,
        default=str(tdir),
        help="目标点云路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="ICP 最大对应点距离阈值",
    )
    parser.add_argument(
        "--max_iteration",
        type=int,
        default=500,
        help="ICP 最大迭代次数",
    )
    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=["none", "merged"],
        default="merged",
        help="可视化模式: 'none' 不显示; 'merged' 显示配准后叠加结果",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出点云PCD文件路径（默认自动生成）",
    )
    parser.add_argument(
        "--output_all",
        type=bool,
        default=True,
        help="是否额外保存合并后的点云（默认保存）",
    )

    return parser.parse_args()


def load_point_clouds(source_path: str, target_path: str):
    """读取源点云和目标点云，并进行基础校验。"""
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    if len(source.points) == 0:
        raise ValueError(f"源点云为空或读取失败: {source_path}")
    if len(target.points) == 0:
        raise ValueError(f"目标点云为空或读取失败: {target_path}")

    print(f"源点云: {source_path}")
    print(f"目标点云: {target_path}")
    print(f"源点数: {len(source.points)}")
    print(f"目标点数: {len(target.points)}")
    return source, target


def run_icp_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float,
    max_iteration: int,
):
    """执行 ICP 配准并返回结果与耗时。"""
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration
    )

    start = time.perf_counter()
    reg_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )
    elapsed = time.perf_counter() - start

    return reg_icp, elapsed





def print_icp_report(reg_icp, elapsed: float, max_iteration: int):
    """输出 ICP 关键统计信息。"""

    print("\n=== ICP 结果 ===")

    print(f"耗时: {elapsed:.6f} 秒")
    print(f"匹配率: {reg_icp.fitness:.6f}")
    print(f"inlier_rmse: {reg_icp.inlier_rmse:.6f}")    #内点对应误差的均方根

    corr_count = len(reg_icp.correspondence_set)
    print(f"对应点对数量: {corr_count}")

    print("变换矩阵:")
    print(reg_icp.transformation)


def visualize_registered_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    window_name: str,
):
    """可视化配准后的叠加结果：源点云红色，目标点云绿色。"""
    source_vis = o3d.geometry.PointCloud(source)
    target_vis = o3d.geometry.PointCloud(target)

    source_vis.transform(transformation)
    source_vis.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    target_vis.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色

    print("打开可视化窗口：红色=配准后源点云, 绿色=目标点云")
    o3d.visualization.draw_geometries(
        [source_vis, target_vis],
        window_name=window_name,
    )


def build_merged_cloud(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
):
    """构建合并点云：先将源点云变换到目标坐标系，再与目标点云拼接。"""
    source_registered = o3d.geometry.PointCloud(source)
    target_copy = o3d.geometry.PointCloud(target)

    source_registered.transform(transformation)
    merged = source_registered + target_copy
    return merged


def refine_register_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float = 0.05,
    max_iteration: int = 500,
):
    """执行 ICP 精配准，返回变换后的 source、ICP 矩阵和统计报告。"""
    reg_icp, elapsed = run_icp_registration(
        source=source,
        target=target,
        threshold=threshold,
        max_iteration=max_iteration,
    )
    registered_source = o3d.geometry.PointCloud(source)
    registered_source.transform(reg_icp.transformation)
    report = {
        "elapsed": elapsed,
        "fitness": float(reg_icp.fitness),
        "inlier_rmse": float(reg_icp.inlier_rmse),
        "correspondence_count": int(len(reg_icp.correspondence_set)),
    }
    return registered_source, reg_icp.transformation, report


def refine_register_file(
    source_path,
    target_path,
    output_path=None,
    output_all=True,
    threshold: float = 0.05,
    max_iteration: int = 500,
):
    """文件级 ICP 精配准接口，供流水线或命令行复用。"""
    sourcedir = Path(source_path)
    targetdir = Path(target_path)
    source, target = load_point_clouds(sourcedir, targetdir)
    registered_source, transformation, report = refine_register_point_clouds(
        source=source,
        target=target,
        threshold=threshold,
        max_iteration=max_iteration,
    )

    if output_path is None or str(output_path) == "":
        output_path = sourcedir.parent / f"{sourcedir.stem}_ICP_registered_source.pcd"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_point_cloud(str(output_path), registered_source)
    if not success:
        raise IOError(f"配准后的源点云保存失败: {output_path}")
    print(f"配准后的源点云已保存: {output_path}")

    merged_output_path = None
    if output_all:
        merged_output_path = sourcedir.parent / f"{sourcedir.stem}_ICP_merged.pcd"
        merged = build_merged_cloud(source, target, transformation)
        merged_success = o3d.io.write_point_cloud(str(merged_output_path), merged)
        if not merged_success:
            raise IOError(f"合并点云保存失败: {merged_output_path}")
        print(f"合并点云已保存: {merged_output_path}")

    return registered_source, transformation, report, output_path, merged_output_path



def main():
    args = parse_args()
    sourcedir = Path(args.source)
    targetdir = Path(args.target)

    source, target = load_point_clouds(sourcedir, targetdir)

    reg_icp, elapsed = run_icp_registration(
        source=source,
        target=target,
        threshold=args.threshold,
        max_iteration=args.max_iteration,
    )

    print_icp_report(reg_icp, elapsed=elapsed, max_iteration=args.max_iteration)

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = sourcedir.parent / f"{sourcedir.stem}_ICP_registered_source.pcd"

    registered_source = o3d.geometry.PointCloud(source)
    registered_source.transform(reg_icp.transformation)

    success = o3d.io.write_point_cloud(str(output_path), registered_source)
    if not success:
        raise IOError(f"配准后的源点云保存失败: {output_path}")
    print(f"配准后的源点云已保存: {output_path}")
    print(f"配准后源点数: {len(registered_source.points)}")

    if args.output_all:
        merged_output_path = sourcedir.parent / f"{sourcedir.stem}_ICP_merged.pcd"
        merged = build_merged_cloud(source, target, reg_icp.transformation)
        merged_success = o3d.io.write_point_cloud(str(merged_output_path), merged)
        if not merged_success:
            raise IOError(f"合并点云保存失败: {merged_output_path}")
        print(f"合并点云已保存: {merged_output_path}")
        print(f"合并后点数: {len(merged.points)}")
    else:
        print("跳过保存合并点云（output_all=False）")

    if args.vis_mode == "merged":
        source_name = Path(args.source).name
        target_name = Path(args.target).name
        win_title = f"ICP Registered: {source_name} -> {target_name}"
        visualize_registered_clouds(
            source=source,
            target=target,
            transformation=reg_icp.transformation,
            window_name=win_title,
        )
    else:
        print("跳过可视化显示")


if __name__ == "__main__":
    main()
