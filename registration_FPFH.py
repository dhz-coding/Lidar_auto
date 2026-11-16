# FPFH + RANSAC粗配准
import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 FPFH + RANSAC 对源点云和目标点云进行粗配准",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sdir = r"C:\Users\xm\Desktop\boatdata\boat2_test\boat3_changed.pcd"
    tdir = r"C:\Users\xm\Desktop\boatdata\boat2_test\boat2_1_colorized.pcd"
    parser.add_argument("--source", type=str, default=str(sdir), help="源点云路径")
    parser.add_argument("--target", type=str, default=str(tdir), help="目标点云路径")
    parser.add_argument("--voxel_size", type=float, default=0.2, help="下采样体素大小")
    parser.add_argument(
        "--normal_radius_ratio",
        type=float,
        default=0.6,
        help="法向量估计搜索半径",
    )
    parser.add_argument(
        "--feature_radius_ratio",
        type=float,
        default=2.0,
        help="FPFH 特征计算搜索半径",
    )
    parser.add_argument(
        "--distance_threshold_ratio",
        type=float,
        default=0.6,
        help="RANSAC 对应点距离阈值",
    )
    parser.add_argument(
        "--ransac_n",
        type=int,
        default=4,
        help="每次 RANSAC 采样点数",
    )
    parser.add_argument(
        "--max_iteration",
        type=int,
        default=1000000,
        help="RANSAC 最大迭代次数",
    )
    parser.add_argument(
        "--edge_length_threshold",
        type=float,
        default=0.9,
        help="边长一致性检查阈值（越接近 1.0 越严格）",
    )
    parser.add_argument(
        "--ransac_confidence",
        type=float,
        default=0.90,
        help="RANSAC 收敛置信度",
    )
    parser.add_argument(
        "--max_validation",
        type=int,
        default=1000,
        help="保留参数：用于兼容旧调用（当前 Open3D 接口未直接使用）",
    )
    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=["none", "merged"],
        default="merged",
        help="可视化模式: 'none' 不显示; 'merged' 显示粗配准后叠加结果",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出配准后的源点云 PCD 文件路径（默认自动生成）",
    )
    parser.add_argument(
        "--output_all",
        type=bool,
        default=False,
        help="是否额外保存合并后的点云（默认不保存）",
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


def preprocess_point_cloud_and_fpfh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius: float,
    feature_radius: float,
):
    """点云下采样、法向量估计并计算 FPFH 特征。"""
    pcd_down = pcd.voxel_down_sample(voxel_size)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )

    return pcd_down, pcd_fpfh


def run_fpfh_coarse_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    distance_threshold: float,
    ransac_n: int,
    max_iteration: int,
    max_validation: int,
    normal_radius: float,
    feature_radius: float,
    edge_length_threshold: float,
    ransac_confidence: float,
):
    """执行 FPFH + RANSAC 粗配准。"""
    start = time.perf_counter()
    source_down, source_fpfh = preprocess_point_cloud_and_fpfh(
        source,
        voxel_size,
        normal_radius,
        feature_radius,
    )
    target_down, target_fpfh = preprocess_point_cloud_and_fpfh(
        target,
        voxel_size,
        normal_radius,
        feature_radius,
    )

    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
            edge_length_threshold
        ),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            distance_threshold
        ),
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        max_iteration=max_iteration,
        confidence=ransac_confidence,
    )

    reg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=ransac_n,
        checkers=checkers,
        criteria=criteria,
    )

    elapsed = time.perf_counter() - start

    return reg, elapsed, source_down, target_down


def print_fpfh_report(reg_result, elapsed: float):
    """输出 FPFH + RANSAC粗配准关键统计信息。"""
    print("\n=== FPFH + RANSAC粗配准结果 ===")
    print(f"耗时: {elapsed:.6f} 秒")
    print(f"匹配率: {reg_result.fitness:.6f}")
    print(f"inlier_rmse: {reg_result.inlier_rmse:.6f}")

    corr_count = len(reg_result.correspondence_set)
    print(f"对应点对数量: {corr_count}")

    print("变换矩阵:")
    print(reg_result.transformation)


def visualize_registered_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    window_name: str,
):
    """可视化粗配准后的叠加结果：源点云红色，目标点云绿色。"""
    source_vis = o3d.geometry.PointCloud(source)
    target_vis = o3d.geometry.PointCloud(target)

    source_vis.transform(transformation)
    source_vis.paint_uniform_color([1.0, 0.0, 0.0])
    target_vis.paint_uniform_color([0.0, 1.0, 0.0])

    print("打开可视化窗口：红色=粗配准后源点云, 绿色=目标点云")
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


def coarse_register_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.2,
    normal_radius_ratio: float = 0.6,
    feature_radius_ratio: float = 2.0,
    distance_threshold_ratio: float = 0.6,
    ransac_n: int = 4,
    max_iteration: int = 1000000,
    max_validation: int = 1000,
    edge_length_threshold: float = 0.9,
    ransac_confidence: float = 0.90,
):
    """执行 FPFH + RANSAC 粗配准，返回变换后的 source、变换矩阵和报告对象。"""
    distance_threshold = distance_threshold_ratio * voxel_size
    feature_radius = feature_radius_ratio * voxel_size
    normal_radius = normal_radius_ratio * voxel_size

    reg, elapsed, source_down, target_down = run_fpfh_coarse_registration(
        source=source,
        target=target,
        voxel_size=voxel_size,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        max_iteration=max_iteration,
        max_validation=max_validation,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        edge_length_threshold=edge_length_threshold,
        ransac_confidence=ransac_confidence,
    )

    registered_source = o3d.geometry.PointCloud(source)
    registered_source.transform(reg.transformation)
    report = {
        "elapsed": elapsed,
        "fitness": float(reg.fitness),
        "inlier_rmse": float(reg.inlier_rmse),
        "correspondence_count": int(len(reg.correspondence_set)),
        "source_down_points": int(len(source_down.points)),
        "target_down_points": int(len(target_down.points)),
    }
    return registered_source, reg.transformation, report


def coarse_register_file(
    source_path,
    target_path,
    output_path=None,
    output_all=False,
    **kwargs,
):
    """文件级 FPFH + RANSAC 粗配准接口，供流水线或命令行复用。"""
    sourcedir = Path(source_path)
    targetdir = Path(target_path)
    source, target = load_point_clouds(str(sourcedir), str(targetdir))
    registered_source, transformation, report = coarse_register_point_clouds(
        source,
        target,
        **kwargs,
    )

    if output_path is None or str(output_path) == "":
        output_path = sourcedir.parent / f"{sourcedir.stem}_fpfh_registered_source.pcd"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_point_cloud(str(output_path), registered_source)
    if not success:
        raise IOError(f"配准后的源点云保存失败: {output_path}")
    print(f"配准后的源点云已保存: {output_path}")

    merged_output_path = None
    if output_all:
        merged_output_path = sourcedir.parent / f"{sourcedir.stem}_fpfh_RANSAC_merged.pcd"
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

    source, target = load_point_clouds(str(sourcedir), str(targetdir))

    distance_threshold = args.distance_threshold_ratio * args.voxel_size
    feature_radius = args.feature_radius_ratio * args.voxel_size
    normal_radius = args.normal_radius_ratio * args.voxel_size

    print(f"体素大小: {args.voxel_size}")
    print(f"RANSAC 距离阈值: {distance_threshold}")
    print(f"法向搜索半径: {normal_radius}")
    print(f"FPFH 搜索半径: {feature_radius}")
    print(f"边长一致性阈值: {args.edge_length_threshold}")
    print(f"RANSAC 置信度: {args.ransac_confidence}")

    reg, elapsed, source_down, target_down = run_fpfh_coarse_registration(
        source=source,
        target=target,
        voxel_size=args.voxel_size,
        distance_threshold=distance_threshold,
        ransac_n=args.ransac_n,
        max_iteration=args.max_iteration,
        max_validation=args.max_validation,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        edge_length_threshold=args.edge_length_threshold,
        ransac_confidence=args.ransac_confidence,
    )

    print(f"下采样后源点数: {len(source_down.points)}")
    print(f"下采样后目标点数: {len(target_down.points)}")
    print_fpfh_report(reg, elapsed=elapsed)

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = sourcedir.parent / f"{sourcedir.stem}_fpfh_registered_source.pcd"

    registered_source = o3d.geometry.PointCloud(source)
    registered_source.transform(reg.transformation)

    success = o3d.io.write_point_cloud(str(output_path), registered_source)
    if not success:
        raise IOError(f"配准后的源点云保存失败: {output_path}")
    print(f"配准后的源点云已保存: {output_path}")
    print(f"配准后源点数: {len(registered_source.points)}")

    if args.output_all:
        merged_output_path = sourcedir.parent / f"{sourcedir.stem}_fpfh_RANSAC_merged.pcd"
        merged = build_merged_cloud(source, target, reg.transformation)
        merged_success = o3d.io.write_point_cloud(str(merged_output_path), merged)
        if not merged_success:
            raise IOError(f"合并点云保存失败: {merged_output_path}")
        print(f"合并点云已保存: {merged_output_path}")
        print(f"合并后点数: {len(merged.points)}")
    else:
        print("跳过保存合并点云（output_all=False）")

    if args.vis_mode == "merged":
        source_name = sourcedir.name
        target_name = targetdir.name
        win_title = f"FPFH + RANSAC Coarse Registered: {source_name} -> {target_name}"
        visualize_registered_clouds(
            source=source,
            target=target,
            transformation=reg.transformation,
            window_name=win_title,
        )
    else:
        print("跳过可视化显示")


if __name__ == "__main__":
    main()

