# 装沙前后点云配准：
# 法向量估计 -> 曲率计算 -> 曲率边缘提取 -> 高度约束 + 法向量约束
# -> 欧式聚类去噪 -> 固定结构 ROI -> FPFH -> RANSAC 全局配准 -> ICP 精配准
# -> 将 T 作用到装沙后完整点云
import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="装沙前后点云配准，并将装沙后完整点云变换到装沙前坐标系",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source_default = r"G:\Lidarf\boat2_data\test_all\boat1a.pcd"
    target_default = r"G:\Lidarf\boat2_data\test_all\boat1b.pcd"

    parser.add_argument("--source", type=str, default=source_default, help="装沙后完整点云路径")
    parser.add_argument("--target", type=str, default=target_default, help="装沙前完整点云路径")
    parser.add_argument("--voxel_size", type=float, default=0.15, help="配准用体素下采样大小")
    parser.add_argument(
        "--source_normal_radius_ratio",
        type=float,
        default=3.0,
        help="装沙后点云法向量估计搜索半径与 voxel_size 的比例",
    )
    parser.add_argument(
        "--target_normal_radius_ratio",
        type=float,
        default=3.0,
        help="装沙前点云法向量估计搜索半径与 voxel_size 的比例",
    )
    parser.add_argument(
        "--feature_radius_ratio",
        type=float,
        default=8.0,
        help="FPFH 特征搜索半径与 voxel_size 的比例",
    )
    parser.add_argument(
        "--curvature_radius_ratio",
        type=float,
        default=4.0,
        help="曲率计算搜索半径与 voxel_size 的比例",
    )
    parser.add_argument(
        "--source_curvature_percentile",
        type=float,
        default=75.0,
        help="装沙后点云曲率边缘提取百分位阈值",
    )
    parser.add_argument(
        "--target_curvature_percentile",
        type=float,
        default=90.0,
        help="装沙前点云曲率边缘提取百分位阈值",
    )
    parser.add_argument(
        "--min_feature_points",
        type=int,
        default=1000,
        help="FPFH 特征点云的最少点数，不足时回退到 ROI 下采样点云",
    )
    parser.add_argument(
        "--source_height_min",
        type=float,
        default=None,
        help="装沙后点云高度约束下限",
    )
    parser.add_argument(
        "--source_height_max",
        type=float,
        default=None,
        help="装沙后点云高度约束上限",
    )
    parser.add_argument(
        "--target_height_min",
        type=float,
        default=None,
        help="装沙前点云高度约束下限",
    )
    parser.add_argument(
        "--target_height_max",
        type=float,
        default=None,
        help="装沙前点云高度约束上限",
    )
    parser.add_argument(
        "--height_axis",
        type=str,
        choices=["x", "y", "z"],
        default="z",
        help="高度约束使用的坐标轴",
    )
    parser.add_argument(
        "--normal_axis",
        type=float,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="法向量方向约束轴，默认不启用；会按绝对夹角判断",
    )
    parser.add_argument(
        "--normal_axis_angle",
        type=float,
        default=35.0,
        help="法向量方向约束的最大夹角，单位：度",
    )
    parser.add_argument(
        "--source_roi",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="装沙后固定结构 ROI 边界，默认不启用",
    )
    parser.add_argument(
        "--target_roi",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="装沙前固定结构 ROI 边界，默认不启用；不填则复用 source_roi",
    )
    parser.add_argument(
        "--cluster_eps_ratio",
        type=float,
        default=4.0,
        help="欧式聚类半径与 voxel_size 的比例",
    )
    parser.add_argument(
        "--source_min_cluster_points",
        type=int,
        default=15,
        help="装沙后点云欧式聚类保留的最少簇点数",
    )
    parser.add_argument(
        "--target_min_cluster_points",
        type=int,
        default=15,
        help="装沙前点云欧式聚类保留的最少簇点数",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=10,
        help="欧式聚类最多保留的簇数量",
    )
    parser.add_argument(
        "--ransac_distance_ratio",
        type=float,
        default=4.0,
        help="RANSAC 最大对应距离与 voxel_size 的比例",
    )
    parser.add_argument("--ransac_n", type=int, default=4, help="RANSAC 每次采样点数")
    parser.add_argument("--ransac_max_iteration", type=int, default=4000000, help="RANSAC 最大迭代次数")
    parser.add_argument("--ransac_confidence", type=float, default=0.999, help="RANSAC 置信度")
    parser.add_argument(
        "--post_normal_angle",
        type=float,
        default=35.0,
        help="全局配准后法向量一致性过滤阈值，单位：度",
    )
    parser.add_argument(
        "--disable_post_normal_filter",
        action="store_true",
        help="关闭全局配准后的法向量一致性过滤",
    )
    parser.add_argument(
        "--icp_distance_ratio",
        type=float,
        default=2.0,
        help="ICP 最大对应距离与当前 ICP 体素大小的比例",
    )
    parser.add_argument("--icp_max_iteration", type=int, default=500, help="每层 ICP 最大迭代次数")
    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=[
            "none",
            "merged",
            "roi",
            "feature",
            "roi_merged",
            "feature_merged",
            "roi_feature_merged",
        ],
        default="feature_merged",
        help="可视化模式：none 不显示，merged 显示配准结果，roi 显示 ROI，feature 显示 FPFH 输入点云",
    )
    parser.add_argument("--output", type=str, default=None, help="配准后的装沙后完整点云输出路径")
    parser.add_argument("--matrix_output", type=str, default=None, help="变换矩阵 txt 输出路径")
    parser.add_argument(
        "--skip_cloud_output",
        action="store_true",
        help="只保存变换矩阵，不保存配准后的完整点云，适合批量参数测试",
    )
    parser.add_argument("--output_all", action="store_true", help="额外保存装沙前后合并点云")

    return parser.parse_args()


def load_point_cloud(path: Path, name: str):
    """读取点云并进行基础校验。"""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"{name} 为空或读取失败: {path}")

    print(f"{name}: {path}")
    print(f"{name} 点数: {len(pcd.points)}")
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 50):
    """估计法向量。"""
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.normalize_normals()


def compute_curvatures(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int = 50,
    min_nn: int = 6,
):
    """基于邻域协方差特征值计算曲率。"""
    points = np.asarray(pcd.points)
    curvatures = np.zeros(len(points), dtype=np.float64)
    if len(points) == 0:
        return curvatures

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i, point in enumerate(points):
        _, idx, _ = kdtree.search_hybrid_vector_3d(point, radius, max_nn)
        if len(idx) < min_nn:
            continue

        neighbors = points[idx]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / max(len(idx) - 1, 1)
        eigvals = np.linalg.eigvalsh(cov)
        eig_sum = np.sum(eigvals)
        if eig_sum > 1e-12:
            curvatures[i] = eigvals[0] / eig_sum

    return curvatures


def select_by_indices(pcd: o3d.geometry.PointCloud, indices):
    """按索引提取点云。"""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return o3d.geometry.PointCloud()
    return pcd.select_by_index(indices.tolist())


def height_mask(points: np.ndarray, axis: str, height_min, height_max):
    """构建高度约束掩码。"""
    if height_min is None and height_max is None:
        return np.ones(len(points), dtype=bool)

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    values = points[:, axis_index]
    mask = np.ones(len(points), dtype=bool)
    if height_min is not None:
        mask &= values >= height_min
    if height_max is not None:
        mask &= values <= height_max
    return mask


def normal_axis_mask(normals: np.ndarray, axis, max_angle_deg: float):
    """构建法向量方向约束掩码。"""
    if axis is None:
        return np.ones(len(normals), dtype=bool)

    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        raise ValueError("normal_axis 不能为零向量")

    axis = axis / axis_norm
    normals_norm = np.linalg.norm(normals, axis=1)
    valid = normals_norm > 1e-12
    unit_normals = np.zeros_like(normals)
    unit_normals[valid] = normals[valid] / normals_norm[valid, None]

    cos_threshold = np.cos(np.deg2rad(max_angle_deg))
    cos_values = np.abs(unit_normals @ axis)
    return valid & (cos_values >= cos_threshold)


def roi_mask(points: np.ndarray, bounds):
    """构建固定结构 ROI 掩码。"""
    if bounds is None:
        return np.ones(len(points), dtype=bool)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )


def extract_curvature_edges(
    pcd: o3d.geometry.PointCloud,
    curvatures: np.ndarray,
    percentile: float,
):
    """按曲率阈值提取边缘候选点。"""
    if len(curvatures) == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=np.int64)

    threshold = np.percentile(curvatures, percentile)
    indices = np.flatnonzero(curvatures >= threshold)
    return select_by_indices(pcd, indices), indices


def euclidean_cluster_filter(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
    max_clusters: int,
):
    """用 DBSCAN 近似欧式聚类去噪，保留最大的若干个有效簇。"""
    if len(pcd.points) == 0:
        return pcd

    labels = np.asarray(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return pcd

    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    keep_labels = unique_labels[order[:max_clusters]]
    keep_indices = np.flatnonzero(np.isin(labels, keep_labels))
    return select_by_indices(pcd, keep_indices)


def prepare_registration_cloud(
    pcd: o3d.geometry.PointCloud,
    name: str,
    voxel_size: float,
    normal_radius: float,
    curvature_radius: float,
    curvature_percentile: float,
    height_axis: str,
    height_min,
    height_max,
    normal_axis,
    normal_axis_angle: float,
    cluster_eps: float,
    min_cluster_points: int,
    max_clusters: int,
    roi_bounds,
    min_feature_points: int,
):
    """生成配准所需的 ROI 点云、曲率边缘点云和 FPFH 输入点云。"""
    print(f"\n=== {name} 预处理 ===")

    down = pcd.voxel_down_sample(voxel_size)
    estimate_normals(down, normal_radius)
    print(f"下采样后点数: {len(down.points)}")

    curvatures = compute_curvatures(down, curvature_radius)
    edge_cloud, _ = extract_curvature_edges(
        down,
        curvatures,
        curvature_percentile,
    )
    print(f"曲率边缘点数: {len(edge_cloud.points)}")

    down_points = np.asarray(down.points)
    down_normals = np.asarray(down.normals)
    stable_mask = height_mask(down_points, height_axis, height_min, height_max)
    stable_mask &= normal_axis_mask(down_normals, normal_axis, normal_axis_angle)
    stable_mask &= roi_mask(down_points, roi_bounds)
    stable_cloud = select_by_indices(down, np.flatnonzero(stable_mask))
    print(f"高度/法向量/ROI 约束后 ROI 点数: {len(stable_cloud.points)}")

    edge_points = np.asarray(edge_cloud.points)
    edge_normals = np.asarray(edge_cloud.normals)
    edge_mask = height_mask(edge_points, height_axis, height_min, height_max)
    edge_mask &= normal_axis_mask(edge_normals, normal_axis, normal_axis_angle)
    edge_mask &= roi_mask(edge_points, roi_bounds)
    constrained_edges = select_by_indices(edge_cloud, np.flatnonzero(edge_mask))
    print(f"约束后边缘点数: {len(constrained_edges.points)}")

    clustered_edges = euclidean_cluster_filter(
        constrained_edges,
        eps=cluster_eps,
        min_points=min_cluster_points,
        max_clusters=max_clusters,
    )
    print(f"欧式聚类去噪后边缘点数: {len(clustered_edges.points)}")

    if len(clustered_edges.points) >= min_feature_points:
        feature_cloud = clustered_edges
        print("FPFH 输入: 曲率边缘聚类点云")
    else:
        feature_cloud = stable_cloud
        print("FPFH 输入: ROI 下采样点云（边缘点过少，自动回退）")

    if len(feature_cloud.points) < 3:
        raise ValueError(f"{name} 可用于配准的点数不足")

    if len(stable_cloud.points) < 3:
        stable_cloud = feature_cloud

    estimate_normals(feature_cloud, normal_radius)
    estimate_normals(stable_cloud, normal_radius)
    return stable_cloud, feature_cloud


def compute_fpfh(
    pcd: o3d.geometry.PointCloud,
    normal_radius: float,
    feature_radius: float,
):
    """计算 FPFH 特征。"""
    estimate_normals(pcd, normal_radius)
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )


def run_ransac_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_fpfh,
    target_fpfh,
    distance_threshold: float,
    ransac_n: int,
    max_iteration: int,
    confidence: float,
):
    """使用 Open3D FPFH + RANSAC 全局配准。"""
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        max_iteration=max_iteration,
        confidence=confidence,
    )

    start = time.perf_counter()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=checkers,
        criteria=criteria,
    )
    elapsed = time.perf_counter() - start
    return result.transformation, result, elapsed


def filter_by_post_normal_consistency(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    distance_threshold: float,
    max_angle_deg: float,
):
    """全局配准后，根据最近邻距离和旋转后的法向量夹角过滤 ICP 输入。"""
    if len(source.points) == 0 or len(target.points) == 0:
        return source, target

    source_points = np.asarray(source.points)
    source_normals = np.asarray(source.normals)
    target_normals = np.asarray(target.normals)

    source_transformed = o3d.geometry.PointCloud(source)
    source_transformed.transform(transformation)
    transformed_points = np.asarray(source_transformed.points)
    rotation = transformation[:3, :3]
    rotated_normals = (rotation @ source_normals.T).T

    target_tree = o3d.geometry.KDTreeFlann(target)
    cos_threshold = np.cos(np.deg2rad(max_angle_deg))
    keep_source = []
    keep_target = []

    for i, point in enumerate(transformed_points):
        _, idx, dist = target_tree.search_knn_vector_3d(point, 1)
        if len(idx) != 1:
            continue
        if np.sqrt(dist[0]) > distance_threshold:
            continue

        src_normal = rotated_normals[i]
        tgt_normal = target_normals[idx[0]]
        src_norm = np.linalg.norm(src_normal)
        tgt_norm = np.linalg.norm(tgt_normal)
        if src_norm < 1e-12 or tgt_norm < 1e-12:
            continue

        cos_value = abs(float(np.dot(src_normal, tgt_normal) / (src_norm * tgt_norm)))
        if cos_value >= cos_threshold:
            keep_source.append(i)
            keep_target.append(idx[0])

    if len(keep_source) < 3:
        print("法向量一致性过滤后点数过少，保留原 ICP 输入")
        return source, target

    return select_by_indices(source, keep_source), select_by_indices(target, keep_target)


def run_multiscale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transformation: np.ndarray,
    voxel_sizes,
    distance_ratio: float,
    source_normal_radius_ratio: float,
    target_normal_radius_ratio: float,
    max_iteration: int,
):
    """执行多尺度 point-to-plane ICP 精配准。"""
    transformation = init_transformation.copy()
    reports = []

    start_all = time.perf_counter()
    for voxel_size in voxel_sizes:
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        source_normal_radius = source_normal_radius_ratio * voxel_size
        target_normal_radius = target_normal_radius_ratio * voxel_size
        estimate_normals(source_down, source_normal_radius)
        estimate_normals(target_down, target_normal_radius)

        threshold = distance_ratio * voxel_size
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration
        )
        reg = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            threshold,
            transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
        transformation = reg.transformation
        reports.append((voxel_size, threshold, reg.fitness, reg.inlier_rmse))

    elapsed = time.perf_counter() - start_all
    return transformation, reports, elapsed


def evaluate_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    threshold: float,
):
    """评估配准结果。"""
    return o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        threshold,
        transformation,
    )


def build_merged_cloud(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
):
    """构建合并点云。"""
    source_registered = o3d.geometry.PointCloud(source)
    target_copy = o3d.geometry.PointCloud(target)
    source_registered.transform(transformation)
    return source_registered + target_copy


def visualize_registered_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    window_name: str,
):
    """可视化配准叠加结果。"""
    source_vis = o3d.geometry.PointCloud(source)
    target_vis = o3d.geometry.PointCloud(target)

    source_vis.transform(transformation)
    source_vis.paint_uniform_color([1.0, 0.0, 0.0])
    target_vis.paint_uniform_color([0.0, 1.0, 0.0])

    print("打开可视化窗口：红色=配准后装沙后点云，绿色=装沙前点云")
    o3d.visualization.draw_geometries(
        [source_vis, target_vis],
        window_name=window_name,
    )


def visualize_roi_clouds(
    source_roi: o3d.geometry.PointCloud,
    target_roi: o3d.geometry.PointCloud,
    window_name: str,
):
    """可视化提取出的配准 ROI 点云。"""
    source_vis = o3d.geometry.PointCloud(source_roi)
    target_vis = o3d.geometry.PointCloud(target_roi)

    source_vis.paint_uniform_color([1.0, 0.0, 0.0])
    target_vis.paint_uniform_color([0.0, 1.0, 0.0])

    print("打开 ROI 可视化窗口：红色=装沙后 ROI，绿色=装沙前 ROI")
    o3d.visualization.draw_geometries(
        [source_vis, target_vis],
        window_name=window_name,
    )


def visualize_feature_clouds(
    source_feature: o3d.geometry.PointCloud,
    target_feature: o3d.geometry.PointCloud,
    window_name: str,
):
    """可视化实际参与 FPFH 的点云。"""
    source_vis = o3d.geometry.PointCloud(source_feature)
    target_vis = o3d.geometry.PointCloud(target_feature)

    source_vis.paint_uniform_color([1.0, 0.0, 0.0])
    target_vis.paint_uniform_color([0.0, 1.0, 0.0])

    print("打开 Feature 可视化窗口：红色=装沙后 FPFH 输入点云，绿色=装沙前 FPFH 输入点云")
    o3d.visualization.draw_geometries(
        [source_vis, target_vis],
        window_name=window_name,
    )


def print_global_report(method: str, transformation: np.ndarray, elapsed: float, evaluation):
    """输出全局配准报告。"""
    print(f"\n=== {method} 全局配准结果 ===")
    print(f"耗时: {elapsed:.6f} 秒")
    print(f"fitness: {evaluation.fitness:.6f}")
    print(f"inlier_rmse: {evaluation.inlier_rmse:.6f}")
    print("初始变换矩阵:")
    print(transformation)


def print_icp_report(transformation: np.ndarray, reports, elapsed: float, evaluation):
    """输出 ICP 配准报告。"""
    print("\n=== ICP 精配准结果 ===")
    for voxel_size, threshold, fitness, rmse in reports:
        print(
            f"voxel={voxel_size:.6f}, threshold={threshold:.6f}, "
            f"fitness={fitness:.6f}, inlier_rmse={rmse:.6f}"
        )

    print(f"ICP 总耗时: {elapsed:.6f} 秒")
    print(f"最终 fitness: {evaluation.fitness:.6f}")
    print(f"最终 inlier_rmse: {evaluation.inlier_rmse:.6f}")
    print("最终变换矩阵 T_after_to_before:")
    print(transformation)


def align_after_to_before(
    source_after: o3d.geometry.PointCloud,
    target_before: o3d.geometry.PointCloud,
    voxel_size: float = 0.15,
    source_normal_radius_ratio: float = 3.0,
    target_normal_radius_ratio: float = 3.0,
    feature_radius_ratio: float = 8.0,
    curvature_radius_ratio: float = 4.0,
    source_curvature_percentile: float = 75.0,
    target_curvature_percentile: float = 90.0,
    min_feature_points: int = 1000,
    source_height_min=None,
    source_height_max=None,
    target_height_min=None,
    target_height_max=None,
    height_axis: str = "z",
    normal_axis=None,
    normal_axis_angle: float = 35.0,
    source_roi=None,
    target_roi=None,
    cluster_eps_ratio: float = 4.0,
    source_min_cluster_points: int = 15,
    target_min_cluster_points: int = 15,
    max_clusters: int = 10,
    ransac_distance_ratio: float = 4.0,
    ransac_n: int = 4,
    ransac_max_iteration: int = 4000000,
    ransac_confidence: float = 0.999,
    post_normal_angle: float = 35.0,
    disable_post_normal_filter: bool = False,
    icp_distance_ratio: float = 2.0,
    icp_max_iteration: int = 500,
):
    """将装沙后完整点云配准到装沙前坐标系，返回配准后点云、矩阵和报告。"""
    source_normal_radius = source_normal_radius_ratio * voxel_size
    target_normal_radius = target_normal_radius_ratio * voxel_size
    feature_radius = feature_radius_ratio * voxel_size
    curvature_radius = curvature_radius_ratio * voxel_size
    cluster_eps = cluster_eps_ratio * voxel_size
    ransac_distance = ransac_distance_ratio * voxel_size
    target_roi = target_roi if target_roi is not None else source_roi

    source_icp, source_feature = prepare_registration_cloud(
        pcd=source_after,
        name="source 装沙后",
        voxel_size=voxel_size,
        normal_radius=source_normal_radius,
        curvature_radius=curvature_radius,
        curvature_percentile=source_curvature_percentile,
        height_axis=height_axis,
        height_min=source_height_min,
        height_max=source_height_max,
        normal_axis=normal_axis,
        normal_axis_angle=normal_axis_angle,
        cluster_eps=cluster_eps,
        min_cluster_points=source_min_cluster_points,
        max_clusters=max_clusters,
        roi_bounds=source_roi,
        min_feature_points=min_feature_points,
    )
    target_icp, target_feature = prepare_registration_cloud(
        pcd=target_before,
        name="target 装沙前",
        voxel_size=voxel_size,
        normal_radius=target_normal_radius,
        curvature_radius=curvature_radius,
        curvature_percentile=target_curvature_percentile,
        height_axis=height_axis,
        height_min=target_height_min,
        height_max=target_height_max,
        normal_axis=normal_axis,
        normal_axis_angle=normal_axis_angle,
        cluster_eps=cluster_eps,
        min_cluster_points=target_min_cluster_points,
        max_clusters=max_clusters,
        roi_bounds=target_roi,
        min_feature_points=min_feature_points,
    )

    source_fpfh = compute_fpfh(source_feature, source_normal_radius, feature_radius)
    target_fpfh = compute_fpfh(target_feature, target_normal_radius, feature_radius)
    global_transformation, ransac_result, global_elapsed = run_ransac_global_registration(
        source=source_feature,
        target=target_feature,
        source_fpfh=source_fpfh,
        target_fpfh=target_fpfh,
        distance_threshold=ransac_distance,
        ransac_n=ransac_n,
        max_iteration=ransac_max_iteration,
        confidence=ransac_confidence,
    )

    global_eval = evaluate_registration(
        source_feature,
        target_feature,
        global_transformation,
        ransac_distance,
    )
    print_global_report("FPFH + RANSAC", global_transformation, global_elapsed, global_eval)

    icp_source = source_icp
    icp_target = target_icp
    if not disable_post_normal_filter:
        icp_source, icp_target = filter_by_post_normal_consistency(
            source=source_icp,
            target=target_icp,
            transformation=global_transformation,
            distance_threshold=ransac_distance,
            max_angle_deg=post_normal_angle,
        )
        print(f"\n法向量一致性过滤后 source ICP 点数: {len(icp_source.points)}")
        print(f"法向量一致性过滤后 target ICP 点数: {len(icp_target.points)}")

    icp_voxel_sizes = [2.0 * voxel_size, voxel_size, 0.5 * voxel_size]
    final_transformation, icp_reports, icp_elapsed = run_multiscale_icp(
        source=icp_source,
        target=icp_target,
        init_transformation=global_transformation,
        voxel_sizes=icp_voxel_sizes,
        distance_ratio=icp_distance_ratio,
        source_normal_radius_ratio=source_normal_radius_ratio,
        target_normal_radius_ratio=target_normal_radius_ratio,
        max_iteration=icp_max_iteration,
    )
    final_eval_threshold = icp_distance_ratio * min(icp_voxel_sizes)
    final_eval = evaluate_registration(
        source_icp,
        target_icp,
        final_transformation,
        final_eval_threshold,
    )
    print_icp_report(final_transformation, icp_reports, icp_elapsed, final_eval)

    registered_source = o3d.geometry.PointCloud(source_after)
    registered_source.transform(final_transformation)
    report = {
        "global_elapsed": float(global_elapsed),
        "global_fitness": float(global_eval.fitness),
        "global_inlier_rmse": float(global_eval.inlier_rmse),
        "ransac_correspondence_count": int(len(ransac_result.correspondence_set)),
        "icp_elapsed": float(icp_elapsed),
        "final_fitness": float(final_eval.fitness),
        "final_inlier_rmse": float(final_eval.inlier_rmse),
        "icp_reports": [
            {
                "voxel_size": float(item[0]),
                "threshold": float(item[1]),
                "fitness": float(item[2]),
                "inlier_rmse": float(item[3]),
            }
            for item in icp_reports
        ],
    }
    return registered_source, final_transformation, report


def align_after_to_before_file(
    source_path,
    target_path,
    output_path=None,
    matrix_output=None,
    output_all=False,
    **kwargs,
):
    """文件级装沙前后配准接口。"""
    sourcedir = Path(source_path)
    targetdir = Path(target_path)
    source = load_point_cloud(sourcedir, "装沙后完整点云 source")
    target = load_point_cloud(targetdir, "装沙前完整点云 target")
    registered_source, transformation, report = align_after_to_before(
        source,
        target,
        **kwargs,
    )

    if output_path is None or str(output_path) == "":
        output_path = sourcedir.parent / f"{sourcedir.stem}_registered_to_before.pcd"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_point_cloud(str(output_path), registered_source)
    if not success:
        raise IOError(f"配准后的装沙后完整点云保存失败: {output_path}")
    print(f"\n配准后的装沙后完整点云已保存: {output_path}")

    if matrix_output is None or str(matrix_output) == "":
        matrix_output = output_path.with_suffix(".txt")
    matrix_output = Path(matrix_output)
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(matrix_output, transformation, fmt="%.10f")
    np.save(matrix_output.with_suffix(".npy"), transformation)
    print(f"变换矩阵 txt 已保存: {matrix_output}")
    print(f"变换矩阵 npy 已保存: {matrix_output.with_suffix('.npy')}")

    merged_output_path = None
    if output_all:
        merged_output_path = output_path.with_name(f"{output_path.stem}_merged.pcd")
        merged = build_merged_cloud(source, target, transformation)
        merged_success = o3d.io.write_point_cloud(str(merged_output_path), merged)
        if not merged_success:
            raise IOError(f"合并点云保存失败: {merged_output_path}")
        print(f"合并点云已保存: {merged_output_path}")

    return registered_source, transformation, report, output_path, matrix_output, merged_output_path


def main():
    args = parse_args()

    sourcedir = Path(args.source)
    targetdir = Path(args.target)

    source = load_point_cloud(sourcedir, "装沙后完整点云 source")
    target = load_point_cloud(targetdir, "装沙前完整点云 target")

    source_normal_radius = args.source_normal_radius_ratio * args.voxel_size
    target_normal_radius = args.target_normal_radius_ratio * args.voxel_size
    feature_radius = args.feature_radius_ratio * args.voxel_size
    curvature_radius = args.curvature_radius_ratio * args.voxel_size
    cluster_eps = args.cluster_eps_ratio * args.voxel_size
    ransac_distance = args.ransac_distance_ratio * args.voxel_size

    target_roi = args.target_roi if args.target_roi is not None else args.source_roi

    print("\n=== 参数 ===")
    print(f"voxel_size: {args.voxel_size}")
    print(f"source_normal_radius: {source_normal_radius}")
    print(f"target_normal_radius: {target_normal_radius}")
    print(f"feature_radius: {feature_radius}")
    print(f"curvature_radius: {curvature_radius}")
    print(f"cluster_eps: {cluster_eps}")
    print(f"ransac_distance: {ransac_distance}")

    source_icp, source_feature = prepare_registration_cloud(
        pcd=source,
        name="source 装沙后",
        voxel_size=args.voxel_size,
        normal_radius=source_normal_radius,
        curvature_radius=curvature_radius,
        curvature_percentile=args.source_curvature_percentile,
        height_axis=args.height_axis,
        height_min=args.source_height_min,
        height_max=args.source_height_max,
        normal_axis=args.normal_axis,
        normal_axis_angle=args.normal_axis_angle,
        cluster_eps=cluster_eps,
        min_cluster_points=args.source_min_cluster_points,
        max_clusters=args.max_clusters,
        roi_bounds=args.source_roi,
        min_feature_points=args.min_feature_points,
    )
    target_icp, target_feature = prepare_registration_cloud(
        pcd=target,
        name="target 装沙前",
        voxel_size=args.voxel_size,
        normal_radius=target_normal_radius,
        curvature_radius=curvature_radius,
        curvature_percentile=args.target_curvature_percentile,
        height_axis=args.height_axis,
        height_min=args.target_height_min,
        height_max=args.target_height_max,
        normal_axis=args.normal_axis,
        normal_axis_angle=args.normal_axis_angle,
        cluster_eps=cluster_eps,
        min_cluster_points=args.target_min_cluster_points,
        max_clusters=args.max_clusters,
        roi_bounds=target_roi,
        min_feature_points=args.min_feature_points,
    )

    if args.vis_mode in {"roi", "roi_merged", "roi_feature_merged"}:
        roi_win_title = f"Extracted ROI: {sourcedir.name} / {targetdir.name}"
        visualize_roi_clouds(
            source_roi=source_icp,
            target_roi=target_icp,
            window_name=roi_win_title,
        )

    if args.vis_mode in {"feature", "feature_merged", "roi_feature_merged"}:
        feature_win_title = f"FPFH Feature Clouds: {sourcedir.name} / {targetdir.name}"
        visualize_feature_clouds(
            source_feature=source_feature,
            target_feature=target_feature,
            window_name=feature_win_title,
        )

    source_fpfh = compute_fpfh(source_feature, source_normal_radius, feature_radius)
    target_fpfh = compute_fpfh(target_feature, target_normal_radius, feature_radius)
    global_transformation, ransac_result, global_elapsed = run_ransac_global_registration(
        source=source_feature,
        target=target_feature,
        source_fpfh=source_fpfh,
        target_fpfh=target_fpfh,
        distance_threshold=ransac_distance,
        ransac_n=args.ransac_n,
        max_iteration=args.ransac_max_iteration,
        confidence=args.ransac_confidence,
    )
    global_method = "FPFH + RANSAC"

    global_eval = evaluate_registration(
        source_feature,
        target_feature,
        global_transformation,
        ransac_distance,
    )
    print_global_report(global_method, global_transformation, global_elapsed, global_eval)

    icp_source = source_icp
    icp_target = target_icp
    if not args.disable_post_normal_filter:
        icp_source, icp_target = filter_by_post_normal_consistency(
            source=source_icp,
            target=target_icp,
            transformation=global_transformation,
            distance_threshold=ransac_distance,
            max_angle_deg=args.post_normal_angle,
        )
        print(f"\n法向量一致性过滤后 source ICP 点数: {len(icp_source.points)}")
        print(f"法向量一致性过滤后 target ICP 点数: {len(icp_target.points)}")

    icp_voxel_sizes = [2.0 * args.voxel_size, args.voxel_size, 0.5 * args.voxel_size]

    final_transformation, icp_reports, icp_elapsed = run_multiscale_icp(
        source=icp_source,
        target=icp_target,
        init_transformation=global_transformation,
        voxel_sizes=icp_voxel_sizes,
        distance_ratio=args.icp_distance_ratio,
        source_normal_radius_ratio=args.source_normal_radius_ratio,
        target_normal_radius_ratio=args.target_normal_radius_ratio,
        max_iteration=args.icp_max_iteration,
    )
    final_eval_threshold = args.icp_distance_ratio * min(icp_voxel_sizes)
    final_eval = evaluate_registration(
        source_icp,
        target_icp,
        final_transformation,
        final_eval_threshold,
    )
    print_icp_report(final_transformation, icp_reports, icp_elapsed, final_eval)

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = sourcedir.parent / f"{sourcedir.stem}_registered_to_before.pcd"

    if args.skip_cloud_output:
        print("\n跳过保存配准后的装沙后完整点云")
    else:
        registered_source = o3d.geometry.PointCloud(source)
        registered_source.transform(final_transformation)
        success = o3d.io.write_point_cloud(str(output_path), registered_source)
        if not success:
            raise IOError(f"配准后的装沙后完整点云保存失败: {output_path}")
        print(f"\n配准后的装沙后完整点云已保存: {output_path}")
        print(f"配准后完整点云点数: {len(registered_source.points)}")

    if args.matrix_output is not None:
        matrix_output_path = Path(args.matrix_output)
    else:
        matrix_output_path = output_path.with_suffix(".txt")

    np.savetxt(matrix_output_path, final_transformation, fmt="%.10f")
    np.save(matrix_output_path.with_suffix(".npy"), final_transformation)
    print(f"变换矩阵 txt 已保存: {matrix_output_path}")
    print(f"变换矩阵 npy 已保存: {matrix_output_path.with_suffix('.npy')}")

    if args.output_all and not args.skip_cloud_output:
        merged_output_path = output_path.with_name(f"{output_path.stem}_merged.pcd")
        merged = build_merged_cloud(source, target, final_transformation)
        merged_success = o3d.io.write_point_cloud(str(merged_output_path), merged)
        if not merged_success:
            raise IOError(f"合并点云保存失败: {merged_output_path}")
        print(f"合并点云已保存: {merged_output_path}")
        print(f"合并后点数: {len(merged.points)}")
    elif args.output_all:
        print("skip_cloud_output=True，跳过保存合并点云")

    if args.vis_mode in {"merged", "roi_merged", "feature_merged", "roi_feature_merged"}:
        win_title = f"Before/After Registration: {sourcedir.name} -> {targetdir.name}"
        visualize_registered_clouds(
            source=source,
            target=target,
            transformation=final_transformation,
            window_name=win_title,
        )
    else:
        print("跳过配准叠加可视化显示")


if __name__ == "__main__":
    main()
