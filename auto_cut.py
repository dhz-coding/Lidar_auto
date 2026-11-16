# 平面拟合交线自动裁剪
# 会根据装沙前点云的 RANSAC 平面拟合结果，自动提取四个舱壁交点并裁剪装沙前后两次的点云。
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "体素质心下采样后使用 RANSAC 提取舱壁平面，"
            "根据两组平行垂直平面的交线确定四个顶点并裁剪点云"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_before_input = r"D:\LiDAR\boat2_data\test_all\boat1b_denoised_40_5.0.pcd"
    default_after_input = r"D:\LiDAR\boat2_data\test_all\boat1a_denoised_40_5.0.pcd"
    parser.add_argument(
        "--input_before",
        type=str,
        default=str(default_before_input),
        help="输入点云 PCD 文件路径",
    )
    parser.add_argument(
        "--input_after",
        type=str,
        default=str(default_after_input),
        help="输入点云 PCD 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出 PCD 路径，默认按输入文件名自动生成",
    )
    parser.add_argument(
        "--output_after",
        type=str,
        default="",
        help="输出泥沙 PCD 路径，默认按输入文件名自动生成",
    )
    parser.add_argument(
        "--wall_output",
        type=str,
        default="",
        help="筛选后的舱壁平面点云输出路径，默认不保存",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.04,
        help="RANSAC 平面拟合前的体素质心下采样大小，0 表示不下采样",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.08,
        help="RANSAC 平面内点距离阈值",
    )
    parser.add_argument(
        "--ransac_n",
        type=int,
        default=3,
        help="RANSAC 每次采样点数",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10000,
        help="RANSAC 迭代次数",
    )
    parser.add_argument(
        "--max_planes",
        type=int,
        default=16,
        help="最多迭代提取的平面数量",
    )
    parser.add_argument(
        "--min_plane_points",
        type=int,
        default=500,
        help="单个平面最少内点数，低于该值停止或丢弃",
    )
    parser.add_argument(
        "--vertical_normal_z_threshold",
        type=float,
        default=0.2,
        help="垂直平面筛选阈值，平面法向量 z 分量绝对值小于该值",
    )
    parser.add_argument(
        "--parallel_angle_threshold",
        type=float,
        default=12.0,
        help="平面归为同一平行组的角度阈值，单位度",
    )
    parser.add_argument(
        "--min_group_angle",
        type=float,
        default=45.0,
        help="两组平行平面之间的最小夹角，单位度",
    )
    parser.add_argument(
        "--min_plane_separation",
        type=float,
        default=0.3,
        help="同组两侧边界平面的最小间距",
    )

    parser.add_argument(
        "--line_search_radius",
        type=float,
        default=0.1,
        help="原始点云中搜索交线附近最高点的半径",
    )
    parser.add_argument(
        "--line_search_expand",
        type=int,
        default=4,
        help="交线附近无点时，搜索半径最大扩张倍数",
    )
    parser.add_argument(
        "--crop_margin_x",
        type=float,
        default=-0.2,
        help="X 轴裁剪边界扩张距离，负值表示向内收缩",
    )
    parser.add_argument(
        "--crop_margin_y",
        type=float,
        default=-0.15,
        help="Y 轴裁剪边界扩张距离，负值表示向内收缩",
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=None,
        help="裁剪 Z 最小值，默认使用原始点云最小 Z",
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=None,
        help="裁剪 Z 最大值，默认使用四条交线附近最高点的最大 Z",
    )
    parser.add_argument(
        "--z_margin",
        type=float,
        default=0.2,
        help="在自动 Z 裁剪上界和下界外额外扩张的距离",
    )

    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=[
            "none",
            "before_after",
            "after_only",
            "wall_only",
            "wall_after",
            "all",
        ],
        default="wall_after",
        help="可视化模式",
    )

    return parser.parse_args()


def load_point_cloud(input_path: str):
    """读取点云，并进行基础校验。"""
    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise ValueError(f"输入点云为空或读取失败: {input_path}")

    pts = np.asarray(pcd.points)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    print("成功读取点云文件")
    print(f"输入路径: {input_path}")
    print(f"点数: {pts.shape[0]}")
    print(
        f"坐标范围: X[{mins[0]:.3f},{maxs[0]:.3f}] "
        f"Y[{mins[1]:.3f},{maxs[1]:.3f}] "
        f"Z[{mins[2]:.3f},{maxs[2]:.3f}]"
    )
    return pcd


def voxel_downsample_centroid(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
):
    """体素质心下采样。"""
    if voxel_size is None or voxel_size <= 0:
        print("不进行体素下采样。")
        return o3d.geometry.PointCloud(pcd)

    print("开始体素质心下采样...")
    print(f"  voxel_size = {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    n_before = len(pcd.points)
    n_after = len(pcd_down.points)
    print(f"原始点数: {n_before}")
    print(f"下采样后点数: {n_after}")
    if n_before > 0:
        print(f"保留比例: {n_after / n_before:.4f} ({n_after / n_before * 100:.2f}%)")

    if n_after == 0:
        raise ValueError("下采样后点云为空，请减小 voxel_size。")
    return pcd_down


def normalize_plane_model(model):
    """将平面方程 ax+by+cz+d=0 标准化。"""
    plane = np.asarray(model, dtype=float)
    normal = plane[:3]
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("RANSAC 返回了无效平面。")
    return plane / norm


def extract_planes_ransac(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    max_planes: int,
    min_plane_points: int,
):
    """迭代 RANSAC 平面分割并提取平面。"""
    remaining = o3d.geometry.PointCloud(pcd)
    planes = []

    print("开始 RANSAC 平面分割...")
    for i in range(max_planes):
        if len(remaining.points) < min_plane_points:
            print("剩余点数不足，停止平面提取。")
            break

        model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if len(inliers) < min_plane_points:
            print(f"第 {i + 1} 个候选平面内点不足，停止平面提取。")
            break

        model = normalize_plane_model(model)
        plane_pcd = remaining.select_by_index(inliers)
        plane_pts = np.asarray(plane_pcd.points)
        centroid = plane_pts.mean(axis=0)

        planes.append(
            {
                "index": i + 1,
                "model": model,
                "pcd": plane_pcd,
                "count": len(inliers),
                "centroid": centroid,
            }
        )

        print(
            f"平面 {i + 1}: 点数={len(inliers)} "
            f"model=({model[0]:.6f}, {model[1]:.6f}, {model[2]:.6f}, {model[3]:.6f})"
        )
        remaining = remaining.select_by_index(inliers, invert=True)

    if len(planes) == 0:
        raise ValueError("未提取到有效平面，请增大 distance_threshold 或降低 min_plane_points。")
    print(f"RANSAC 共提取平面数量: {len(planes)}")
    return planes


def filter_vertical_planes(planes, normal_z_threshold: float):
    """筛选垂直平面。"""
    vertical_planes = []
    print("筛选垂直平面...")
    for plane in planes:
        model = plane["model"]
        abs_z = abs(float(model[2]))
        if abs_z <= normal_z_threshold:
            vertical_planes.append(plane)
            print(
                f"  保留平面 {plane['index']}: |nz|={abs_z:.6f}, 点数={plane['count']}"
            )
        else:
            print(
                f"  丢弃平面 {plane['index']}: |nz|={abs_z:.6f}, 点数={plane['count']}"
            )

    if len(vertical_planes) < 4:
        raise ValueError(
            "垂直平面数量不足 4 个，请增大 max_planes、降低 min_plane_points，"
            "或放宽 vertical_normal_z_threshold。"
        )
    return vertical_planes


def plane_xy_direction(plane):
    """获取垂直平面在 XY 中的单位法向方向。"""
    model = plane["model"]
    xy = np.asarray([model[0], model[1]], dtype=float)
    norm = np.linalg.norm(xy)
    if norm < 1e-12:
        return None
    return xy / norm


def cluster_parallel_planes(vertical_planes, angle_threshold_deg: float):
    """根据 XY 法向方向将垂直平面分为若干平行组。"""
    cos_threshold = np.cos(np.deg2rad(angle_threshold_deg))
    sorted_planes = sorted(vertical_planes, key=lambda item: item["count"], reverse=True)
    groups = []

    for plane in sorted_planes:
        direction = plane_xy_direction(plane)
        if direction is None:
            continue

        target_group = None
        target_sign = 1.0
        for group in groups:
            dot = float(np.dot(direction, group["direction"]))
            if abs(dot) >= cos_threshold:
                target_group = group
                target_sign = 1.0 if dot >= 0 else -1.0
                break

        if target_group is None:
            groups.append(
                {
                    "direction": direction,
                    "members": [plane],
                    "direction_sum": direction * plane["count"],
                    "total_count": plane["count"],
                }
            )
        else:
            target_group["members"].append(plane)
            target_group["direction_sum"] += target_sign * direction * plane["count"]
            target_group["total_count"] += plane["count"]
            new_dir = target_group["direction_sum"]
            target_group["direction"] = new_dir / np.linalg.norm(new_dir)

    valid_groups = []
    print("平行平面分组结果:")
    for i, group in enumerate(groups, start=1):
        member_ids = [member["index"] for member in group["members"]]
        direction = group["direction"]
        selected_planes = select_two_largest_parallel_planes(group, cos_threshold)
        print(
            f"  组 {i}: 平面={member_ids}, 总点数={group['total_count']}, "
            f"XY法向=({direction[0]:.6f}, {direction[1]:.6f})"
        )
        if selected_planes is None:
            print("    可用平面不足 2 个，丢弃该组。")
            continue

        plane_a, plane_b, parallel_score = selected_planes
        group["members"] = [plane_a, plane_b]
        group["total_count"] = plane_a["count"] + plane_b["count"]
        group["low"] = plane_a
        group["high"] = plane_b
        group["parallel_score"] = parallel_score
        print(
            f"    保留平面 {plane_a['index']} / {plane_b['index']}, "
            f"点数={group['total_count']}, 平行度={parallel_score:.6f}"
        )
        valid_groups.append(group)

    valid_groups = sorted(valid_groups, key=lambda item: item["total_count"], reverse=True)

    if len(valid_groups) < 2:
        raise ValueError(
            "可用平行平面组不足 2 组，请放宽 parallel_angle_threshold "
            "或检查 RANSAC 提取出的垂直平面。"
        )
    return valid_groups


def select_two_largest_parallel_planes(group, cos_threshold: float):
    """在一个平行组内保留两个最大且最平行的平面。"""
    members = group["members"]
    if len(members) < 2:
        return None

    best = None
    best_score = -np.inf
    for i in range(len(members)):
        direction_i = plane_xy_direction(members[i])
        if direction_i is None:
            continue
        for j in range(i + 1, len(members)):
            direction_j = plane_xy_direction(members[j])
            if direction_j is None:
                continue

            parallel_score = abs(float(np.dot(direction_i, direction_j)))
            if parallel_score < cos_threshold:
                continue

            size_score = members[i]["count"] + members[j]["count"]
            score = size_score * parallel_score
            if score > best_score:
                best_score = score
                best = (members[i], members[j], parallel_score)

    return best


def plane_offset_at_z(plane, group_direction, z_ref: float):
    """计算平面在指定 z 截面中沿组法向的有向偏移。"""
    model = plane["model"]
    xy_norm = np.linalg.norm(model[:2])
    if xy_norm < 1e-12:
        return 0.0

    xy_direction = model[:2] / xy_norm
    d_xy = (model[3] + model[2] * z_ref) / xy_norm
    if np.dot(xy_direction, group_direction) < 0:
        d_xy = -d_xy
    return -float(d_xy)


def choose_two_parallel_groups(groups, z_ref: float, min_group_angle: float, min_separation: float):
    """选择两组平行平面，直接使用每组内保留的两个平面。"""
    best = None
    best_score = -np.inf

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            direction_a = groups[i]["direction"]
            direction_b = groups[j]["direction"]
            dot = abs(float(np.dot(direction_a, direction_b)))
            angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
            if angle < min_group_angle:
                continue

            boundary_a = build_boundary_from_group(groups[i], z_ref)
            boundary_b = build_boundary_from_group(groups[j], z_ref)
            if (
                boundary_a["separation"] < min_separation
                or boundary_b["separation"] < min_separation
            ):
                continue

            score = (
                boundary_a["total_count"]
                + boundary_b["total_count"]
                + 100.0 * boundary_a["separation"] * boundary_b["separation"]
            )
            if score > best_score:
                best_score = score
                best = (boundary_a, boundary_b, angle)

    if best is None:
        raise ValueError(
            "未找到满足条件的两组平行舱壁平面，"
            "请降低 min_group_angle 或 min_plane_separation。"
        )

    boundary_a, boundary_b, angle = best
    print("选中的两组舱壁边界平面:")
    print(
        f"  A组: 平面 {boundary_a['low']['index']} / {boundary_a['high']['index']}, "
        f"间距={boundary_a['separation']:.6f}"
    )
    print(
        f"  B组: 平面 {boundary_b['low']['index']} / {boundary_b['high']['index']}, "
        f"间距={boundary_b['separation']:.6f}"
    )
    print(f"  两组法向夹角: {angle:.3f} deg")
    return boundary_a, boundary_b


def build_boundary_from_group(group, z_ref: float):
    """根据组内已保留的两个平面构建边界描述。"""
    plane_a = group["members"][0]
    plane_b = group["members"][1]
    offset_a = plane_offset_at_z(plane_a, group["direction"], z_ref)
    offset_b = plane_offset_at_z(plane_b, group["direction"], z_ref)

    if offset_a <= offset_b:
        low_plane = plane_a
        high_plane = plane_b
        low_offset = offset_a
        high_offset = offset_b
    else:
        low_plane = plane_b
        high_plane = plane_a
        low_offset = offset_b
        high_offset = offset_a

    return {
        "low": low_plane,
        "high": high_plane,
        "low_offset": low_offset,
        "high_offset": high_offset,
        "separation": high_offset - low_offset,
        "direction": group["direction"],
        "total_count": group["total_count"],
        "parallel_score": group.get("parallel_score", 0.0),
    }


def intersect_two_planes(model_a, model_b):
    """计算两个平面的交线，返回交线上的一点和方向。"""
    n1 = model_a[:3]
    n2 = model_b[:3]
    d1 = model_a[3]
    d2 = model_b[3]

    direction = np.cross(n1, n2)
    norm2 = float(np.dot(direction, direction))
    if norm2 < 1e-12:
        raise ValueError("两个平面近似平行，无法计算交线。")

    point = np.cross(d2 * n1 - d1 * n2, direction) / norm2
    direction = direction / np.sqrt(norm2)
    return point, direction


def find_highest_point_near_line(
    pcd: o3d.geometry.PointCloud,
    line_point: np.ndarray,
    line_direction: np.ndarray,
    radius: float,
    expand_times: int,
):
    """在原始点云交线附近搜索 z 值最高点，并投影回交线。"""
    points = np.asarray(pcd.points)
    direction = line_direction / np.linalg.norm(line_direction)
    best_point = None
    best_z = -np.inf
    best_count = 0

    radius_list = [radius * (i + 1) for i in range(max(expand_times, 1))]
    nearest_point = None
    nearest_distance = np.inf
    chunk_size = 200000

    for current_radius in radius_list:
        radius2 = current_radius * current_radius
        current_best_point = None
        current_best_z = -np.inf
        current_count = 0

        for start in range(0, points.shape[0], chunk_size):
            chunk = points[start:start + chunk_size]
            vectors = chunk - line_point
            t = vectors @ direction
            closest = line_point + t[:, None] * direction
            diff = chunk - closest
            dist2 = np.einsum("ij,ij->i", diff, diff)

            local_min = int(np.argmin(dist2))
            if dist2[local_min] < nearest_distance:
                nearest_distance = float(dist2[local_min])
                nearest_point = chunk[local_min]

            mask = dist2 <= radius2
            if np.any(mask):
                selected = chunk[mask]
                local_idx = int(np.argmax(selected[:, 2]))
                candidate = selected[local_idx]
                current_count += selected.shape[0]
                if candidate[2] > current_best_z:
                    current_best_z = float(candidate[2])
                    current_best_point = candidate

        if current_best_point is not None:
            best_point = current_best_point
            best_z = current_best_z
            best_count = current_count
            print(
                f"交线附近搜索半径 {current_radius:.6f}: "
                f"邻近点数={best_count}, 最高Z={best_z:.6f}"
            )
            break

    if best_point is None:
        best_point = nearest_point
        best_z = float(nearest_point[2])
        print(
            f"交线附近未找到半径内点，使用最近点，"
            f"距离={np.sqrt(nearest_distance):.6f}, Z={best_z:.6f}"
        )

    t = float(np.dot(best_point - line_point, direction))
    corner = line_point + t * direction
    return corner, best_point, best_count


def order_polygon_points_xy(points_3d: np.ndarray):
    """按 XY 中心角排序四个顶点。"""
    center = points_3d[:, :2].mean(axis=0)
    angles = np.arctan2(points_3d[:, 1] - center[1], points_3d[:, 0] - center[0])
    return points_3d[np.argsort(angles)]


def expand_polygon_xy(points_3d: np.ndarray, margin_x: float, margin_y: float):
    """分别沿 X/Y 轴扩张或收缩 XY 顶点。"""
    if abs(margin_x) < 1e-12 and abs(margin_y) < 1e-12:
        return points_3d

    expanded = points_3d.copy()
    center = expanded[:, :2].mean(axis=0)
    directions = np.sign(expanded[:, :2] - center)
    margins = np.array([margin_x, margin_y], dtype=float)
    expanded[:, :2] += directions * margins
    return expanded


def calculate_corner_points(
    pcd_original: o3d.geometry.PointCloud,
    boundary_a,
    boundary_b,
    line_search_radius: float,
    line_search_expand: int,
    crop_margin_x: float,
    crop_margin_y: float,
):
    """由两组边界平面计算四条交线和四个角点。"""
    plane_pairs = [
        (boundary_a["low"], boundary_b["low"]),
        (boundary_a["high"], boundary_b["low"]),
        (boundary_a["high"], boundary_b["high"]),
        (boundary_a["low"], boundary_b["high"]),
    ]

    corners = []
    line_sets = []
    print("开始计算平面交线和四个角点...")
    for i, (plane_a, plane_b) in enumerate(plane_pairs, start=1):
        line_point, line_direction = intersect_two_planes(
            plane_a["model"],
            plane_b["model"],
        )
        corner, raw_top_point, near_count = find_highest_point_near_line(
            pcd_original,
            line_point,
            line_direction,
            radius=line_search_radius,
            expand_times=line_search_expand,
        )
        corners.append(corner)
        line_sets.append((line_point, line_direction, corner))
        print(
            f"  角点 {i}: 平面 {plane_a['index']} x {plane_b['index']} -> "
            f"({corner[0]:.6f}, {corner[1]:.6f}, {corner[2]:.6f}), "
            f"原始最高点=({raw_top_point[0]:.6f}, {raw_top_point[1]:.6f}, {raw_top_point[2]:.6f}), "
            f"邻近点数={near_count}"
        )

    corner_points = order_polygon_points_xy(np.asarray(corners, dtype=float))
    corner_points = expand_polygon_xy(corner_points, crop_margin_x, crop_margin_y)

    print("自动提取的四个裁剪顶点:")
    for i, point in enumerate(corner_points.tolist(), start=1):
        print(f"  P{i}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")

    return corner_points, line_sets


def polygon_mask_xy(points_xy: np.ndarray, polygon_xy: np.ndarray):
    """XY 平面点在多边形内判断。"""
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    poly = polygon_xy
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    cond = (y0 > y[:, None]) != (y1 > y[:, None])
    x_intersect = (x1 - x0) * (y[:, None] - y0) / (y1 - y0 + 1e-20) + x0
    crossings = cond & (x[:, None] < x_intersect)
    inside = crossings.sum(axis=1) % 2 == 1

    return inside


def crop_by_corner_points(
    pcd: o3d.geometry.PointCloud,
    corner_points: np.ndarray,
    z_min,
    z_max,
    z_margin: float,
):
    """根据四个角点进行 XY 多边形裁剪，并使用角点最高 Z 作为默认上界。"""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=int), (0.0, 0.0)

    raw_z_min = float(pts[:, 2].min()) if z_min is None else float(z_min)
    raw_z_max = float(corner_points[:, 2].max()) if z_max is None else float(z_max)
    crop_z_min = raw_z_min - z_margin
    crop_z_max = raw_z_max + z_margin

    mask_xy = polygon_mask_xy(pts[:, :2], corner_points[:, :2])
    mask_z = (pts[:, 2] >= crop_z_min) & (pts[:, 2] <= crop_z_max)
    mask = mask_xy & mask_z

    ind = np.flatnonzero(mask).astype(int)
    cropped = pcd.select_by_index(ind.tolist())

    print("根据四个舱壁交点自动裁剪点云...")
    print(f"Z 裁剪范围: [{crop_z_min:.6f}, {crop_z_max:.6f}]")
    print(f"原始点数: {pts.shape[0]}")
    print(f"保留点数: {ind.size}")
    print(f"移除点数: {pts.shape[0] - ind.size}")
    if pts.shape[0] > 0:
        ratio = ind.size / pts.shape[0]
        print(f"保留比例: {ratio:.4f} ({ratio * 100:.2f}%)")

    return cropped, ind, (crop_z_min, crop_z_max)


def merge_plane_clouds(planes):
    """合并多个平面点云。"""
    merged = o3d.geometry.PointCloud()
    for plane in planes:
        merged += plane["pcd"]
    return merged


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str) -> None:
    """保存点云到指定路径。"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=True)
    if not success:
        raise RuntimeError(f"点云保存失败: {out_path}")
    print(f"点云已保存到: {out_path}")


def build_polygon_lines(corner_points: np.ndarray):
    """构建四点边界线框。"""
    points = corner_points
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([0.0, 0.0, 1.0])
    return line_set


def build_intersection_lines(line_sets, z_min: float, z_max: float):
    """构建交线可视化线段。"""
    points = []
    lines = []
    for line_point, line_direction, _ in line_sets:
        direction = line_direction / np.linalg.norm(line_direction)
        if abs(direction[2]) < 1e-6:
            continue
        t_min = (z_min - line_point[2]) / direction[2]
        t_max = (z_max - line_point[2]) / direction[2]
        p0 = line_point + t_min * direction
        p1 = line_point + t_max * direction
        idx = len(points)
        points.extend([p0, p1])
        lines.append([idx, idx + 1])

    if len(points) == 0:
        return None

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points)),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([0.0, 0.0, 1.0])
    return line_set


def visualize_result(
    pcd_original: o3d.geometry.PointCloud,
    pcd_cropped: o3d.geometry.PointCloud,
    wall_pcd: o3d.geometry.PointCloud,
    corner_points: np.ndarray,
    line_sets,
    z_range,
    vis_mode: str,
    window_name: str,
):
    """可视化原始点云、裁剪结果、舱壁平面和交线。"""
    if vis_mode == "none":
        print("跳过可视化显示。")
        return

    geometries = []
    if vis_mode in ("before_after", "all"):
        pcd_orig_vis = o3d.geometry.PointCloud(pcd_original)
        pcd_orig_vis.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(pcd_orig_vis)

    if vis_mode in ("after_only", "before_after", "wall_after", "all"):
        if len(pcd_cropped.points) == 0:
            print("裁剪后点云为空，跳过裁剪结果可视化。")
        else:
            pcd_crop_vis = o3d.geometry.PointCloud(pcd_cropped)
            pcd_crop_vis.paint_uniform_color([0.0, 1.0, 0.0])
            geometries.append(pcd_crop_vis)

    if vis_mode in ("wall_only", "wall_after", "all"):
        wall_vis = o3d.geometry.PointCloud(wall_pcd)
        wall_vis.paint_uniform_color([0.0, 0.0, 1.0])
        geometries.append(wall_vis)

    if vis_mode in ("wall_only", "wall_after", "all", "before_after"):
        polygon_lines = build_polygon_lines(corner_points)
        geometries.append(polygon_lines)
        intersection_lines = build_intersection_lines(line_sets, z_range[0], z_range[1])
        if intersection_lines is not None:
            geometries.append(intersection_lines)

    if len(geometries) == 0:
        print("没有可显示的几何对象。")
        return

    print("打开可视化窗口：红色=原始点云，绿色=裁剪结果，蓝色=舱壁平面/边界线")
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def derive_crop_spec(
    pcd_before: o3d.geometry.PointCloud,
    voxel_size: float = 0.04,
    distance_threshold: float = 0.08,
    ransac_n: int = 3,
    num_iterations: int = 10000,
    max_planes: int = 16,
    min_plane_points: int = 500,
    vertical_normal_z_threshold: float = 0.2,
    parallel_angle_threshold: float = 12.0,
    min_group_angle: float = 45.0,
    min_plane_separation: float = 0.3,
    line_search_radius: float = 0.1,
    line_search_expand: int = 4,
    crop_margin_x: float = -0.2,
    crop_margin_y: float = -0.15,
    z_min=None,
    z_max=None,
    z_margin: float = 0.2,
):
    """从装沙前点云自动推导裁剪范围，返回裁剪规格和可视化辅助对象。"""
    pcd_down = voxel_downsample_centroid(pcd_before, voxel_size)
    planes = extract_planes_ransac(
        pcd_down,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        max_planes=max_planes,
        min_plane_points=min_plane_points,
    )
    vertical_planes = filter_vertical_planes(
        planes,
        normal_z_threshold=vertical_normal_z_threshold,
    )
    groups = cluster_parallel_planes(
        vertical_planes,
        angle_threshold_deg=parallel_angle_threshold,
    )

    pts = np.asarray(pcd_before.points)
    z_ref = float(np.median(pts[:, 2]))
    boundary_a, boundary_b = choose_two_parallel_groups(
        groups,
        z_ref=z_ref,
        min_group_angle=min_group_angle,
        min_separation=min_plane_separation,
    )

    selected_wall_planes = [
        boundary_a["low"],
        boundary_a["high"],
        boundary_b["low"],
        boundary_b["high"],
    ]
    wall_pcd = merge_plane_clouds(selected_wall_planes)
    corner_points, line_sets = calculate_corner_points(
        pcd_before,
        boundary_a,
        boundary_b,
        line_search_radius=line_search_radius,
        line_search_expand=line_search_expand,
        crop_margin_x=crop_margin_x,
        crop_margin_y=crop_margin_y,
    )
    _, _, z_range = crop_by_corner_points(
        pcd_before,
        corner_points,
        z_min=z_min,
        z_max=z_max,
        z_margin=z_margin,
    )
    spec = {
        "corner_points": corner_points.tolist(),
        "polygon_xy": corner_points[:, :2].tolist(),
        "z_range": [float(z_range[0]), float(z_range[1])],
        "z_min": float(z_range[0]),
        "z_max": float(z_range[1]),
        "parameters": {
            "voxel_size": voxel_size,
            "distance_threshold": distance_threshold,
            "ransac_n": ransac_n,
            "num_iterations": num_iterations,
            "max_planes": max_planes,
            "min_plane_points": min_plane_points,
            "vertical_normal_z_threshold": vertical_normal_z_threshold,
            "parallel_angle_threshold": parallel_angle_threshold,
            "min_group_angle": min_group_angle,
            "min_plane_separation": min_plane_separation,
            "line_search_radius": line_search_radius,
            "line_search_expand": line_search_expand,
            "crop_margin_x": crop_margin_x,
            "crop_margin_y": crop_margin_y,
            "z_margin": z_margin,
        },
    }
    return spec, wall_pcd, line_sets


def crop_with_spec(pcd: o3d.geometry.PointCloud, crop_spec):
    """使用已保存的裁剪规格裁剪点云。"""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=int)

    polygon_xy = np.asarray(
        crop_spec.get("polygon_xy", np.asarray(crop_spec["corner_points"])[:, :2]),
        dtype=float,
    )
    z_range = crop_spec.get("z_range", [crop_spec["z_min"], crop_spec["z_max"]])
    z_min, z_max = float(z_range[0]), float(z_range[1])

    mask_xy = polygon_mask_xy(pts[:, :2], polygon_xy)
    mask_z = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    ind = np.flatnonzero(mask_xy & mask_z).astype(int)
    cropped = pcd.select_by_index(ind.tolist())
    print("使用固定裁剪规格裁剪点云...")
    print(f"Z 裁剪范围: [{z_min:.6f}, {z_max:.6f}]")
    print(f"原始点数: {pts.shape[0]}")
    print(f"保留点数: {ind.size}")
    return cropped, ind


def save_crop_spec(crop_spec, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(crop_spec, f, ensure_ascii=False, indent=2)
    print(f"裁剪规格已保存到: {output_path}")


def load_crop_spec(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_crop_pair_files(
    input_before,
    input_after=None,
    output_before=None,
    output_after=None,
    crop_spec_output=None,
    **kwargs,
):
    """文件级自动裁剪接口：从 before 推导范围，并用同一范围裁剪 before/after。"""
    input_before = Path(input_before)
    pcd_before = load_point_cloud(str(input_before))
    pcd_after = None
    if input_after is not None and str(input_after) != "":
        pcd_after = load_point_cloud(str(input_after))

    crop_spec, wall_pcd, line_sets = derive_crop_spec(pcd_before, **kwargs)
    before_cropped, _ = crop_with_spec(pcd_before, crop_spec)
    if output_before is None or str(output_before) == "":
        output_before = input_before.parent / f"{input_before.stem}_autoptf2_crop.pcd"
    save_point_cloud(before_cropped, str(output_before))

    after_cropped = None
    if pcd_after is not None:
        input_after = Path(input_after)
        after_cropped, _ = crop_with_spec(pcd_after, crop_spec)
        if output_after is None or str(output_after) == "":
            output_after = input_after.parent / f"{input_after.stem}_autoptf2_crop.pcd"
        save_point_cloud(after_cropped, str(output_after))

    if crop_spec_output is not None and str(crop_spec_output) != "":
        save_crop_spec(crop_spec, crop_spec_output)

    return before_cropped, after_cropped, crop_spec, wall_pcd, line_sets


def main():
    args = parse_args()

    input_before_path = Path(args.input_before)
    input_after_path = Path(args.input_after)
    pcd = load_point_cloud(str(input_before_path))
    pcd_down = voxel_downsample_centroid(pcd, args.voxel_size)

    if input_after_path != "":
        pcd_after = load_point_cloud(str(input_after_path))

    planes = extract_planes_ransac(
        pcd_down,
        distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n,
        num_iterations=args.num_iterations,
        max_planes=args.max_planes,
        min_plane_points=args.min_plane_points,
    )
    vertical_planes = filter_vertical_planes(
        planes,
        normal_z_threshold=args.vertical_normal_z_threshold,
    )
    groups = cluster_parallel_planes(
        vertical_planes,
        angle_threshold_deg=args.parallel_angle_threshold,
    )

    pts = np.asarray(pcd.points)
    z_ref = float(np.median(pts[:, 2]))
    boundary_a, boundary_b = choose_two_parallel_groups(
        groups,
        z_ref=z_ref,
        min_group_angle=args.min_group_angle,
        min_separation=args.min_plane_separation,
    )

    selected_wall_planes = [
        boundary_a["low"],
        boundary_a["high"],
        boundary_b["low"],
        boundary_b["high"],
    ]
    wall_pcd = merge_plane_clouds(selected_wall_planes)

    corner_points, line_sets = calculate_corner_points(
        pcd,
        boundary_a,
        boundary_b,
        line_search_radius=args.line_search_radius,
        line_search_expand=args.line_search_expand,
        crop_margin_x=args.crop_margin_x,
        crop_margin_y=args.crop_margin_y,
    )

    pcd_cropped, _, z_range = crop_by_corner_points(
        pcd,
        corner_points,
        z_min=args.z_min,
        z_max=args.z_max,
        z_margin=args.z_margin,
    )
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_before_path.parent / f"{input_before_path.stem}_autoptf2_crop.pcd"
    save_point_cloud(pcd_cropped, str(output_path))

    if args.wall_output:
        save_point_cloud(wall_pcd, args.wall_output)

    if input_after_path != "":
        pcd_after_cropped, _, z_range = crop_by_corner_points(
            pcd_after,
            corner_points,
            z_min=args.z_min,
            z_max=args.z_max,
            z_margin=args.z_margin,
        )
        if args.output_after:
            output_after_path = Path(args.output_after)
        else:
            output_after_path = input_after_path.parent / f"{input_after_path.stem}_autoptf2_crop.pcd"
        save_point_cloud(pcd_after_cropped, str(output_after_path))


    print("用于裁剪的四个顶点如下:")
    for i, point in enumerate(corner_points.tolist(), start=1):
        print(f"  P{i}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")

    window_name = f"{input_before_path.name} Auto PTF 2"
    visualize_result(
        pcd_original=pcd,
        pcd_cropped=pcd_cropped,
        wall_pcd=wall_pcd,
        corner_points=corner_points,
        line_sets=line_sets,
        z_range=z_range,
        vis_mode=args.vis_mode,
        window_name=window_name,
    )

    if input_after_path != "":
        window_name2 = f"{input_after_path.name} Auto PTF 2"
        visualize_result(
            pcd_original=pcd_after,
            pcd_cropped=pcd_after_cropped,
            wall_pcd=wall_pcd,
            corner_points=corner_points,
            line_sets=line_sets,
            z_range=z_range,
            vis_mode=args.vis_mode,
            window_name=window_name2,
        )



if __name__ == "__main__":
    main()
