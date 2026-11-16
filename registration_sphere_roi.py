"""Spherical ROI filtering used only for registration inputs."""

import numpy as np
import open3d as o3d


DEFAULT_SPHERE_ROI = {
    "enabled": False,
    "center": [0.0, 0.0, 0.0],
    "radius": 8.0,
}


def normalize_sphere_roi(settings=None):
    """Validate and normalize spherical ROI settings."""
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("sphere_roi 必须是 YAML 字典")

    result = {**DEFAULT_SPHERE_ROI, **settings}
    center = np.asarray(result["center"], dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("sphere_roi.center 必须是包含 3 个有限数值的数组")

    radius = float(result["radius"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("sphere_roi.radius 必须是大于 0 的有限数值")

    return {
        "enabled": bool(result["enabled"]),
        "center": center.tolist(),
        "radius": radius,
    }


def filter_point_cloud_by_sphere(pcd, settings=None, label=""):
    """Return a filtered copy and a JSON-serializable filtering report."""
    normalized = normalize_sphere_roi(settings)
    input_points = len(pcd.points)

    if not normalized["enabled"]:
        report = {
            **normalized,
            "input_points": input_points,
            "kept_points": input_points,
            "removed_points": 0,
            "keep_ratio": 1.0 if input_points else 0.0,
        }
        return o3d.geometry.PointCloud(pcd), report

    points = np.asarray(pcd.points)
    center = np.asarray(normalized["center"], dtype=np.float64)
    radius = normalized["radius"]
    finite = np.isfinite(points).all(axis=1)
    distances_squared = np.sum((points - center) ** 2, axis=1)
    mask = finite & (distances_squared <= radius * radius)
    filtered = pcd.select_by_index(np.flatnonzero(mask).tolist())
    kept_points = len(filtered.points)

    report = {
        **normalized,
        "input_points": input_points,
        "kept_points": kept_points,
        "removed_points": input_points - kept_points,
        "keep_ratio": kept_points / input_points if input_points else 0.0,
    }
    prefix = f"{label}: " if label else ""
    print(
        f"[球形 ROI] {prefix}中心={normalized['center']}，半径={radius:g} m，"
        f"原始点数={input_points}，保留={kept_points} "
        f"({report['keep_ratio']:.1%})",
        flush=True,
    )
    return filtered, report
