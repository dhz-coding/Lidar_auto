# 空洞补全_IDW_连通域
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


DEFAULT_PC_PATH = Path(r"C:\Users\xm\Desktop\boatdata\boat2\after\boat23_voxel_0.02.pcd")

GRID_SIZE = 0.1
N_MIN = 1

SMALL_HOLE_MAX_CELLS = 20
MEDIUM_HOLE_MAX_CELLS = 50

IDW_POWER_SMALL = 2.5
IDW_RADIUS_SMALL = 3

IDW_POWER_MEDIUM = 2.0
IDW_RADIUS_MEDIUM = 5

IDW_POWER_LARGE = 1.5
IDW_RADIUS_LARGE = 9

MIN_COMPONENT_CELLS = 1
MAX_COMPONENT_CELLS = 300
MIN_BOUNDARY_SUPPORT_RATIO = 0.2
MIN_BOUNDARY_SUPPORT_CELLS = 5
REJECT_TOUCHING_GRID_EDGE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用栅格连通域 + IDW 对点云空洞进行补全",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_PC_PATH), help="输入点云 PCD 文件路径")
    parser.add_argument("--output", type=str, default="", help="输出点云 PCD 文件路径")
    parser.add_argument("--grid_size", type=float, default=GRID_SIZE, help="栅格分辨率")
    parser.add_argument("--n_min", type=int, default=N_MIN, help="栅格内最少点数")
    parser.add_argument("--small_hole_max_cells", type=int, default=SMALL_HOLE_MAX_CELLS)
    parser.add_argument("--medium_hole_max_cells", type=int, default=MEDIUM_HOLE_MAX_CELLS)
    parser.add_argument("--idw_power_small", type=float, default=IDW_POWER_SMALL)
    parser.add_argument("--idw_radius_small", type=int, default=IDW_RADIUS_SMALL)
    parser.add_argument("--idw_power_medium", type=float, default=IDW_POWER_MEDIUM)
    parser.add_argument("--idw_radius_medium", type=int, default=IDW_RADIUS_MEDIUM)
    parser.add_argument("--idw_power_large", type=float, default=IDW_POWER_LARGE)
    parser.add_argument("--idw_radius_large", type=int, default=IDW_RADIUS_LARGE)
    parser.add_argument("--min_component_cells", type=int, default=MIN_COMPONENT_CELLS)
    parser.add_argument("--max_component_cells", type=int, default=MAX_COMPONENT_CELLS)
    parser.add_argument("--min_boundary_support_ratio", type=float, default=MIN_BOUNDARY_SUPPORT_RATIO)
    parser.add_argument("--min_boundary_support_cells", type=int, default=MIN_BOUNDARY_SUPPORT_CELLS)
    parser.add_argument("--reject_touching_grid_edge", action="store_true")
    parser.add_argument(
        "--vis_mode",
        choices=["none", "after_only"],
        default="after_only",
        help="可视化模式",
    )
    return parser.parse_args()


def _classify_by_cells(cell_count, small_hole_max_cells, medium_hole_max_cells):
    if cell_count < small_hole_max_cells:
        return "small"
    if cell_count < medium_hole_max_cells:
        return "medium"
    return "large"


def fill_holes_idw(
    pcd: o3d.geometry.PointCloud,
    grid_size: float = GRID_SIZE,
    n_min: int = N_MIN,
    small_hole_max_cells: int = SMALL_HOLE_MAX_CELLS,
    medium_hole_max_cells: int = MEDIUM_HOLE_MAX_CELLS,
    idw_power_small: float = IDW_POWER_SMALL,
    idw_radius_small: int = IDW_RADIUS_SMALL,
    idw_power_medium: float = IDW_POWER_MEDIUM,
    idw_radius_medium: int = IDW_RADIUS_MEDIUM,
    idw_power_large: float = IDW_POWER_LARGE,
    idw_radius_large: int = IDW_RADIUS_LARGE,
    min_component_cells: int = MIN_COMPONENT_CELLS,
    max_component_cells: int = MAX_COMPONENT_CELLS,
    min_boundary_support_ratio: float = MIN_BOUNDARY_SUPPORT_RATIO,
    min_boundary_support_cells: int = MIN_BOUNDARY_SUPPORT_CELLS,
    reject_touching_grid_edge: bool = REJECT_TOUCHING_GRID_EDGE,
):
    """使用空洞连通域筛选和 IDW 插值补全点云，返回补全后的点云和统计信息。"""
    points = np.asarray(pcd.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise ValueError("输入点云为空或不包含有效点。")
    if grid_size <= 0:
        raise ValueError("grid_size 必须大于 0。")

    print("Original points:", points.shape)

    xmin, ymin = points[:, 0].min(), points[:, 1].min()
    xmax, ymax = points[:, 0].max(), points[:, 1].max()

    nx = max(int(np.ceil((xmax - xmin) / grid_size)), 1)
    ny = max(int(np.ceil((ymax - ymin) / grid_size)), 1)

    ix = ((points[:, 0] - xmin) / grid_size).astype(int)
    iy = ((points[:, 1] - ymin) / grid_size).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    count = np.zeros((nx, ny), dtype=int)
    for i, j in zip(ix, iy):
        count[i, j] += 1

    occupied = count >= n_min
    empty = ~occupied
    labels = -np.ones((nx, ny), dtype=np.int32)
    components = []
    neighbors8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    label_id = 0
    for i in range(nx):
        for j in range(ny):
            if not empty[i, j] or labels[i, j] != -1:
                continue

            q = deque([(i, j)])
            labels[i, j] = label_id
            comp_cells = []
            touches_edge = False
            while q:
                ci, cj = q.popleft()
                comp_cells.append((ci, cj))
                if ci == 0 or ci == nx - 1 or cj == 0 or cj == ny - 1:
                    touches_edge = True
                for di, dj in neighbors8:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        if empty[ni, nj] and labels[ni, nj] == -1:
                            labels[ni, nj] = label_id
                            q.append((ni, nj))

            components.append({
                "label": label_id,
                "cells": comp_cells,
                "touches_edge": touches_edge,
            })
            label_id += 1

    hole_mask = np.zeros((nx, ny), dtype=bool)
    accepted_components = 0
    for comp in components:
        cells = comp["cells"]
        cell_count = len(cells)
        if cell_count < min_component_cells or cell_count > max_component_cells:
            continue
        if reject_touching_grid_edge and comp["touches_edge"]:
            continue

        boundary_cells = set()
        for ci, cj in cells:
            for di, dj in neighbors8:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < nx and 0 <= nj < ny and labels[ni, nj] != comp["label"]:
                    boundary_cells.add((ni, nj))

        if not boundary_cells:
            continue

        support_count = sum(1 for bi, bj in boundary_cells if occupied[bi, bj])
        support_ratio = support_count / len(boundary_cells)
        if support_count < min_boundary_support_cells:
            continue
        if support_ratio < min_boundary_support_ratio:
            continue

        for ci, cj in cells:
            hole_mask[ci, cj] = True
        accepted_components += 1

    print("Empty components found:", len(components))
    print("Accepted hole components:", accepted_components)
    print("Hole grids detected:", int(np.sum(hole_mask)))

    label_to_metrics = {
        comp["label"]: {"size": len(comp["cells"])}
        for comp in components
    }

    xy = points[:, :2]
    z = points[:, 2]
    tree = cKDTree(xy)

    filled_points = []
    filled_by_tier = {"small": 0, "medium": 0, "large": 0}
    for i, j in zip(*np.where(hole_mask)):
        cx = xmin + (i + 0.5) * grid_size
        cy = ymin + (j + 0.5) * grid_size

        comp_label = labels[i, j]
        comp_size = label_to_metrics.get(comp_label, {"size": 0})["size"]
        tier = _classify_by_cells(
            comp_size,
            small_hole_max_cells=small_hole_max_cells,
            medium_hole_max_cells=medium_hole_max_cells,
        )

        if tier == "small":
            idw_power = idw_power_small
            idw_radius = idw_radius_small
        elif tier == "medium":
            idw_power = idw_power_medium
            idw_radius = idw_radius_medium
        else:
            idw_power = idw_power_large
            idw_radius = idw_radius_large

        idx = tree.query_ball_point([cx, cy], idw_radius * grid_size)
        if len(idx) < 3:
            continue

        dists = np.linalg.norm(xy[idx] - np.array([cx, cy]), axis=1)
        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / (dists ** idw_power)
        z_interp = np.sum(weights * z[idx]) / np.sum(weights)

        filled_points.append([cx, cy, z_interp])
        filled_by_tier[tier] += 1

    filled_points = np.asarray(filled_points, dtype=np.float64)
    print("Interpolated points:", filled_points.shape)
    print(
        "Filled grids by tier: "
        f"small={filled_by_tier['small']}, "
        f"medium={filled_by_tier['medium']}, "
        f"large={filled_by_tier['large']}"
    )

    final_points = points
    if len(filled_points) > 0:
        final_points = np.vstack([points, filled_points])

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(final_points)

    report = {
        "original_points": int(points.shape[0]),
        "final_points": int(final_points.shape[0]),
        "interpolated_points": int(filled_points.shape[0]),
        "empty_components": int(len(components)),
        "accepted_components": int(accepted_components),
        "hole_grids": int(np.sum(hole_mask)),
        "filled_small": int(filled_by_tier["small"]),
        "filled_medium": int(filled_by_tier["medium"]),
        "filled_large": int(filled_by_tier["large"]),
    }
    return pcd_out, report


def fill_holes_file(input_path, output_path=None, **kwargs):
    input_path = Path(input_path)
    if output_path is None or str(output_path) == "":
        output_path = input_path.with_name(input_path.stem + "_grid_idw_filled_once.pcd")
    output_path = Path(output_path)

    pcd = o3d.io.read_point_cloud(str(input_path))
    filled, report = fill_holes_idw(pcd, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(output_path), filled)
    if not ok:
        raise RuntimeError(f"点云保存失败: {output_path}")
    print("Final point count:", report["final_points"])
    print("Saved to:", output_path)
    return filled, report, output_path


def main():
    args = parse_args()
    pcd_out, _, _ = fill_holes_file(
        args.input,
        output_path=args.output,
        grid_size=args.grid_size,
        n_min=args.n_min,
        small_hole_max_cells=args.small_hole_max_cells,
        medium_hole_max_cells=args.medium_hole_max_cells,
        idw_power_small=args.idw_power_small,
        idw_radius_small=args.idw_radius_small,
        idw_power_medium=args.idw_power_medium,
        idw_radius_medium=args.idw_radius_medium,
        idw_power_large=args.idw_power_large,
        idw_radius_large=args.idw_radius_large,
        min_component_cells=args.min_component_cells,
        max_component_cells=args.max_component_cells,
        min_boundary_support_ratio=args.min_boundary_support_ratio,
        min_boundary_support_cells=args.min_boundary_support_cells,
        reject_touching_grid_edge=args.reject_touching_grid_edge,
    )
    if args.vis_mode == "after_only":
        o3d.visualization.draw_geometries([pcd_out], window_name="Hole-filled point cloud")
    else:
        print("跳过可视化显示")


if __name__ == "__main__":
    main()
