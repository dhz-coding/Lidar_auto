"""Fixed full-cloud nearest-neighbor validation; no ground truth is needed."""

import math
import numpy as np
from scipy.spatial import cKDTree
import registration_curvature_roi as roi


DEFAULT_VALIDATION = {
    "voxel_size": 0.1,
    "distance_threshold": 0.2,
    "min_fitness": 0.85,
    "max_rmse": 0.08,
    "review_min_fitness": 0.7,
    "review_max_rmse": 0.1,
    "min_correspondences": 100,
    "max_translation_scene_ratio": 2.0,
}


def validation_parameters(parameters=None):
    supplied = dict(parameters or {})
    if set(supplied) - set(DEFAULT_VALIDATION):
        raise ValueError(f"Unknown validation parameters: {sorted(set(supplied) - set(DEFAULT_VALIDATION))}")
    result = {**DEFAULT_VALIDATION, **supplied}
    for key, value in result.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
    for key in ("voxel_size", "distance_threshold", "max_rmse", "review_max_rmse", "max_translation_scene_ratio"):
        if result[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= result["review_min_fitness"] <= result["min_fitness"] <= 1:
        raise ValueError("Require 0 <= review_min_fitness <= min_fitness <= 1")
    if result["max_rmse"] > result["review_max_rmse"]:
        raise ValueError("max_rmse must not exceed review_max_rmse")
    if result["min_correspondences"] < 3 or int(result["min_correspondences"]) != result["min_correspondences"]:
        raise ValueError("min_correspondences must be an integer >= 3")
    return result


class FixedPairValidator:
    """Freeze a source/target snapshot once for all parameter retries of a pair."""

    def __init__(self, source, target, parameters=None):
        self.parameters = validation_parameters(parameters)
        voxel = self.parameters["voxel_size"]
        self.clouds = [cloud.voxel_down_sample(voxel) for cloud in (source, target)]
        self.points = [np.asarray(cloud.points).copy() for cloud in self.clouds]
        if any(len(points) < 3 or not np.isfinite(points).all() for points in self.points):
            raise ValueError("Validation clouds must contain at least three finite points")
        for points in self.points:
            points.setflags(write=False)
        self.trees = [cKDTree(points) for points in self.points]
        minimum = np.minimum(self.points[0].min(axis=0), self.points[1].min(axis=0))
        maximum = np.maximum(self.points[0].max(axis=0), self.points[1].max(axis=0))
        self.scene_diag = float(np.linalg.norm(maximum - minimum))

    def _direction(self, source_index, target_index, transformation):
        tau = self.parameters["distance_threshold"]
        transformed = self.points[source_index] @ transformation[:3, :3].T + transformation[:3, 3]
        distances, _ = self.trees[target_index].query(
            transformed, k=1, distance_upper_bound=tau, workers=1,
        )
        matched = np.isfinite(distances) & (distances < tau)
        count = int(matched.sum())
        return {
            "fitness": count / len(distances),
            "inlier_rmse": float(np.sqrt(np.mean(distances[matched] ** 2))) if count else None,
            "correspondence_count": count,
            "truncated_rmse": float(np.sqrt(np.mean(np.where(matched, distances, tau) ** 2))),
            "point_count": len(distances),
        }

    def evaluate(self, transformation):
        transformation = np.asarray(transformation, dtype=float)
        result = {"success_basis": "geometry_proxy", "parameters": self.parameters,
                  "source_points": len(self.points[0]), "target_points": len(self.points[1])}
        if not roi.valid_transform(transformation):
            return {**result, "state": "rejected", "passed": False, "failure_reasons": ["invalid_rigid_transform"]}
        if np.linalg.norm(transformation[:3, 3]) > self.parameters["max_translation_scene_ratio"] * self.scene_diag:
            return {**result, "state": "rejected", "passed": False, "failure_reasons": ["translation_outside_scene_limit"]}
        forward = self._direction(0, 1, transformation)
        reverse = self._direction(1, 0, np.linalg.inv(transformation))
        metrics = [forward, reverse]
        enough = all(m["correspondence_count"] >= self.parameters["min_correspondences"] for m in metrics)
        fitness = min(m["fitness"] for m in metrics)
        rmse = max(m["inlier_rmse"] if m["inlier_rmse"] is not None else math.inf for m in metrics)
        if enough and fitness >= self.parameters["min_fitness"] and rmse <= self.parameters["max_rmse"]:
            state, reasons = "geometry_passed", []
        elif enough and fitness >= self.parameters["review_min_fitness"] and rmse <= self.parameters["review_max_rmse"]:
            state, reasons = "needs_review", ["borderline_fixed_geometry"]
        else:
            state, reasons = "rejected", ["insufficient_fixed_geometry"]
        return {**result, "state": state, "passed": state == "geometry_passed",
                "failure_reasons": reasons, "forward": forward, "reverse": reverse,
                "symmetric_truncated_rmse": float(np.sqrt(np.mean([m["truncated_rmse"] ** 2 for m in metrics])))}

