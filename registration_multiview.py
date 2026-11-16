"""Two-pass registration: fixed verification, retained candidates and raw-pair evidence.

All transforms map source to target. The reference frame remains the first scan.
No standard transforms, predefined surfaces or weighted acceptance scores are used.
"""

import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from registration_curvature_roi import curvature_roi_parameters, register_curvature_roi, valid_transform
from registration_validation import FixedPairValidator


def transform_difference(actual, expected):
    delta = np.linalg.inv(expected) @ actual
    angle = math.degrees(math.acos(float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    return {"translation_error": float(np.linalg.norm(delta[:3, 3])), "rotation_error_deg": angle}


def within_difference(error, settings):
    return (error["translation_error"] <= settings.get("translation_threshold", .2)
            and error["rotation_error_deg"] <= settings.get("rotation_threshold_deg", 2.0))


def candidate_order(candidate):
    metrics = candidate["fixed_evaluation"]["final"]
    return (0 if metrics["state"] == "geometry_passed" else 1,
            metrics["symmetric_truncated_rmse"],
            -min(metrics[d]["fitness"] for d in ("forward", "reverse")))


def retain_candidates(candidates, settings):
    retained = []
    for candidate in sorted(candidates, key=candidate_order):
        if any(within_difference(transform_difference(np.array(candidate["transformation"]), np.array(other["transformation"])),
                                 {"translation_threshold": settings.get("deduplicate_translation", .01),
                                  "rotation_threshold_deg": settings.get("deduplicate_rotation_deg", .1)}) for other in retained):
            continue
        retained.append(candidate)
    return retained[:int(settings.get("max_candidates", 4))]


def attempt_pair_registration(source, target, pair_name, params):
    """Try the configured solver scales; retain distinct review candidates."""
    started = time.perf_counter()
    validator = FixedPairValidator(source, target, params.get("validation"))
    base = curvature_roi_parameters(params.get("curvature_roi"))
    voxels = params.get("retry_voxel_multipliers", [1.0])
    thresholds = params.get("retry_icp_threshold_multipliers", [1.0])
    if not voxels or not thresholds:
        raise ValueError("重试倍率列表不能为空")
    limit = int(params.get("max_pair_attempts", max(len(voxels), len(thresholds))))
    if limit < 1 or int(params.get("candidate_review", {}).get("max_candidates", 4)) < 1:
        raise ValueError("配准尝试次数和候选数量必须至少为 1")
    attempts, retained = [], []
    for index in range(limit):
        parameters = dict(base)
        parameters["voxel_size"] *= float(voxels[min(index, len(voxels) - 1)])
        parameters["icp_distance_ratio"] *= float(thresholds[min(index, len(thresholds) - 1)])
        attempt_started = time.perf_counter()
        row = {"attempt_index": index + 1, "candidate_id": f"{pair_name}#{index + 1}", "parameters": parameters}
        print(f"[多点位配准] {pair_name}，尝试 {index + 1}/{limit}，求解体素={parameters['voxel_size']:g}；"
              f"固定验证体素={validator.parameters['voxel_size']:g}，距离={validator.parameters['distance_threshold']:g}", flush=True)
        try:
            total, coarse, delta, registration = register_curvature_roi(source, target, parameters)
            check = validator.evaluate(total)
            row.update({"state": check["state"], "passed": check["passed"], "failure_reasons": check["failure_reasons"],
                        "transformation": total.tolist(), "fpfh_transformation": coarse.tolist(),
                        "icp_delta_transformation": delta.tolist(), "registration": registration,
                        "fixed_evaluation": {"coarse": validator.evaluate(coarse), "final": check}})
        except Exception as error:
            row.update({"state": "rejected", "passed": False, "failure_reasons": ["solver_exception"],
                        "exception": repr(error), "registration": getattr(error, "report", None)})
        row["elapsed_seconds"] = time.perf_counter() - attempt_started
        attempts.append(row)
        if row["state"] != "rejected":
            retained = retain_candidates(retained + [row], params.get("candidate_review", {}))
        print(f"[多点位配准] {pair_name}：{row['state']}，保留 {len(retained)} 个候选", flush=True)
        if row["passed"]:
            return True, total, coarse, delta, {"pair_name": pair_name, "success": True,
                "selected_attempt": index + 1, "attempts": attempts, "candidates": retained,
                "transformation": total.tolist(), "elapsed_seconds": time.perf_counter() - started}
    return False, None, None, None, {"pair_name": pair_name, "success": False,
        "attempts": attempts, "candidates": retained, "elapsed_seconds": time.perf_counter() - started}


class TwoPassRegistration:
    def __init__(self, clouds, paths, label, params, pair_solver=None, on_merge=None):
        self.clouds = clouds
        self.paths = [Path(p) for p in paths]
        self.label = label
        self.params = params
        self.solve = pair_solver or attempt_pair_registration
        self.on_merge = on_merge
        self.merged = o3d.geometry.PointCloud(clouds[0])
        self.transforms = [np.eye(4)] + [None] * (len(clouds) - 1)
        self.records = [{"path": str(p), "role": "reference" if i == 0 else "source",
                         "status": "reference" if i == 0 else "pending"} for i, p in enumerate(self.paths)]
        self.records[0]["transform_to_global"] = np.eye(4).tolist()
        self.proof_cache = {}

    def merge(self, index, total, coarse, delta, phase):
        source = o3d.geometry.PointCloud(self.clouds[index])
        source.transform(total)
        record = self.records[index]
        record.update({"status": "registered" if phase == "first_pass" else "registered_after_review",
                       "phase": phase, "transform_to_global": total.tolist(),
                       "fpfh_to_global": coarse.tolist(), "icp_delta_global": delta.tolist()})
        if self.on_merge:
            self.on_merge(index, source, self.merged, record)
        self.merged += source
        voxel = float(self.params.get("merge_voxel_size", 0))
        if voxel > 0:
            self.merged = self.merged.voxel_down_sample(voxel)
        self.transforms[index] = total.copy()
        print(f"[多点位配准] {self.label} scan {index + 1}：{phase} 已合并", flush=True)

    def first_pass(self):
        for index in range(1, len(self.clouds)):
            members = [i for i, t in enumerate(self.transforms) if t is not None]
            print(f"\n=== {self.label} 第一遍: scan {index + 1} -> Global Cloud ===")
            passed, total, coarse, delta, pair = self.solve(self.clouds[index], self.merged,
                f"{self.label} scan {index + 1} -> Global Cloud", self.params)
            pair.update({"source": str(self.paths[index]), "target": "current_global_cloud", "target_members": members})
            self.records[index].update({"pair_report": pair, "target_members": members})
            if passed:
                self.merge(index, total, coarse, delta, "first_pass")
            else:
                self.records[index]["status"] = "pending_review" if pair.get("candidates") else "registration_failed_not_merged"

    def independent_pair(self, source, anchor):
        key = (source, anchor)
        if key not in self.proof_cache:
            parameters = dict(self.params)
            parameters["max_pair_attempts"] = int(self.params.get("candidate_review", {}).get("max_pair_attempts", 2))
            self.proof_cache[key] = self.solve(self.clouds[source], self.clouds[anchor],
                f"{self.label} 复核: scan {source + 1} -> scan {anchor + 1}", parameters)
        return self.proof_cache[key]

    def second_pass(self):
        # Frozen references: recovered scans never become evidence for one another.
        anchors = [i for i, t in enumerate(self.transforms) if t is not None]
        reference_cloud = o3d.geometry.PointCloud(self.merged)
        reference_transforms = {i: self.transforms[i].copy() for i in anchors}
        review_settings = self.params.get("candidate_review", {})
        max_references = int(review_settings.get("max_reference_scans", 2))
        if max_references < 1:
            raise ValueError("max_reference_scans 必须至少为 1")
        for index in range(1, len(self.clouds)):
            record = self.records[index]
            if self.transforms[index] is not None:
                continue
            candidates = record["pair_report"].get("candidates", [])
            review = {"reference_indices": anchors, "reference_paths": [str(self.paths[i]) for i in anchors],
                      "checks": [], "accepted": False}
            record["review"] = review
            if not candidates:
                review["reason"] = "no_retained_candidate"
                continue
            members = record["target_members"]
            # If the original target was just one scan, repeating that exact pair is not new evidence.
            available = [i for i in anchors if i != index and members != [i]]
            available.sort(key=lambda i: (i in members, -i))
            if not available:
                review["reason"] = "no_independent_reference"
                record["status"] = "unconfirmed_not_merged"
                continue
            validator = FixedPairValidator(self.clouds[index], reference_cloud, self.params.get("validation"))
            print(f"\n=== {self.label} 第二遍: scan {index + 1}，{len(candidates)} 个候选 ===")
            for anchor in available[:max_references]:
                passed, local, local_coarse, local_delta, pair = self.independent_pair(index, anchor)
                check = {"anchor_index": anchor, "anchor": str(self.paths[anchor]),
                         "independent_registration": pair, "independent_geometry_passed": passed}
                review["checks"].append(check)
                if not passed:
                    check["reason"] = "independent_registration_unconfirmed"
                    continue
                anchor_global = reference_transforms[anchor]
                predicted = anchor_global @ local
                comparisons = [{"candidate_id": c["candidate_id"],
                                **transform_difference(np.array(c["transformation"]), predicted)} for c in candidates]
                matches = [c for c in comparisons if within_difference(c, review_settings)]
                check["candidate_comparisons"] = comparisons
                if not matches:
                    check["reason"] = "independent_transform_disagrees"
                    continue
                placement = validator.evaluate(predicted)
                check["adopted_transform_validation"] = placement
                # The raw independent pair is strictly verified. The global placement must
                # still meet at least the review floor; no standard or scene model is used.
                if placement["state"] == "rejected":
                    check["reason"] = "independent_transform_fails_global_placement"
                    continue
                review.update({"accepted": True, "anchor_index": anchor, "matched_candidate_ids": [c["candidate_id"] for c in matches],
                               "adopted_transformation": predicted.tolist(), "reason": "independent_geometry_and_transform_agree"})
                # Express the ICP correction in the global frame as well.
                coarse_global = anchor_global @ local_coarse
                delta_global = anchor_global @ local_delta @ np.linalg.inv(anchor_global)
                self.merge(index, predicted, coarse_global, delta_global, "second_pass")
                break
            if not review["accepted"]:
                review.setdefault("reason", "no_confirming_evidence_within_budget")
                record["status"] = "unconfirmed_not_merged"

    def loop_check(self):
        settings = self.params.get("loop_closure", {})
        if not settings.get("enabled", True):
            return {"enabled": False, "loop_passed": True, "independent_cycle": False, "reason": "disabled_by_configuration"}
        successful = [i for i, t in enumerate(self.transforms) if t is not None]
        if len(successful) < 2:
            return {"enabled": True, "loop_passed": False, "independent_cycle": False, "reason": "no_successful_scan_beyond_reference"}
        source = successful[-1]
        record = self.records[source]
        direct_path = (record.get("phase") == "first_pass" and record.get("target_members") == [0]) or (
            record.get("phase") == "second_pass" and record.get("review", {}).get("anchor_index") == 0)
        cached = self.proof_cache.get((source, 0)) if not direct_path else None
        if cached and not cached[0]:
            cached = None
        if cached:
            result = cached
        else:
            result = self.solve(self.clouds[source], self.clouds[0],
                f"{self.label} 闭环检查: scan {source + 1} -> scan 1", self.params)
        passed, direct, _, _, pair = result
        report = {"enabled": True, "source_index": source + 1, "target_index": 1,
                  "source": str(self.paths[source]), "target": str(self.paths[0]),
                  "direct_registration_success": passed, "direct_registration_report": pair,
                  "independent_cycle": not direct_path, "evidence_reused": cached is not None,
                  "check_type": "pair_repeatability" if direct_path else "loop_consistency",
                  "translation_threshold": float(settings.get("translation_threshold", .5)),
                  "rotation_threshold_deg": float(settings.get("rotation_threshold_deg", 5))}
        if not passed or direct is None:
            return {**report, "loop_passed": False, "reason": "direct_registration_unconfirmed"}
        error = transform_difference(self.transforms[source], direct)
        loop_passed = within_difference(error, report)
        return {**report, **error, "loop_passed": loop_passed,
                "direct_transformation": direct.tolist(), "previous_global_transformation": self.transforms[source].tolist(),
                "reason": "within_thresholds" if loop_passed else "transform_disagreement"}

    def run(self):
        self.first_pass()
        self.second_pass()
        loop = self.loop_check()
        missing = [str(p) for p, t in zip(self.paths, self.transforms) if t is None]
        status = ("complete" if not missing else "partial") if loop["loop_passed"] else "failed"
        print(f"[多点位配准] {self.label}：{status}，已合并 {len(self.clouds) - len(missing)}/{len(self.clouds)}，"
              f"闭环={loop['loop_passed']}，类型={loop.get('check_type', 'unavailable')}", flush=True)
        return self.merged, self.transforms, {
            "reference": str(self.paths[0]), "method": "two_pass_fixed_validation",
            "success_basis": "geometry_and_consistency_proxy", "registration_status": status,
            "all_scans_registered": not missing, "scans": self.records, "failed_scans": missing,
            "success_count": len(self.clouds) - len(missing),
            "transformations": [t.tolist() if t is not None else None for t in self.transforms],
            "loop_closure": loop, "validation": self.params.get("validation", {}),
            "partial_result_allowed": True,
        }
