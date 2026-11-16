import argparse
import ctypes
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

import yaml


from auto_cut import crop_with_spec, derive_crop_spec, save_crop_spec
from bottom_equation_fit import fit_bottom_plane_file
from denoise_statistical import statistical_denoise_file
from fill_holes import fill_holes_file
from pcd2png import show_point_cloud_file, show_point_cloud_files
from registration_multiview import TwoPassRegistration, attempt_pair_registration

from registration_b_a import align_after_to_before
from tin_volume_calculate import calculate_tin_volume_file
from voxel_downsample_centroid import voxel_downsample_file


DEFAULT_CONFIG_PATH = "config.yaml"
ACTIVE_STAGE_TIMERS = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="船舱泥沙体积测量完整流程",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="YAML 配置文件路径")
    parser.add_argument("--before_dir", type=str, default="", help="装沙前 PCD 目录")
    parser.add_argument("--after_dir", type=str, default="", help="装沙后 PCD 目录")
    parser.add_argument("--output_dir", type=str, default="", help="输出目录")
    parser.add_argument("--console_level", type=str, choices=["simple", "detailed"], default=None, help="控制台输出级别，空值表示使用 config.yaml 中的配置",)
    return parser.parse_args()


def load_config(args):

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件内容必须是 YAML 字典: {config_path}")

    config["_config_dir"] = str(config_path.resolve().parent)
    config.setdefault("visualization", {})
    config.setdefault("logging", {})
    config["logging"].setdefault("console_level", "detailed")
    config["logging"].setdefault("file_name", "v_measure_full.txt")

    if args.before_dir:
        config["before_dir"] = args.before_dir
    if args.after_dir:
        config["after_dir"] = args.after_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.console_level:
        config.setdefault("logging", {})["console_level"] = args.console_level

    if not config.get("before_dir"):
        raise ValueError("必须提供装沙前点云目录：--before_dir 或 config.before_dir")
    if not config.get("after_dir"):
        raise ValueError("必须提供装沙后点云目录：--after_dir 或 config.after_dir")

    output_base_dir = resolve_config_path(config["output_dir"], config)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    config["_output_base_dir"] = str(output_base_dir)
    config["_run_timestamp"] = run_timestamp
    config["output_dir"] = str(output_base_dir / run_timestamp)
    return config


def resolve_config_path(path, config):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(config.get("_config_dir", ".")).joinpath(path).resolve()


def resolve_output_path(path, config):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(config["output_dir"]).joinpath(path).resolve()


def collect_pcd_files(directory, config, label):
    directory = resolve_config_path(directory, config)
    if not directory.exists():
        raise FileNotFoundError(f"{label} 点云目录不存在: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} 路径不是目录: {directory}")

    pcd_files = sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pcd"],
        key=lambda path: path.name.lower(),
    )
    if not pcd_files:
        raise FileNotFoundError(f"{label} 点云目录中没有 .pcd 文件: {directory}")

    print(f"{label} 点云目录: {directory}")
    print(f"{label} PCD 文件数: {len(pcd_files)}")
    for idx, path in enumerate(pcd_files, start=1):
        role = "基准" if idx == 1 else "待配准"
        print(f"  {idx:02d}. [{role}] {path.name}")
    return pcd_files


class PipelineLogger:
    def __init__(self, file_handle, console_stream, console_level="detailed"):
        self.file_handle = file_handle
        self.console_stream = console_stream
        self.console_level = console_level
        self._buffer = ""

    def _console_should_print(self, line):
        if self.console_level == "detailed":
            return True

        text = line.strip()
        if not text:
            return True

        prefixes = (
            "===",
            "[多点位配准]",
            "[曲率 ROI]",
            "Traceback",
            "File ",
            "ValueError",
            "RuntimeError",
            "FileNotFoundError",
            "NotADirectoryError",
            "ModuleNotFoundError",
            "TypeError",
            "最终体积",
            "保存点云:",
            "保存矩阵:",
            "保存 JSON:",
            "读取点云:",
            "点云目录:",
            "PCD 文件数:",
            "结果已保存到:",
            "底面平面方程:",
            "Estimated Volume",
            "Saved mesh:",
            "配准后的",
            "合并点云已保存:",
            "跳过",
            "打开 TIN",
        )
        return text.startswith(prefixes)

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)

        self.file_handle.write(data)
        self.file_handle.flush()

        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line += "\n"
            if self._console_should_print(line):
                self.console_stream.write(line)
                self.console_stream.flush()

    def flush(self):
        self.file_handle.flush()
        if self._buffer:
            if self._console_should_print(self._buffer):
                self.console_stream.write(self._buffer)
                self.console_stream.flush()
            self._buffer = ""
        self.console_stream.flush()


@contextmanager
def tee_pipeline_output(log_path, console_level="detailed"):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8", buffering=1) as fh:
        tee = PipelineLogger(fh, original_stdout, console_level=console_level)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield log_path
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            tee.flush()


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("page_fault_count", ctypes.c_uint32),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def get_process_memory_usage_mb():
    psutil_memory = get_process_memory_usage_with_psutil()
    if psutil_memory is not None:
        return psutil_memory

    windows_memory = get_process_memory_usage_with_windows_api()
    if windows_memory is not None:
        return windows_memory

    proc_memory = get_process_memory_usage_with_proc_status()
    if proc_memory is not None:
        return proc_memory

    return {
        "rss_mb": None,
        "peak_rss_mb": None,
        "pagefile_mb": None,
        "peak_pagefile_mb": None,
        "source": "unavailable",
    }


def get_process_memory_usage_with_psutil():
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        peak_rss = getattr(memory_info, "peak_wset", None)
        pagefile = getattr(memory_info, "vms", None)
        peak_pagefile = getattr(memory_info, "peak_pagefile", None)
        mb = 1024 * 1024
        return {
            "rss_mb": memory_info.rss / mb,
            "peak_rss_mb": peak_rss / mb if peak_rss is not None else None,
            "pagefile_mb": pagefile / mb if pagefile is not None else None,
            "peak_pagefile_mb": peak_pagefile / mb if peak_pagefile is not None else None,
            "source": "psutil",
        }
    except Exception:
        return None


def get_process_memory_usage_with_windows_api():
    if not hasattr(ctypes, "WinDLL"):
        return None
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        try:
            get_memory_info = kernel32.K32GetProcessMemoryInfo
            source = "kernel32.K32GetProcessMemoryInfo"
        except AttributeError:
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_memory_info = psapi.GetProcessMemoryInfo
            source = "psapi.GetProcessMemoryInfo"
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_uint32,
        ]
        get_memory_info.restype = ctypes.c_int
        ok = get_memory_info(handle, ctypes.byref(counters), counters.cb)
    except Exception:
        return None
    if not ok:
        return None
    mb = 1024 * 1024
    return {
        "rss_mb": counters.working_set_size / mb,
        "peak_rss_mb": counters.peak_working_set_size / mb,
        "pagefile_mb": counters.pagefile_usage / mb,
        "peak_pagefile_mb": counters.peak_pagefile_usage / mb,
        "source": source,
    }


def get_process_memory_usage_with_proc_status():
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    values = {}
    try:
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmPeak:")):
                name, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if parts:
                    values[name] = float(parts[0]) / 1024.0
    except Exception:
        return None
    if "VmRSS" not in values:
        return None
    return {
        "rss_mb": values.get("VmRSS"),
        "peak_rss_mb": values.get("VmHWM"),
        "pagefile_mb": values.get("VmSize"),
        "peak_pagefile_mb": values.get("VmPeak"),
        "source": "/proc/self/status",
    }


@contextmanager
def measured_stage(report, key, name):
    timer = {"visualization_seconds": 0.0}
    ACTIVE_STAGE_TIMERS.append(timer)
    start_time = time.perf_counter()
    start_memory = get_process_memory_usage_mb()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        ACTIVE_STAGE_TIMERS.pop()
        end_memory = get_process_memory_usage_mb()
        visualization_seconds = timer["visualization_seconds"]
        duration = max(0.0, end_time - start_time - visualization_seconds)
        memory_delta = None
        if start_memory["rss_mb"] is not None and end_memory["rss_mb"] is not None:
            memory_delta = end_memory["rss_mb"] - start_memory["rss_mb"]
        stage_report = {
            "name": name,
            "elapsed_seconds": duration,
            "visualization_excluded_seconds": visualization_seconds,
            "memory_start_mb": start_memory["rss_mb"],
            "memory_end_mb": end_memory["rss_mb"],
            "memory_delta_mb": memory_delta,
            "memory_process_peak_mb": end_memory["peak_rss_mb"],
            "pagefile_start_mb": start_memory["pagefile_mb"],
            "pagefile_end_mb": end_memory["pagefile_mb"],
            "pagefile_process_peak_mb": end_memory["peak_pagefile_mb"],
            "memory_source": end_memory.get("source"),
        }
        report.setdefault("stage_metrics", {})[key] = stage_report
        print(
            "[阶段统计] "
            f"{name}: 用时 {duration:.3f}s, "
            f"内存 {format_mb(stage_report['memory_start_mb'])} -> "
            f"{format_mb(stage_report['memory_end_mb'])}, "
            f"进程峰值 {format_mb(stage_report['memory_process_peak_mb'])}"
        )


def format_mb(value):
    if value is None:
        return "N/A"
    return f"{value:.2f} MB"


@contextmanager
def excluded_from_stage_timing():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        for timer in ACTIVE_STAGE_TIMERS:
            timer["visualization_seconds"] += elapsed


def read_point_cloud(path):
    path = Path(path)
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"点云为空或读取失败: {path}")
    print(f"读取点云: {path}")
    print(f"点数: {len(pcd.points)}")
    return pcd


def write_point_cloud(path, pcd, write_ascii=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=write_ascii)
    if not ok:
        raise RuntimeError(f"点云保存失败: {path}")
    print(f"保存点云: {path}")
    return path


def display_point_cloud_result(path, title, config):
    vis_config = config.get("visualization", {})
    if not vis_config.get("enabled", True):
        return

    with excluded_from_stage_timing():
        show_point_cloud_file(
            path,
            window_name=title,
            width=vis_config.get("width", 960),
            height=vis_config.get("height", 720),
            left=vis_config.get("left", 100),
            top=vis_config.get("top", 100),
            clear_colors=vis_config.get("clear_colors", False),
            show_normals=vis_config.get("show_normals", False),
            print_info=vis_config.get("print_info", True),
        )


def display_point_cloud_results(paths, title, config, colors=None):
    vis_config = config.get("visualization", {})
    if not vis_config.get("enabled", True):
        return

    with excluded_from_stage_timing():
        show_point_cloud_files(
            paths,
            window_name=title,
            colors=colors,
            width=vis_config.get("width", 960),
            height=vis_config.get("height", 720),
            left=vis_config.get("left", 100),
            top=vis_config.get("top", 100),
            clear_colors=True,
            show_normals=vis_config.get("show_normals", False),
            print_info=vis_config.get("print_info", True),
        )


def display_registration_pair_result(source_registered, global_cloud, title, config):
    vis_config = config.get("visualization", {})
    if not vis_config.get("enabled", True):
        return
    if not vis_config.get("show_multiview_pair", True):
        return

    source_vis = point_cloud_copy(source_registered)
    global_vis = point_cloud_copy(global_cloud)
    source_vis.paint_uniform_color([1.0, 0.0, 0.0])
    global_vis.paint_uniform_color([0.0, 1.0, 0.0])

    print("打开配准结果窗口：红色=配准后源点云, 绿色=全局点云")
    with excluded_from_stage_timing():
        o3d.visualization.draw_geometries(
            [global_vis, source_vis],
            window_name=title,
            width=vis_config.get("width", 960),
            height=vis_config.get("height", 720),
            left=vis_config.get("left", 100),
            top=vis_config.get("top", 100),
        )


def display_cloud_object_result(pcd, title, config):
    vis_config = config.get("visualization", {})
    if not vis_config.get("enabled", True):
        return
    if not vis_config.get("show_multiview_final", True):
        return

    print("打开点云可视化窗口...")
    with excluded_from_stage_timing():
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=vis_config.get("width", 960),
            height=vis_config.get("height", 720),
            left=vis_config.get("left", 100),
            top=vis_config.get("top", 100),
        )


def display_mesh_result(mesh, title, config):
    vis_config = config.get("visualization", {})
    if not vis_config.get("enabled", True):
        return
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("TIN 网格为空，跳过体积计算图像显示。")
        return

    print("打开 TIN 体积计算网格可视化窗口...")
    with excluded_from_stage_timing():
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=title,
            width=vis_config.get("width", 960),
            height=vis_config.get("height", 720),
            left=vis_config.get("left", 100),
            top=vis_config.get("top", 100),
        )


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"保存 JSON: {path}")
    return path


def point_cloud_copy(pcd):
    return o3d.geometry.PointCloud(pcd)


def save_matrix(matrix, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path.with_suffix(".txt"), matrix, fmt="%.10f")
    np.save(output_path.with_suffix(".npy"), matrix)
    print(f"保存矩阵: {output_path.with_suffix('.txt')}")
    print(f"保存矩阵: {output_path.with_suffix('.npy')}")


def maybe_downsample(pcd, voxel_size):
    if voxel_size is None or voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size)


def attempt_multiview_registration(source, target, pair_name, params):
    """固定验证、保留多个候选；无需标准变换或预置几何结构。"""
    return attempt_pair_registration(source, target, pair_name, params)


def register_and_merge_scans_once(
    clouds, scan_paths, label, transform_dir, params, config, pipeline_attempt_index,
):
    """第一遍严格验收，第二遍独立配准复核，最后检查末个已合并扫描。"""
    def on_merge(index, registered_source, global_cloud, record):
        display_registration_pair_result(
            registered_source, global_cloud,
            f"{label} scan {index + 1} registered: source(red) / global(green)", config,
        )
        prefix = f"{label}_attempt_{pipeline_attempt_index}_scan_{index + 1:02d}"
        save_matrix(np.array(record["fpfh_to_global"]), transform_dir / f"{prefix}_fpfh_to_global")
        save_matrix(np.array(record["icp_delta_global"]), transform_dir / f"{prefix}_icp_delta")
        save_matrix(np.array(record["transform_to_global"]), transform_dir / f"{prefix}_to_global")

    engine = TwoPassRegistration(
        clouds, scan_paths, label, params,
        pair_solver=attempt_multiview_registration, on_merge=on_merge,
    )
    merged, transformations, report = engine.run()
    report["pipeline_attempt_index"] = int(pipeline_attempt_index)
    return merged, transformations, report


def register_and_merge_scans(scan_paths, label, output_path, transform_dir, params, config):
    """两遍配准，允许部分成功；闭环失败仍从头执行整组。"""
    scan_paths = [Path(path) for path in scan_paths]
    output_path = Path(output_path)
    transform_dir.mkdir(parents=True, exist_ok=True)
    clouds = [read_point_cloud(path) for path in scan_paths]

    if len(scan_paths) == 1:
        pcd = point_cloud_copy(clouds[0])
        write_point_cloud(output_path, pcd)
        save_matrix(np.eye(4), transform_dir / f"{label}_scan_01_to_global")
        print(f"[多点位配准] {label} 只有 1 个 PCD 文件，跳过多点位配准和闭环检查。")
        return pcd, {
            "reference": str(scan_paths[0]),
            "registration_status": "complete",
            "pipeline_success": True,
            "all_scans_registered": True,
            "success_count": 1,
            "failed_scans": [],
            "scans": [{"path": str(scan_paths[0]), "transform_to_global": np.eye(4).tolist()}],
            "loop_closure": {
                "enabled": False,
                "loop_passed": True,
                "reason": "只有 1 个点云",
            },
        }

    save_matrix(np.eye(4), transform_dir / f"{label}_scan_01_to_global")
    loop_params = params.get("loop_closure", {})
    max_pipeline_attempts = max(1, int(loop_params.get("max_pipeline_attempts", 3)))
    pipeline_attempts = []

    for pipeline_attempt_index in range(1, max_pipeline_attempts + 1):
        print(f"\n=== {label} 多点位整体配准第 {pipeline_attempt_index} 轮 ===")
        merged, _, report = register_and_merge_scans_once(
            clouds=clouds,
            scan_paths=scan_paths,
            label=label,
            transform_dir=transform_dir,
            params=params,
            config=config,
            pipeline_attempt_index=pipeline_attempt_index,
        )
        loop_passed = bool(report.get("loop_closure", {}).get("loop_passed"))
        pipeline_attempts.append(
            {
                "pipeline_attempt_index": pipeline_attempt_index,
                "loop_passed": loop_passed,
                "registration_status": report["registration_status"],
                "failed_scans": report.get("failed_scans", []),
                "success_count": report.get("success_count"),
                "loop_reason": report.get("loop_closure", {}).get("reason"),
            }
        )
        report["pipeline_attempts"] = pipeline_attempts
        report["pipeline_success"] = loop_passed
        report["retry_count"] = pipeline_attempt_index - 1
        report_path = output_path.with_name(
            f"{output_path.stem}_attempt_{pipeline_attempt_index}_registration.json"
        )
        write_json(report_path, report)

        if loop_passed:
            write_point_cloud(output_path, merged)
            display_cloud_object_result(
                merged,
                f"{label} multiview registration complete",
                config,
            )
            return merged, report

        if pipeline_attempt_index < max_pipeline_attempts:
            print(f"{label} 闭环检查未通过，重新从头执行多点位配准。")

    raise RuntimeError(
        f"{label} 多点位配准闭环检查连续 {max_pipeline_attempts} 轮未通过，停止后续流程。"
    )


def run_pipeline(config):
    pipeline_start_time = time.perf_counter()
    pipeline_start_memory = get_process_memory_usage_mb()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_dir = output_dir / "transforms"

    pipeline_report = {
        "coordinate_system": "装沙前首个点云坐标系",
        "before_dir": str(resolve_config_path(config["before_dir"], config)),
        "after_dir": str(resolve_config_path(config["after_dir"], config)),
        "output_base_dir": config.get("_output_base_dir"),
        "run_timestamp": config.get("_run_timestamp"),
        "outputs": {},
        "stage_metrics": {},
    }

    with measured_stage(pipeline_report, "00_collect_inputs", "输入点云扫描"):
        before_scans = collect_pcd_files(config["before_dir"], config, "装沙前")
        after_scans = collect_pcd_files(config["after_dir"], config, "装沙后")

    print("\n=== 1. 装沙前多点位配准合并 ===")
    with measured_stage(pipeline_report, "01_before_multiview_registration", "装沙前多点位配准合并"):
        before_full_path = output_dir / "01_before_full.pcd"
        before_full, before_merge_report = register_and_merge_scans(
            before_scans,
            label="before",
            output_path=before_full_path,
            transform_dir=transform_dir,
            params=config["multi_view_registration"],
            config=config,
        )
    pipeline_report["before_merge"] = before_merge_report
    pipeline_report["outputs"]["before_full"] = str(before_full_path)
    display_point_cloud_result(before_full_path, "Step 1 - before_full", config)

    print("\n=== 1. 装沙后多点位配准合并 ===")
    with measured_stage(pipeline_report, "01_after_multiview_registration", "装沙后多点位配准合并"):
        after_full_native_path = output_dir / "02_after_full_native.pcd"
        after_full_native, after_merge_report = register_and_merge_scans(
            after_scans,
            label="after",
            output_path=after_full_native_path,
            transform_dir=transform_dir,
            params=config["multi_view_registration"],
            config=config,
        )
    pipeline_report["after_merge"] = after_merge_report
    pipeline_report["multiview_registration_status"] = (
        "partial" if any(r.get("registration_status") == "partial" for r in (before_merge_report, after_merge_report)) else "complete"
    )
    if pipeline_report["multiview_registration_status"] == "partial":
        print("[多点位配准] 当前为部分成功，将使用已确认合并的点云继续后续处理；缺失扫描已记录在报告中。")
    pipeline_report["outputs"]["after_full_native"] = str(after_full_native_path)
    display_point_cloud_result(after_full_native_path, "Step 1 - after_full_native", config)

    print("\n=== 2. 装沙前后整体点云配准 ===")
    with measured_stage(pipeline_report, "02_before_after_registration", "装沙前后整体点云配准"):
        after_aligned, t_after_to_before, ba_report = align_after_to_before(
            source_after=after_full_native,
            target_before=before_full,
            **config["before_after_registration"],
        )
        after_aligned_path = output_dir / "03_after_aligned_to_before.pcd"
        write_point_cloud(after_aligned_path, after_aligned)
        save_matrix(t_after_to_before, transform_dir / "03_T_after_to_before")
        write_json(output_dir / "03_before_after_registration_report.json", ba_report)
    pipeline_report["before_after_registration"] = ba_report
    pipeline_report["outputs"]["after_aligned_to_before"] = str(after_aligned_path)
    display_point_cloud_result(after_aligned_path, "Step 2 - after_aligned_to_before", config)
    display_point_cloud_results(
        [before_full_path, after_aligned_path],
        "Step 2 - before_full(green) / after_aligned(red)",
        config,
        colors=[
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )

    print("\n=== 3. 自动裁剪 ===")
    with measured_stage(pipeline_report, "03_auto_crop", "自动裁剪"):
        crop_spec, _, _ = derive_crop_spec(
            before_full,
            **config["auto_cut"],
        )
        crop_spec_path = output_dir / "04_crop_spec.json"
        save_crop_spec(crop_spec, crop_spec_path)

        before_crop, _ = crop_with_spec(before_full, crop_spec)
        after_crop, _ = crop_with_spec(after_aligned, crop_spec)
        before_crop_path = output_dir / "04_before_crop.pcd"
        after_crop_path = output_dir / "04_after_crop.pcd"
        write_point_cloud(before_crop_path, before_crop)
        write_point_cloud(after_crop_path, after_crop)
    pipeline_report["outputs"]["crop_spec"] = str(crop_spec_path)
    pipeline_report["outputs"]["before_crop"] = str(before_crop_path)
    pipeline_report["outputs"]["after_crop"] = str(after_crop_path)
    display_point_cloud_result(before_crop_path, "Step 3 - before_crop", config)
    display_point_cloud_result(after_crop_path, "Step 3 - after_crop", config)

    print("\n=== 4. 船舱底面平面拟合 ===")
    with measured_stage(pipeline_report, "04_bottom_plane", "船舱底面平面拟合"):
        bottom_filled_path = output_dir / "05_before_bottom_filled.pcd"
        bottom_plane_path = output_dir / "05_bottom_plane.json"
        _, bottom_plane, bottom_report, _ = fit_bottom_plane_file(
            before_crop_path,
            output_path=bottom_filled_path,
            plane_output=bottom_plane_path,
            visualize=False,
            **config["bottom_plane"],
        )
    pipeline_report["bottom_plane"] = bottom_report
    pipeline_report["outputs"]["bottom_filled"] = str(bottom_filled_path)
    pipeline_report["outputs"]["bottom_plane"] = str(bottom_plane_path)
    display_point_cloud_result(bottom_filled_path, "Step 4 - before_bottom_filled", config)

    print("\n=== 5. 纯泥沙点云处理 ===")
    with measured_stage(pipeline_report, "05_sand_processing", "纯泥沙点云处理"):
        sand_input = config.get("sand_surface_input") or str(after_crop_path)
        current_sand_path = Path(sand_input)

        denoise_config = config["sand_processing"].get("denoise", {})
        if denoise_config.get("enabled", True):
            with measured_stage(pipeline_report, "05_01_sand_denoise", "纯泥沙点云去噪"):
                denoise_path = output_dir / "06_sand_denoised.pcd"
                _, _, current_sand_path = statistical_denoise_file(
                    current_sand_path,
                    output_path=denoise_path,
                    nb_neighbors=denoise_config.get("nb_neighbors", 60),
                    std_ratio=denoise_config.get("std_ratio", 3.5),
                )
            pipeline_report["outputs"]["sand_denoised"] = str(current_sand_path)
            display_point_cloud_result(current_sand_path, "Step 5 - sand_denoised", config)

        voxel_config = config["sand_processing"].get("voxel", {})
        if voxel_config.get("enabled", True):
            with measured_stage(pipeline_report, "05_02_sand_voxel", "纯泥沙点云体素下采样"):
                voxel_path = output_dir / "06_sand_voxel.pcd"
                _, current_sand_path = voxel_downsample_file(
                    current_sand_path,
                    output_path=voxel_path,
                    voxel_size=voxel_config.get("voxel_size", 0.02),
                )
            pipeline_report["outputs"]["sand_voxel"] = str(current_sand_path)
            display_point_cloud_result(current_sand_path, "Step 5 - sand_voxel", config)

        fill_config = dict(config["sand_processing"].get("fill_holes", {}))
        fill_enabled = fill_config.pop("enabled", True)
        if fill_enabled:
            with measured_stage(pipeline_report, "05_03_fill_holes", "泥沙表面孔洞填补"):
                filled_path = output_dir / "06_sand_surface_final.pcd"
                _, fill_report, current_sand_path = fill_holes_file(
                    current_sand_path,
                    output_path=filled_path,
                    **fill_config,
                )
            pipeline_report["fill_holes"] = fill_report
            display_point_cloud_result(current_sand_path, "Step 5 - sand_surface_final", config)

    pipeline_report["outputs"]["sand_surface_final"] = str(current_sand_path)
    if not fill_enabled:
        display_point_cloud_result(current_sand_path, "Step 5 - sand_surface_final", config)

    print("\n=== 6. TIN 体积计算 ===")
    with measured_stage(pipeline_report, "06_tin_volume", "TIN 体积计算"):
        volume_config = config["volume"]
        mesh_path = output_dir / "07_tin_mesh.ply"
        volume_report_path = output_dir / "07_volume_report.json"
        volume, tin_mesh, volume_report = calculate_tin_volume_file(
            current_sand_path,
            plane_model=bottom_plane,
            max_edge=volume_config.get("max_edge", 0.1),
            mesh_output=mesh_path,
            report_output=volume_report_path,
        )
    display_mesh_result(tin_mesh, "Step 6 - TIN volume mesh", config)
    pipeline_report["volume"] = volume_report
    pipeline_report["outputs"]["tin_mesh"] = str(mesh_path)
    pipeline_report["outputs"]["volume_report"] = str(volume_report_path)
    pipeline_report["volume_m3"] = float(volume)
    pipeline_end_memory = get_process_memory_usage_mb()
    pipeline_memory_delta = None
    if (
        pipeline_start_memory["rss_mb"] is not None
        and pipeline_end_memory["rss_mb"] is not None
    ):
        pipeline_memory_delta = (
            pipeline_end_memory["rss_mb"] - pipeline_start_memory["rss_mb"]
        )
    pipeline_report["total_metrics"] = {
        "elapsed_seconds": time.perf_counter() - pipeline_start_time,
        "memory_start_mb": pipeline_start_memory["rss_mb"],
        "memory_end_mb": pipeline_end_memory["rss_mb"],
        "memory_delta_mb": pipeline_memory_delta,
        "memory_process_peak_mb": pipeline_end_memory["peak_rss_mb"],
        "pagefile_start_mb": pipeline_start_memory["pagefile_mb"],
        "pagefile_end_mb": pipeline_end_memory["pagefile_mb"],
        "pagefile_process_peak_mb": pipeline_end_memory["peak_pagefile_mb"],
        "memory_source": pipeline_end_memory.get("source"),
    }
    print(
        "[阶段统计] 总流程: "
        f"用时 {pipeline_report['total_metrics']['elapsed_seconds']:.3f}s, "
        f"内存 {format_mb(pipeline_report['total_metrics']['memory_start_mb'])} -> "
        f"{format_mb(pipeline_report['total_metrics']['memory_end_mb'])}, "
        f"进程峰值 {format_mb(pipeline_report['total_metrics']['memory_process_peak_mb'])}"
    )

    write_json(output_dir / "v_measure_report.json", pipeline_report)
    print(f"\n最终体积: {volume:.3f} m^3")
    return pipeline_report


def main():
    args = parse_args()
    config = load_config(args)
    log_path = resolve_output_path(
        config["logging"].get("file_name", "v_measure_full.txt"),
        config,
    )
    console_level = config.get("logging", {}).get("console_level", "detailed")
    with tee_pipeline_output(log_path, console_level=console_level):
        print(f"日志文件: {log_path}")
        print(f"结果已保存到: {config['output_dir']}")
        print(f"控制台级别: {console_level}")
        run_pipeline(config)


if __name__ == "__main__":
    main()
