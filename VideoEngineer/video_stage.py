from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video_engine_blender import run_blender_python_script


STAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = STAGE_DIR.parent
CAMERA_OUTPUT_DIR = WORKSPACE_DIR / "Cinematographer" / "output"
DEFAULT_BLENDER_EXE = Path(os.getenv("STORYBLENDER_BLENDER_EXE", "blender"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: Any, path: Path) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def latest_camera_handoff() -> Path:
    candidates = sorted((item for item in CAMERA_OUTPUT_DIR.iterdir() if item.is_dir()), key=lambda item: item.name)
    if not candidates:
        raise FileNotFoundError("No Cinematographer output folders found. Run Cinematographer/run_cinematographer.py first.")
    return candidates[-1] / "outputs" / "camera_handoff_v1.json"


def default_output_root(run_id: str | None = None) -> Path:
    suffix = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return STAGE_DIR / "output" / suffix


@dataclass(slots=True)
class VideoEngineerConfig:
    camera_handoff_path: str
    output_root: str
    run_id: str = ""
    fps: int = 24
    resolution_x: int = 960
    resolution_y: int = 540
    blender_exe: str = ""
    render_engine: str = "BLENDER_EEVEE_NEXT"
    render_samples: int = 8
    camera_quality: str = "fast"


def parse_args() -> VideoEngineerConfig:
    parser = argparse.ArgumentParser(description="Run the isolated VideoEngineer stage.")
    parser.add_argument("--camera-handoff-path", default=str(latest_camera_handoff()))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
    parser.add_argument("--blender-exe", default="")
    parser.add_argument("--render-engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--render-samples", type=int, default=8)
    parser.add_argument("--camera-quality", choices=("fast", "quality"), default="fast")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(args.run_id).resolve()
    return VideoEngineerConfig(
        camera_handoff_path=args.camera_handoff_path,
        output_root=str(output_root),
        run_id=args.run_id,
        fps=args.fps,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        blender_exe=args.blender_exe,
        render_engine=args.render_engine,
        render_samples=args.render_samples,
        camera_quality=args.camera_quality,
    )


def ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"


def sorted_frame_paths(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("frame_*.png"))


def detect_start_number(frame_dir: Path) -> int:
    frame_paths = sorted_frame_paths(frame_dir)
    if not frame_paths:
        return 1
    stem = frame_paths[0].stem
    suffix = stem.split("_")[-1]
    try:
        return int(suffix)
    except ValueError:
        return 1


def encode_frames(frame_dir: Path, fps: int, output_path: Path) -> subprocess.CompletedProcess[str]:
    start_number = detect_start_number(frame_dir)
    command = [
        ffmpeg_path(),
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def default_blender_exe(configured_path: str) -> Path:
    if configured_path:
        return Path(configured_path).resolve()
    bundled = STAGE_DIR.parent.parent.parent.parent / "blender" / "blender.exe"
    if bundled.exists():
        return bundled.resolve()
    return DEFAULT_BLENDER_EXE


def resolve_blend_file(camera_handoff: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    director_handoff_path = Path(str(camera_handoff["director_handoff_path"])).resolve()
    director_handoff = load_json(director_handoff_path)
    source_inventory_path = Path(str((director_handoff.get("files") or {}).get("source_inventory_path") or "")).resolve()
    source_inventory = load_json(source_inventory_path) if source_inventory_path.exists() else {}
    selected_sources = source_inventory.get("selected_sources") or {}
    for candidate in selected_sources.get("blend_paths") or []:
        candidate_path = Path(str(candidate))
        if candidate_path.exists():
            return candidate_path.resolve(), director_handoff, source_inventory
    demo_root = Path(str(director_handoff.get("demo_root") or "")).resolve()
    fallback = demo_root / "The Godfather.blend"
    return fallback.resolve(), director_handoff, source_inventory


def classify_blender_failure(stderr_path: Path) -> str:
    try:
        text = stderr_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return "unknown"
    crash_tokens = (
        "exception_access_violation",
        "nvoglv64.dll",
        "access violation",
        "segmentation fault",
        "gpu",
    )
    if any(token in text for token in crash_tokens):
        return "blender_driver_or_gpu_crash"
    if "out of memory" in text or "cuda error" in text:
        return "blender_memory_or_cuda_failure"
    return "blender_process_failed"


def run_blender_renders(
    *,
    config: VideoEngineerConfig,
    camera_handoff_path: Path,
    output_root: Path,
    blend_file: Path,
) -> dict[str, Any]:
    attempts = [
        {
            "label": "initial",
            "render_engine": str(config.render_engine),
            "render_samples": max(1, int(config.render_samples)),
            "stdout_path": output_root / "blender_render_stdout.log",
            "stderr_path": output_root / "blender_render_stderr.log",
        },
        {
            "label": "retry_low_samples",
            "render_engine": str(config.render_engine),
            "render_samples": max(1, int(config.render_samples) // 2),
            "stdout_path": output_root / "blender_render_retry_stdout.log",
            "stderr_path": output_root / "blender_render_retry_stderr.log",
        },
    ]
    last_result: dict[str, Any] | None = None
    attempt_reports: list[dict[str, Any]] = []
    for attempt in attempts:
        script_args = [
            "--camera-handoff-path",
            str(camera_handoff_path),
            "--output-root",
            str(output_root),
            "--fps",
            str(config.fps),
            "--resolution-x",
            str(config.resolution_x),
            "--resolution-y",
            str(config.resolution_y),
            "--render-engine",
            str(attempt["render_engine"]),
            "--render-samples",
            str(attempt["render_samples"]),
        ]
        result = run_blender_python_script(
            blender_exe=default_blender_exe(config.blender_exe),
            blend_file=blend_file,
            python_script=STAGE_DIR / "blender_render_worker.py",
            script_args=script_args,
            workdir=STAGE_DIR,
            stdout_path=attempt["stdout_path"],
            stderr_path=attempt["stderr_path"],
            timeout_seconds=7200,
            background=True,
        )
        result["attempt_label"] = str(attempt["label"])
        result["failure_class"] = "" if result.get("success") else classify_blender_failure(Path(str(result.get("stderr_path") or "")))
        result["render_report_path"] = str(output_root / "outputs" / "blender_render_report_v1.json")
        attempt_reports.append(
            {
                "attempt_label": result["attempt_label"],
                "success": bool(result.get("success")),
                "returncode": result.get("returncode"),
                "failure_class": result.get("failure_class"),
                "stdout_path": result.get("stdout_path"),
                "stderr_path": result.get("stderr_path"),
            }
        )
        last_result = result
        if result.get("success"):
            break
        if result.get("failure_class") not in {"blender_driver_or_gpu_crash", "blender_memory_or_cuda_failure"}:
            break
    final_result = dict(last_result or {})
    final_result["attempts"] = attempt_reports
    final_result["render_report_path"] = str(output_root / "outputs" / "blender_render_report_v1.json")
    return final_result


def poster_frame(frame_dir: Path, clip_dir: Path) -> Path | None:
    frame_paths = sorted_frame_paths(frame_dir)
    if not frame_paths:
        return None
    poster_path = clip_dir / "poster.png"
    shutil.copy2(frame_paths[len(frame_paths) // 2], poster_path)
    return poster_path


ROTATION_CONTINUITY_THRESHOLD_RAD = 0.35


def _lens_float(value: Any, fallback: float = 35.0) -> float:
    try:
        lens = float(value)
        if math.isfinite(lens) and lens > 0.0:
            return lens
    except Exception:
        pass
    return float(fallback)


def _vector3(values: Any, default: list[float] | None = None) -> list[float]:
    fallback = default or [0.0, 0.0, 0.0]
    try:
        result = [float(value) for value in list(values)[:3]]
        if len(result) == 3 and all(math.isfinite(value) for value in result):
            return result
    except Exception:
        pass
    return list(fallback)


def _normalize_quaternion(values: Any) -> tuple[float, float, float, float] | None:
    try:
        quat = tuple(float(value) for value in list(values)[:4])
        if len(quat) != 4:
            return None
        length = math.sqrt(sum(value * value for value in quat))
        if length <= 1e-8 or not math.isfinite(length):
            return None
        return tuple(value / length for value in quat)  # type: ignore[return-value]
    except Exception:
        return None


def _euler_xyz_to_quaternion(rotation_euler: Any) -> tuple[float, float, float, float]:
    x, y, z = _vector3(rotation_euler)
    cx, sx = math.cos(x * 0.5), math.sin(x * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cz, sz = math.cos(z * 0.5), math.sin(z * 0.5)
    quat = (
        cx * cy * cz - sx * sy * sz,
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
    )
    return _normalize_quaternion(quat) or (1.0, 0.0, 0.0, 0.0)


def _keyframe_quaternion(keyframe: dict[str, Any]) -> tuple[float, float, float, float]:
    return _normalize_quaternion(keyframe.get("rotation_quaternion")) or _euler_xyz_to_quaternion(keyframe.get("rotation_euler"))


def _quat_angle(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dot = abs(sum(a[index] * b[index] for index in range(4)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _allows_lens_ramp(payload: dict[str, Any]) -> bool:
    preset = str(payload.get("preset_name") or "").strip().lower()
    lens_policy = payload.get("lens_policy") if isinstance(payload.get("lens_policy"), dict) else {}
    return preset == "static_subtle_zoom" and bool(lens_policy.get("uses_subtle_lens_ramp", True))


def trajectory_motion_qc(trajectory_path: Path) -> dict[str, Any]:
    if not trajectory_path.exists():
        return {
            "trajectory_exists": False,
            "keyframe_count": 0,
            "has_location_change": False,
            "has_lens_change": False,
            "has_unallowed_lens_change": False,
            "lens_policy_valid": False,
            "rotation_continuity_valid": False,
            "max_adjacent_rotation_delta": 0.0,
            "trajectory_preset": "",
        }
    payload = load_json(trajectory_path)
    keyframes = list(payload.get("keyframes") or [])
    has_location_change = False
    has_lens_change = False
    has_rotation_change = False
    has_unallowed_lens_change = False
    lens_policy_valid = True
    rotation_continuity_valid = True
    max_adjacent_rotation_delta = 0.0
    lens_values: list[float] = []
    travel_distance = 0.0
    duration_seconds = 0.0
    speed_policy: dict[str, Any] = {}
    if len(keyframes) >= 2:
        locations = [_vector3(keyframe.get("location")) for keyframe in keyframes]
        has_location_change = any(
            math.sqrt(sum((locations[index + 1][axis] - locations[index][axis]) ** 2 for axis in range(3))) > 1e-5
            for index in range(len(locations) - 1)
        )
        travel_distance = sum(
            math.sqrt(sum((locations[index + 1][axis] - locations[index][axis]) ** 2 for axis in range(3)))
            for index in range(len(locations) - 1)
        )
        base_lens = _lens_float((payload.get("lens_policy") or {}).get("base_lens_mm") if isinstance(payload.get("lens_policy"), dict) else None, _lens_float(keyframes[0].get("lens_mm"), 35.0))
        inherited_lens = base_lens
        for keyframe in keyframes:
            inherited_lens = _lens_float(keyframe.get("lens_mm"), inherited_lens)
            lens_values.append(round(inherited_lens, 4))
        has_lens_change = any(abs(value - lens_values[0]) > 1e-4 for value in lens_values[1:]) if lens_values else False
        allows_lens_ramp = _allows_lens_ramp(payload)
        lens_ramp_monotonic = all(lens_values[index + 1] + 1e-4 >= lens_values[index] for index in range(max(len(lens_values) - 1, 0)))
        has_unallowed_lens_change = has_lens_change and not allows_lens_ramp
        lens_policy_valid = (not has_unallowed_lens_change) and (lens_ramp_monotonic if allows_lens_ramp else True)
        rotations = [_keyframe_quaternion(keyframe) for keyframe in keyframes]
        adjacent_rotation_deltas = [_quat_angle(rotations[index], rotations[index + 1]) for index in range(len(rotations) - 1)]
        max_adjacent_rotation_delta = max(adjacent_rotation_deltas) if adjacent_rotation_deltas else 0.0
        has_rotation_change = any(delta > 1e-4 for delta in adjacent_rotation_deltas)
        rotation_continuity_valid = max_adjacent_rotation_delta <= ROTATION_CONTINUITY_THRESHOLD_RAD + 1e-6
        start_frame = int(payload.get("start_frame") or keyframes[0].get("frame") or 1)
        end_frame = int(payload.get("end_frame") or keyframes[-1].get("frame") or start_frame)
        timing_policy = payload.get("timing_policy") if isinstance(payload.get("timing_policy"), dict) else {}
        try:
            fps = int(timing_policy.get("fps") or 24)
        except Exception:
            fps = 24
        duration_seconds = max((end_frame - start_frame + 1) / float(max(fps, 1)), 0.0)
        speed_policy = timing_policy.get("motion_speed_policy") if isinstance(timing_policy.get("motion_speed_policy"), dict) else {}
    safety_report = payload.get("safety_report") if isinstance(payload.get("safety_report"), dict) else {}
    visibility_report = safety_report.get("trajectory_visibility_report") if isinstance(safety_report.get("trajectory_visibility_report"), dict) else {}
    return {
        "trajectory_exists": True,
        "keyframe_count": len(keyframes),
        "has_location_change": has_location_change,
        "has_lens_change": has_lens_change,
        "has_rotation_change": has_rotation_change,
        "has_unallowed_lens_change": has_unallowed_lens_change,
        "lens_policy_valid": bool(lens_policy_valid),
        "rotation_continuity_valid": bool(rotation_continuity_valid),
        "max_adjacent_rotation_delta": round(max_adjacent_rotation_delta, 6),
        "rotation_delta_threshold_rad": ROTATION_CONTINUITY_THRESHOLD_RAD,
        "lens_values": lens_values,
        "trajectory_preset": str(payload.get("preset_name") or ""),
        "trajectory_selection_source": str(payload.get("trajectory_selection_source") or ""),
        "safety_report": safety_report,
        "travel_distance": round(travel_distance, 6),
        "duration_seconds": round(duration_seconds, 3),
        "location_speed_units_per_second": round(travel_distance / max(duration_seconds, 1e-6), 6) if duration_seconds else 0.0,
        "motion_speed_policy": speed_policy,
        "speed_limited": bool(speed_policy.get("speed_limited")) if isinstance(speed_policy, dict) else False,
        "visibility_guard_applied": bool(speed_policy.get("visibility_guard_applied")) if isinstance(speed_policy, dict) else False,
        "trajectory_visibility_valid": visibility_report.get("valid") if visibility_report else None,
        "start_frame": payload.get("start_frame"),
        "end_frame": payload.get("end_frame"),
    }


def clip_trim_policy(camera: dict[str, Any], trajectory_path: Path, frame_count: int, fps: int) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if trajectory_path.exists():
        try:
            payload = load_json(trajectory_path)
        except Exception:
            payload = {}
    timing_policy = payload.get("timing_policy") if isinstance(payload.get("timing_policy"), dict) else {}
    handles = camera.get("editor_handles_seconds") or timing_policy.get("editor_handles_seconds") or {}
    if isinstance(handles, (int, float)):
        head_seconds = tail_seconds = float(handles)
    elif isinstance(handles, dict):
        head_seconds = float(handles.get("head") or handles.get("start") or 0.0)
        tail_seconds = float(handles.get("tail") or handles.get("end") or 0.0)
    else:
        head_seconds = tail_seconds = 0.0
    trim_start = max(0, int(round(head_seconds * float(fps or 24))))
    trim_end = max(0, int(round(tail_seconds * float(fps or 24))))
    if frame_count - trim_start - trim_end < 8:
        trim_start = 0
        trim_end = 0
    return {
        "editor_handles_seconds": {"head": round(head_seconds, 3), "tail": round(tail_seconds, 3)},
        "trim_start_frames": trim_start,
        "trim_end_frames": trim_end,
    }


def render_clip_record(
    *,
    camera: dict[str, Any],
    render_entry: dict[str, Any],
    output_root: Path,
    fps: int,
) -> dict[str, Any]:
    clip_dir = ensure_directory(
        output_root
        / "clips"
        / f"scene_{camera['scene_id']}_shot_{camera['shot_id']}"
        / str(camera["camera_name"])
    )
    frame_dir = Path(str(render_entry.get("frame_dir") or ""))
    poster_path = poster_frame(frame_dir, clip_dir)
    clip_path = clip_dir / "clip.mp4"
    encode_result = encode_frames(frame_dir, fps, clip_path)
    if encode_result.returncode != 0:
        encode_result = subprocess.run(
            [
                ffmpeg_path(),
                "-y",
                "-framerate",
                str(fps),
                "-start_number",
                str(detect_start_number(frame_dir)),
                "-i",
                str(frame_dir / "frame_%04d.png"),
                "-c:v",
                "mpeg4",
                str(clip_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    frame_count = int(render_entry.get("frame_count") or len(sorted_frame_paths(frame_dir)))
    duration_seconds = round(frame_count / float(fps), 3) if fps else 0.0
    qc = {
        "render_success": encode_result.returncode == 0 and clip_path.exists(),
        "source_kind": "blender_scene_render",
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "clip_size_bytes": clip_path.stat().st_size if clip_path.exists() else 0,
        "encoder_returncode": encode_result.returncode,
        "scene_name": render_entry.get("scene_name"),
        "focus_source": render_entry.get("focus_source"),
        "trajectory_plan_path": render_entry.get("trajectory_plan_path"),
        "motion_synthesis": render_entry.get("motion_synthesis") or {},
    }
    trajectory_path = Path(str(render_entry.get("trajectory_plan_path") or ""))
    motion_qc = trajectory_motion_qc(trajectory_path)
    trim_policy = clip_trim_policy(camera, trajectory_path, frame_count, fps)
    motion_qc["meaningful_motion"] = bool(
        motion_qc.get("has_location_change")
        or motion_qc.get("has_rotation_change")
        or motion_qc.get("has_lens_change")
    )
    return {
        "scene_id": camera["scene_id"],
        "shot_id": camera["shot_id"],
        "camera_name": camera["camera_name"],
        "camera_index": camera.get("camera_index"),
        "movement_tag": camera.get("movement_tag"),
        "target_duration_seconds": camera.get("target_duration_seconds"),
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "frame_dir": str(frame_dir),
        "clip_path": str(clip_path),
        "poster_frame_path": str(poster_path) if poster_path and poster_path.exists() else "",
        "trajectory_plan_path": str(trajectory_path) if trajectory_path.exists() else "",
        "trajectory_preset": motion_qc.get("trajectory_preset") or "",
        "trajectory_selection_source": motion_qc.get("trajectory_selection_source") or camera.get("trajectory_selection_source") or "",
        "motion_qc": motion_qc,
        "editor_handles_seconds": trim_policy["editor_handles_seconds"],
        "trim_start_frames": trim_policy["trim_start_frames"],
        "trim_end_frames": trim_policy["trim_end_frames"],
        "cut_in_frame": trim_policy["trim_start_frames"],
        "cut_out_frame": max(frame_count - trim_policy["trim_end_frames"] - 1, 0),
        "cut_reason": "handle_trim",
        "continuity_match_score": None,
        "same_scene_match_cut": False,
        "quality_qc": camera.get("quality_qc") or {},
        "quality_candidate_board_path": camera.get("quality_candidate_board_path") or "",
        "same_shot_diversity_review": camera.get("same_shot_diversity_review") or {},
        "same_shot_diversity_status": camera.get("same_shot_diversity_status") or "",
        "similar_to_camera_name": camera.get("similar_to_camera_name") or "",
        "diversity_reselection_trace": camera.get("diversity_reselection_trace") or [],
        "editor_recommended_omit": bool(camera.get("editor_recommended_omit")),
        "transition_to_next": camera.get("bridge_to_next"),
        "render_qc": qc,
    }


def run_video_engineer(config: VideoEngineerConfig) -> dict[str, Any]:
    output_root = ensure_directory(Path(config.output_root).resolve())
    outputs_dir = ensure_directory(output_root / "outputs")
    camera_handoff_path = Path(config.camera_handoff_path).resolve()
    camera_handoff = load_json(camera_handoff_path)
    blend_file, director_handoff, source_inventory = resolve_blend_file(camera_handoff)
    demo_root = Path(str(director_handoff.get("demo_root") or "")).resolve()

    blender_result = run_blender_renders(
        config=config,
        camera_handoff_path=camera_handoff_path,
        output_root=output_root,
        blend_file=blend_file,
    )
    if not blender_result.get("success"):
        raise RuntimeError(
            "Blender scene render failed. "
            f"See {blender_result.get('stderr_path') or (output_root / 'blender_render_stderr.log')}"
        )

    render_report_path = Path(str(blender_result["render_report_path"])).resolve()
    render_report_payload = load_json(render_report_path)
    render_entries = render_report_payload.get("render_report") or []
    render_lookup = {
        (int(item.get("scene_id") or 0), int(item.get("shot_id") or 0), str(item.get("camera_name") or "")): item
        for item in render_entries
    }

    clips: list[dict[str, Any]] = []
    shot_timelines: list[dict[str, Any]] = []
    for shot in camera_handoff.get("shots") or []:
        shot_clips: list[dict[str, Any]] = []
        for camera in shot.get("cameras") or []:
            key = (int(camera.get("scene_id") or 0), int(camera.get("shot_id") or 0), str(camera.get("camera_name") or ""))
            render_entry = render_lookup.get(key)
            if not render_entry or not render_entry.get("success"):
                raise RuntimeError(f"Missing successful Blender render for camera {key}")
            clip = render_clip_record(
                camera=camera,
                render_entry=render_entry,
                output_root=output_root,
                fps=config.fps,
            )
            shot_clips.append(clip)
            clips.append(clip)
        shot_timelines.append(
            {
                "scene_id": shot.get("scene_id"),
                "shot_id": shot.get("shot_id"),
                "clip_names": [clip["camera_name"] for clip in shot_clips],
                "clip_paths": [clip["clip_path"] for clip in shot_clips],
                "transition_plan": [clip.get("transition_to_next") for clip in shot_clips if clip.get("transition_to_next")],
            }
        )

    clips_manifest_path = save_json(
        {
            "schema_version": "storyblender.clips_manifest.v1",
            "generated_at": utc_now(),
            "fps": config.fps,
            "resolution": {"width": config.resolution_x, "height": config.resolution_y},
            "camera_quality": config.camera_quality,
            "clips": clips,
        },
        outputs_dir / "clips_manifest_v1.json",
    )
    video_handoff_path = save_json(
        {
            "schema_version": "storyblender.video_handoff.v1",
            "generated_at": utc_now(),
            "run_id": config.run_id or output_root.name,
            "camera_handoff_path": str(camera_handoff_path),
            "fps": config.fps,
            "resolution": {"width": config.resolution_x, "height": config.resolution_y},
            "camera_quality": config.camera_quality,
            "clip_manifest_path": str(clips_manifest_path),
            "clips": clips,
            "shot_timelines": shot_timelines,
            "audio_refs": [],
            "blender_render_report_path": str(render_report_path),
            "blender_render_stdout_path": str(blender_result.get("stdout_path") or ""),
            "blender_render_stderr_path": str(blender_result.get("stderr_path") or ""),
        },
        outputs_dir / "video_handoff_v1.json",
    )
    manifest_path = save_json(
        {
            "schema_version": "storyblender.video_manifest.v1",
            "generated_at": utc_now(),
            "success": True,
            "run_id": config.run_id or output_root.name,
            "camera_handoff_path": str(camera_handoff_path),
            "blend_file": str(blend_file),
            "output_root": str(output_root),
            "clips_manifest_path": str(clips_manifest_path),
            "video_handoff_path": str(video_handoff_path),
            "blender_render_result": blender_result,
            "source_inventory_path": str(Path(str((director_handoff.get("files") or {}).get("source_inventory_path") or ""))),
            "blend_paths": list((source_inventory.get("selected_sources") or {}).get("blend_paths") or []),
        },
        output_root / "manifest.json",
    )
    return {
        "success": True,
        "manifest_path": str(manifest_path),
        "clips_manifest_path": str(clips_manifest_path),
        "video_handoff_path": str(video_handoff_path),
    }


def main() -> int:
    config = parse_args()
    result = run_video_engineer(config)
    print("VideoEngineer completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
