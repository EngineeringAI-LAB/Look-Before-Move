from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cinematographer_engine_blender import run_blender_python_script
from cinematographer_llm_adapter import call_json_response, image_path_to_data_url, llm_ready


STAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = STAGE_DIR.parent
DIRECTOR_OUTPUT_DIR = WORKSPACE_DIR / "Director" / "output"
CAMERA_DIR = WORKSPACE_DIR.parent.parent.parent
DEFAULT_BLENDER_EXE = Path(os.getenv("STORYBLENDER_BLENDER_EXE", "blender"))
CONFIG_DIR = WORKSPACE_DIR / "config"
CONFIG_CANDIDATE_PATHS = [
    CONFIG_DIR / "runtime_config.json",
    CONFIG_DIR / "storyblender_runtime_config.json",
    CONFIG_DIR / "local_config.json",
]


# Module-level ablation flags, populated by `run_cinematographer` at startup.
# Used by helpers (e.g. `build_camera_trajectory_plan`) that are not given the
# CinematographerConfig object directly. Keep keys in sync with
# CinematographerConfig fields.
_ABLATION_FLAGS: dict[str, bool] = {
    "disable_vlm_reflection": False,
    "disable_trajectory_grounding": False,
    "disable_semantic_height_adjust": False,
}


def _set_ablation_flags(config: "CinematographerConfig") -> None:
    _ABLATION_FLAGS["disable_vlm_reflection"] = bool(getattr(config, "disable_vlm_reflection", False))
    _ABLATION_FLAGS["disable_trajectory_grounding"] = bool(getattr(config, "disable_trajectory_grounding", False))
    _ABLATION_FLAGS["disable_semantic_height_adjust"] = bool(getattr(config, "disable_semantic_height_adjust", False))


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


def bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def start_frame_contract(camera_package: dict[str, Any]) -> dict[str, Any]:
    return dict(((camera_package.get("shot_contract") or {}).get("start_frame_contract") or {}))


def persist_camera_package(camera: dict[str, Any]) -> None:
    package_path = Path(str(camera.get("camera_package_path") or ""))
    if package_path.exists():
        save_json(camera, package_path)


def load_runtime_defaults(stage_key: str) -> dict[str, Any]:
    for path in CONFIG_CANDIDATE_PATHS:
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        defaults: dict[str, Any] = {}
        shared = payload.get("shared")
        stage_payload = payload.get(stage_key)
        if isinstance(shared, dict):
            defaults.update(shared)
        if isinstance(stage_payload, dict):
            defaults.update(stage_payload)
        defaults["_config_path"] = str(path)
        return defaults
    return {}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def canonical_direction(raw_direction: Any, angle: str = "") -> str:
    text = str(raw_direction or "").strip().lower()
    angle_text = str(angle or "").strip().lower()
    if "top" in text or "bird" in angle_text or "top" in angle_text:
        return "top"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "back" in text or "rear" in text:
        return "back"
    return "front"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    rows: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in rows:
            rows.append(text)
    return rows


_NONHUMAN_FOCUS_TOKENS = (
    "font", "bed", "cart", "car", "sedan", "table", "desk", "door", "wall",
    "window", "gun", "weapon", "lamp", "chair", "barrel", "bottle", "glass",
)


def likely_human_focus_id(asset_id: str) -> bool:
    text = str(asset_id or "").lower()
    return bool(text) and not any(token in text for token in _NONHUMAN_FOCUS_TOKENS)


def canonical_movement(raw_movement: Any) -> str:
    text = str(raw_movement or "").strip().lower().replace("-", " ").replace("_", " ")
    if not text:
        return "static"
    if "lock" in text:
        return "locked_off"
    if "push in" in text or "zoom in" in text or "dolly in" in text:
        return "push_in"
    if "push out" in text or "zoom out" in text or "dolly out" in text:
        return "push_out"
    if "orbit" in text:
        return "orbit"
    if "pan" in text:
        return "pan"
    if "truck" in text:
        return "truck"
    if "pedestal" in text or "crane" in text:
        return "pedestal"
    if "static" in text or "hold" in text:
        return "static"
    return text.replace(" ", "_")


def latest_director_handoff() -> Path:
    candidates = sorted((item for item in DIRECTOR_OUTPUT_DIR.iterdir() if item.is_dir()), key=lambda item: item.name)
    if not candidates:
        raise FileNotFoundError("No Director output folders found. Run Director/run_director.py first.")
    return candidates[-1] / "outputs" / "director_handoff_v1.json"


def default_output_root(run_id: str | None = None) -> Path:
    suffix = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return STAGE_DIR / "output" / suffix


@dataclass(slots=True)
class CinematographerConfig:
    director_handoff_path: str
    output_root: str
    run_id: str = ""
    fps: int = 24
    plate_width: int = 1600
    plate_height: int = 900
    frame_width: int = 960
    frame_height: int = 540
    camera_quality: str = "fast"
    blender_exe: str = ""
    quality_resolution_x: int = 480
    quality_resolution_y: int = 270
    quality_render_engine: str = "BLENDER_EEVEE_NEXT"
    quality_render_samples: int = 8
    quality_top_k: int = 20
    quality_max_depth: int = 5
    quality_frontier_limit: int = 24
    quality_timeout_seconds: int = 7200
    enable_candidate_validation: bool = False
    validation_failed_top_per_channel: int = 20
    validation_success_near_threshold_per_channel: int = 20
    validation_reason_samples: int = 5
    preview_resolution_x: int = 480
    preview_resolution_y: int = 270
    preview_render_samples: int = 8
    vision_model: str = "gemini-3-flash-preview"
    anyllm_api_key: str = ""
    anyllm_api_base: str = "https://yunwu.ai"
    anyllm_provider: str = "gemini"
    llm_micro_adjust_max_rounds: int = 10
    per_channel_top_k: int = 20
    llm_max_workers: int = 4
    llm_retry_count: int = 2
    final_review_retry_count: int = 2
    run_pre_continuity_story_judge: bool = False
    resume_existing: bool = False
    # --- ablation switches (2026-05-03 quality_and_ablations batch) ---
    disable_vlm_reflection: bool = False
    disable_trajectory_grounding: bool = False
    disable_semantic_height_adjust: bool = False


def parse_args() -> CinematographerConfig:
    runtime_defaults = load_runtime_defaults("cinematographer")
    parser = argparse.ArgumentParser(description="Run the isolated Cinematographer stage.")
    parser.add_argument("--director-handoff-path", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--plate-width", type=int, default=1600)
    parser.add_argument("--plate-height", type=int, default=900)
    parser.add_argument("--frame-width", type=int, default=960)
    parser.add_argument("--frame-height", type=int, default=540)
    parser.add_argument("--camera-quality", choices=("fast", "quality"), default="fast")
    parser.add_argument("--blender-exe", default=str(runtime_defaults.get("blender_exe") or os.getenv("STORYBLENDER_BLENDER_EXE", "")))
    parser.add_argument("--quality-resolution-x", type=int, default=480)
    parser.add_argument("--quality-resolution-y", type=int, default=270)
    parser.add_argument("--quality-render-engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--quality-render-samples", type=int, default=8)
    parser.add_argument("--quality-top-k", type=int, default=20)
    parser.add_argument("--quality-max-depth", type=int, default=5)
    parser.add_argument("--quality-frontier-limit", type=int, default=24)
    parser.add_argument("--quality-timeout-seconds", type=int, default=7200)
    parser.add_argument("--enable-candidate-validation", action="store_true")
    parser.add_argument("--validation-failed-top-per-channel", type=int, default=20)
    parser.add_argument("--validation-success-near-threshold-per-channel", type=int, default=20)
    parser.add_argument("--validation-reason-samples", type=int, default=5)
    parser.add_argument("--preview-resolution-x", type=int, default=480)
    parser.add_argument("--preview-resolution-y", type=int, default=270)
    parser.add_argument("--preview-render-samples", type=int, default=8)
    parser.add_argument("--vision-model", default=str(runtime_defaults.get("vision_model") or os.getenv("STORYBLENDER_VISION_MODEL", "gemini-3-flash-preview")))
    parser.add_argument("--anyllm-api-key", default=str(runtime_defaults.get("anyllm_api_key") or os.getenv("ANYLLM_API_KEY", "")))
    parser.add_argument("--anyllm-api-base", default=str(runtime_defaults.get("anyllm_api_base") or os.getenv("ANYLLM_API_BASE", "https://yunwu.ai")))
    parser.add_argument("--anyllm-provider", default=str(runtime_defaults.get("anyllm_provider") or os.getenv("ANYLLM_PROVIDER", "gemini")))
    parser.add_argument("--llm-micro-adjust-max-rounds", type=int, default=10)
    parser.add_argument("--per-channel-top-k", type=int, default=20)
    parser.add_argument("--llm-max-workers", type=int, default=4)
    parser.add_argument("--llm-retry-count", type=int, default=2)
    parser.add_argument("--final-review-retry-count", type=int, default=2)
    parser.add_argument("--run-pre-continuity-story-judge", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--disable-vlm-reflection",
        action="store_true",
        help="Ablation: skip llm_micro_adjust loops and final_preview_review pipeline.",
    )
    parser.add_argument(
        "--disable-trajectory-grounding",
        action="store_true",
        help="Ablation: replace physics-aware trajectory plan with naive 2-keyframe linear interpolation.",
    )
    parser.add_argument(
        "--disable-semantic-height-adjust",
        action="store_true",
        help="Ablation: always look at geometric center (no face/feet/hand semantic Z offset). Forwarded to quality worker.",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(args.run_id).resolve()
    director_handoff_path = args.director_handoff_path or str(latest_director_handoff())
    return CinematographerConfig(
        director_handoff_path=director_handoff_path,
        output_root=str(output_root),
        run_id=args.run_id,
        fps=args.fps,
        plate_width=args.plate_width,
        plate_height=args.plate_height,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        camera_quality=args.camera_quality,
        blender_exe=args.blender_exe,
        quality_resolution_x=args.quality_resolution_x,
        quality_resolution_y=args.quality_resolution_y,
        quality_render_engine=args.quality_render_engine,
        quality_render_samples=args.quality_render_samples,
        quality_top_k=args.quality_top_k,
        quality_max_depth=args.quality_max_depth,
        quality_frontier_limit=args.quality_frontier_limit,
        quality_timeout_seconds=args.quality_timeout_seconds,
        enable_candidate_validation=bool(args.enable_candidate_validation),
        validation_failed_top_per_channel=args.validation_failed_top_per_channel,
        validation_success_near_threshold_per_channel=args.validation_success_near_threshold_per_channel,
        validation_reason_samples=args.validation_reason_samples,
        preview_resolution_x=args.preview_resolution_x,
        preview_resolution_y=args.preview_resolution_y,
        preview_render_samples=args.preview_render_samples,
        vision_model=args.vision_model,
        anyllm_api_key=args.anyllm_api_key,
        anyllm_api_base=args.anyllm_api_base,
        anyllm_provider=args.anyllm_provider,
        llm_micro_adjust_max_rounds=args.llm_micro_adjust_max_rounds,
        per_channel_top_k=args.per_channel_top_k,
        llm_max_workers=args.llm_max_workers,
        llm_retry_count=args.llm_retry_count,
        final_review_retry_count=args.final_review_retry_count,
        run_pre_continuity_story_judge=bool(args.run_pre_continuity_story_judge),
        resume_existing=bool(args.resume_existing),
        disable_vlm_reflection=bool(args.disable_vlm_reflection),
        disable_trajectory_grounding=bool(args.disable_trajectory_grounding),
        disable_semantic_height_adjust=bool(args.disable_semantic_height_adjust),
    )


def is_closeup(distance_label: str) -> bool:
    label = str(distance_label or "").strip().lower()
    return label in {"extreme close-up", "close-up", "medium close-up"}


def subtle_motion(distance_label: str) -> tuple[str, float]:
    label = str(distance_label or "").strip().lower()
    if label in {"extreme close-up", "close-up", "medium close-up"}:
        return "push_out", 0.02
    if label in {"medium shot", "full shot"}:
        return "push_in", 0.03
    return "push_in", 0.04


def movement_duration_seconds(movement_tag: str) -> float:
    movement = str(movement_tag or "").strip().lower()
    if movement in {"static", "locked_off"}:
        return 2.5
    if movement in {"push_in", "push_out"}:
        return 3.0
    if movement == "truck":
        return 3.2
    if movement in {"pan", "pedestal"}:
        return 3.6
    if movement == "orbit":
        return 4.2
    return 3.0


def width_fraction_for_distance(distance_label: str) -> float:
    mapping = {
        "extreme close-up": 0.34,
        "close-up": 0.42,
        "medium close-up": 0.52,
        "medium shot": 0.64,
        "full shot": 0.76,
        "wide shot": 0.88,
        "long shot": 0.94,
    }
    return mapping.get(str(distance_label or "").strip().lower(), 0.64)


def lens_mm_for_distance(distance_label: str) -> float:
    mapping = {
        "extreme close-up": 85.0,
        "close-up": 65.0,
        "medium close-up": 50.0,
        "medium shot": 35.0,
        "full shot": 32.0,
        "wide shot": 28.0,
        "long shot": 24.0,
    }
    return mapping.get(str(distance_label or "").strip().lower(), 35.0)


def angle_pitch(angle_label: str) -> float:
    text = str(angle_label or "").strip().lower()
    if "bird" in text or "top" in text:
        return -55.0
    if "high" in text:
        return -16.0
    if "low" in text:
        return 14.0
    return 0.0


def load_director_context(director_handoff_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    handoff = load_json(director_handoff_path)
    blocking_plans_path = Path(handoff["files"]["blocking_plans_path"])
    blocking_plans = load_json(blocking_plans_path)
    return handoff, blocking_plans


def contract_lookup(shot_contracts: list[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    for contract in shot_contracts or []:
        try:
            scene_id = int(contract.get("scene_id") or 0)
            shot_id = int(contract.get("shot_id") or 0)
        except (TypeError, ValueError):
            continue
        camera_name = str(contract.get("camera_name") or "").strip()
        if camera_name:
            rows[(scene_id, shot_id, camera_name)] = contract
    return rows


def layout_bounds(scene_details: dict[str, Any]) -> tuple[float, float, float, float]:
    scene_size = scene_details.get("scene_size") or {}
    x_min = float(scene_size.get("x_negative", -10.0))
    x_max = float(scene_size.get("x", 10.0))
    y_min = float(scene_size.get("y_negative", -10.0))
    y_max = float(scene_size.get("y", 10.0))
    if x_min == x_max:
        x_min, x_max = -10.0, 10.0
    if y_min == y_max:
        y_min, y_max = -10.0, 10.0
    return x_min, x_max, y_min, y_max


def find_focus_centroid(focus_ids: list[str], scene_details: dict[str, Any]) -> tuple[float, float, float]:
    layout_assets = scene_details.get("layout_assets") or []
    matches: list[tuple[float, float, float]] = []
    for asset in layout_assets:
        asset_id = str(asset.get("asset_id") or "").strip()
        if asset_id not in focus_ids:
            continue
        location = asset.get("location") or {}
        matches.append(
            (
                float(location.get("x", 0.0)),
                float(location.get("y", 0.0)),
                float(location.get("z", 0.0)),
            )
        )
    if not matches:
        return 0.0, 0.0, 0.0
    count = float(len(matches))
    return (
        sum(item[0] for item in matches) / count,
        sum(item[1] for item in matches) / count,
        sum(item[2] for item in matches) / count,
    )


def primary_focus_layout_asset(focus_ids: list[str], scene_details: dict[str, Any]) -> dict[str, Any]:
    if not focus_ids:
        return {}
    primary_focus_id = str(focus_ids[0] or "").strip()
    for asset in scene_details.get("layout_assets") or []:
        if str(asset.get("asset_id") or "").strip() == primary_focus_id:
            return dict(asset)
    return {}


def subject_relative_axes(focus_ids: list[str], scene_details: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    asset = primary_focus_layout_asset(focus_ids, scene_details)
    rotation = asset.get("rotation") or {}
    try:
        yaw_degrees = float(rotation.get("z", 0.0) or 0.0)
    except (TypeError, ValueError):
        yaw_degrees = 0.0
    yaw = math.radians(yaw_degrees)
    front = (math.sin(yaw), -math.cos(yaw))
    right = (math.cos(yaw), math.sin(yaw))
    return front, right


def frame_center_from_focus(focus_ids: list[str], scene_details: dict[str, Any]) -> tuple[float, float]:
    focus_x, focus_y, focus_z = find_focus_centroid(focus_ids, scene_details)
    x_min, x_max, y_min, y_max = layout_bounds(scene_details)
    x_norm = (focus_x - x_min) / max(x_max - x_min, 0.001)
    y_norm = (focus_y - y_min) / max(y_max - y_min, 0.001)
    center_x = clamp(0.18 + x_norm * 0.64, 0.18, 0.82)
    center_y = clamp(0.60 - (y_norm - 0.5) * 0.12 - focus_z * 0.02, 0.28, 0.72)
    return center_x, center_y


def frame_window(
    *,
    focus_ids: list[str],
    scene_details: dict[str, Any],
    distance_label: str,
) -> dict[str, float]:
    center_x, center_y = frame_center_from_focus(focus_ids, scene_details)
    width_fraction = width_fraction_for_distance(distance_label)
    return {
        "center_x": round(center_x, 5),
        "center_y": round(center_y, 5),
        "width": round(width_fraction, 5),
        "height": round(width_fraction, 5),
    }


def apply_movement_to_window(
    start_window: dict[str, float],
    movement_tag: str,
    magnitude: float,
    direction_tag: str,
) -> dict[str, float]:
    end_window = dict(start_window)
    if movement_tag == "push_in":
        end_window["width"] = clamp(end_window["width"] * (1.0 - magnitude), 0.24, 0.98)
        end_window["height"] = clamp(end_window["height"] * (1.0 - magnitude), 0.24, 0.98)
    elif movement_tag == "push_out":
        end_window["width"] = clamp(end_window["width"] * (1.0 + magnitude), 0.24, 0.98)
        end_window["height"] = clamp(end_window["height"] * (1.0 + magnitude), 0.24, 0.98)
    elif movement_tag == "pan":
        delta = 0.03 if direction_tag == "right" else -0.03
        end_window["center_x"] = clamp(end_window["center_x"] + delta, 0.16, 0.84)
    elif movement_tag == "truck":
        delta = 0.08 if direction_tag == "right" else -0.08
        end_window["center_x"] = clamp(end_window["center_x"] + delta, 0.16, 0.84)
        end_window["width"] = clamp(end_window["width"] * 0.98, 0.24, 0.98)
        end_window["height"] = clamp(end_window["height"] * 0.98, 0.24, 0.98)
    elif movement_tag == "orbit":
        delta = 0.02 if direction_tag == "right" else -0.02
        end_window["center_x"] = clamp(end_window["center_x"] + delta, 0.16, 0.84)
    elif movement_tag == "pedestal":
        end_window["center_y"] = clamp(end_window["center_y"] - 0.02, 0.24, 0.76)
    return {key: round(float(value), 5) for key, value in end_window.items()}


def camera_transform(
    *,
    focus_ids: list[str],
    scene_details: dict[str, Any],
    distance_label: str,
    angle_label: str,
    direction_tag: str,
    frame: dict[str, float],
) -> dict[str, Any]:
    focus_x, focus_y, focus_z = find_focus_centroid(focus_ids, scene_details)
    distance_map = {
        "extreme close-up": 1.0,
        "close-up": 1.3,
        "medium close-up": 1.9,
        "medium shot": 3.0,
        "full shot": 4.1,
        "wide shot": 5.6,
        "long shot": 6.8,
    }
    distance = distance_map.get(str(distance_label or "").strip().lower(), 3.0)
    front_axis, right_axis = subject_relative_axes(focus_ids, scene_details)
    if direction_tag == "top":
        x_offset = 0.0
        y_offset = 0.0
    elif direction_tag == "back":
        x_offset = -front_axis[0] * distance * 0.9
        y_offset = -front_axis[1] * distance * 0.9
    elif direction_tag == "left":
        x_offset = -right_axis[0] * distance * 0.75
        y_offset = -right_axis[1] * distance * 0.75
    elif direction_tag == "right":
        x_offset = right_axis[0] * distance * 0.75
        y_offset = right_axis[1] * distance * 0.75
    else:
        x_offset = front_axis[0] * distance
        y_offset = front_axis[1] * distance
    z_offset = 1.55 if direction_tag != "top" else 7.0
    yaw = {"left": 24.0, "right": -24.0, "back": 180.0, "top": 0.0, "front": 0.0}.get(direction_tag, 0.0)
    return {
        "location": [round(focus_x + x_offset, 4), round(focus_y + y_offset, 4), round(focus_z + z_offset, 4)],
        "rotation_euler": [round(math.radians(angle_pitch(angle_label)), 6), round(math.radians(yaw), 6), 0.0],
        "framing": frame,
    }


def _rotation3(values: Any, default: list[float] | None = None) -> list[float]:
    fallback = list(default or [0.0, 0.0, 0.0])
    try:
        if values is None:
            raise ValueError
        result = [float(value) for value in list(values)[:3]]
        if len(result) != 3 or not all(math.isfinite(value) for value in result):
            raise ValueError
        return result
    except Exception:
        return fallback


def _unwrap_angle_near(reference: float, candidate: float) -> float:
    angle = float(candidate)
    while angle - reference > math.pi:
        angle -= math.tau
    while angle - reference < -math.pi:
        angle += math.tau
    return angle


def _align_rotation_near(reference: Any, candidate: Any) -> list[float]:
    reference_values = _rotation3(reference)
    candidate_values = _rotation3(candidate, reference_values)
    return [
        round(_unwrap_angle_near(reference_values[index], candidate_values[index]), 6)
        for index in range(3)
    ]


def _lens_mm(value: Any, fallback: float = 35.0) -> float:
    try:
        lens = float(value)
        if math.isfinite(lens) and lens > 0:
            return lens
    except Exception:
        pass
    return float(fallback)


def _trajectory_allows_lens_ramp(preset_name: str | None) -> bool:
    return str(preset_name or "").strip().lower() == "static_subtle_zoom"


def build_trajectory_keyframes(
    *,
    start_transform: dict[str, Any],
    end_transform: dict[str, Any],
    movement_tag: str,
    motion_profile: str,
    frame_count: int,
    fps: int,
    preset_name: str | None = None,
) -> list[dict[str, Any]]:
    preset = normalize_trajectory_preset(preset_name or preset_for_motion(movement_tag, "", motion_profile))
    start_lens = round(_lens_mm(start_transform.get("lens_mm"), _lens_mm(end_transform.get("lens_mm"), 35.0)), 4)
    start_transform = dict(start_transform)
    start_transform["lens_mm"] = start_lens
    if not _trajectory_allows_lens_ramp(preset):
        end_transform = dict(end_transform)
        end_transform["lens_mm"] = start_lens
    if preset == "static_hold":
        end_transform = dict(start_transform)
    elif preset == "static_subtle_zoom":
        end_transform = dict(start_transform)
        end_transform["lens_mm"] = round(min(start_lens * 1.04, 85.0), 4)
    elif motion_profile == "closeup_static" and preset == "static_subtle_zoom":
        preset = "static_subtle_zoom"
        end_transform = dict(start_transform)
        end_transform["lens_mm"] = round(min(start_lens * 1.04, 85.0), 4)

    if preset in {"static_hold", "static_subtle_zoom"}:
        return [
            {"frame": 0, "t": 0.0, "transform": start_transform, "easing": "linear"},
            {"frame": frame_count - 1, "t": 1.0, "transform": end_transform, "easing": "linear"},
        ]
    start_loc = start_transform.get("location") or [0.0, 0.0, 0.0]
    end_loc = end_transform.get("location") or [0.0, 0.0, 0.0]
    start_rot = _rotation3(start_transform.get("rotation_euler"))
    end_rot = _align_rotation_near(start_rot, end_transform.get("rotation_euler") or start_rot)
    if preset in {"pan_left", "pan_right"}:
        end_loc = list(start_loc)
    aligned_end_transform = dict(end_transform)
    aligned_end_transform["location"] = [round(float(value), 6) for value in end_loc]
    aligned_end_transform["rotation_euler"] = end_rot
    aligned_end_transform["lens_mm"] = start_lens
    keyframes = [
        {"frame": 0, "t": 0.0, "transform": start_transform, "easing": "ease_in"},
    ]
    if (movement_tag in {"orbit", "truck", "pan", "pedestal"} or preset in MIDPOINT_TRAJECTORY_PRESETS) and frame_count > 48:
        mid_frame = frame_count // 2
        mid_loc = [round((start_loc[i] + end_loc[i]) * 0.5, 4) for i in range(3)]
        if preset in {"pedestal_up", "pedestal_down"}:
            delta = 0.04 if preset == "pedestal_up" else -0.04
            mid_loc[2] = round(mid_loc[2] + delta, 4)
        elif preset in {"rise_reveal", "drop_reveal"}:
            delta = 0.05 if preset == "rise_reveal" else -0.05
            mid_loc[2] = round(mid_loc[2] + delta, 4)
        elif preset == "s_curve":
            mid_loc[0] = round(mid_loc[0] + 0.03, 4)
        mid_rot = _align_rotation_near(start_rot, [(start_rot[i] + end_rot[i]) * 0.5 for i in range(3)])
        mid_framing = {}
        for key in ("center_x", "center_y", "width", "height"):
            start_val = float((start_transform.get("framing") or {}).get(key, 0.5))
            end_val = float((end_transform.get("framing") or {}).get(key, 0.5))
            mid_framing[key] = round((start_val + end_val) * 0.5, 5)
        keyframes.append({
            "frame": mid_frame,
            "t": 0.5,
            "transform": {"location": mid_loc, "rotation_euler": mid_rot, "lens_mm": start_lens, "framing": mid_framing},
            "easing": "ease_in_out",
        })
    keyframes.append({
        "frame": frame_count - 1,
        "t": 1.0,
        "transform": aligned_end_transform,
        "easing": "ease_out",
    })
    return keyframes


TRAJECTORY_PRESET_ALIASES = {
    "static": "static_hold",
    "locked_off": "static_hold",
    "locked off": "static_hold",
    "hold": "static_hold",
    "push_in": "push_in_arc",
    "push in": "push_in_arc",
    "zoom in": "push_in_arc",
    "push_out": "pull_out_arc",
    "push out": "pull_out_arc",
    "zoom out": "pull_out_arc",
    "pull out": "pull_out_arc",
}

MIDPOINT_TRAJECTORY_PRESETS = {
    "orbit_left_arc",
    "orbit_right_arc",
    "pan_left",
    "pan_right",
    "truck_left",
    "truck_right",
    "pedestal_up",
    "pedestal_down",
    "rise_reveal",
    "drop_reveal",
    "s_curve",
}

HIGH_RISK_TRAJECTORY_PRESETS = {
    "orbit_left_arc",
    "orbit_right_arc",
    "rise_reveal",
    "drop_reveal",
    "s_curve",
}

TRAJECTORY_TRAVEL_LIMITS = {
    "straight_ease": 0.95,
    "push_in_arc": 0.85,
    "pull_out_arc": 0.85,
    "truck_left": 1.05,
    "truck_right": 1.05,
    "orbit_left_arc": 0.8,
    "orbit_right_arc": 0.8,
    "pedestal_up": 0.35,
    "pedestal_down": 0.35,
    "rise_reveal": 0.42,
    "drop_reveal": 0.42,
    "s_curve": 0.85,
}

TRAJECTORY_MAX_SPEED = {
    "straight_ease": 0.28,
    "push_in_arc": 0.25,
    "pull_out_arc": 0.25,
    "truck_left": 0.30,
    "truck_right": 0.30,
    "pan_left": 0.0,
    "pan_right": 0.0,
    "pedestal_up": 0.12,
    "pedestal_down": 0.12,
    "orbit_left_arc": 0.22,
    "orbit_right_arc": 0.22,
    "rise_reveal": 0.12,
    "drop_reveal": 0.12,
    "s_curve": 0.22,
}


def normalize_trajectory_preset(raw_preset: Any) -> str:
    text = str(raw_preset or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return "straight_ease"
    return TRAJECTORY_PRESET_ALIASES.get(text, text)


def preset_for_motion(movement_tag: Any, direction_tag: Any = "", motion_profile: Any = "") -> str:
    profile = str(motion_profile or "").strip().lower()
    if profile == "closeup_static":
        return "static_subtle_zoom"
    movement = str(movement_tag or "").strip().lower()
    direction = str(direction_tag or "").strip().lower()
    if movement in {"static", "locked_off"}:
        return "static_hold"
    if movement == "push_in":
        return "push_in_arc"
    if movement == "push_out":
        return "pull_out_arc"
    if movement == "orbit":
        return "orbit_left_arc" if direction == "left" else "orbit_right_arc"
    if movement == "pan":
        return "pan_left" if direction == "left" else "pan_right"
    if movement == "truck":
        return "truck_left" if direction == "left" else "truck_right"
    if movement == "pedestal":
        return "pedestal_up"
    return "straight_ease"


def trajectory_duration_seconds(
    preset_name: Any,
    *,
    travel_distance: float = 0.0,
    closeup: bool = False,
) -> float:
    preset = normalize_trajectory_preset(preset_name)
    if preset in {"static_hold", "static_subtle_zoom"}:
        return 2.5
    if preset in {"straight_ease", "push_in_arc", "pull_out_arc"}:
        base_duration = 3.0
    elif preset in {"truck_left", "truck_right"}:
        base_duration = 3.6
    elif preset in {"pan_left", "pan_right", "pedestal_up", "pedestal_down"}:
        base_duration = 3.8
    elif preset in {"orbit_left_arc", "orbit_right_arc", "rise_reveal", "drop_reveal", "s_curve"}:
        base_duration = 4.4
    else:
        base_duration = 3.0
    if closeup:
        base_duration = max(base_duration, 3.2)
    max_speed = TRAJECTORY_MAX_SPEED.get(preset)
    if max_speed and travel_distance > 0.0:
        base_duration = max(base_duration, travel_distance / max_speed)
    if preset in {"orbit_left_arc", "orbit_right_arc", "rise_reveal", "drop_reveal", "s_curve"}:
        return round(min(base_duration, 5.0), 3)
    return round(min(base_duration, 4.8), 3)


def closeup_motion_context(camera: dict[str, Any]) -> bool:
    return bool(camera.get("closeup_required")) or is_closeup(str(camera.get("distance_label") or ""))


def trajectory_motion_limit(preset_name: Any, *, closeup: bool = False) -> float:
    preset = normalize_trajectory_preset(preset_name)
    if preset in {"static_hold", "static_subtle_zoom", "pan_left", "pan_right"}:
        return 0.0
    limit = float(TRAJECTORY_TRAVEL_LIMITS.get(preset, 1.4))
    if closeup:
        return min(limit, 0.25)
    return limit


def limit_trajectory_motion(
    *,
    camera: dict[str, Any],
    preset_name: Any,
    start_transform: dict[str, Any],
    end_transform: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    preset = normalize_trajectory_preset(preset_name)
    start_location = _vector3(start_transform.get("location"), [0.0, -3.0, 1.6])
    end_location = _vector3(end_transform.get("location"), start_location)
    closeup = closeup_motion_context(camera)
    original_travel = _distance_between(start_location, end_location)
    max_travel = trajectory_motion_limit(preset, closeup=closeup)
    limited_end = dict(end_transform)
    speed_limited = False
    if preset in {"static_hold", "static_subtle_zoom", "pan_left", "pan_right"}:
        limited_end["location"] = [round(value, 6) for value in start_location]
        speed_limited = original_travel > 0.001
    elif max_travel > 0.0 and original_travel > max_travel:
        ratio = max_travel / max(original_travel, 1e-6)
        limited_end["location"] = [
            round(start_location[index] + (end_location[index] - start_location[index]) * ratio, 6)
            for index in range(3)
        ]
        speed_limited = True
    limited_travel = _distance_between(start_location, limited_end.get("location"))
    return limited_end, {
        "preset_name": preset,
        "closeup": closeup,
        "max_travel": round(max_travel, 6),
        "original_travel_distance": round(original_travel, 6),
        "limited_travel_distance": round(limited_travel, 6),
        "speed_limited": speed_limited,
    }


def _vector3(values: Any, default: list[float] | None = None) -> list[float]:
    fallback = default or [0.0, 0.0, 0.0]
    try:
        if values is None:
            raise ValueError
        result = [float(value) for value in list(values)[:3]]
        if len(result) != 3 or not all(math.isfinite(value) for value in result):
            raise ValueError
        return result
    except Exception:
        return list(fallback)


def _distance_between(a: Any, b: Any) -> float:
    av = _vector3(a)
    bv = _vector3(b)
    return math.sqrt(sum((av[index] - bv[index]) ** 2 for index in range(3)))


def _subtract3(a: Any, b: Any) -> list[float]:
    av = _vector3(a)
    bv = _vector3(b)
    return [av[index] - bv[index] for index in range(3)]


def _length3(values: Any) -> float:
    vector = _vector3(values)
    return math.sqrt(sum(value * value for value in vector))


def _dot3(a: Any, b: Any) -> float:
    av = _vector3(a)
    bv = _vector3(b)
    return sum(av[index] * bv[index] for index in range(3))


def _unit3(values: Any) -> list[float]:
    vector = _vector3(values)
    length = _length3(vector)
    if length <= 1e-6:
        return [0.0, 0.0, 0.0]
    return [value / length for value in vector]


def _direction_alignment(a: Any, b: Any) -> float:
    av = _unit3(a)
    bv = _unit3(b)
    if _length3(av) <= 1e-6 or _length3(bv) <= 1e-6:
        return 0.0
    return max(-1.0, min(1.0, _dot3(av, bv)))


def candidate_visible_fraction(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    projection = candidate.get("projection") or {}
    for key in ("visible_fraction", "raw_visibility"):
        try:
            value = float(projection.get(key))
            if math.isfinite(value):
                return max(0.0, min(value, 1.0))
        except (TypeError, ValueError):
            pass
    scores = candidate.get("scores") or {}
    for key in ("raw_visibility", "visibility"):
        try:
            value = float(scores.get(key))
            if math.isfinite(value):
                return max(0.0, min(value, 1.0))
        except (TypeError, ValueError):
            pass
    return 0.0


def candidate_line_of_sight(candidate: dict[str, Any] | None) -> float:
    scores = (candidate or {}).get("scores") or {}
    try:
        value = float(scores.get("line_of_sight") or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(value, 1.0))


def candidate_readability_threshold(camera: dict[str, Any], candidate: dict[str, Any] | None) -> float:
    source = str((candidate or {}).get("source") or "").lower()
    try:
        explicit = float((candidate or {}).get("readability_threshold"))
        if math.isfinite(explicit) and explicit > 0.0:
            return explicit
    except (TypeError, ValueError):
        pass
    if source.startswith("semantic_face") or source.startswith("semantic_feet"):
        return 0.08
    return 0.02


def candidate_readability_grade(camera: dict[str, Any], candidate: dict[str, Any] | None) -> str:
    visible_fraction = candidate_visible_fraction(candidate)
    threshold = candidate_readability_threshold(camera, candidate)
    if visible_fraction <= 0.0:
        return "blocked"
    if visible_fraction < threshold:
        return "marginal"
    if visible_fraction < threshold * 1.8:
        return "readable"
    return "strong"


def candidate_rotation(candidate: dict[str, Any] | None) -> list[float]:
    return _rotation3((candidate or {}).get("rotation_euler"))


def candidate_viewpoint_similarity(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
    *,
    position_threshold: float = 0.18,
    lens_threshold: float = 10.0,
) -> bool:
    if not a or not b:
        return False
    if _distance_between((a.get("location") or []), (b.get("location") or [])) > position_threshold:
        return False
    try:
        lens_a = float(a.get("lens_mm") or 35.0)
        lens_b = float(b.get("lens_mm") or 35.0)
    except (TypeError, ValueError):
        lens_a = lens_b = 35.0
    return abs(lens_a - lens_b) <= lens_threshold


_BACK_VIEW_PHRASES = (
    "back view",
    "back-view",
    "back-facing",
    "back facing",
    "from behind",
    "behind him",
    "behind her",
    "over shoulder",
    "over-the-shoulder",
    "ots",
    "back of head",
    "rear view",
    "rear-view",
)


def _is_back_view_intent_camera(camera: dict[str, Any]) -> bool:
    """Decide whether the camera package legitimately asks for a back-of-head
    or over-the-shoulder framing. Used by the Phase-1 LLM filter so that
    back-facing candidates are only allowed when the shot contract intends
    them. Pure metadata sniff; never raises."""
    if not isinstance(camera, dict):
        return False
    semantic_target = str(
        (camera.get("semantic_contract") or {}).get("primary_semantic_target")
        or camera.get("primary_semantic_target")
        or ""
    ).strip().lower().replace(" ", "_")
    if semantic_target == "back_of_head":
        return True
    pieces: list[str] = []
    for key in (
        "shot_description",
        "scene_description",
        "camera_role",
        "coverage_role",
        "coverage_type",
        "movement_tag",
    ):
        pieces.append(str(camera.get(key) or ""))
    contract = camera.get("shot_contract") or {}
    if isinstance(contract, dict):
        try:
            pieces.append(json.dumps(contract, ensure_ascii=False))
        except (TypeError, ValueError):
            pass
    semantic_contract = camera.get("semantic_contract") or {}
    if isinstance(semantic_contract, dict):
        try:
            pieces.append(json.dumps(semantic_contract, ensure_ascii=False))
        except (TypeError, ValueError):
            pass
    text = " ".join(pieces).lower()
    return any(phrase in text for phrase in _BACK_VIEW_PHRASES)


def candidate_readability_reasons(camera: dict[str, Any], candidate: dict[str, Any] | None) -> list[str]:
    if not candidate:
        return ["missing_candidate"]
    projection = candidate.get("projection") or {}
    scores = candidate.get("scores") or {}
    reasons: list[str] = []
    visible_fraction = candidate_visible_fraction(candidate)
    threshold = candidate_readability_threshold(camera, candidate)
    if visible_fraction <= 0.0:
        reasons.append("visible_fraction_zero")
    if not bool(projection.get("valid", True)):
        reasons.append("projection_invalid")
    if float(projection.get("area_ratio") or 0.0) <= 0.0:
        reasons.append("projection_area_zero")
    preview_qc = candidate.get("preview_image_qc")
    if isinstance(preview_qc, dict) and str(preview_qc.get("status") or "") == "failed":
        reasons.append(str(preview_qc.get("reason") or "preview_qc_failed"))
    preview_path = str(candidate.get("preview_image_path") or "").strip()
    if not preview_path or not Path(preview_path).exists():
        reasons.append("preview_missing")
    source = str(candidate.get("source") or "").lower()
    direction = " ".join(
        str(value or "").lower()
        for value in (candidate.get("direction"), candidate.get("internal_direction"))
    )
    semantic_contract = camera.get("semantic_contract") or {}
    semantic_target = str(
        semantic_contract.get("primary_semantic_target")
        or camera.get("primary_semantic_target")
        or ""
    ).lower()
    must_show_face = bool(semantic_contract.get("must_show_face")) or semantic_target in {"face", "eyes", "head", "back_of_head"}
    if source.startswith("semantic_chest"):
        reasons.append("semantic_chest_disabled")
    if source.startswith("semantic_face"):
        if "back" in direction:
            reasons.append("face_target_back_facing")
        if visible_fraction < threshold:
            reasons.append("face_detail_unreadable")
    if source.startswith("semantic_feet"):
        if visible_fraction < threshold:
            reasons.append("semantic_detail_unreadable")
    if float(scores.get("primary_visible_fraction") or 0.0) < min(threshold, 0.02):
        reasons.append("primary_visible_below_floor")
    if float(scores.get("primary_area_ratio") or 0.0) <= 0.0:
        reasons.append("primary_area_zero")
    if must_show_face and "back" in direction and source.startswith("semantic"):
        reasons.append("semantic_face_unreadable")
    if not source.startswith("semantic") and visible_fraction < threshold:
        reasons.append("action_visibility_below_threshold")
    return sorted(set(reasons))


def candidate_selection_validity(camera: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    reasons = candidate_readability_reasons(camera, candidate)
    return {
        "valid": not reasons,
        "candidate_id": (candidate or {}).get("candidate_id"),
        "visible_fraction": round(candidate_visible_fraction(candidate), 6),
        "line_of_sight": round(candidate_line_of_sight(candidate), 6),
        "readability_threshold": round(candidate_readability_threshold(camera, candidate), 4),
        "readability_grade": candidate_readability_grade(camera, candidate),
        "reasons": reasons,
    }


PROTECTED_SEMANTIC_SELECTION_REASON_PREFIX = "deterministic_fallback_closeup_semantic_weighted"
PROTECTED_SEMANTIC_SOURCE_PREFIXES = ("semantic_face", "semantic_feet")


def candidate_key(candidate: dict[str, Any] | None) -> str:
    return str((candidate or {}).get("candidate_id") or "").strip()


def is_protected_semantic_seed_candidate(
    camera: dict[str, Any],
    candidate: dict[str, Any] | None,
    *,
    selection_reason: str = "",
) -> bool:
    if not candidate:
        return False
    reason = str(selection_reason or "").strip()
    if not reason.startswith(PROTECTED_SEMANTIC_SELECTION_REASON_PREFIX):
        return False
    if str(candidate.get("channel") or "").strip().lower() != "semantic":
        return False
    source = str(candidate.get("source") or "").strip().lower()
    if not source.startswith(PROTECTED_SEMANTIC_SOURCE_PREFIXES):
        return False
    return bool(candidate_selection_validity(camera, candidate).get("valid"))


def iter_row_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for candidate in (row.get("selected_candidate"), *(row.get("top_candidates") or [])):
        if isinstance(candidate, dict):
            candidates.append(candidate)
    for channel_info in (row.get("channel_boards") or {}).values():
        if not isinstance(channel_info, dict):
            continue
        candidates.extend(candidate for candidate in (channel_info.get("selected") or []) if isinstance(candidate, dict))
    return candidates


def protected_semantic_seed_candidate(camera: dict[str, Any], row: dict[str, Any] | None = None) -> dict[str, Any] | None:
    report = camera.get("quality_candidate_report") or {}
    reason = str(
        camera.get("protected_semantic_seed_selection_reason")
        or report.get("protected_semantic_seed_selection_reason")
        or report.get("selection_reason")
        or camera.get("selection_reason")
        or ""
    )
    stored = camera.get("protected_semantic_seed_candidate") or report.get("protected_semantic_seed_candidate")
    if is_protected_semantic_seed_candidate(camera, stored, selection_reason=reason):
        return stored
    current = camera.get("selected_candidate") or {}
    if is_protected_semantic_seed_candidate(camera, current, selection_reason=reason):
        return current
    if row:
        row_reason = str(row.get("selection_reason") or reason)
        selected = row.get("selected_candidate") or {}
        if is_protected_semantic_seed_candidate(camera, selected, selection_reason=row_reason):
            return selected
        protected_id = str(
            camera.get("protected_semantic_seed_candidate_id")
            or report.get("protected_semantic_seed_candidate_id")
            or candidate_key(selected)
        ).strip()
        if protected_id:
            for candidate in iter_row_candidates(row):
                if candidate_key(candidate) == protected_id and is_protected_semantic_seed_candidate(
                    camera,
                    candidate,
                    selection_reason=row_reason,
                ):
                    return candidate
    return None


def remember_protected_semantic_seed(camera: dict[str, Any], candidate: dict[str, Any], selection_reason: str) -> None:
    camera["protected_semantic_seed_candidate"] = candidate
    camera["protected_semantic_seed_candidate_id"] = candidate_key(candidate)
    camera["protected_semantic_seed_candidate_preview_path"] = str(candidate.get("preview_image_path") or "")
    camera["protected_semantic_seed_selection_reason"] = selection_reason
    report = camera.get("quality_candidate_report")
    if isinstance(report, dict):
        report["protected_semantic_seed_candidate_id"] = camera["protected_semantic_seed_candidate_id"]
        report["protected_semantic_seed_candidate_preview_path"] = camera["protected_semantic_seed_candidate_preview_path"]
        report["protected_semantic_seed_selection_reason"] = selection_reason


def prioritize_protected_candidate(
    candidates: list[dict[str, Any]],
    protected_candidate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    protected_id = candidate_key(protected_candidate)
    if not protected_id:
        return list(candidates)
    first = None
    rest: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate_key(candidate) == protected_id:
            first = candidate
        else:
            rest.append(candidate)
    if first is None and protected_candidate:
        first = protected_candidate
    return ([first] if first else []) + rest


def current_candidate_is_protected_semantic_seed(camera: dict[str, Any], row: dict[str, Any] | None = None) -> bool:
    protected = protected_semantic_seed_candidate(camera, row)
    return bool(protected and candidate_key(protected) == candidate_key(camera.get("selected_candidate") or {}))


def candidate_quality_rank(
    camera: dict[str, Any],
    candidate: dict[str, Any],
    *,
    previous_candidate: dict[str, Any] | None = None,
    prefer_non_semantic: bool = False,
) -> float:
    scores = candidate.get("scores") or {}
    score = float(scores.get("final") or 0.0)
    score += 0.35 * candidate_visible_fraction(candidate)
    score += 0.18 * candidate_line_of_sight(candidate)
    score += 0.08 * float(scores.get("upper_body_visibility") or 0.0)
    if candidate_readability_reasons(camera, candidate):
        score -= 1.0
    if prefer_non_semantic and str(candidate.get("channel") or "") != "semantic":
        score += 0.18
    if previous_candidate:
        if candidate_viewpoint_similarity(previous_candidate, candidate):
            score -= 0.22
        else:
            score += 0.08
    return score


def row_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [candidate for candidate in (row.get("top_candidates") or []) if candidate.get("preview_image_path")]


def is_closeup_camera(camera: dict[str, Any]) -> bool:
    semantic_contract = camera.get("semantic_contract") or {}
    semantic_target = str(
        semantic_contract.get("primary_semantic_target")
        or camera.get("primary_semantic_target")
        or ""
    ).strip().lower()
    semantic_detail_target = semantic_target in {"face", "eyes", "head", "back_of_head", "feet", "foot", "shoe", "shoes"}
    return bool(camera.get("closeup_required")) or is_closeup(str(camera.get("distance_label") or "")) or semantic_detail_target


def movement_requires_dynamic(camera: dict[str, Any]) -> bool:
    if is_closeup_camera(camera):
        return False
    movement = str(camera.get("movement_tag") or "").strip().lower()
    return movement in {"pan", "truck", "push_in", "push_out", "orbit", "pedestal"} or "follow" in movement


def trajectory_direction_hint(
    camera: dict[str, Any],
    start_candidate: dict[str, Any] | None,
    endpoint_candidate: dict[str, Any] | None,
) -> str:
    for candidate in (endpoint_candidate, start_candidate):
        text = " ".join(
            str(value or "").lower()
            for value in ((candidate or {}).get("direction"), (candidate or {}).get("internal_direction"))
        )
        if "left" in text:
            return "left"
        if "right" in text:
            return "right"
    start_loc = _vector3((start_candidate or {}).get("location"))
    end_loc = _vector3((endpoint_candidate or {}).get("location"), start_loc)
    if end_loc[0] < start_loc[0] - 0.02:
        return "left"
    if end_loc[0] > start_loc[0] + 0.02:
        return "right"
    return str(camera.get("direction_tag") or "right")


def derived_motion_end_transform(camera: dict[str, Any], start_transform: dict[str, Any]) -> dict[str, Any]:
    start_location = _vector3(start_transform.get("location"), [0.0, -3.0, 1.6])
    target = _vector3(start_transform.get("target"), [start_location[0], start_location[1] + 1.0, start_location[2]])
    direction = [target[index] - start_location[index] for index in range(3)]
    length = math.sqrt(sum(value * value for value in direction))
    if length <= 1e-6:
        direction = [0.0, 1.0, 0.0]
        length = 1.0
    direction = [value / length for value in direction]
    location = list(start_location)
    movement = str(camera.get("movement_tag") or "").strip().lower()
    distance = max(length, 1.0)
    if movement == "push_in":
        for index in range(3):
            location[index] += direction[index] * max(distance * 0.035, 0.06)
    elif movement == "push_out":
        for index in range(3):
            location[index] -= direction[index] * max(distance * 0.035, 0.06)
    elif movement in {"truck", "pan"}:
        right = [direction[1], -direction[0], 0.0]
        right_len = math.sqrt(sum(value * value for value in right))
        if right_len <= 1e-6:
            right = [1.0, 0.0, 0.0]
            right_len = 1.0
        right = [value / right_len for value in right]
        sign = -1.0 if trajectory_direction_hint(camera, None, None) == "left" else 1.0
        for index in range(3):
            location[index] += right[index] * sign * max(distance * 0.04, 0.08)
    elif movement == "orbit":
        rel_x = start_location[0] - target[0]
        rel_y = start_location[1] - target[1]
        angle = math.radians(-4.0 if trajectory_direction_hint(camera, None, None) == "left" else 4.0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        location[0] = target[0] + rel_x * cos_a - rel_y * sin_a
        location[1] = target[1] + rel_x * sin_a + rel_y * cos_a
    elif movement == "pedestal":
        location[2] += 0.08
    end_transform = dict(start_transform)
    end_transform["location"] = [round(value, 6) for value in location]
    end_transform["rotation_euler"] = candidate_rotation({"rotation_euler": start_transform.get("rotation_euler")})
    end_transform["lens_mm"] = float(start_transform.get("lens_mm") or camera.get("lens_mm") or 35.0)
    return end_transform


def trajectory_has_meaningful_motion(
    camera: dict[str, Any],
    preset_name: Any,
    start_transform: dict[str, Any],
    end_transform: dict[str, Any],
) -> bool:
    preset = normalize_trajectory_preset(preset_name)
    location_delta = _distance_between(start_transform.get("location"), end_transform.get("location"))
    start_rotation = candidate_rotation({"rotation_euler": start_transform.get("rotation_euler")})
    end_rotation = _align_rotation_near(start_rotation, end_transform.get("rotation_euler") or start_rotation)
    rotation_delta = sum(abs(end_rotation[index] - start_rotation[index]) for index in range(3))
    if preset in {"static_hold", "static_subtle_zoom"}:
        return not movement_requires_dynamic(camera)
    if preset in {"pan_left", "pan_right"}:
        return rotation_delta >= 0.03
    if preset in {"truck_left", "truck_right", "push_in_arc", "pull_out_arc", "straight_ease"}:
        return location_delta >= 0.05
    if preset in {"orbit_left_arc", "orbit_right_arc", "pedestal_up", "pedestal_down", "rise_reveal", "drop_reveal", "s_curve"}:
        return location_delta >= 0.05 or rotation_delta >= 0.04
    return location_delta >= 0.04 or rotation_delta >= 0.03


def choose_replacement_candidate(
    *,
    row: dict[str, Any],
    camera: dict[str, Any],
    current_candidate: dict[str, Any] | None = None,
    previous_candidate: dict[str, Any] | None = None,
    prefer_non_semantic: bool = False,
) -> dict[str, Any] | None:
    candidates = row_candidates(row)
    current_id = str((current_candidate or {}).get("candidate_id") or "")
    ranked = sorted(
        candidates,
        key=lambda candidate: candidate_quality_rank(
            camera,
            candidate,
            previous_candidate=previous_candidate,
            prefer_non_semantic=prefer_non_semantic,
        ),
        reverse=True,
    )
    for candidate in ranked:
        if current_id and str(candidate.get("candidate_id") or "") == current_id:
            continue
        if candidate_selection_validity(camera, candidate).get("valid"):
            return candidate
    return None


DIVERSITY_POSITION_THRESHOLD = 0.35
DIVERSITY_VIEW_ANGLE_THRESHOLD_DEGREES = 12.0
DIVERSITY_LENS_THRESHOLD_MM = 12.0
DIVERSITY_PREVIEW_DELTA_THRESHOLD = 0.055


def _unit_vector(values: list[float]) -> list[float] | None:
    length = math.sqrt(sum(float(value) * float(value) for value in values))
    if length <= 1e-6:
        return None
    return [float(value) / length for value in values]


def candidate_view_vector(candidate: dict[str, Any] | None) -> list[float] | None:
    source = candidate or {}
    location = _vector3(source.get("location"))
    target = source.get("target")
    if target is not None:
        target_vec = _vector3(target, location)
        return _unit_vector([target_vec[index] - location[index] for index in range(3)])
    rotation = candidate_rotation(source)
    pitch = float(rotation[0])
    yaw = float(rotation[2])
    return _unit_vector([
        math.sin(yaw) * math.cos(pitch),
        math.cos(yaw) * math.cos(pitch),
        -math.sin(pitch),
    ])


def vector_angle_degrees(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    dot = max(-1.0, min(1.0, sum(float(a[index]) * float(b[index]) for index in range(3))))
    return math.degrees(math.acos(dot))


def candidate_preview_delta(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    path_a = Path(str((a or {}).get("preview_image_path") or ""))
    path_b = Path(str((b or {}).get("preview_image_path") or ""))
    if not path_a.exists() or not path_b.exists():
        return None
    try:
        from PIL import Image, ImageChops, ImageStat

        image_a = Image.open(path_a).convert("L").resize((96, 54))
        image_b = Image.open(path_b).convert("L").resize((96, 54))
        diff = ImageChops.difference(image_a, image_b)
        return float(ImageStat.Stat(diff).mean[0]) / 255.0
    except Exception:
        return None


def same_shot_candidate_similarity(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    if not a or not b:
        return {
            "similar": False,
            "criteria_matches": [],
            "position_distance": None,
            "view_angle_degrees": None,
            "lens_delta_mm": None,
            "preview_delta": None,
        }
    position_distance = _distance_between(a.get("location"), b.get("location"))
    view_angle = vector_angle_degrees(candidate_view_vector(a), candidate_view_vector(b))
    try:
        lens_delta = abs(float(a.get("lens_mm") or 35.0) - float(b.get("lens_mm") or 35.0))
    except (TypeError, ValueError):
        lens_delta = 0.0
    preview_delta = candidate_preview_delta(a, b)
    matches: list[str] = []
    if position_distance < DIVERSITY_POSITION_THRESHOLD and view_angle is not None and view_angle < DIVERSITY_VIEW_ANGLE_THRESHOLD_DEGREES and lens_delta < DIVERSITY_LENS_THRESHOLD_MM:
        matches.append("position_and_angle_and_lens_below_threshold")
    if preview_delta is not None and preview_delta < DIVERSITY_PREVIEW_DELTA_THRESHOLD:
        matches.append("preview_delta_below_threshold")
    return {
        "similar": bool(matches),
        "criteria_matches": matches,
        "position_distance": round(position_distance, 6),
        "view_angle_degrees": round(view_angle, 6) if view_angle is not None else None,
        "lens_delta_mm": round(lens_delta, 6),
        "preview_delta": round(preview_delta, 6) if preview_delta is not None else None,
        "thresholds": {
            "position_distance": DIVERSITY_POSITION_THRESHOLD,
            "view_angle_degrees": DIVERSITY_VIEW_ANGLE_THRESHOLD_DEGREES,
            "lens_delta_mm": DIVERSITY_LENS_THRESHOLD_MM,
            "preview_delta": DIVERSITY_PREVIEW_DELTA_THRESHOLD,
        },
    }


def candidate_similar_to_any(candidate: dict[str, Any], previous_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    for previous in previous_candidates:
        review = same_shot_candidate_similarity(previous, candidate)
        comparisons.append(
            {
                "previous_candidate_id": previous.get("candidate_id"),
                **review,
            }
        )
    first_match = next((item for item in comparisons if item.get("similar")), None)
    return {
        "similar": first_match is not None,
        "first_match": first_match or {},
        "comparisons": comparisons,
    }


def diversity_role_bonus(candidate: dict[str, Any], camera_ordinal: int) -> float:
    source_text = " ".join(
        str(value or "").lower()
        for value in (
            candidate.get("source"),
            candidate.get("channel"),
            candidate.get("direction"),
            candidate.get("internal_direction"),
        )
    )
    bonus = 0.0
    if camera_ordinal == 1:
        if any(token in source_text for token in ("side", "over", "shoulder", "reaction", "medium", "direction", "preset")):
            bonus += 0.2
        if "semantic_face" in source_text:
            bonus -= 0.12
    else:
        if any(token in source_text for token in ("semantic_feet", "detail", "insert", "pov", "reaction", "side")):
            bonus += 0.22
        if "front_0deg" in source_text:
            bonus -= 0.08
    return bonus


def choose_same_shot_diverse_candidate(
    *,
    row: dict[str, Any],
    camera: dict[str, Any],
    current_candidate: dict[str, Any],
    previous_cameras: list[dict[str, Any]],
    camera_ordinal: int,
) -> dict[str, Any] | None:
    previous_candidates = [cam.get("selected_candidate") or {} for cam in previous_cameras if cam.get("selected_candidate")]
    current_id = str(current_candidate.get("candidate_id") or "")
    candidates = row_candidates(row)
    ranked = sorted(
        candidates,
        key=lambda candidate: candidate_quality_rank(camera, candidate, prefer_non_semantic=camera_ordinal == 1)
        + diversity_role_bonus(candidate, camera_ordinal),
        reverse=True,
    )
    for candidate in ranked:
        if current_id and str(candidate.get("candidate_id") or "") == current_id:
            continue
        if not candidate_selection_validity(camera, candidate).get("valid"):
            continue
        if candidate_similar_to_any(candidate, previous_candidates).get("similar"):
            continue
        return candidate
    return None


def choose_replacement_endpoint(
    *,
    row: dict[str, Any],
    camera: dict[str, Any],
    start_candidate: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = safe_endpoint_candidates(row, start_candidate)
    start_id = str(start_candidate.get("candidate_id") or "")
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate_quality_rank(camera, candidate, previous_candidate=start_candidate),
            _distance_between(candidate.get("location"), start_candidate.get("location")),
        ),
        reverse=True,
    )
    for candidate in ranked:
        if str(candidate.get("candidate_id") or "") == start_id:
            continue
        if candidate_selection_validity(camera, candidate).get("valid") and not candidate_viewpoint_similarity(start_candidate, candidate):
            return candidate
    return None


def repeated_static_view(previous_camera: dict[str, Any] | None, camera: dict[str, Any]) -> bool:
    if not previous_camera:
        return False
    if int(previous_camera.get("scene_id") or 0) != int(camera.get("scene_id") or 0):
        return False
    if int(previous_camera.get("shot_id") or 0) == int(camera.get("shot_id") or 0):
        return False
    previous_preset = normalize_trajectory_preset((previous_camera.get("trajectory_plan") or {}).get("preset_name"))
    current_preset = normalize_trajectory_preset((camera.get("trajectory_plan") or {}).get("preset_name"))
    if previous_preset not in {"static_hold", "static_subtle_zoom"} or current_preset not in {"static_hold", "static_subtle_zoom"}:
        return False
    return candidate_viewpoint_similarity(previous_camera.get("selected_candidate") or {}, camera.get("selected_candidate") or {})


def camera_requires_hands(camera: dict[str, Any]) -> bool:
    return False


def shot_requires_interaction(camera: dict[str, Any]) -> bool:
    contract = start_frame_contract(camera)
    if bool(contract.get("must_show_interaction")):
        return True
    description = str(camera.get("shot_description") or "").strip().lower()
    interaction_phrases = (
        "embrace",
        "whisper",
        "whispers",
        "hands ",
        "gives ",
        "offers ",
        "passes ",
        "shakes hands",
        "leans into",
        "leans toward",
        "stands directly in front of",
        "hug",
        "kisses",
        "touches ",
    )
    return any(phrase in description for phrase in interaction_phrases)


def camera_requires_secondary_subject(camera: dict[str, Any]) -> bool:
    contract = start_frame_contract(camera)
    secondary_ids = contract.get("secondary_focus_ids") or camera.get("secondary_focus_ids") or []
    return bool(secondary_ids) or shot_requires_interaction(camera)


def deterministic_quality_gate(camera: dict[str, Any]) -> dict[str, Any]:
    report = camera.get("quality_candidate_report") or {}
    if not isinstance(report, dict) or not report:
        return {"valid": True, "reasons": []}
    reasons: list[str] = []
    if report.get("success") is False:
        reasons.append("quality_report_failed")
    eligible_count = int(report.get("candidate_count_eligible") or 0)
    retained_count = int(report.get("candidate_count_retained") or 0)
    if eligible_count <= 0:
        reasons.append("quality_no_eligible_candidate")
    if retained_count <= 0:
        reasons.append("quality_no_retained_candidate")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "candidate_count_eligible": eligible_count,
        "candidate_count_retained": retained_count,
        "quality_success": bool(report.get("success")),
    }


def quality_gate_vetoed(validation: dict[str, Any]) -> bool:
    return str(validation.get("error_class") or "") == "deterministic_quality_gate"


SEVERE_CAMERA_TERMS = (
    "out of frame",
    "not in frame",
    "completely cropped",
    "severely cropped",
    "completely cut off",
    "wrong subject",
    "primary subject is missing",
    "primary subject missing",
    "primary subject not visible",
    "primary subject not in",
    "primary subject completely",
    "drastically wrong shot",
    "drastically wrong angle",
    "composition completely off",
    "unusable",
)

FRAMING_ONLY_SOFT_TERMS = (
    "shot size",
    "medium shot",
    "long shot",
    "wide shot",
    "full shot",
    "eye-level",
    "eye level",
    "high-angle",
    "high angle",
    "low-angle",
    "low angle",
    "too far",
    "too close",
    "camera angle",
)

PRIMARY_VISIBILITY_FAILURE_TERMS = (
    "out of frame",
    "not in frame",
    "not visible",
    "missing",
    "wrong subject",
    "severely cropped",
    "completely cropped",
    "cut off",
    "occluded",
    "unreadable",
    "cannot see",
    "can't see",
)


def hard_block_is_subject_readable_framing_only(review: dict[str, Any]) -> bool:
    """Downgrade distance/angle-only complaints when the primary subject is visible."""
    if bool_or_none(review.get("primary_subject_visible")) is not True:
        return False
    hard_block = str(review.get("hard_block_camera_reason") or "").lower()
    camera_issue = str(review.get("camera_issue") or "").lower()
    reason = str(review.get("reason") or "").lower()
    review_text = " ".join([hard_block, camera_issue, reason]).strip()
    if not review_text:
        return False
    if not any(term in review_text for term in FRAMING_ONLY_SOFT_TERMS):
        return False
    hard_block_text = hard_block or camera_issue
    if any(term in hard_block_text for term in PRIMARY_VISIBILITY_FAILURE_TERMS):
        return False
    return True

CAMERA_TERMS = (
    "camera",
    "framing",
    "frame",
    "shot size",
    "close-up",
    "medium shot",
    "wide shot",
    "long shot",
    "full shot",
    "angle",
    "eye-level",
    "high-angle",
    "low-angle",
    "top-down",
    "overhead",
    "crop",
    "cropped",
    "cut off",
    "occlusion",
    "occluded",
    "too far",
    "too close",
    "composition",
    "viewpoint",
    "behind the subject",
    "behind him",
    "behind her",
    "facing away",
    "back of",
    "pan",
    "tracking",
    "orbit",
)

NON_CAMERA_TERMS = (
    "environment",
    "set dressing",
    "background",
    "garden",
    "prop",
    "props",
    "lighting",
    "action",
    "acting",
    "animation",
    "pose",
    "gesture",
    "facial expression",
    "expression",
    "emotion",
    "interaction",
    "hands",
    "semantic",
    "story",
    "narrative",
    "character is not",
    "not performing",
    "does not match",
    "missing from the scene",
    "missing object",
    "idle pose",
    "neutral standing",
    "t-pose",
    "standing still",
    "model quality",
    "render quality",
    "workbench",
    "low resolution",
    "asset",
    "costume",
    "clothing",
    "hair",
    "material",
    "texture",
)


def classify_review_reason(review: dict[str, Any]) -> dict[str, Any]:
    """Classify a review into camera_controllable / non_camera / mixed / unknown.

    Returns dict with:
        category: 'camera_controllable' | 'non_camera' | 'mixed' | 'unknown'
        severity: 'severe_camera' | 'minor_camera' | 'non_camera_only' | 'unknown'
        is_hard_block: bool - True only for severe camera issues
        camera_keywords_found: list[str]
        non_camera_keywords_found: list[str]
    """
    reason = str(review.get("reason") or "").lower()
    camera_issue = str(review.get("camera_issue") or "").lower()
    non_camera_issue = str(review.get("non_camera_issue") or "").lower()
    hard_block = str(review.get("hard_block_camera_reason") or "").lower()

    # If the LLM explicitly provided structured fields, prefer them
    if hard_block and hard_block not in ("null", "none", ""):
        if hard_block_is_subject_readable_framing_only(review):
            return {
                "category": "camera_controllable",
                "severity": "minor_camera",
                "is_hard_block": False,
                "hard_block_camera_reason": None,
                "camera_keywords_found": [term for term in CAMERA_TERMS if term in " ".join([reason, camera_issue, hard_block])],
                "non_camera_keywords_found": [term for term in NON_CAMERA_TERMS if term in non_camera_issue],
            }
        return {
            "category": "camera_controllable",
            "severity": "severe_camera",
            "is_hard_block": True,
            "hard_block_camera_reason": review.get("hard_block_camera_reason"),
            "camera_keywords_found": [],
            "non_camera_keywords_found": [],
        }

    # Combine reason + camera_issue for keyword scanning
    all_text = " ".join([reason, camera_issue, non_camera_issue])
    if not all_text.strip():
        return {
            "category": "unknown",
            "severity": "unknown",
            "is_hard_block": False,
            "hard_block_camera_reason": None,
            "camera_keywords_found": [],
            "non_camera_keywords_found": [],
        }

    camera_kw_found = [term for term in CAMERA_TERMS if term in all_text]
    non_camera_kw_found = [term for term in NON_CAMERA_TERMS if term in all_text]
    severe_kw_found = [term for term in SEVERE_CAMERA_TERMS if term in all_text]

    has_camera = bool(camera_kw_found)
    has_non_camera = bool(non_camera_kw_found)
    has_severe = bool(severe_kw_found)

    # If LLM explicitly set camera_issue but no hard_block_camera_reason, it's minor
    if camera_issue and camera_issue not in ("null", "none", "") and not has_severe:
        if non_camera_issue and non_camera_issue not in ("null", "none", ""):
            return {
                "category": "mixed",
                "severity": "minor_camera",
                "is_hard_block": False,
                "hard_block_camera_reason": None,
                "camera_keywords_found": camera_kw_found,
                "non_camera_keywords_found": non_camera_kw_found,
            }
        return {
            "category": "camera_controllable",
            "severity": "minor_camera",
            "is_hard_block": False,
            "hard_block_camera_reason": None,
            "camera_keywords_found": camera_kw_found,
            "non_camera_keywords_found": non_camera_kw_found,
        }

    # Keyword-based classification (fallback for old-format reviews without new fields)
    if has_severe and not has_non_camera:
        return {
            "category": "camera_controllable",
            "severity": "severe_camera",
            "is_hard_block": True,
            "hard_block_camera_reason": "; ".join(severe_kw_found),
            "camera_keywords_found": camera_kw_found,
            "non_camera_keywords_found": non_camera_kw_found,
        }

    if has_severe and has_non_camera:
        # Severe camera + non-camera: still hard block for the camera part
        return {
            "category": "mixed",
            "severity": "severe_camera",
            "is_hard_block": True,
            "hard_block_camera_reason": "; ".join(severe_kw_found),
            "camera_keywords_found": camera_kw_found,
            "non_camera_keywords_found": non_camera_kw_found,
        }

    if has_camera and not has_non_camera:
        # Camera-only keywords, but not severe → minor camera
        return {
            "category": "camera_controllable",
            "severity": "minor_camera",
            "is_hard_block": False,
            "hard_block_camera_reason": None,
            "camera_keywords_found": camera_kw_found,
            "non_camera_keywords_found": non_camera_kw_found,
        }

    if has_non_camera and not has_camera:
        return {
            "category": "non_camera",
            "severity": "non_camera_only",
            "is_hard_block": False,
            "hard_block_camera_reason": None,
            "camera_keywords_found": camera_kw_found,
            "non_camera_keywords_found": non_camera_kw_found,
        }

    if has_camera and has_non_camera:
        # Mixed but no severe camera keywords → soft warning
        return {
            "category": "mixed",
            "severity": "minor_camera",
            "is_hard_block": False,
            "hard_block_camera_reason": None,
            "camera_keywords_found": camera_kw_found,
            "non_camera_keywords_found": non_camera_kw_found,
        }

    return {
        "category": "unknown",
        "severity": "unknown",
        "is_hard_block": False,
        "hard_block_camera_reason": None,
        "camera_keywords_found": camera_kw_found,
        "non_camera_keywords_found": non_camera_kw_found,
    }


def camera_blocking_review_reason(review: dict[str, Any]) -> bool:
    """Legacy wrapper: returns True only if the review has a severe camera-controllable hard block."""
    classification = classify_review_reason(review)
    return classification.get("is_hard_block", False)


HANDOFF_POLICY_DEFAULT = "camera_only"
HANDOFF_POLICY_SEMANTIC_ONLY = "semantic_only"
VALID_HANDOFF_POLICIES = (HANDOFF_POLICY_DEFAULT, HANDOFF_POLICY_SEMANTIC_ONLY)


def final_preview_review_validity(
    camera: dict[str, Any],
    review: dict[str, Any],
    *,
    handoff_policy: str = HANDOFF_POLICY_DEFAULT,
) -> dict[str, Any]:
    # ---- helpers ----
    _semantic = handoff_policy == HANDOFF_POLICY_SEMANTIC_ONLY

    # ---- deterministic quality gate ----
    quality_gate = deterministic_quality_gate(camera)
    if not quality_gate.get("valid"):
        # In semantic_only mode, quality gate is only a hard block when no
        # preview/render exists at all (file missing).  If we have a preview
        # file on disk we let it through with a warning.
        preview_exists = bool(
            (camera.get("preview_frame_path") and Path(str(camera["preview_frame_path"])).exists())
            or (camera.get("final_preview_path") and Path(str(camera["final_preview_path"])).exists())
        )
        if _semantic and preview_exists:
            # downgrade to warning
            pass  # fall through to LLM review below
        else:
            return {
                "valid": False,
                "verdict": "blocked_quality_gate",
                "reasons": ["final_preview_deterministic_veto", *[str(reason) for reason in (quality_gate.get("reasons") or [])]],
                "warnings": [],
                "error_class": "deterministic_quality_gate",
                "handoff_policy": handoff_policy,
                "review_source": review.get("review_source") or "",
                "preview_path": review.get("preview_path") or "",
                "deterministic_quality_gate": quality_gate,
            }

    # ---- review not available ----
    reasons: list[str] = []
    warnings: list[str] = []

    if not quality_gate.get("valid"):
        # quality gate failed but we are in semantic_only with a preview → warning
        warnings.append("quality_gate_downgraded_warning")

    if not review.get("success"):
        error = str(review.get("error") or "final_preview_review_failed")
        if error == "llm_unavailable_for_story_consistency":
            if _semantic:
                # In semantic_only we pass when LLM is unavailable – we have a preview.
                return {
                    "valid": True,
                    "verdict": "passed_with_warning",
                    "reasons": [],
                    "warnings": ["llm_unavailable_passthrough"],
                    "error_class": "llm_unavailable_geometry_fallback",
                    "handoff_policy": handoff_policy,
                    "review_source": review.get("review_source") or "",
                    "preview_path": review.get("preview_path") or "",
                }
            deterministic = candidate_selection_validity(camera, camera.get("selected_candidate") or {})
            fallback_reasons = ["llm_unavailable_geometry_fallback", *[str(reason) for reason in (deterministic.get("reasons") or [])]]
            is_valid = bool(deterministic.get("valid"))
            return {
                "valid": is_valid,
                "verdict": "passed_clean" if is_valid else "hard_blocked_camera_issue",
                "reasons": fallback_reasons,
                "warnings": [],
                "error_class": "llm_unavailable_geometry_fallback",
                "handoff_policy": handoff_policy,
                "review_source": review.get("review_source") or "",
                "preview_path": review.get("preview_path") or "",
            }
        error_class = "llm_transport_error" if "llm_transport_error" in error else "llm_review_error"
        if _semantic:
            # transport/parse error but we want lenient – pass with warning
            return {
                "valid": True,
                "verdict": "passed_with_warning",
                "reasons": [],
                "warnings": [f"llm_error_downgraded:{error}"],
                "error_class": error_class,
                "handoff_policy": handoff_policy,
                "review_source": review.get("review_source") or "",
                "preview_path": review.get("preview_path") or "",
            }
        reasons.append(error)
        return {
            "valid": False,
            "verdict": "hard_blocked_camera_issue",
            "reasons": reasons,
            "warnings": [],
            "error_class": error_class,
            "handoff_policy": handoff_policy,
            "review_source": review.get("review_source") or "",
            "preview_path": review.get("preview_path") or "",
        }

    # ---- extract LLM fields ----
    primary_subject_visible = bool_or_none(review.get("primary_subject_visible"))
    framing_matches_intent = bool_or_none(review.get("framing_matches_intent"))
    needs_reshoot = bool_or_none(review.get("needs_reshoot"))
    hands_visible = bool_or_none(review.get("hands_visible"))
    interaction_readable = bool_or_none(review.get("interaction_readable"))
    secondary_subject_visible = bool_or_none(review.get("secondary_subject_visible"))
    classification = classify_review_reason(review)
    is_hard_block = classification.get("is_hard_block", False)

    # ================================================================
    #  SEMANTIC_ONLY policy — only block when primary subject absent
    # ================================================================
    if _semantic:
        # Hard block ONLY when LLM explicitly says primary subject is NOT visible
        if primary_subject_visible is False:
            reasons.append("primary_subject_not_visible")

        # Everything else → warning, never hard block
        if framing_matches_intent is not True:
            warnings.append("framing_mismatch_warning")
        if needs_reshoot is True:
            warnings.append("needs_reshoot_warning")
        if review.get("hard_block_camera_reason"):
            warnings.append("camera_issue_downgraded_warning")
        if review.get("camera_issue"):
            warnings.append("camera_issue_warning")
        if review.get("non_camera_issue"):
            warnings.append("non_camera_issue_warning")
        if secondary_subject_visible is False:
            warnings.append("secondary_subject_visible_false")
        if interaction_readable is False:
            warnings.append("interaction_readable_false")
        if hands_visible is False:
            warnings.append("hands_visible_false")

        has_hard_block = bool(reasons)
        if has_hard_block:
            verdict = "hard_blocked_subject_missing"
        elif warnings:
            verdict = "passed_with_warning"
        else:
            verdict = "passed_clean"

        return {
            "valid": not has_hard_block,
            "verdict": verdict,
            "reasons": reasons,
            "warnings": warnings,
            "scope": "semantic_only",
            "handoff_policy": handoff_policy,
            "classification": classification,
            "review_source": review.get("review_source") or "",
            "preview_path": review.get("preview_path") or "",
            "primary_subject_visible": primary_subject_visible,
            "secondary_subject_visible": secondary_subject_visible,
            "hands_visible": hands_visible,
            "interaction_readable": interaction_readable,
            "framing_matches_intent": framing_matches_intent,
            "needs_reshoot": needs_reshoot,
            "consistency_score": float(review.get("consistency_score") or 0.0),
            "camera_issue": review.get("camera_issue"),
            "non_camera_issue": review.get("non_camera_issue"),
            "hard_block_camera_reason": review.get("hard_block_camera_reason"),
        }

    # ================================================================
    #  CAMERA_ONLY policy (default, previous behaviour)
    # ================================================================
    # Primary subject not visible → hard block (always camera-controllable)
    if primary_subject_visible is False:
        reasons.append("primary_subject_visible_false")

    # Structured hard_block_camera_reason from LLM: hard block only after
    # classifier confirms the issue affects primary-subject usability.
    if review.get("hard_block_camera_reason"):
        if is_hard_block:
            reasons.append("hard_block_camera_reason")
        else:
            warnings.append("hard_block_camera_reason_downgraded_warning")

    # framing_matches_intent=false: only hard block if classifier confirms severe camera issue
    if framing_matches_intent is not True:
        if is_hard_block:
            reasons.append("framing_matches_intent_false")
        else:
            # Non-camera or minor camera issue → soft warning, does not block
            warnings.append("framing_matches_intent_false_soft_warning")

    # needs_reshoot → only hard block if classifier confirms severe camera AND framing is also bad
    if needs_reshoot is True and not reasons:
        if is_hard_block and framing_matches_intent is not True:
            reasons.append("needs_reshoot_camera_hard_block")
        else:
            warnings.append("needs_reshoot_non_camera_warning")

    # Non-camera diagnostic fields → always warnings
    if secondary_subject_visible is False:
        warnings.append("secondary_subject_visible_false")
    if interaction_readable is False:
        warnings.append("interaction_readable_false")
    if hands_visible is False:
        warnings.append("hands_visible_false")

    # Non-camera issue from LLM → warning
    if review.get("non_camera_issue"):
        warnings.append("non_camera_issue_reported")

    # Camera issue from LLM but not severe → warning
    if review.get("camera_issue") and not is_hard_block:
        warnings.append("minor_camera_issue_warning")

    has_hard_block = bool(reasons)
    if has_hard_block:
        verdict = "hard_blocked_camera_issue"
    elif warnings:
        verdict = "passed_with_warning"
    else:
        verdict = "passed_clean"

    return {
        "valid": not has_hard_block,
        "verdict": verdict,
        "reasons": reasons,
        "warnings": warnings,
        "scope": "camera_only",
        "handoff_policy": handoff_policy,
        "classification": classification,
        "ignored_non_camera_fields": {
            "needs_reshoot": needs_reshoot,
            "hands_visible": hands_visible,
            "secondary_subject_visible": secondary_subject_visible,
            "interaction_readable": interaction_readable,
        },
        "review_source": review.get("review_source") or "",
        "preview_path": review.get("preview_path") or "",
        "primary_subject_visible": primary_subject_visible,
        "secondary_subject_visible": secondary_subject_visible,
        "hands_visible": hands_visible,
        "interaction_readable": interaction_readable,
        "framing_matches_intent": framing_matches_intent,
        "needs_reshoot": needs_reshoot,
        "consistency_score": float(review.get("consistency_score") or 0.0),
        "camera_issue": review.get("camera_issue"),
        "non_camera_issue": review.get("non_camera_issue"),
        "hard_block_camera_reason": review.get("hard_block_camera_reason"),
    }


def choose_final_preview_replacement_candidate(
    *,
    row: dict[str, Any],
    camera: dict[str, Any],
    current_candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if current_candidate_is_protected_semantic_seed(camera, row):
        return None
    dynamic_face = choose_dynamic_face_closeup_candidate(row=row, camera=camera, current_candidate=current_candidate)
    if dynamic_face is not None:
        return dynamic_face
    candidates = row_candidates(row)
    current_id = str((current_candidate or {}).get("candidate_id") or "")
    ranked = sorted(
        candidates,
        key=lambda candidate: candidate_quality_rank(
            camera,
            candidate,
            previous_candidate=current_candidate,
            prefer_non_semantic=False,
        ),
        reverse=True,
    )
    for candidate in ranked:
        if current_id and str(candidate.get("candidate_id") or "") == current_id:
            continue
        if current_candidate and candidate_viewpoint_similarity(current_candidate, candidate):
            continue
        if candidate_selection_validity(camera, candidate).get("valid"):
            return candidate
    return None


def choose_dynamic_face_closeup_candidate(
    *,
    row: dict[str, Any],
    camera: dict[str, Any],
    current_candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not closeup_motion_context(camera):
        return None
    current_source = str((current_candidate or {}).get("source") or "")
    if current_source.startswith("semantic_face_s3_extreme_macro_dynamic"):
        return None
    candidates = [
        candidate
        for candidate in row_candidates(row)
        if str(candidate.get("source") or "").startswith("semantic_face_s3_extreme_macro_dynamic")
        and candidate_selection_validity(camera, candidate).get("valid")
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda candidate: candidate_quality_rank(
            camera,
            candidate,
            previous_candidate=current_candidate,
            prefer_non_semantic=False,
        ),
        reverse=True,
    )
    return candidates[0]


def same_shot_handoff_preset(camera: dict[str, Any]) -> str:
    trajectory_plan = camera.get("trajectory_plan") or {}
    preset = normalize_trajectory_preset(
        trajectory_plan.get("preset_name")
        or camera.get("trajectory_preset")
        or preset_for_motion(camera.get("movement_tag"), camera.get("direction_tag"), camera.get("motion_profile"))
    )
    if preset in {"static_hold", "static_subtle_zoom", "pan_left", "pan_right"}:
        return "straight_ease"
    return preset


def apply_same_shot_handoff_motion(
    *,
    camera: dict[str, Any],
    next_camera: dict[str, Any],
    fps: int,
) -> dict[str, Any]:
    camera_name = str(camera.get("camera_name") or "")
    next_camera_name = str(next_camera.get("camera_name") or "")
    start_transform = dict(camera.get("start_transform") or {})
    end_transform = dict(camera.get("end_transform") or start_transform)
    next_start_transform = dict(next_camera.get("start_transform") or {})
    start_location = _vector3(start_transform.get("location"), [0.0, -3.0, 1.6])
    current_end_location = _vector3(end_transform.get("location"), start_location)
    next_start_location = _vector3(next_start_transform.get("location"), current_end_location)
    vector_to_next = _subtract3(next_start_location, start_location)
    distance_to_next = _length3(vector_to_next)
    result = {
        "camera_name": camera_name,
        "to_camera_name": next_camera_name,
        "status": "skipped",
        "distance_to_next_start": round(distance_to_next, 6),
    }
    if distance_to_next <= 1e-4:
        result["reason"] = "next_start_too_close"
        camera["same_shot_handoff_motion"] = result
        persist_camera_package(camera)
        return result

    existing_motion = _subtract3(current_end_location, start_location)
    existing_travel = _length3(existing_motion)
    existing_alignment = _direction_alignment(existing_motion, vector_to_next)
    closeup = closeup_motion_context(camera)
    minimum_travel = 0.06 if closeup else 0.12
    max_travel = 0.25 if closeup else 0.75
    travel_ratio = 0.12 if closeup else 0.22
    desired_travel = distance_to_next * travel_ratio
    desired_travel = max(minimum_travel, desired_travel)
    desired_travel = min(desired_travel, max_travel, distance_to_next * 0.75)
    if desired_travel <= 1e-4:
        result["reason"] = "desired_motion_too_small"
        camera["same_shot_handoff_motion"] = result
        persist_camera_package(camera)
        return result

    aligned_enough = existing_travel >= minimum_travel and existing_alignment >= 0.35
    result.update(
        {
            "existing_travel_distance": round(existing_travel, 6),
            "existing_alignment_to_next": round(existing_alignment, 6),
            "closeup_motion_context": closeup,
            "minimum_travel": round(minimum_travel, 6),
            "max_guided_travel": round(max_travel, 6),
            "travel_ratio": round(travel_ratio, 6),
            "desired_travel_distance": round(desired_travel, 6),
        }
    )
    if aligned_enough:
        result["status"] = "aligned_existing_motion"
        camera["same_shot_handoff_motion"] = result
        persist_camera_package(camera)
        return result

    direction = _unit3(vector_to_next)
    guided_end_location = [
        round(start_location[index] + direction[index] * desired_travel, 6)
        for index in range(3)
    ]
    guided_end_transform = dict(end_transform)
    guided_end_transform["location"] = guided_end_location
    if "lens_mm" not in guided_end_transform:
        guided_end_transform["lens_mm"] = float(start_transform.get("lens_mm") or camera.get("lens_mm") or 35.0)
    camera["end_transform"] = guided_end_transform
    preset = same_shot_handoff_preset(camera)
    previous_source = str(camera.get("trajectory_selection_source") or camera.get("selection_source") or "selection")
    refresh_camera_trajectory(
        camera,
        preset_name=preset,
        start_candidate=camera.get("selected_candidate") or {},
        endpoint_candidate=None,
        selection_source=f"{previous_source}|same_shot_handoff",
        reason=f"same_shot_handoff_toward_{next_camera_name}",
        fps=fps,
    )
    final_motion = _subtract3((camera.get("end_transform") or {}).get("location"), start_location)
    final_alignment = _direction_alignment(final_motion, vector_to_next)
    result.update(
        {
            "status": "adjusted",
            "preset_name": preset,
            "guided_end_location": guided_end_location,
            "final_end_location": _vector3((camera.get("end_transform") or {}).get("location"), guided_end_location),
            "final_travel_distance": round(_length3(final_motion), 6),
            "final_alignment_to_next": round(final_alignment, 6),
        }
    )
    camera["same_shot_handoff_motion"] = result
    persist_camera_package(camera)
    return result


def repair_same_shot_handoff_motion(
    shot_outputs: list[dict[str, Any]],
    *,
    fps: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for shot in shot_outputs:
        eligible_cameras = [
            camera
            for camera in (shot.get("cameras") or [])
            if bool(camera.get("downstream_eligible", True))
        ]
        for index in range(len(eligible_cameras) - 1):
            rows.append(
                apply_same_shot_handoff_motion(
                    camera=eligible_cameras[index],
                    next_camera=eligible_cameras[index + 1],
                    fps=fps,
                )
            )
    adjusted = [row for row in rows if row.get("status") == "adjusted"]
    aligned = [row for row in rows if row.get("status") == "aligned_existing_motion"]
    skipped = [row for row in rows if row.get("status") == "skipped"]
    return {
        "schema_version": "storyblender.same_shot_handoff_motion.v1",
        "pair_count": len(rows),
        "adjusted_count": len(adjusted),
        "aligned_existing_count": len(aligned),
        "skipped_count": len(skipped),
        "rows": rows,
    }


def refresh_camera_row_metadata(camera_rows: list[dict[str, Any]], shot_outputs: list[dict[str, Any]]) -> None:
    metadata_by_camera: dict[str, dict[str, Any]] = {}
    for shot in shot_outputs:
        for camera in shot.get("cameras") or []:
            metadata_by_camera[str(camera.get("camera_name") or "")] = {
                "preview_frame_path": camera.get("preview_frame_path") or "",
                "final_preview_path": camera.get("final_preview_path") or camera.get("preview_frame_path") or "",
                "downstream_eligible": bool(camera.get("downstream_eligible", True)),
                "selection_source": camera.get("selection_source") or "",
                "same_shot_diversity_status": camera.get("same_shot_diversity_status") or "",
                "editor_recommended_omit": bool(camera.get("editor_recommended_omit")),
            }
    for row in camera_rows:
        name = str(row.get("camera_name") or "")
        metadata = metadata_by_camera.get(name)
        if not metadata:
            continue
        row.update(metadata)


def build_downstream_shots(shot_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    downstream_shots: list[dict[str, Any]] = []
    for shot in shot_outputs:
        eligible_cameras = [dict(camera) for camera in (shot.get("cameras") or []) if bool(camera.get("downstream_eligible", True))]
        if not eligible_cameras:
            continue
        bridges = [build_bridge(eligible_cameras[index], eligible_cameras[index + 1]) for index in range(len(eligible_cameras) - 1)]
        for index, camera in enumerate(eligible_cameras):
            camera["bridge_to_next"] = bridges[index] if index < len(bridges) else None
        downstream_shot = dict(shot)
        downstream_shot["cameras"] = eligible_cameras
        downstream_shot["bridge_trajectories"] = bridges
        downstream_shots.append(downstream_shot)
    return downstream_shots


def apply_validated_camera_selection(
    *,
    camera: dict[str, Any],
    start_candidate: dict[str, Any],
    endpoint_candidate: dict[str, Any] | None,
    trajectory_choice: Any,
    selection_source: str,
    selection_reason: Any,
    fps: int,
) -> None:
    start_transform = candidate_transform(start_candidate, camera.get("start_transform") or {})
    if start_candidate.get("lens_mm"):
        camera["lens_mm"] = float(start_candidate.get("lens_mm"))
    preset = normalize_trajectory_preset(trajectory_choice)
    start_lens = round(_lens_mm(start_transform.get("lens_mm"), _lens_mm(camera.get("lens_mm"), 35.0)), 4)
    start_transform["lens_mm"] = start_lens
    if preset in {"static_hold", "static_subtle_zoom"} and not movement_requires_dynamic(camera):
        endpoint_candidate = start_candidate
    if endpoint_candidate:
        end_transform = candidate_transform(endpoint_candidate, camera.get("end_transform") or start_transform)
        if endpoint_candidate.get("lens_mm") and _trajectory_allows_lens_ramp(preset):
            end_transform["lens_mm"] = float(endpoint_candidate.get("lens_mm"))
    elif movement_requires_dynamic(camera):
        end_transform = derived_motion_end_transform(camera, start_transform)
    else:
        end_transform = dict(start_transform)
    if not _trajectory_allows_lens_ramp(preset):
        end_transform = dict(end_transform)
        end_transform["lens_mm"] = start_lens
    camera["selected_candidate"] = start_candidate
    camera["start_transform"] = start_transform
    camera["end_transform"] = end_transform
    refresh_camera_trajectory(
        camera,
        preset_name=preset,
        start_candidate=start_candidate,
        endpoint_candidate=endpoint_candidate,
        selection_source=selection_source,
        reason=selection_reason,
        fps=fps,
    )


def candidate_transform(candidate: dict[str, Any] | None, fallback: dict[str, Any], *, keep_framing: bool = True) -> dict[str, Any]:
    source = candidate or {}
    transform = {
        "location": _vector3(source.get("location"), _vector3(fallback.get("location"), [0.0, -3.0, 1.6])),
        "rotation_euler": _vector3(source.get("rotation_euler"), _vector3(fallback.get("rotation_euler"), [0.0, 0.0, 0.0])),
        "framing": dict(fallback.get("framing") or {}) if keep_framing else {},
        "lens_mm": float(source.get("lens_mm") or fallback.get("lens_mm") or 35.0),
    }
    if source.get("target"):
        transform["target"] = _vector3(source.get("target"))
    return transform


def video_keyframes_from_camera_keyframes(keyframes: list[dict[str, Any]], frame_count: int) -> list[dict[str, Any]]:
    video_keyframes: list[dict[str, Any]] = []
    inherited_lens = 35.0
    if keyframes:
        inherited_lens = _lens_mm((keyframes[0].get("transform") or {}).get("lens_mm"), 35.0)
    for keyframe in keyframes:
        transform = keyframe.get("transform") or {}
        frame_number = int(keyframe.get("frame") or 0) + 1
        frame_number = max(1, min(frame_number, max(int(frame_count), 1)))
        inherited_lens = round(_lens_mm(transform.get("lens_mm"), inherited_lens), 4)
        video_keyframes.append(
            {
                "frame": frame_number,
                "location": _vector3(transform.get("location"), [0.0, -3.0, 1.6]),
                "rotation_euler": _vector3(transform.get("rotation_euler"), [0.0, 0.0, 0.0]),
                "lens_mm": inherited_lens,
            }
        )
    return video_keyframes


def high_risk_allowed(camera: dict[str, Any], preset_name: str, reason: Any = "") -> bool:
    preset = normalize_trajectory_preset(preset_name)
    if preset not in HIGH_RISK_TRAJECTORY_PRESETS:
        return True
    text = " ".join(
        str(value or "").lower()
        for value in (
            camera.get("shot_description"),
            camera.get("scene_description"),
            camera.get("movement_tag"),
            camera.get("authored_movement"),
            reason,
        )
    )
    return any(term in text for term in ("orbit", "reveal", "rise", "drop", "arc", "curve", "circle", "around"))


def trajectory_safety_report(
    *,
    camera: dict[str, Any],
    preset_name: str,
    start_candidate: dict[str, Any] | None,
    endpoint_candidate: dict[str, Any] | None,
    start_transform: dict[str, Any],
    end_transform: dict[str, Any],
    reason: Any = "",
    duration_seconds: float | None = None,
    motion_speed_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preset = normalize_trajectory_preset(preset_name)
    start_visible = candidate_visible_fraction(start_candidate)
    endpoint_visible = candidate_visible_fraction(endpoint_candidate)
    travel = _distance_between(start_transform.get("location"), end_transform.get("location"))
    start_location = _vector3(start_transform.get("location"), [0.0, -3.0, 1.6])
    end_location = _vector3(end_transform.get("location"), start_location)
    height_delta = abs(end_location[2] - start_location[2])
    closeup = bool(camera.get("closeup_required")) or is_closeup(str(camera.get("distance_label") or ""))
    warnings: list[str] = []
    allowed = high_risk_allowed(camera, preset, reason)
    if preset in HIGH_RISK_TRAJECTORY_PRESETS and not allowed:
        warnings.append("high_risk_preset_without_story_support")
    if endpoint_candidate and endpoint_visible <= 0.0 and endpoint_candidate.get("candidate_id") != (start_candidate or {}).get("candidate_id"):
        warnings.append("endpoint_visible_fraction_zero")
    if closeup and travel > 0.75:
        warnings.append("closeup_endpoint_travel_too_large")
    if closeup and height_delta > 0.35:
        warnings.append("closeup_height_delta_too_large")
    duration = float(duration_seconds or trajectory_duration_seconds(preset, travel_distance=travel, closeup=closeup))
    speed = travel / max(duration, 1e-6)
    max_speed = TRAJECTORY_MAX_SPEED.get(preset)
    if max_speed and speed > max_speed * 1.05:
        warnings.append("trajectory_speed_too_fast")
    return {
        "valid": not warnings,
        "warnings": warnings,
        "preset_name": preset,
        "start_candidate_id": (start_candidate or {}).get("candidate_id"),
        "endpoint_candidate_id": (endpoint_candidate or {}).get("candidate_id"),
        "start_visible_fraction": round(start_visible, 6),
        "endpoint_visible_fraction": round(endpoint_visible, 6),
        "travel_distance": round(travel, 6),
        "height_delta": round(height_delta, 6),
        "duration_seconds": round(duration, 3),
        "location_speed_units_per_second": round(speed, 6),
        "max_speed_units_per_second": round(float(max_speed or 0.0), 6),
        "motion_speed_policy": dict(motion_speed_policy or {}),
        "closeup": closeup,
        "high_risk": preset in HIGH_RISK_TRAJECTORY_PRESETS,
        "high_risk_allowed": allowed,
    }


def build_camera_trajectory_plan(
    *,
    camera: dict[str, Any],
    preset_name: Any,
    start_transform: dict[str, Any],
    end_transform: dict[str, Any],
    start_candidate: dict[str, Any] | None,
    endpoint_candidate: dict[str, Any] | None,
    selection_source: str,
    reason: Any = "",
    fps: int,
) -> dict[str, Any]:
    preset = normalize_trajectory_preset(preset_name)
    if _ABLATION_FLAGS.get("disable_trajectory_grounding"):
        # Ablation: naive linear interpolation between unmodified start/end.
        # Skip motion limiting, frame-count duration heuristics, midpoint
        # easing, and safety report.
        duration_seconds = float(camera.get("target_duration_seconds") or 2.5)
        frame_count = max(24, int(round(duration_seconds * int(fps or 24))))
        naive_keyframes = [
            {"frame": 0, "t": 0.0, "transform": dict(start_transform), "easing": "linear"},
            {
                "frame": max(0, frame_count - 1),
                "t": 1.0,
                "transform": dict(end_transform),
                "easing": "linear",
            },
        ]
        return {
            "schema_version": "plan_a.video_trajectory.v1",
            "preset_name": preset,
            "selection_reason": str(reason or selection_source),
            "trajectory_selection_source": selection_source,
            "start_candidate_id": (start_candidate or {}).get("candidate_id"),
            "endpoint_candidate_id": (endpoint_candidate or {}).get("candidate_id"),
            "start_frame": 1,
            "end_frame": frame_count,
            "duration_seconds": round(duration_seconds, 3),
            "timing_policy": {
                "fps": int(fps or 24),
                "editor_handles_seconds": {"head": 0.25, "tail": 0.25},
                "motion_speed_policy": {"ablation_disabled": True},
            },
            "lens_policy": {
                "base_lens_mm": round(float(start_transform.get("lens_mm") or camera.get("lens_mm") or 35.0), 4),
                "uses_subtle_lens_ramp": False,
            },
            "safety_report": {
                "ablation_disabled": True,
                "warnings": ["trajectory_grounding_disabled_by_ablation"],
            },
            "keyframes": video_keyframes_from_camera_keyframes(naive_keyframes, frame_count),
            "ablation_disabled": True,
        }
    limited_end_transform, motion_speed_policy = limit_trajectory_motion(
        camera=camera,
        preset_name=preset,
        start_transform=start_transform,
        end_transform=end_transform,
    )
    duration_seconds = trajectory_duration_seconds(
        preset,
        travel_distance=float(motion_speed_policy.get("limited_travel_distance") or 0.0),
        closeup=bool(motion_speed_policy.get("closeup")),
    )
    frame_count = max(24, int(round(duration_seconds * int(fps or 24))))
    keyframes = build_trajectory_keyframes(
        start_transform=start_transform,
        end_transform=limited_end_transform,
        movement_tag=str(camera.get("movement_tag") or ""),
        motion_profile=str(camera.get("motion_profile") or ""),
        frame_count=frame_count,
        fps=int(fps or 24),
        preset_name=preset,
    )
    effective_end_transform = dict((keyframes[-1].get("transform") if keyframes else limited_end_transform) or limited_end_transform)
    safety = trajectory_safety_report(
        camera=camera,
        preset_name=preset,
        start_candidate=start_candidate,
        endpoint_candidate=endpoint_candidate,
        start_transform=start_transform,
        end_transform=effective_end_transform,
        reason=reason,
        duration_seconds=duration_seconds,
        motion_speed_policy=motion_speed_policy,
    )
    return {
        "schema_version": "plan_a.video_trajectory.v1",
        "preset_name": preset,
        "selection_reason": str(reason or selection_source),
        "trajectory_selection_source": selection_source,
        "start_candidate_id": (start_candidate or {}).get("candidate_id"),
        "endpoint_candidate_id": (endpoint_candidate or {}).get("candidate_id"),
        "start_frame": 1,
        "end_frame": frame_count,
        "duration_seconds": round(duration_seconds, 3),
        "timing_policy": {
            "fps": int(fps or 24),
            "editor_handles_seconds": {"head": 0.25, "tail": 0.25},
            "motion_speed_policy": motion_speed_policy,
        },
        "lens_policy": {
            "base_lens_mm": round(float(start_transform.get("lens_mm") or camera.get("lens_mm") or 35.0), 4),
            "uses_subtle_lens_ramp": preset == "static_subtle_zoom",
        },
        "safety_report": safety,
        "keyframes": video_keyframes_from_camera_keyframes(keyframes, frame_count),
    }


def refresh_camera_trajectory(
    camera: dict[str, Any],
    *,
    preset_name: Any | None = None,
    start_candidate: dict[str, Any] | None = None,
    endpoint_candidate: dict[str, Any] | None = None,
    selection_source: str,
    reason: Any = "",
    fps: int,
) -> None:
    preset = normalize_trajectory_preset(
        preset_name
        or preset_for_motion(camera.get("movement_tag"), camera.get("direction_tag"), camera.get("motion_profile"))
    )
    start_transform = camera.get("start_transform") or {}
    end_transform = camera.get("end_transform") or start_transform
    if _ABLATION_FLAGS.get("disable_trajectory_grounding"):
        # Ablation: keep raw end_transform, build trivial linear keyframes.
        limited_end_transform = dict(end_transform)
        camera["end_transform"] = limited_end_transform
        duration_seconds = float(camera.get("target_duration_seconds") or 2.5)
        frame_count = max(24, int(round(duration_seconds * int(fps or 24))))
        camera["target_duration_seconds"] = round(duration_seconds, 3)
        camera["target_frame_count"] = frame_count
        camera["trajectory_keyframes"] = [
            {"frame": 0, "t": 0.0, "transform": dict(start_transform), "easing": "linear"},
            {
                "frame": max(0, frame_count - 1),
                "t": 1.0,
                "transform": dict(limited_end_transform),
                "easing": "linear",
            },
        ]
    else:
        limited_end_transform, motion_speed_policy = limit_trajectory_motion(
            camera=camera,
            preset_name=preset,
            start_transform=start_transform,
            end_transform=end_transform,
        )
        camera["end_transform"] = limited_end_transform
        duration_seconds = trajectory_duration_seconds(
            preset,
            travel_distance=float(motion_speed_policy.get("limited_travel_distance") or 0.0),
            closeup=bool(motion_speed_policy.get("closeup")),
        )
        frame_count = max(24, int(round(duration_seconds * int(fps or 24))))
        camera["target_duration_seconds"] = round(duration_seconds, 3)
        camera["target_frame_count"] = frame_count
        camera["trajectory_keyframes"] = build_trajectory_keyframes(
            start_transform=start_transform,
            end_transform=limited_end_transform,
            movement_tag=str(camera.get("movement_tag") or ""),
            motion_profile=str(camera.get("motion_profile") or ""),
            frame_count=frame_count,
            fps=int(fps or 24),
            preset_name=preset,
        )
    if camera["trajectory_keyframes"]:
        camera["end_transform"] = dict(camera["trajectory_keyframes"][-1].get("transform") or camera.get("end_transform") or {})
    trajectory_plan = build_camera_trajectory_plan(
        camera=camera,
        preset_name=preset,
        start_transform=start_transform,
        end_transform=limited_end_transform,
        start_candidate=start_candidate,
        endpoint_candidate=endpoint_candidate,
        selection_source=selection_source,
        reason=reason,
        fps=int(fps or 24),
    )
    camera["trajectory_plan"] = trajectory_plan
    camera["trajectory_selection_source"] = selection_source
    camera["trajectory_safety_report"] = trajectory_plan.get("safety_report") or {}
    camera["editor_handles_seconds"] = {"head": 0.25, "tail": 0.25}


def build_bridge(from_plan: dict[str, Any], to_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "from_camera_name": from_plan["camera_name"],
        "to_camera_name": to_plan["camera_name"],
        "transition_type": "motivated_cut" if from_plan["primary_focus_id"] == to_plan["primary_focus_id"] else "contrast_cut",
        "overlap_frames": 6,
        "start_transform": from_plan["end_transform"],
        "end_transform": to_plan["start_transform"],
    }


def default_blender_exe(configured_path: str) -> Path:
    if configured_path:
        return Path(configured_path).resolve()
    local_blender = CAMERA_DIR / "blender" / "blender.exe"
    if local_blender.exists():
        return local_blender.resolve()
    return DEFAULT_BLENDER_EXE


def resolve_blend_file(director_handoff: dict[str, Any]) -> Path:
    files = director_handoff.get("files") or {}
    source_inventory_path = Path(str(files.get("source_inventory_path") or ""))
    source_inventory = load_json(source_inventory_path) if source_inventory_path.exists() else {}
    selected_sources = source_inventory.get("selected_sources") or {}
    for candidate in selected_sources.get("blend_paths") or []:
        candidate_path = Path(str(candidate))
        if candidate_path.exists():
            return candidate_path.resolve()
    demo_root = Path(str(director_handoff.get("demo_root") or "")).resolve()
    return (demo_root / "The Godfather.blend").resolve()


def run_quality_candidate_worker(
    *,
    config: CinematographerConfig,
    director_handoff: dict[str, Any],
    camera_handoff_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    blend_file = resolve_blend_file(director_handoff)
    stdout_path = output_root / "camera_quality_stdout.log"
    stderr_path = output_root / "camera_quality_stderr.log"
    script_args = [
        "--camera-handoff-path",
        str(camera_handoff_path),
        "--output-root",
        str(output_root),
        "--resolution-x",
        str(config.quality_resolution_x),
        "--resolution-y",
        str(config.quality_resolution_y),
        "--render-engine",
        config.quality_render_engine,
        "--render-samples",
        str(config.quality_render_samples),
        "--top-k",
        str(config.quality_top_k),
        "--max-depth",
        str(config.quality_max_depth),
        "--frontier-limit",
        str(config.quality_frontier_limit),
        "--per-channel-top-k",
        str(config.per_channel_top_k),
    ]
    if config.enable_candidate_validation:
        script_args.extend(
            [
                "--enable-candidate-validation",
                "--validation-failed-top-per-channel",
                str(config.validation_failed_top_per_channel),
                "--validation-success-near-threshold-per-channel",
                str(config.validation_success_near_threshold_per_channel),
                "--validation-reason-samples",
                str(config.validation_reason_samples),
            ]
        )
    if config.disable_semantic_height_adjust:
        script_args.append("--disable-semantic-height-adjust")
    result = run_blender_python_script(
        blender_exe=default_blender_exe(config.blender_exe),
        blend_file=blend_file,
        python_script=STAGE_DIR / "cinematographer_quality_worker.py",
        script_args=script_args,
        workdir=STAGE_DIR,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=config.quality_timeout_seconds,
        background=True,
    )
    result["quality_report_path"] = str(output_root / "outputs" / "camera_quality_report_v1.json")
    if config.enable_candidate_validation:
        result["candidate_validation_manifest_path"] = str(output_root / "outputs" / "candidate_validation_manifest_v1.jsonl")
        result["candidate_blind_review_input_path"] = str(output_root / "outputs" / "candidate_blind_review_input_v1.jsonl")
        result["candidate_blind_review_result_path"] = str(output_root / "outputs" / "candidate_blind_review_result_v1.jsonl")
        result["candidate_filter_confusion_report_path"] = str(output_root / "outputs" / "candidate_filter_confusion_report_v1.json")
    result["blend_file"] = str(blend_file)
    return result


def run_preview_render(
    *,
    config: CinematographerConfig,
    director_handoff: dict[str, Any],
    camera_handoff_path: Path,
    output_root: Path,
    report_filename: str = "camera_preview_report_v1.json",
) -> dict[str, Any]:
    """Render one preview frame per camera for visual verification."""
    blend_file = resolve_blend_file(director_handoff)
    stdout_path = output_root / "camera_preview_stdout.log"
    stderr_path = output_root / "camera_preview_stderr.log"
    result = run_blender_python_script(
        blender_exe=default_blender_exe(config.blender_exe),
        blend_file=blend_file,
        python_script=STAGE_DIR / "cinematographer_preview_worker.py",
        script_args=[
            "--camera-handoff-path",
            str(camera_handoff_path),
            "--output-root",
            str(output_root),
            "--resolution-x",
            str(config.preview_resolution_x),
            "--resolution-y",
            str(config.preview_resolution_y),
            "--render-engine",
            config.quality_render_engine,
            "--render-samples",
            str(config.preview_render_samples),
            "--report-filename",
            report_filename,
        ],
        workdir=STAGE_DIR,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=config.quality_timeout_seconds,
        background=True,
    )
    report_path = output_root / "outputs" / report_filename
    result["preview_report_path"] = str(report_path)
    return result


def apply_preview_paths(shot_outputs: list[dict[str, Any]], camera_rows: list[dict[str, Any]], preview_report: dict[str, Any]) -> None:
    """Write preview_frame_path back into camera packages and rows."""
    path_map: dict[str, str] = {}
    for item in preview_report.get("results") or []:
        if item.get("success") and item.get("preview_path"):
            path_map[str(item["camera_name"])] = str(item["preview_path"])
    for shot in shot_outputs:
        for camera in shot.get("cameras") or []:
            name = str(camera.get("camera_name") or "")
            if name in path_map:
                camera["preview_frame_path"] = path_map[name]
                pkg_path = Path(str(camera.get("camera_package_path") or ""))
                if pkg_path.exists():
                    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
                    pkg["preview_frame_path"] = path_map[name]
                    save_json(pkg, pkg_path)
    for row in camera_rows:
        name = str(row.get("camera_name") or "")
        if name in path_map:
            row["preview_frame_path"] = path_map[name]


def render_and_apply_previews(
    *,
    shot_outputs: list[dict[str, Any]],
    camera_rows: list[dict[str, Any]],
    config: CinematographerConfig,
    director_handoff: dict[str, Any],
    director_handoff_path: Path,
    output_root: Path,
    outputs_dir: Path,
    handoff_filename: str,
    report_filename: str = "camera_preview_report_v1.json",
    camera_names: set[str] | None = None,
) -> dict[str, Any]:
    render_shot_outputs = shot_outputs
    if camera_names is not None:
        render_shot_outputs = []
        for shot in shot_outputs:
            filtered_cameras = [
                camera for camera in (shot.get("cameras") or [])
                if str(camera.get("camera_name") or "") in camera_names
            ]
            if filtered_cameras:
                shot_copy = dict(shot)
                shot_copy["cameras"] = filtered_cameras
                render_shot_outputs.append(shot_copy)
    preview_handoff_path = save_json(
        {
            "schema_version": "storyblender.camera_handoff.v1",
            "generated_at": utc_now(),
            "run_id": config.run_id or output_root.name,
            "director_handoff_path": str(director_handoff_path),
            "shots": render_shot_outputs,
            "cameras": [camera for shot in render_shot_outputs for camera in shot["cameras"]],
        },
        outputs_dir / handoff_filename,
    )
    preview_result = run_preview_render(
        config=config,
        director_handoff=director_handoff,
        camera_handoff_path=preview_handoff_path,
        output_root=output_root,
        report_filename=report_filename,
    )
    if preview_result.get("success"):
        preview_report_path = Path(str(preview_result.get("preview_report_path") or ""))
        if preview_report_path.exists():
            preview_report = load_json(preview_report_path)
            apply_preview_paths(shot_outputs, camera_rows, preview_report)
            success_count = preview_report.get("success_count", 0)
            total_count = preview_report.get("total", 0)
            print(f"[preview] {success_count}/{total_count} camera previews rendered.")
        else:
            print("[preview] Preview report not found, skipping preview path injection.")
    else:
        print(f"[preview] Preview render failed. See {preview_result.get('stderr_path', '')}")
    for shot in shot_outputs:
        for camera in shot.get("cameras") or []:
            if camera_names is not None and str(camera.get("camera_name") or "") not in camera_names:
                continue
            camera["final_preview_path"] = camera.get("preview_frame_path") or ""
            persist_camera_package(camera)
    refresh_camera_row_metadata(camera_rows, shot_outputs)
    return preview_result


def run_final_preview_review_pipeline(
    *,
    shot_outputs: list[dict[str, Any]],
    rows_by_camera: dict[str, dict[str, Any]],
    camera_rows: list[dict[str, Any]],
    config: CinematographerConfig,
    director_handoff: dict[str, Any],
    director_handoff_path: Path,
    output_root: Path,
    outputs_dir: Path,
) -> dict[str, Any]:
    all_cameras = [camera for shot in shot_outputs for camera in shot.get("cameras") or []]
    if config.disable_vlm_reflection:
        ablation_rows: list[dict[str, Any]] = []
        for camera in all_cameras:
            current_candidate = camera.get("selected_candidate") or {}
            review = {
                "success": True,
                "review_source": "disabled_by_ablation_vlm_reflection",
                "ablation_disabled": True,
            }
            validation = {
                "valid": True,
                "verdict": "passed_ablation_vlm_reflection_disabled",
                "reasons": [],
                "warnings": ["vlm_reflection_disabled_by_ablation"],
                "ablation_disabled": True,
            }
            review["validation"] = validation
            camera["final_preview_path"] = camera.get("preview_frame_path") or ""
            camera["final_preview_llm_review"] = review
            camera["final_render_source_candidate_id"] = current_candidate.get("candidate_id") or ""
            camera["final_render_source_candidate_preview_path"] = current_candidate.get("preview_image_path") or ""
            camera["final_selected_candidate_preview_path"] = current_candidate.get("preview_image_path") or ""
            camera["final_preview_repair_trace"] = [
                {
                    "attempt_index": 0,
                    "candidate_id": current_candidate.get("candidate_id"),
                    "candidate_preview_path": current_candidate.get("preview_image_path") or "",
                    "preview_path": camera.get("preview_frame_path") or "",
                    "review_source": "disabled_by_ablation_vlm_reflection",
                    "validation": validation,
                }
            ]
            camera["downstream_eligible"] = True
            persist_camera_package(camera)
            ablation_rows.append(
                {
                    "camera_name": str(camera.get("camera_name") or ""),
                    "review": review,
                    "validation": validation,
                }
            )
        print(
            f"[ablation] disable_vlm_reflection=True -> skipped final_preview_review for "
            f"{len(all_cameras)} cameras",
            flush=True,
        )
        return {
            "rows": ablation_rows,
            "blocked_camera_names": [],
            "repaired_camera_names": [],
            "ablation_disabled": True,
        }
    max_workers = max(1, min(config.llm_max_workers, len(all_cameras)))
    final_rows: list[dict[str, Any]] = []
    repaired_cameras: list[str] = []
    blocked_cameras: list[str] = []

    def _review_camera(cam: dict[str, Any]) -> dict[str, Any]:
        review = llm_story_consistency_judge(
            camera_package=cam,
            config=config,
            preview_path=str(cam.get("preview_frame_path") or ""),
            allow_candidate_preview_fallback=False,
            review_source="final_preview_frame",
        )
        validation = final_preview_review_validity(cam, review)
        return {
            "camera_name": cam.get("camera_name"),
            "review": review,
            "validation": validation,
        }

    review_map: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_review_camera, cam): cam for cam in all_cameras}
        for future in as_completed(futures):
            result = future.result()
            review_map[str(result["camera_name"])] = result

    cameras_requiring_repair: list[dict[str, Any]] = []
    for camera in all_cameras:
        camera_name = str(camera.get("camera_name") or "")
        result = review_map.get(camera_name) or {
            "review": {"success": False, "error": "final_preview_review_missing"},
            "validation": {"valid": False, "reasons": ["final_preview_review_missing"]},
        }
        review = dict(result.get("review") or {})
        validation = dict(result.get("validation") or {})
        review["validation"] = validation
        current_candidate = camera.get("selected_candidate") or {}
        camera["final_preview_path"] = camera.get("preview_frame_path") or ""
        camera["final_preview_llm_review"] = review
        camera["final_render_source_candidate_id"] = current_candidate.get("candidate_id") or ""
        camera["final_render_source_candidate_preview_path"] = current_candidate.get("preview_image_path") or ""
        camera["final_selected_candidate_preview_path"] = current_candidate.get("preview_image_path") or ""
        camera["final_preview_repair_trace"] = [
            {
                "attempt_index": 0,
                "candidate_id": current_candidate.get("candidate_id"),
                "candidate_preview_path": current_candidate.get("preview_image_path") or "",
                "preview_path": camera.get("preview_frame_path") or "",
                "review_source": review.get("review_source") or "final_preview_frame",
                "validation": validation,
            }
        ]
        if validation.get("valid"):
            camera["downstream_eligible"] = True
        else:
            if quality_gate_vetoed(validation):
                camera["downstream_eligible"] = False
                camera["selection_source"] = "final_preview_blocked"
                camera["selection_degrade_reason"] = "; ".join(
                    value
                    for value in (
                        camera.get("selection_degrade_reason") or "",
                        "final_preview_blocked",
                        *[str(reason) for reason in (validation.get("reasons") or [])],
                    )
                    if value
                )
                blocked_cameras.append(camera_name)
                persist_camera_package(camera)
                continue
            row = rows_by_camera.get(camera_name) or {}
            if current_candidate_is_protected_semantic_seed(camera, row):
                camera["downstream_eligible"] = False
                camera["selection_source"] = "final_preview_protected_semantic_blocked"
                camera["selection_degrade_reason"] = "; ".join(
                    value
                    for value in (
                        camera.get("selection_degrade_reason") or "",
                        "protected_semantic_seed_not_repaired",
                        *[str(reason) for reason in (validation.get("reasons") or [])],
                    )
                    if value
                )
                camera["final_preview_repair_trace"].append(
                    {
                        "attempt_index": 1,
                        "status": "skipped",
                        "reason": "protected_semantic_seed_not_repaired",
                        "protected_candidate_id": candidate_key(camera.get("selected_candidate") or {}),
                    }
                )
                blocked_cameras.append(camera_name)
                persist_camera_package(camera)
                continue
            replacement = choose_final_preview_replacement_candidate(
                row=row,
                camera=camera,
                current_candidate=camera.get("selected_candidate") or {},
            )
            if replacement is None:
                camera["downstream_eligible"] = False
                camera["selection_source"] = "final_preview_blocked"
                camera["selection_degrade_reason"] = "; ".join(
                    value
                    for value in (
                        camera.get("selection_degrade_reason") or "",
                        "final_preview_blocked",
                        *[str(reason) for reason in (validation.get("reasons") or [])],
                    )
                    if value
                )
                blocked_cameras.append(camera_name)
            else:
                repaired_cameras.append(camera_name)
                endpoint = None
                trajectory_choice = normalize_trajectory_preset((camera.get("trajectory_plan") or {}).get("preset_name"))
                if movement_requires_dynamic(camera):
                    endpoint = choose_replacement_endpoint(
                        row=row,
                        camera=camera,
                        start_candidate=replacement,
                    )
                    trajectory_choice = normalize_trajectory_preset(
                        preset_for_motion(
                            camera.get("movement_tag"),
                            trajectory_direction_hint(camera, replacement, endpoint),
                            camera.get("motion_profile"),
                        )
                    )
                apply_validated_camera_selection(
                    camera=camera,
                    start_candidate=replacement,
                    endpoint_candidate=endpoint,
                    trajectory_choice=trajectory_choice,
                    selection_source="final_preview_repair",
                    selection_reason="; ".join(validation.get("reasons") or []) or "final_preview_repair",
                    fps=config.fps,
                )
                camera["preview_render_label"] = "repair"
                camera["final_render_source_candidate_id"] = replacement.get("candidate_id") or ""
                camera["final_render_source_candidate_preview_path"] = replacement.get("preview_image_path") or ""
                camera["final_selected_candidate_preview_path"] = replacement.get("preview_image_path") or ""
                camera["selection_source"] = "final_preview_repair"
                camera["selection_validity"] = candidate_selection_validity(camera, replacement)
                camera["selection_degrade_reason"] = "; ".join(
                    value
                    for value in (
                        camera.get("selection_degrade_reason") or "",
                        "final_preview_repair",
                        *[str(reason) for reason in (validation.get("reasons") or [])],
                    )
                    if value
                )
                camera["downstream_eligible"] = False
                camera["final_preview_repair_trace"].append(
                    {
                        "attempt_index": 1,
                        "replacement_candidate_id": replacement.get("candidate_id"),
                        "replacement_candidate_preview_path": replacement.get("preview_image_path") or "",
                        "replacement_endpoint_candidate_id": (endpoint or {}).get("candidate_id"),
                        "selection_source": "final_preview_repair",
                        "failure_reasons": validation.get("reasons") or [],
                    }
                )
                cameras_requiring_repair.append(camera)
        persist_camera_package(camera)

    if cameras_requiring_repair:
        render_and_apply_previews(
            shot_outputs=shot_outputs,
            camera_rows=camera_rows,
            config=config,
            director_handoff=director_handoff,
            director_handoff_path=director_handoff_path,
            output_root=output_root,
            outputs_dir=outputs_dir,
            handoff_filename="camera_handoff_preview_repair_input_v1.json",
            report_filename="camera_preview_repair_report_v1.json",
            camera_names={str(cam.get("camera_name") or "") for cam in cameras_requiring_repair},
        )
        review_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_review_camera, cam): cam for cam in cameras_requiring_repair}
            for future in as_completed(futures):
                result = future.result()
                review_map[str(result["camera_name"])] = result
        for camera in cameras_requiring_repair:
            camera_name = str(camera.get("camera_name") or "")
            result = review_map.get(camera_name) or {
                "review": {"success": False, "error": "final_preview_repair_review_missing"},
                "validation": {"valid": False, "reasons": ["final_preview_repair_review_missing"]},
            }
            review = dict(result.get("review") or {})
            validation = dict(result.get("validation") or {})
            review["validation"] = validation
            current_candidate = camera.get("selected_candidate") or {}
            camera["final_preview_path"] = camera.get("preview_frame_path") or ""
            camera["final_preview_llm_review"] = review
            camera["final_preview_repair_trace"].append(
                {
                    "attempt_index": 2,
                    "candidate_id": current_candidate.get("candidate_id"),
                    "candidate_preview_path": current_candidate.get("preview_image_path") or "",
                    "preview_path": camera.get("preview_frame_path") or "",
                    "review_source": review.get("review_source") or "final_preview_frame",
                    "validation": validation,
                }
            )
            if validation.get("valid"):
                camera["downstream_eligible"] = True
            else:
                camera["downstream_eligible"] = False
                camera["selection_source"] = "final_preview_blocked"
                camera["selection_degrade_reason"] = "; ".join(
                    value
                    for value in (
                        camera.get("selection_degrade_reason") or "",
                        "final_preview_blocked",
                        *[str(reason) for reason in (validation.get("reasons") or [])],
                    )
                    if value
                )
                blocked_cameras.append(camera_name)
            persist_camera_package(camera)

    blocked_set = set(blocked_cameras)
    for camera in all_cameras:
        review = dict(camera.get("final_preview_llm_review") or {})
        validation = dict((review.get("validation") or {}))
        final_rows.append(
            {
                "camera_name": camera.get("camera_name"),
                "success": bool(validation.get("valid")),
                "verdict": validation.get("verdict") or ("passed_clean" if validation.get("valid") else "hard_blocked_camera_issue"),
                "preview_path": camera.get("preview_frame_path") or "",
                "review_source": review.get("review_source") or "",
                "consistency_score": review.get("consistency_score"),
                "needs_reshoot": review.get("needs_reshoot"),
                "reason": review.get("reason"),
                "camera_issue": review.get("camera_issue"),
                "non_camera_issue": review.get("non_camera_issue"),
                "hard_block_camera_reason": review.get("hard_block_camera_reason"),
                "validation": validation,
                "warnings": validation.get("warnings") or [],
                "deterministic_quality_gate": validation.get("deterministic_quality_gate") or deterministic_quality_gate(camera),
                "downstream_eligible": bool(camera.get("downstream_eligible", True)),
                "final_preview_repair_trace": camera.get("final_preview_repair_trace") or [],
            }
        )
        if camera.get("camera_name") in blocked_set:
            camera["downstream_eligible"] = False
    refresh_camera_row_metadata(camera_rows, shot_outputs)
    return {
        "rows": final_rows,
        "repaired_camera_names": repaired_cameras,
        "blocked_camera_names": sorted(set(blocked_cameras)),
    }


def quality_rows_by_camera(report_payload: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in report_payload.get("rows") or []:
        key = (int(row.get("scene_id") or 0), int(row.get("shot_id") or 0), str(row.get("camera_name") or ""))
        rows[key] = row
    return rows


def apply_quality_results(shot_outputs: list[dict[str, Any]], report_payload: dict[str, Any], config: CinematographerConfig) -> None:
    lookup = quality_rows_by_camera(report_payload)
    for shot in shot_outputs:
        for camera in shot.get("cameras") or []:
            key = (int(camera.get("scene_id") or 0), int(camera.get("shot_id") or 0), str(camera.get("camera_name") or ""))
            row = lookup.get(key)
            if not row:
                camera["quality_mode"] = "quality_missing"
                continue
            has_quality_selection = bool(row.get("success") and row.get("selected_candidate"))
            camera["quality_mode"] = "quality"
            camera["quality_candidate_report"] = {
                "success": bool(row.get("success")),
                "candidate_count_raw": int(row.get("candidate_count_raw") or 0),
                "candidate_count_eligible": int(row.get("candidate_count_eligible") or 0),
                "candidate_count_retained": int(row.get("candidate_count_retained") or 0),
                "candidate_count_deduplicated": int(row.get("candidate_count_deduplicated") or 0),
                "candidate_board_path": row.get("candidate_board_path") or "",
                "channel_boards": row.get("channel_boards") or {},
                "selection_source": row.get("selection_source") or "",
                "selection_reason": row.get("selection_reason") or "",
                "semantic_contract": row.get("semantic_contract") or {},
                "closeup_required": bool(row.get("closeup_required")),
                "semantic_weight_reason": row.get("semantic_weight_reason") or "",
            }
            if row.get("selection_source"):
                camera["selection_source"] = row.get("selection_source") or ""
            camera["closeup_required"] = bool(row.get("closeup_required"))
            camera["semantic_weight_reason"] = row.get("semantic_weight_reason") or ""
            if row.get("quality_qc"):
                camera["quality_qc"] = row.get("quality_qc") or {}
            if row.get("candidate_board_path"):
                camera["quality_candidate_board_path"] = row.get("candidate_board_path") or ""
            if has_quality_selection:
                camera["selected_candidate"] = row.get("selected_candidate") or {}
                selection_reason = str(row.get("selection_reason") or "")
                if is_protected_semantic_seed_candidate(
                    camera,
                    camera["selected_candidate"],
                    selection_reason=selection_reason,
                ):
                    remember_protected_semantic_seed(camera, camera["selected_candidate"], selection_reason)
            if row.get("top_candidates"):
                camera["top_candidates"] = row.get("top_candidates") or []
            if has_quality_selection and row.get("start_transform"):
                camera["start_transform"] = row["start_transform"]
                if row["start_transform"].get("lens_mm") is not None:
                    camera["lens_mm"] = row["start_transform"].get("lens_mm")
            if has_quality_selection and row.get("end_transform"):
                camera["end_transform"] = row["end_transform"]
            if has_quality_selection:
                refresh_camera_trajectory(
                    camera,
                    start_candidate=camera.get("selected_candidate") or {},
                    endpoint_candidate=None,
                    selection_source=row.get("selection_source") or "eligible_candidate",
                    reason=row.get("selection_reason") or "quality_worker_selected",
                    fps=config.fps,
                )
            package_path = Path(str(camera.get("camera_package_path") or ""))
            if package_path.exists():
                save_json(camera, package_path)


LEGAL_TRAJECTORY_PRESETS = [
    "static_hold",
    "static_subtle_zoom",
    "straight_ease",
    "push_in_arc",
    "pull_out_arc",
    "orbit_left_arc",
    "orbit_right_arc",
    "pan_left",
    "pan_right",
    "truck_left",
    "truck_right",
    "pedestal_up",
    "pedestal_down",
    "rise_reveal",
    "drop_reveal",
    "s_curve",
]


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "channel": candidate.get("channel") or "unknown",
        "source": candidate.get("source") or "unknown",
        "direction": candidate.get("direction"),
        "operation_depth": candidate.get("operation_depth"),
        "operation_sequence": candidate.get("operation_sequence") or [],
        "scores": candidate.get("scores") or {},
        "projection": candidate.get("projection") or {},
        "lens_mm": candidate.get("lens_mm"),
        "visible_fraction": candidate_visible_fraction(candidate),
    }


def safe_endpoint_candidates(row: dict[str, Any], selected: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [c for c in row.get("top_candidates") or [] if c.get("preview_image_path")]
    safe = [
        candidate
        for candidate in candidates
        if candidate.get("location")
        and candidate.get("rotation_euler")
        and candidate_selection_validity(row, candidate).get("valid")
    ]
    selected_id = str(selected.get("candidate_id") or "")
    selected_valid = candidate_selection_validity(row, selected).get("valid")
    if selected_id and selected_valid and all(str(candidate.get("candidate_id") or "") != selected_id for candidate in safe):
        safe.insert(0, selected)
    return safe or ([selected] if selected and selected_valid else [])


def llm_select_from_board(
    *,
    row: dict[str, Any],
    config: CinematographerConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Use vision LLM to select the best candidate from the merged board."""
    candidates = [c for c in row.get("top_candidates") or [] if c.get("preview_image_path")]
    protected_candidate = row.get("protected_semantic_seed_candidate") or row.get("protected_candidate") or {}
    protected_candidate_id = candidate_key(protected_candidate)
    board_path = str(row.get("candidate_board_path") or "").strip()
    if not candidates:
        return None, {"success": False, "error": "no_rendered_candidates"}
    if not board_path or not Path(board_path).exists():
        return None, {"success": False, "error": "candidate_board_missing"}
    if not llm_ready(model=config.vision_model, api_key=config.anyllm_api_key):
        return None, {"success": False, "error": "llm_unavailable_for_candidate_selection"}

    system_prompt = "You are a cinematographer selecting one start camera from a rendered candidate board."
    user_content = [
        {
            "type": "text",
            "text": (
                "Choose the single best start camera candidate.  Optimize for story semantics, "
                "focus readability, composition, and usable surrounding scene context.  The board "
                "tiles are the truth source. The semantic channel contains close-up/detail candidates "
                "only; prefer it only when the script or shot contract calls for a close-up/detail. "
                "If the script asks for a wider or revealing shot, choose direction/preset unless a "
                "semantic close-up better satisfies the described start frame. When this shot has "
                "multiple cameras, choose coverage with a clear narrative function that can differ "
                "from neighboring cameras: master coverage, action follow, reaction, detail, "
                "over-shoulder, POV, or complementary angle. Return strict JSON only."
            ),
        },
        {
            "type": "text",
            "text": json.dumps(
                {
                    "camera_name": row.get("camera_name"),
                    "camera_index": row.get("camera_index"),
                    "shot_description": row.get("shot_description") or "",
                    "scene_description": row.get("scene_description") or "",
                    "camera_role": row.get("camera_role") or "",
                    "distance_label": row.get("distance_label") or "",
                    "movement_tag": row.get("movement_tag") or "",
                    "shot_contract": row.get("shot_contract") or {},
                    "semantic_contract": row.get("semantic_contract") or {},
                    "closeup_required": bool(row.get("closeup_required")),
                    "semantic_weight_reason": row.get("semantic_weight_reason") or "",
                    "protected_candidate_id": protected_candidate_id,
                    "protected_candidate_policy": (
                        "When protected_candidate_id is present, it is a deterministic semantic detail seed. "
                        "Use it only as a fallback if no visually better valid candidate is available."
                    )
                    if protected_candidate_id
                    else "",
                    "candidate_count_on_board": len(candidates),
                    "candidates": [_candidate_summary(c) for c in candidates],
                    "response_schema": {
                        "selected_candidate_id": "candidate id from candidates",
                        "reason": "short reason",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
        {"type": "image_url", "image_url": {"url": image_path_to_data_url(board_path)}},
    ]
    parsed, error, raw_text = call_json_response(
        model=config.vision_model,
        system_prompt=system_prompt,
        user_content=user_content,
        api_key=config.anyllm_api_key,
        api_base=config.anyllm_api_base,
        provider=config.anyllm_provider,
        reasoning_effort="medium",
        retry_count=config.llm_retry_count,
    )
    if error or not isinstance(parsed, dict):
        return None, {"success": False, "error": error or "llm_selection_parse_failed", "raw_response": raw_text}
    selected_id = str(parsed.get("selected_candidate_id") or "").strip()
    by_id = {str(c.get("candidate_id")): c for c in candidates}
    board_candidate_ids = list(by_id)
    selected = by_id.get(selected_id)
    if selected is None:
        fallback = by_id.get(protected_candidate_id) if protected_candidate_id else None
        fallback_source = "protected_semantic_seed_fallback" if fallback is not None else "llm_unknown_candidate_fallback"
        if fallback is None:
            fallback = candidates[0] if candidates else None
        return fallback, {
            "success": fallback is not None,
            "warning": "llm_selected_unknown_candidate",
            "selected_candidate_id": selected_id,
            "fallback_candidate_id": fallback.get("candidate_id") if fallback else None,
            "protected_candidate_id": protected_candidate_id,
            "valid_candidate_ids": board_candidate_ids,
            "board_candidate_ids": board_candidate_ids,
            "raw_response": raw_text,
            "selection_source": fallback_source,
        }
    return selected, {
        "success": True,
        "selected_candidate_id": selected_id,
        "reason": parsed.get("reason"),
        "protected_candidate_id": protected_candidate_id,
        "board_candidate_ids": board_candidate_ids,
        "raw_response": raw_text,
        "selection_source": "llm_candidate_board_selection",
    }


def _build_board_from_previews(candidates: list[dict[str, Any]], output_path: Path) -> str | None:
    """Build a tile-grid board image from candidate preview images (PIL, stage-side)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    previews = [(c, Path(str(c.get("preview_image_path") or ""))) for c in candidates]
    previews = [(c, p) for c, p in previews if p.exists()]
    if not previews:
        return None
    rendered = []
    for candidate, path in previews:
        try:
            rendered.append((candidate, Image.open(path)))
        except Exception:
            continue
    if not rendered:
        return None
    # Fix G1: render a strictly text-free board so the LLM judges from
    # the images alone. Thumbnails are arranged row-major and correspond
    # 1:1 to the candidate list passed alongside in the prompt.
    tile_w = rendered[0][1].width
    tile_h = rendered[0][1].height
    cols = min(len(rendered), 5)
    rows = (len(rendered) + cols - 1) // cols
    board = Image.new("RGB", (cols * tile_w, rows * tile_h), (244, 242, 238))
    for idx, (_candidate, img) in enumerate(rendered):
        row, col = divmod(idx, cols)
        left = col * tile_w
        top = row * tile_h
        board.paste(img.resize((tile_w, tile_h)), (left, top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(str(output_path))
    return str(output_path)


def _balanced_candidates_for_llm(candidates: list[dict[str, Any]], per_channel_limit: int = 8, total_limit: int = 30) -> list[dict[str, Any]]:
    """Keep the LLM board balanced across direction, preset, and semantic channels."""
    grouped: dict[str, list[dict[str, Any]]] = {"direction": [], "preset": [], "semantic": []}
    for candidate in candidates:
        grouped.setdefault(str(candidate.get("channel") or "direction"), []).append(candidate)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for channel_name in ("direction", "preset", "semantic"):
        for candidate in grouped.get(channel_name, [])[:per_channel_limit]:
            selected.append(candidate)
            seen.add(id(candidate))
    if len(selected) < total_limit:
        for candidate in candidates:
            if len(selected) >= total_limit:
                break
            if id(candidate) in seen:
                continue
            selected.append(candidate)
            seen.add(id(candidate))
    return selected[:total_limit]


def llm_filter_channel_board(
    *,
    channel_name: str,
    channel_info: dict[str, Any],
    camera_package: dict[str, Any],
    config: CinematographerConfig,
) -> dict[str, Any]:
    """Phase-1 LLM: filter a single channel's board, returning survivors and rejections."""
    candidates = channel_info.get("selected") or []
    board_path = str(channel_info.get("board_path") or "")
    rendered = [c for c in candidates if c.get("preview_image_path") and Path(str(c["preview_image_path"])).exists()]
    if not rendered:
        return {"channel": channel_name, "success": False, "error": "no_rendered_previews", "survivors": [], "rejected": []}
    if not board_path or not Path(board_path).exists():
        return {"channel": channel_name, "success": False, "error": "board_image_missing", "survivors": rendered, "rejected": []}
    if not llm_ready(model=config.vision_model, api_key=config.anyllm_api_key):
        return {"channel": channel_name, "success": False, "error": "llm_unavailable", "survivors": rendered, "rejected": []}

    shot_desc = str(camera_package.get("shot_description") or "")
    scene_desc = str(camera_package.get("scene_description") or "")
    distance = str(camera_package.get("distance_label") or "")
    angle = str(camera_package.get("angle_label") or "")
    semantic_target = str(camera_package.get("primary_semantic_target") or "")
    quality_info = camera_package.get("quality_candidate_report") or {}

    # Fix D: instruct the LLM to judge subject facing FROM THE IMAGE and to
    # treat the per-candidate ``direction`` metadata as untrusted. The
    # rig-derived direction labels are unreliable on rigs where the
    # area-based front/back probe gets fooled, so the only authoritative
    # facing signal is the rendered preview itself. Back-of-head shots are
    # rejected unless the shot contract explicitly requests a back / OTS
    # framing.
    back_view_intended = _is_back_view_intent_camera(camera_package)
    system_prompt = (
        "You are a cinematographer reviewing camera candidate previews for one channel. "
        "Select ALL candidates that match the shooting requirements. Reject any that have "
        "bad composition, wrong framing, occluded subjects, or mismatched story intent. "
        "The semantic channel is only for close-up/detail candidates; keep semantic candidates "
        "when the script or shot contract needs a close-up/detail, and reject them when they "
        "fight a wider story intent. "
        "FACING RULE: Determine subject facing strictly from the rendered image, not from the "
        "candidate's `direction` metadata field, which can be wrong because the rig-axis probe "
        "may flip front and back. If the subject's back of head, nape, or shoulder blades fill "
        "the frame and no face is visible, treat the candidate as a back-facing shot. "
        "Reject every back-facing or strongly back-three-quarter candidate UNLESS the shot "
        "contract or shot description explicitly asks for a back view, over-the-shoulder, "
        "from-behind, or back-of-head framing. "
        "Return strict JSON."
    )
    facing_hint = (
        "FACING CHECK: This shot does NOT request a back view. Reject any candidate where the "
        "image clearly shows the subject's back of head / nape / spine and no face is visible."
        if not back_view_intended
        else "FACING CHECK: This shot explicitly requests a back-facing or OTS framing; back views are allowed."
    )
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Channel: {channel_name}. Review each of the {len(rendered)} candidate previews "
                f"on the board. The board image is intentionally text-free (no labels, ids, or "
                f"scores are drawn on it); thumbnails are arranged row-major (left to right, then "
                f"top to bottom) and correspond 1:1 to the `candidates` list below in the same "
                f"order. Use the JSON list to look up the `candidate_id` for the i-th thumbnail. "
                f"Keep every candidate that is acceptable for this shot. Reject candidates that "
                f"clearly fail the shooting requirements based on what you see in the image. "
                f"{facing_hint} "
                f"Judge composition and facing from the rendered image only."
            ),
        },
        {
            "type": "text",
            "text": json.dumps(
                {
                    "camera_name": camera_package.get("camera_name"),
                    "shot_description": shot_desc,
                    "scene_description": scene_desc,
                    "distance_label": distance,
                    "angle_label": angle,
                    "primary_semantic_target": semantic_target,
                    "camera_role": camera_package.get("camera_role") or "",
                    "movement_tag": camera_package.get("movement_tag") or "",
                    "shot_contract": camera_package.get("shot_contract") or {},
                    "semantic_contract": quality_info.get("semantic_contract") or {},
                    "closeup_required": bool(quality_info.get("closeup_required")),
                    "semantic_weight_reason": quality_info.get("semantic_weight_reason") or "",
                    "channel": channel_name,
                    "candidate_count": len(rendered),
                    "candidates": [
                        # Fix G1: ordered row-major to match the text-free board image.
                        # `direction` and per-candidate `scores` are intentionally omitted so
                        # the LLM judges from the images and the shot intent, not from
                        # potentially-wrong rig metadata or upstream scoring.
                        {
                            "position": index + 1,
                            "candidate_id": c.get("candidate_id"),
                            "channel": c.get("channel") or channel_name,
                            "source": c.get("source") or "unknown",
                            "lens_mm": c.get("lens_mm"),
                        }
                        for index, c in enumerate(rendered)
                    ],
                    "response_schema": {
                        "accepted_candidate_ids": ["list of accepted candidate IDs"],
                        "rejections": [
                            {"candidate_id": "...", "reason": "short reason for rejection"},
                        ],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
        {"type": "image_url", "image_url": {"url": image_path_to_data_url(board_path)}},
    ]
    parsed, error, raw_text = call_json_response(
        model=config.vision_model,
        system_prompt=system_prompt,
        user_content=user_content,
        api_key=config.anyllm_api_key,
        api_base=config.anyllm_api_base,
        provider=config.anyllm_provider,
        reasoning_effort="medium",
        retry_count=config.llm_retry_count,
    )
    if error or not isinstance(parsed, dict):
        return {
            "channel": channel_name,
            "success": False,
            "error": error or "channel_filter_parse_failed",
            "survivors": rendered,
            "rejected": [],
            "raw_response": raw_text,
        }
    accepted_ids = set(str(aid) for aid in (parsed.get("accepted_candidate_ids") or []))
    by_id = {str(c.get("candidate_id")): c for c in rendered}
    survivors = [by_id[cid] for cid in accepted_ids if cid in by_id]
    rejected_entries = parsed.get("rejections") or []
    rejected_ids = set(str(r.get("candidate_id") or "") for r in rejected_entries if isinstance(r, dict))
    for c in rendered:
        cid = str(c.get("candidate_id"))
        if cid not in accepted_ids and cid not in rejected_ids:
            accepted_ids.add(cid)
            survivors.append(c)
    if not survivors:
        survivors = rendered

    return {
        "channel": channel_name,
        "success": True,
        "survivor_count": len(survivors),
        "rejected_count": len(rendered) - len(survivors),
        "survivors": survivors,
        "rejected": rejected_entries,
        "raw_response": raw_text,
    }


def _candidate_after_operation(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
    operation: str,
) -> dict[str, Any] | None:
    base_sequence = list(selected.get("operation_sequence") or [])
    target_sequence = base_sequence + [operation]
    for c in candidates:
        if list(c.get("operation_sequence") or []) == target_sequence:
            return c
    matches = [
        c for c in candidates
        if list(c.get("operation_sequence") or [])[-1:] == [operation]
        and c.get("preview_image_path")
    ]
    matches.sort(key=lambda c: float((c.get("scores") or {}).get("final") or 0.0), reverse=True)
    return matches[0] if matches else None


def llm_micro_adjust(
    *,
    row: dict[str, Any],
    selected: dict[str, Any],
    config: CinematographerConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """LLM micro-adjustment: up to N rounds of single-step camera operations."""
    if config.disable_vlm_reflection:
        return selected, {
            "applied": False,
            "stop_reason": "disabled_by_ablation_vlm_reflection",
            "rounds": [],
            "ablation_disabled": True,
        }
    candidates = [c for c in row.get("top_candidates") or [] if c.get("preview_image_path")]
    board_path = str(row.get("candidate_board_path") or "").strip()
    max_rounds = config.llm_micro_adjust_max_rounds
    if not llm_ready(model=config.vision_model, api_key=config.anyllm_api_key):
        return selected, {"applied": False, "stop_reason": "llm_unavailable_for_micro_adjustment", "rounds": []}
    current = selected
    rounds: list[dict[str, Any]] = []
    seen_transitions: set[tuple[str, str]] = set()
    seen_candidate_ids: list[str] = [str(selected.get("candidate_id") or "")]
    for round_index in range(1, max_rounds + 1):
        preview_path = str(current.get("preview_image_path") or "")
        if not preview_path or not Path(preview_path).exists():
            return current, {"applied": bool(rounds), "stop_reason": "preview_image_missing", "rounds": rounds}
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Review this selected camera preview in its surrounding scene context. "
                    "If it is good enough, return satisfied=true. If it needs one small change, "
                    "choose exactly one operation from the operation list. Do not ask for zoom if "
                    "the surrounding context would disappear."
                ),
            },
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "camera_name": row.get("camera_name"),
                        "current_candidate": _candidate_summary(current),
                        "allowed_operations": [
                            "pan_left", "pan_right", "pan_up", "pan_down",
                            "orbit_left", "orbit_right", "orbit_up", "orbit_down",
                            "truck_left", "truck_right",
                            "dolly_in", "dolly_out",
                            "zoom_in", "zoom_out",
                            "pedestal_up", "pedestal_down",
                        ],
                        "response_schema": {
                            "satisfied": True,
                            "operation": "one allowed operation or empty",
                            "reason": "short reason",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        if board_path and Path(board_path).exists():
            user_content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(board_path)}})
        user_content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(preview_path)}})
        parsed, error, raw_text = call_json_response(
            model=config.vision_model,
            system_prompt="You are a camera micro-adjustment reviewer. Return strict JSON.",
            user_content=user_content,
            api_key=config.anyllm_api_key,
            api_base=config.anyllm_api_base,
            provider=config.anyllm_provider,
            reasoning_effort="low",
            retry_count=config.llm_retry_count,
        )
        round_row: dict[str, Any] = {
            "round_index": round_index,
            "candidate_id_before": current.get("candidate_id"),
            "raw_response": raw_text,
        }
        if error or not isinstance(parsed, dict):
            round_row.update({"status": "llm_failed", "error": error or "micro_adjust_parse_failed"})
            rounds.append(round_row)
            return current, {"applied": bool(rounds), "stop_reason": "llm_failed", "rounds": rounds}
        operation = str(parsed.get("operation") or "").strip()
        round_row["llm_decision"] = parsed
        if bool(parsed.get("satisfied")) or not operation:
            round_row["status"] = "satisfied"
            rounds.append(round_row)
            return current, {"applied": bool(rounds), "stop_reason": "llm_satisfied", "rounds": rounds}
        next_candidate = _candidate_after_operation(candidates, current, operation)
        if next_candidate is None:
            round_row["status"] = "operation_not_available_on_rendered_board"
            round_row["requested_operation"] = operation
            rounds.append(round_row)
            return current, {"applied": bool(rounds), "stop_reason": "requested_operation_not_available", "rounds": rounds}
        current_id = str(current.get("candidate_id") or "")
        next_id = str(next_candidate.get("candidate_id") or "")
        transition_key = (current_id, next_id)
        if transition_key in seen_transitions or (next_id and next_id in seen_candidate_ids[:-1]):
            round_row["status"] = "candidate_oscillation"
            round_row["requested_operation"] = operation
            round_row["candidate_id_after"] = next_id
            rounds.append(round_row)
            return current, {"applied": bool(rounds), "stop_reason": "candidate_oscillation", "rounds": rounds}
        seen_transitions.add(transition_key)
        seen_candidate_ids.append(next_id)
        current = next_candidate
        round_row["status"] = "applied_board_candidate_operation"
        round_row["candidate_id_after"] = current.get("candidate_id")
        rounds.append(round_row)
    return current, {"applied": bool(rounds), "stop_reason": "max_rounds_reached", "rounds": rounds}


def llm_choose_trajectory(
    *,
    row: dict[str, Any],
    selected: dict[str, Any],
    camera: dict[str, Any],
    config: CinematographerConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """LLM chooses a trajectory preset and endpoint candidate."""
    candidates = safe_endpoint_candidates(row, selected)
    if not llm_ready(model=config.vision_model, api_key=config.anyllm_api_key):
        return None, None, {"success": False, "error": "llm_unavailable_for_trajectory_selection"}
    if len(candidates) <= 1:
        plan = {
            "trajectory_choice": "static_subtle_zoom",
            "start_candidate_id": selected.get("candidate_id"),
            "endpoint_candidate_id": selected.get("candidate_id"),
            "speed_plan": {"start": "settle", "middle": "subtle lens drift", "end": "hold readable composition"},
            "camera_motion": {"translation": "none", "rotation": "none", "height": "stable"},
            "use_seed_candidate_as_endpoint": False,
            "reason": "Only one valid preview exists, so keep the selected framing and add a subtle lens ramp.",
            "plan_source": "single_preview_static_subtle_zoom",
        }
        return plan, selected, {"success": True, "mode": "single_preview_static_subtle_zoom"}
    board_path = str(row.get("candidate_board_path") or "").strip()
    if not board_path or not Path(board_path).exists():
        return None, None, {"success": False, "error": "trajectory_candidate_board_missing"}
    user_content = [
        {
            "type": "text",
            "text": (
                "Choose a safe trajectory for this shot. The selected start preview is fixed. "
                "You may choose one endpoint candidate from the same rendered board when the motion "
                "needs two previews. Prefer a readable path that preserves the focus group and scene context. "
                "Do not choose endpoints with zero visibility. For close-ups, avoid large side moves, back-of-head "
                "moves, or any endpoint that pushes the face out of frame. Use orbit/rise/drop/s_curve only when "
                "the story or camera movement clearly supports that risk."
            ),
        },
        {
            "type": "text",
            "text": json.dumps(
                {
                    "camera_name": row.get("camera_name"),
                    "shot_description": camera.get("shot_description") or row.get("shot_description") or "",
                    "distance_label": camera.get("distance_label") or row.get("distance_label") or "",
                    "movement_tag": camera.get("movement_tag") or "",
                    "selected_start_candidate": _candidate_summary(selected),
                    "legal_trajectory_presets": LEGAL_TRAJECTORY_PRESETS,
                    "high_risk_presets": sorted(HIGH_RISK_TRAJECTORY_PRESETS),
                    "endpoint_candidates": [_candidate_summary(c) for c in candidates],
                    "response_schema": {
                        "trajectory_choice": "one legal preset",
                        "start_candidate_id": "selected start candidate id",
                        "endpoint_candidate_id": "candidate id from endpoint_candidates",
                        "speed_plan": {"start": "...", "middle": "...", "end": "..."},
                        "camera_motion": {"translation": "...", "rotation": "...", "height": "..."},
                        "reason": "short reason",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
        {"type": "image_url", "image_url": {"url": image_path_to_data_url(board_path)}},
        {"type": "image_url", "image_url": {"url": image_path_to_data_url(str(selected.get("preview_image_path")))}},
    ]
    parsed, error, raw_text = call_json_response(
        model=config.vision_model,
        system_prompt="You are a cinematography trajectory planner. Return strict JSON.",
        user_content=user_content,
        api_key=config.anyllm_api_key,
        api_base=config.anyllm_api_base,
        provider=config.anyllm_provider,
        reasoning_effort="medium",
        retry_count=config.llm_retry_count,
    )
    if error or not isinstance(parsed, dict):
        return None, None, {"success": False, "error": error or "trajectory_parse_failed", "raw_response": raw_text}
    preset = normalize_trajectory_preset(parsed.get("trajectory_choice"))
    if preset not in LEGAL_TRAJECTORY_PRESETS:
        return None, None, {
            "success": False,
            "error": "llm_selected_illegal_trajectory_preset",
            "trajectory_choice": preset,
            "raw_response": raw_text,
        }
    endpoint_id = str(parsed.get("endpoint_candidate_id") or selected.get("candidate_id") or "").strip()
    by_id = {str(c.get("candidate_id")): c for c in candidates}
    endpoint = by_id.get(endpoint_id)
    if endpoint is None:
        return None, None, {
            "success": False,
            "error": "llm_selected_unknown_endpoint_candidate",
            "endpoint_candidate_id": endpoint_id,
            "raw_response": raw_text,
        }
    reason = parsed.get("reason") or ""
    if preset in HIGH_RISK_TRAJECTORY_PRESETS and not high_risk_allowed(camera, preset, reason):
        preset = "static_subtle_zoom" if is_closeup(str(camera.get("distance_label") or "")) else "straight_ease"
        endpoint = selected
        endpoint_id = str(selected.get("candidate_id") or "")
        reason = f"Downgraded high-risk trajectory without story support. Original reason: {reason}"
    plan = {
        "trajectory_choice": preset,
        "start_candidate_id": selected.get("candidate_id"),
        "speed_plan": parsed.get("speed_plan") if isinstance(parsed.get("speed_plan"), dict) else {},
        "camera_motion": parsed.get("camera_motion") if isinstance(parsed.get("camera_motion"), dict) else {},
        "use_seed_candidate_as_endpoint": endpoint.get("candidate_id") != selected.get("candidate_id"),
        "reason": reason,
        "plan_source": "llm_focus_board_trajectory_selection",
        "endpoint_candidate_id": endpoint_id,
    }
    return plan, endpoint, {"success": True, "raw_response": raw_text}


def llm_story_consistency_judge(
    *,
    camera_package: dict[str, Any],
    config: CinematographerConfig,
    preview_path: str | None = None,
    allow_candidate_preview_fallback: bool = True,
    review_source: str = "selected_candidate_preview",
) -> dict[str, Any]:
    """LLM evaluates if the selected camera framing matches the shot's story intent."""
    if not llm_ready(model=config.vision_model, api_key=config.anyllm_api_key):
        return {"success": False, "error": "llm_unavailable_for_story_consistency"}
    resolved_preview_path = str(preview_path or camera_package.get("preview_frame_path") or "")
    resolved_review_source = review_source
    if not resolved_preview_path or not Path(resolved_preview_path).exists():
        if not allow_candidate_preview_fallback:
            return {
                "success": False,
                "error": "no_final_preview_image_for_story_judge",
                "review_source": review_source,
                "preview_path": resolved_preview_path,
            }
        selected = camera_package.get("selected_candidate") or {}
        resolved_preview_path = str(selected.get("preview_image_path") or "")
        resolved_review_source = "selected_candidate_preview"
    if not resolved_preview_path or not Path(resolved_preview_path).exists():
        return {
            "success": False,
            "error": "no_preview_image_for_story_judge",
            "review_source": resolved_review_source,
            "preview_path": resolved_preview_path,
        }
    contract = start_frame_contract(camera_package)
    user_content = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "review_source": resolved_review_source,
                    "camera_name": camera_package.get("camera_name"),
                    "shot_description": camera_package.get("shot_description") or "",
                    "scene_description": camera_package.get("scene_description") or "",
                    "primary_focus_id": camera_package.get("primary_focus_id") or contract.get("primary_focus_id") or "",
                    "focus_ids": camera_package.get("focus_ids") or contract.get("start_focus_ids") or [],
                    "secondary_focus_ids": camera_package.get("secondary_focus_ids") or contract.get("secondary_focus_ids") or [],
                    "primary_semantic_target": camera_package.get("primary_semantic_target") or "",
                    "distance_label": camera_package.get("distance_label") or "",
                    "angle_label": camera_package.get("angle_label") or "",
                    "movement_tag": camera_package.get("movement_tag") or "",
                    "response_schema": {
                        "consistency_score": "float 0-1, 1=strong camera/framing match",
                        "primary_subject_visible": "bool, is the primary subject visible in the frame",
                        "framing_matches_intent": "bool, ONLY judge camera framing/angle/shot-size/composition; ignore character actions, poses, props, expressions, environment content",
                        "camera_issue": "string|null, describe ONLY camera-controllable problems (wrong angle, wrong shot size, subject out of frame, severe crop, wrong composition); null if no camera issue",
                        "non_camera_issue": "string|null, describe non-camera problems (wrong action, missing prop, wrong expression, environment mismatch, asset quality); null if no non-camera issue",
                        "hard_block_camera_reason": "string|null, set ONLY if the primary subject is missing, unreadable, or the camera setup makes the shot unusable; shot-size/angle mismatch alone is not a hard block when primary_subject_visible=true",
                        "reason": "concise explanation covering both camera and non-camera observations",
                        "needs_reshoot": "bool, true ONLY for severe primary-subject camera failures listed in hard_block_camera_reason",
                        "secondary_subject_visible": "bool|null, diagnostic only, never drives needs_reshoot",
                        "hands_visible": "bool|null, diagnostic only",
                        "interaction_readable": "bool|null, diagnostic only",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
        {"type": "image_url", "image_url": {"url": image_path_to_data_url(resolved_preview_path)}},
    ]
    parsed, error, raw_text = call_json_response(
        model=config.vision_model,
        system_prompt=(
            "You are a camera/framing QA judge for a cinematography pipeline. "
            "Your job is to decide whether the CAMERA SETUP (framing, angle, shot size, composition, subject visibility) "
            "is acceptable for handoff. You are NOT judging whether the story, animation, or scene is perfect.\n\n"
            "CAMERA-CONTROLLABLE issues (can cause hard block):\n"
            "- Primary subject completely out of frame or severely cropped\n"
            "- Primary subject unreadable because the shot size is extreme\n"
            "- Camera angle makes the primary subject unreadable or shows the wrong side when a semantic side is required\n"
            "- Camera movement clearly wrong type\n"
            "- Composition completely off (subject at edge, severe occlusion from camera angle)\n\n"
            "NON-CAMERA issues (NEVER cause hard block, report as non_camera_issue only):\n"
            "- Character not performing the described action or pose\n"
            "- Facial expression or emotion not matching description\n"
            "- Props or objects missing from the scene\n"
            "- Environment/set dressing not matching description\n"
            "- Lighting, material, or render quality issues\n"
            "- Secondary character missing (this is an asset/scene issue)\n"
            "- Hand pose or gesture not matching\n"
            "- Story/narrative intent not fully conveyed\n"
            "- Workbench or low-resolution preview appearance\n\n"
            "RULES:\n"
            "- The primary focus id is the required filming subject. Secondary subjects are context unless explicitly required.\n"
            "- Set framing_matches_intent=true if the primary subject is visible and readable, "
            "even if distance/angle is not exact or non-camera issues exist.\n"
            "- Set framing_matches_intent=false ONLY for camera-controllable framing failures.\n"
            "- Set needs_reshoot=true ONLY for severe camera failures that make the primary subject unusable.\n"
            "- Put camera problems in camera_issue field, non-camera problems in non_camera_issue field.\n"
            "- Set hard_block_camera_reason ONLY for severe primary-subject camera failures; null otherwise.\n"
            "- Secondary subject, hands, interaction fields are diagnostic only.\n"
            "Return strict JSON."
        ),
        user_content=user_content,
        api_key=config.anyllm_api_key,
        api_base=config.anyllm_api_base,
        provider=config.anyllm_provider,
        reasoning_effort="medium",
        retry_count=config.final_review_retry_count,
    )
    if error or not isinstance(parsed, dict):
        return {
            "success": False,
            "error": error or "story_judge_parse_failed",
            "raw_response": raw_text,
            "review_source": resolved_review_source,
            "preview_path": resolved_preview_path,
        }
    primary_subject_visible = bool_or_none(parsed.get("primary_subject_visible"))
    if primary_subject_visible is None:
        primary_subject_visible = bool_or_none(parsed.get("character_visible"))
    return {
        "success": True,
        "consistency_score": float(parsed.get("consistency_score") or 0.0),
        "character_visible": primary_subject_visible,
        "primary_subject_visible": primary_subject_visible,
        "secondary_subject_visible": bool_or_none(parsed.get("secondary_subject_visible")),
        "hands_visible": bool_or_none(parsed.get("hands_visible")),
        "interaction_readable": bool_or_none(parsed.get("interaction_readable")),
        "framing_matches_intent": bool_or_none(parsed.get("framing_matches_intent")),
        "camera_issue": parsed.get("camera_issue") or None,
        "non_camera_issue": parsed.get("non_camera_issue") or None,
        "hard_block_camera_reason": parsed.get("hard_block_camera_reason") or None,
        "reason": parsed.get("reason"),
        "needs_reshoot": bool_or_none(parsed.get("needs_reshoot")),
        "raw_response": raw_text,
        "review_source": resolved_review_source,
        "preview_path": resolved_preview_path,
    }


def _process_single_camera_llm(
    camera: dict[str, Any],
    row: dict[str, Any],
    config: CinematographerConfig,
    output_root: Path,
    phase1_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Process one camera through the full two-phase LLM pipeline (runs in thread)."""
    camera_name = str(camera.get("camera_name") or "")

    # ── Phase 1: per-channel LLM filtering ──────────────────────────
    protected_seed = protected_semantic_seed_candidate(camera, row)
    protected_seed_id = candidate_key(protected_seed)
    if protected_seed is not None:
        remember_protected_semantic_seed(
            camera,
            protected_seed,
            str(row.get("selection_reason") or camera.get("protected_semantic_seed_selection_reason") or ""),
        )
    channel_boards = row.get("channel_boards") or {}
    phase1_results = dict(phase1_results or {})
    all_survivors: list[dict[str, Any]] = []

    if not phase1_results:
        for ch_name, ch_info in channel_boards.items():
            print(f"[llm] Phase-1 filtering {camera_name}/{ch_name} ({len(ch_info.get('selected') or [])} candidates)...", flush=True)
            ch_result = llm_filter_channel_board(
                channel_name=ch_name,
                channel_info=ch_info,
                camera_package=camera,
                config=config,
            )
            phase1_results[ch_name] = ch_result

    for ch_result in phase1_results.values():
        all_survivors.extend(ch_result.get("survivors") or [])

    if protected_seed is not None:
        all_survivors = prioritize_protected_candidate(all_survivors, protected_seed)

    if not all_survivors:
        return {"camera_name": camera_name, "success": False, "error": "no_survivors_after_channel_filtering", "phase1": phase1_results}

    # ── Phase 2: Iterative merging and filtering ────────────────────
    current_survivors = all_survivors
    candidate_dir = output_root / "quality_candidates" / f"scene_{camera.get('scene_id', 0)}_shot_{camera.get('shot_id', 0)}" / camera_name
    iteration_count = 0
    merged_board_path = ""
    merged_row = dict(row)
    merged_row["top_candidates"] = current_survivors
    if protected_seed is not None:
        merged_row["protected_semantic_seed_candidate"] = protected_seed
    selection_decision = {}
    selected_llm = None
    llm_board_history: list[dict[str, Any]] = []

    if len(current_survivors) == 1:
        selected_llm = current_survivors[0]
        llm_board_history.append(
            {
                "iteration": 0,
                "candidate_ids": [candidate_key(selected_llm)],
                "board_path": "",
            }
        )
        selection_decision = {
            "success": True,
            "selected_candidate_id": selected_llm.get("candidate_id"),
            "selection_source": "single_survivor_after_channel_filtering",
            "board_candidate_ids": [candidate_key(selected_llm)],
        }

    while len(current_survivors) > 1 and iteration_count < 5:
        iteration_count += 1
        if protected_seed is not None:
            current_survivors = prioritize_protected_candidate(current_survivors, protected_seed)
        # Take up to 20 candidates for the current board
        batch_size = min(20, len(current_survivors))
        current_batch = current_survivors[:batch_size]
        
        merged_board_path = _build_board_from_previews(
            current_batch,
            candidate_dir / f"{camera_name}_llm_merged_board_iter_{iteration_count}.png",
        )
        
        merged_row = dict(row)
        merged_row["top_candidates"] = current_batch
        if protected_seed is not None:
            merged_row["protected_semantic_seed_candidate"] = protected_seed
        if merged_board_path:
            merged_row["candidate_board_path"] = merged_board_path
        llm_board_history.append(
            {
                "iteration": iteration_count,
                "candidate_ids": [candidate_key(candidate) for candidate in current_batch],
                "board_path": merged_board_path or "",
            }
        )

        print(f"[llm] Phase-2 iteration {iteration_count} for {camera_name} ({len(current_batch)} candidates)...", flush=True)
        selected_llm, selection_decision = llm_select_from_board(row=merged_row, config=config)
        
        if selected_llm is None:
            break
            
        # If there are more survivors to process, put the winner back into the pool for the next round
        if len(current_survivors) > batch_size:
            current_survivors = [selected_llm] + current_survivors[batch_size:]
            if protected_seed is not None:
                current_survivors = prioritize_protected_candidate(current_survivors, protected_seed)
        else:
            # All candidates processed, we have our final winner
            break

    if selected_llm is None:
        return {"camera_name": camera_name, "success": False, "phase1": phase1_results, **selection_decision}

    if protected_seed is not None and candidate_selection_validity(camera, protected_seed).get("valid"):
        if candidate_key(selected_llm) != protected_seed_id:
            selection_decision = dict(selection_decision)
            selection_decision["protected_semantic_seed_available"] = True
            selection_decision["protected_candidate_id"] = protected_seed_id
            selection_decision["llm_selected_over_protected_seed"] = candidate_key(selected_llm)
        selection_decision.setdefault("protected_candidate_id", protected_seed_id)
        selection_decision["protected_semantic_seed_active"] = candidate_key(selected_llm) == protected_seed_id

    # ── Micro-adjust ────────────────────────────────────────────────
    prefer_non_semantic = str((selected_llm or {}).get("channel") or "") == "semantic" and not is_closeup_camera(camera)
    local_operation_candidates = row_candidates(row)
    known_candidate_ids = {str(c.get("candidate_id") or "") for c in local_operation_candidates}
    for candidate in [*all_survivors, *current_survivors, selected_llm]:
        candidate_id = str((candidate or {}).get("candidate_id") or "")
        if candidate and candidate_id and candidate_id not in known_candidate_ids:
            local_operation_candidates.append(candidate)
            known_candidate_ids.add(candidate_id)

    def _micro_adjust_local(current_candidate: dict[str, Any]) -> dict[str, Any]:
        """Perform up to 5 steps of local micro-adjustment using mathematical ranking without LLM."""
        best_candidate = current_candidate
        best_score = candidate_quality_rank(camera, best_candidate, prefer_non_semantic=prefer_non_semantic)
        
        for _ in range(5):
            improved = False
            for operation in ["pan_left", "pan_right", "pan_up", "pan_down", "truck_left", "truck_right", "pedestal_up", "pedestal_down", "dolly_in", "dolly_out", "orbit_left", "orbit_right", "orbit_up", "orbit_down", "zoom_in", "zoom_out"]:
                next_candidate = _candidate_after_operation(local_operation_candidates, best_candidate, operation)
                if next_candidate is None:
                    continue
                score = candidate_quality_rank(camera, next_candidate, prefer_non_semantic=prefer_non_semantic)
                if score > best_score + 0.05:  # Require meaningful improvement
                    best_score = score
                    best_candidate = next_candidate
                    improved = True
            if not improved:
                break
        return best_candidate

    protected_seed_active = protected_seed is not None and candidate_key(selected_llm) == protected_seed_id
    if protected_seed_active:
        adjusted = selected_llm
        micro_trace = {
            "applied": False,
            "stop_reason": "protected_semantic_seed",
            "rounds": [],
            "protected_candidate_id": protected_seed_id,
        }
        micro_adjust_status = {
            "applied": False,
            "stop_reason": "protected_semantic_seed",
            "round_count": 0,
        }
        failed_micro_adjust = False
    else:
        print(f"[llm] Local micro-adjusting {camera_name} (selected: {selected_llm.get('candidate_id')})...", flush=True)
        local_adjusted = _micro_adjust_local(selected_llm)
        if str(local_adjusted.get("candidate_id")) != str(selected_llm.get("candidate_id")):
            print(f"[llm] Local adjustment improved candidate to {local_adjusted.get('candidate_id')}", flush=True)

        print(f"[llm] LLM micro-adjusting {camera_name} (selected: {local_adjusted.get('candidate_id')})...", flush=True)
        adjusted, micro_trace = llm_micro_adjust(row=merged_row, selected=local_adjusted, config=config)
        micro_stop_reason = str(micro_trace.get("stop_reason") or "")
        micro_adjust_status = {
            "applied": bool(micro_trace.get("applied")),
            "stop_reason": micro_stop_reason,
            "round_count": len(micro_trace.get("rounds") or []),
        }
        failed_micro_adjust = micro_stop_reason in {
            "candidate_oscillation",
            "requested_operation_not_available",
            "max_rounds_reached",
            "preview_image_missing",
            "llm_failed",
        }
    degrade_reasons: list[str] = []
    if failed_micro_adjust:
        degrade_reasons.append(f"micro_adjust_{micro_stop_reason}")
    selected_validity = candidate_selection_validity(camera, selected_llm)
    adjusted_validity = candidate_selection_validity(camera, adjusted)
    prefer_non_semantic = str((adjusted or selected_llm).get("channel") or "") == "semantic" and not is_closeup_camera(camera)
    final_start = adjusted
    if (failed_micro_adjust and selected_validity.get("valid")) or not adjusted_validity.get("valid"):
        if selected_validity.get("valid"):
            final_start = selected_llm
            if str(adjusted.get("candidate_id") or "") != str(selected_llm.get("candidate_id") or ""):
                degrade_reasons.append("reverted_to_llm_selected_candidate")
        else:
            replacement = choose_replacement_candidate(
                row=merged_row,
                camera=camera,
                current_candidate=adjusted,
                prefer_non_semantic=prefer_non_semantic,
            )
            if replacement is not None:
                final_start = replacement
                degrade_reasons.append("reselected_readable_candidate")
    dynamic_face_candidate = None
    if not protected_seed_active:
        dynamic_face_candidate = choose_dynamic_face_closeup_candidate(
            row=row,
            camera=camera,
            current_candidate=final_start,
        )
    if dynamic_face_candidate is not None:
        final_start = dynamic_face_candidate
        degrade_reasons.append("dynamic_face_closeup_guard")
    selection_validity = candidate_selection_validity(camera, final_start)
    if not selection_validity.get("valid"):
        replacement = choose_replacement_candidate(
            row=merged_row,
            camera=camera,
            current_candidate=final_start,
            prefer_non_semantic=prefer_non_semantic,
        )
        if replacement is not None:
            final_start = replacement
            selection_validity = candidate_selection_validity(camera, final_start)
            degrade_reasons.append("fallback_to_best_readable_candidate")
    if not selection_validity.get("valid"):
        return {
            "camera_name": camera_name,
            "success": False,
            "error": "no_valid_candidate_after_validation",
            "phase1": phase1_results,
            "selection_decision": selection_decision,
            "micro_adjustment_trace": micro_trace,
            "selection_validity": selection_validity,
        }

    # ── Trajectory ──────────────────────────────────────────────────
    print(f"[llm] Choosing trajectory for {camera_name}...", flush=True)
    trajectory_plan, endpoint, trajectory_decision = llm_choose_trajectory(
        row=merged_row, selected=final_start, camera=camera, config=config,
    )
    trajectory_choice = normalize_trajectory_preset(
        (trajectory_plan or {}).get("trajectory_choice")
        or preset_for_motion(camera.get("movement_tag"), camera.get("direction_tag"), camera.get("motion_profile"))
    )
    trajectory_source = "llm_focus_board_trajectory_selection" if trajectory_plan else "trajectory_fallback_after_validation"
    if trajectory_plan is None:
        degrade_reasons.append(str((trajectory_decision or {}).get("error") or "trajectory_planner_failed"))
    start_transform = candidate_transform(final_start, camera.get("start_transform") or {})
    if endpoint and trajectory_choice in {"static_hold", "static_subtle_zoom"} and not movement_requires_dynamic(camera):
        endpoint = final_start
    endpoint_validity = candidate_selection_validity(camera, endpoint) if endpoint else {"valid": not movement_requires_dynamic(camera)}
    end_transform = candidate_transform(endpoint, camera.get("end_transform") or start_transform) if endpoint else derived_motion_end_transform(camera, start_transform)
    if trajectory_choice in {"static_hold", "static_subtle_zoom"} and movement_requires_dynamic(camera):
        endpoint_validity = {"valid": False, "reasons": ["motion_requires_non_static"]}
        degrade_reasons.append("trajectory_static_on_motion_beat")
    if movement_requires_dynamic(camera) and not trajectory_has_meaningful_motion(camera, trajectory_choice, start_transform, end_transform):
        endpoint_validity = {"valid": False, "reasons": ["trajectory_motion_not_meaningful"]}
    if not endpoint_validity.get("valid"):
        replacement_endpoint = choose_replacement_endpoint(
            row=merged_row,
            camera=camera,
            start_candidate=final_start,
        )
        if replacement_endpoint is not None:
            endpoint = replacement_endpoint
            trajectory_choice = normalize_trajectory_preset(
                preset_for_motion(
                    camera.get("movement_tag"),
                    trajectory_direction_hint(camera, final_start, replacement_endpoint),
                    camera.get("motion_profile"),
                )
            )
            trajectory_source = "validated_endpoint_reselection"
            degrade_reasons.append("trajectory_endpoint_reselected")
        elif movement_requires_dynamic(camera):
            endpoint = None
            trajectory_choice = normalize_trajectory_preset(
                preset_for_motion(camera.get("movement_tag"), camera.get("direction_tag"), camera.get("motion_profile"))
            )
            trajectory_source = "validated_authored_motion_fallback"
            degrade_reasons.append("trajectory_authored_motion_fallback")
        else:
            endpoint = final_start
            trajectory_choice = "static_subtle_zoom"
            trajectory_source = "validated_static_hold_fallback"
            degrade_reasons.append("trajectory_static_fallback")

    # ── Write back to camera package ────────────────────────────────
    final_candidate_preview_path = str(final_start.get("preview_image_path") or selected_llm.get("preview_image_path") or "")
    camera["llm_phase1_results"] = {ch: {"survivor_count": r.get("survivor_count", 0), "rejected_count": r.get("rejected_count", 0)} for ch, r in phase1_results.items()}
    camera["llm_merged_board_path"] = merged_board_path or ""
    camera["llm_input_board_path"] = merged_board_path or ""
    camera["llm_selected_candidate"] = final_start
    camera["llm_selected_candidate_preview_path"] = final_candidate_preview_path
    camera["llm_selection_decision"] = selection_decision
    camera["board_selection_decision"] = selection_decision
    camera["llm_board_history"] = llm_board_history
    camera["llm_micro_adjustment_trace"] = micro_trace
    camera["protected_semantic_seed_active"] = bool(protected_seed_active)
    camera["llm_trajectory_plan"] = trajectory_plan or {
        "trajectory_choice": trajectory_choice,
        "start_candidate_id": final_start.get("candidate_id"),
        "endpoint_candidate_id": (endpoint or {}).get("candidate_id"),
        "reason": "; ".join(degrade_reasons),
        "plan_source": trajectory_source,
    }
    camera["llm_trajectory_decision"] = trajectory_decision
    camera["selection_source"] = selection_decision.get("selection_source") or "llm_candidate_board_selection"
    camera["selection_validity"] = selection_validity
    camera["selection_degrade_reason"] = "; ".join(reason for reason in degrade_reasons if reason)
    camera["micro_adjust_status"] = micro_adjust_status
    camera["final_preview_path"] = ""
    camera["final_preview_llm_review"] = {}
    camera["final_preview_repair_trace"] = []
    camera["downstream_eligible"] = True
    camera["continuity_review"] = {
        "repeated_static_view": False,
        "cast_discontinuity": False,
        "reselected_for_continuity": False,
        "reasons": [],
    }
    apply_validated_camera_selection(
        camera=camera,
        start_candidate=final_start,
        endpoint_candidate=endpoint,
        trajectory_choice=trajectory_choice,
        selection_source=trajectory_source,
        selection_reason=(trajectory_plan or {}).get("reason") or (trajectory_plan or {}).get("plan_source") or "; ".join(degrade_reasons),
        fps=config.fps,
    )

    persist_camera_package(camera)

    return {
        "camera_name": camera_name,
        "success": True,
        "phase1_survivor_total": len(all_survivors),
        "selected_candidate_id": final_start.get("candidate_id"),
        "start_candidate_id": final_start.get("candidate_id"),
        "endpoint_candidate_id": (endpoint or {}).get("candidate_id"),
        "trajectory_choice": camera.get("trajectory_plan", {}).get("preset_name") or trajectory_choice,
        "trajectory_safety_report": camera.get("trajectory_safety_report") or {},
        "selection_validity": selection_validity,
        "selection_degrade_reason": camera.get("selection_degrade_reason") or "",
        "micro_adjust_status": micro_adjust_status,
        "phase1": {ch: {"survivor_count": r.get("survivor_count", 0), "rejected_count": r.get("rejected_count", 0)} for ch, r in phase1_results.items()},
        "selection_decision": selection_decision,
        "board_selection_decision": selection_decision,
        "llm_board_history": llm_board_history,
        "protected_semantic_seed_active": bool(protected_seed_active),
        "protected_semantic_seed_candidate_id": protected_seed_id if protected_seed_active else "",
        "llm_input_board_path": merged_board_path or "",
        "selected_candidate_preview_path": final_candidate_preview_path,
        "micro_adjustment_trace": micro_trace,
        "trajectory_decision": trajectory_decision,
    }


def repair_scene_continuity(
    *,
    shot_outputs: list[dict[str, Any]],
    rows_by_camera: dict[str, dict[str, Any]],
    config: CinematographerConfig,
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    previous_camera: dict[str, Any] | None = None
    for shot in shot_outputs:
        for camera in shot.get("cameras") or []:
            camera_name = str(camera.get("camera_name") or "")
            current_candidate = camera.get("selected_candidate") or {}
            current_validity = candidate_selection_validity(camera, current_candidate)
            story = camera.get("story_consistency") or {}
            same_view = repeated_static_view(previous_camera, camera)
            cast_discontinuity = bool(same_view and previous_camera and set(previous_camera.get("focus_ids") or []) != set(camera.get("focus_ids") or []))
            needs_repair = bool(
                not current_validity.get("valid")
                or bool(story.get("needs_reshoot"))
                or same_view
                or cast_discontinuity
            )
            review = {
                "repeated_static_view": same_view,
                "cast_discontinuity": cast_discontinuity,
                "reselected_for_continuity": False,
                "reasons": [],
            }
            if not current_validity.get("valid"):
                review["reasons"].extend(current_validity.get("reasons") or [])
            if bool(story.get("needs_reshoot")):
                review["reasons"].append("story_consistency_needs_reshoot")
            if same_view:
                review["reasons"].append("same_scene_repeated_static_view")
            if cast_discontinuity:
                review["reasons"].append("same_view_cast_discontinuity")
            row = rows_by_camera.get(camera_name) or {}
            if needs_repair and current_candidate_is_protected_semantic_seed(camera, row):
                review["protected_semantic_seed_active"] = True
                review["reasons"].append("protected_semantic_seed_not_reselected")
                needs_repair = False
            if needs_repair:
                replacement = choose_dynamic_face_closeup_candidate(
                    row=row,
                    camera=camera,
                    current_candidate=current_candidate,
                ) or choose_replacement_candidate(
                    row=row,
                    camera=camera,
                    current_candidate=current_candidate,
                    previous_candidate=(previous_camera or {}).get("selected_candidate") or None,
                    prefer_non_semantic=True,
                )
                if replacement is not None:
                    endpoint = None
                    trajectory_choice = normalize_trajectory_preset((camera.get("trajectory_plan") or {}).get("preset_name"))
                    if movement_requires_dynamic(camera):
                        endpoint = choose_replacement_endpoint(
                            row=row,
                            camera=camera,
                            start_candidate=replacement,
                        )
                        trajectory_choice = normalize_trajectory_preset(
                            preset_for_motion(
                                camera.get("movement_tag"),
                                trajectory_direction_hint(camera, replacement, endpoint),
                                camera.get("motion_profile"),
                            )
                        )
                    apply_validated_camera_selection(
                        camera=camera,
                        start_candidate=replacement,
                        endpoint_candidate=endpoint,
                        trajectory_choice=trajectory_choice,
                        selection_source="continuity_readability_repair",
                        selection_reason="; ".join(review["reasons"]) or "continuity_repair",
                        fps=config.fps,
                    )
                    camera["selection_source"] = "continuity_readability_repair"
                    camera["selection_validity"] = candidate_selection_validity(camera, replacement)
                    camera["selection_degrade_reason"] = "; ".join(
                        value for value in (camera.get("selection_degrade_reason") or "", "continuity_readability_repair") if value
                    )
                    review["reselected_for_continuity"] = True
                    current_candidate = replacement
                    repairs.append(
                        {
                            "camera_name": camera_name,
                            "replacement_candidate_id": replacement.get("candidate_id"),
                            "reasons": review["reasons"],
                        }
                    )
            camera["continuity_review"] = review
            package_path = Path(str(camera.get("camera_package_path") or ""))
            if package_path.exists():
                save_json(camera, package_path)
            previous_camera = camera
    return repairs


def repair_same_shot_diversity(
    *,
    shot_outputs: list[dict[str, Any]],
    rows_by_camera: dict[str, dict[str, Any]],
    config: CinematographerConfig,
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for shot in shot_outputs:
        cameras = sorted(
            list(shot.get("cameras") or []),
            key=lambda item: int(item.get("camera_index") or 0),
        )
        accepted_cameras: list[dict[str, Any]] = []
        for camera_ordinal, camera in enumerate(cameras):
            camera_name = str(camera.get("camera_name") or "")
            current_candidate = camera.get("selected_candidate") or {}
            if camera_ordinal == 0 or not accepted_cameras:
                camera["same_shot_diversity_status"] = "master_coverage"
                camera["same_shot_diversity_review"] = {
                    "status": "master_coverage",
                    "reasons": ["first_camera_in_shot"],
                    "comparisons": [],
                }
                camera["similar_to_camera_name"] = ""
                camera["diversity_reselection_trace"] = []
                camera["editor_recommended_omit"] = False
                persist_camera_package(camera)
                accepted_cameras.append(camera)
                continue

            previous_candidates = [prev.get("selected_candidate") or {} for prev in accepted_cameras if prev.get("selected_candidate")]
            similarity = candidate_similar_to_any(current_candidate, previous_candidates)
            matched = similarity.get("first_match") or {}
            similar_camera = ""
            if matched:
                matched_id = str(matched.get("previous_candidate_id") or "")
                for prev in accepted_cameras:
                    if str((prev.get("selected_candidate") or {}).get("candidate_id") or "") == matched_id:
                        similar_camera = str(prev.get("camera_name") or "")
                        break
            review = {
                "status": "passed",
                "reasons": [],
                "camera_ordinal": camera_ordinal,
                "comparisons": similarity.get("comparisons") or [],
            }
            trace: list[dict[str, Any]] = []
            if not similarity.get("similar"):
                camera["same_shot_diversity_status"] = "passed"
                camera["same_shot_diversity_review"] = review
                camera["similar_to_camera_name"] = ""
                camera["diversity_reselection_trace"] = trace
                camera["editor_recommended_omit"] = False
                persist_camera_package(camera)
                accepted_cameras.append(camera)
                continue

            review["status"] = "duplicate_candidate_detected"
            review["reasons"] = list(matched.get("criteria_matches") or ["same_shot_candidate_too_similar"])
            row = rows_by_camera.get(camera_name) or {}
            if current_candidate_is_protected_semantic_seed(camera, row):
                trace.append(
                    {
                        "current_candidate_id": current_candidate.get("candidate_id"),
                        "similar_to_camera_name": similar_camera,
                        "similarity": matched,
                        "status": "protected_semantic_seed_kept",
                    }
                )
                camera["same_shot_diversity_status"] = "protected_semantic_seed_kept"
                camera["editor_recommended_omit"] = False
                review["status"] = "protected_semantic_seed_kept"
                review["protected_semantic_seed_active"] = True
                review["reasons"].append("protected_semantic_seed_not_reselected")
                camera["same_shot_diversity_review"] = review
                camera["similar_to_camera_name"] = similar_camera
                camera["diversity_reselection_trace"] = trace
                persist_camera_package(camera)
                accepted_cameras.append(camera)
                continue
            replacement = choose_dynamic_face_closeup_candidate(
                row=row,
                camera=camera,
                current_candidate=current_candidate,
            ) or choose_same_shot_diverse_candidate(
                row=row,
                camera=camera,
                current_candidate=current_candidate,
                previous_cameras=accepted_cameras,
                camera_ordinal=camera_ordinal,
            )
            trace.append(
                {
                    "current_candidate_id": current_candidate.get("candidate_id"),
                    "similar_to_camera_name": similar_camera,
                    "similarity": matched,
                    "replacement_candidate_id": (replacement or {}).get("candidate_id"),
                }
            )
            if replacement is not None:
                endpoint = None
                trajectory_choice = normalize_trajectory_preset((camera.get("trajectory_plan") or {}).get("preset_name"))
                if movement_requires_dynamic(camera):
                    endpoint = choose_replacement_endpoint(row=row, camera=camera, start_candidate=replacement)
                    trajectory_choice = normalize_trajectory_preset(
                        preset_for_motion(
                            camera.get("movement_tag"),
                            trajectory_direction_hint(camera, replacement, endpoint),
                            camera.get("motion_profile"),
                        )
                    )
                apply_validated_camera_selection(
                    camera=camera,
                    start_candidate=replacement,
                    endpoint_candidate=endpoint,
                    trajectory_choice=trajectory_choice,
                    selection_source="same_shot_diversity_repair",
                    selection_reason="; ".join(review["reasons"]) or "same_shot_diversity_repair",
                    fps=config.fps,
                )
                replacement_similarity = candidate_similar_to_any(
                    camera.get("selected_candidate") or {},
                    [prev.get("selected_candidate") or {} for prev in accepted_cameras if prev.get("selected_candidate")],
                )
                trace.append(
                    {
                        "after_reselection_candidate_id": (camera.get("selected_candidate") or {}).get("candidate_id"),
                        "still_similar": bool(replacement_similarity.get("similar")),
                        "comparisons": replacement_similarity.get("comparisons") or [],
                    }
                )
                if replacement_similarity.get("similar"):
                    camera["same_shot_diversity_status"] = "limited_survivor_pool"
                    camera["editor_recommended_omit"] = True
                    review["status"] = "limited_survivor_pool"
                    review["reasons"].append("replacement_still_similar")
                else:
                    camera["same_shot_diversity_status"] = "reselected"
                    camera["editor_recommended_omit"] = False
                    review["status"] = "reselected"
                camera["selection_source"] = "same_shot_diversity_repair"
                camera["selection_validity"] = candidate_selection_validity(camera, camera.get("selected_candidate") or {})
                camera["selection_degrade_reason"] = "; ".join(
                    value for value in (camera.get("selection_degrade_reason") or "", "same_shot_diversity_repair") if value
                )
                repairs.append(
                    {
                        "camera_name": camera_name,
                        "replacement_candidate_id": replacement.get("candidate_id"),
                        "status": camera["same_shot_diversity_status"],
                        "similar_to_camera_name": similar_camera,
                        "reasons": review["reasons"],
                    }
                )
            else:
                camera["same_shot_diversity_status"] = "limited_survivor_pool"
                camera["editor_recommended_omit"] = True
                review["status"] = "limited_survivor_pool"
                review["reasons"].append("no_diverse_readable_candidate")
                repairs.append(
                    {
                        "camera_name": camera_name,
                        "replacement_candidate_id": "",
                        "status": "limited_survivor_pool",
                        "similar_to_camera_name": similar_camera,
                        "reasons": review["reasons"],
                    }
                )
            camera["same_shot_diversity_review"] = review
            camera["similar_to_camera_name"] = similar_camera
            camera["diversity_reselection_trace"] = trace
            persist_camera_package(camera)
            accepted_cameras.append(camera)
    return repairs


def run_llm_selection_pipeline(
    *,
    quality_report_payload: dict[str, Any],
    shot_outputs: list[dict[str, Any]],
    config: CinematographerConfig,
    output_root: Path,
) -> dict[str, Any]:
    """Two-phase LLM pipeline: channel filter → merge → final select, parallelized across shots."""
    if not llm_ready(model=config.vision_model, api_key=config.anyllm_api_key):
        print("[llm] LLM not available (no API key or model). Skipping LLM selection.")
        return {"success": False, "error": "llm_unavailable", "rows": []}

    rows_by_camera: dict[str, dict[str, Any]] = {}
    for row in quality_report_payload.get("rows") or []:
        camera_name = str(row.get("camera_name") or "")
        if camera_name:
            rows_by_camera[camera_name] = row

    all_cameras = [cam for shot in shot_outputs for cam in shot.get("cameras") or []]
    max_workers = max(1, min(config.llm_max_workers, len(all_cameras)))
    selection_rows: list[dict[str, Any]] = []

    print(f"[llm] Starting two-phase LLM pipeline for {len(all_cameras)} cameras (max_workers={max_workers})...", flush=True)

    # ── Parallel camera processing ──────────────────────────────────
    phase1_results_by_camera: dict[str, dict[str, dict[str, Any]]] = {}
    eligible_cameras: list[tuple[dict[str, Any], dict[str, Any]]] = []
    phase1_future_map: dict[Any, tuple[str, str]] = {}
    phase2_future_map: dict[Any, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for camera in all_cameras:
            camera_name = str(camera.get("camera_name") or "")
            row = rows_by_camera.get(camera_name)
            if not row:
                selection_rows.append({"camera_name": camera_name, "success": False, "error": "quality_row_missing"})
                continue
            if not row.get("success"):
                selection_rows.append(
                    {
                        "camera_name": camera_name,
                        "success": False,
                        "error": "quality_no_candidate_retained",
                        "candidate_count_raw": int(row.get("candidate_count_raw") or 0),
                        "candidate_count_eligible": int(row.get("candidate_count_eligible") or 0),
                        "candidate_count_retained": int(row.get("candidate_count_retained") or 0),
                        "candidate_board_path": row.get("candidate_board_path") or "",
                        "rejection_report_path": row.get("rejection_report_path") or "",
                    }
                )
                continue
            eligible_cameras.append((camera, row))
            for ch_name, ch_info in (row.get("channel_boards") or {}).items():
                print(f"[llm] Phase-1 queued {camera_name}/{ch_name} ({len(ch_info.get('selected') or [])} candidates)...", flush=True)
                future = executor.submit(
                    llm_filter_channel_board,
                    channel_name=ch_name,
                    channel_info=ch_info,
                    camera_package=camera,
                    config=config,
                )
                phase1_future_map[future] = (camera_name, ch_name)

        for future in as_completed(phase1_future_map):
            camera_name, ch_name = phase1_future_map[future]
            try:
                ch_result = future.result()
            except Exception as exc:
                ch_result = {
                    "channel": ch_name,
                    "success": False,
                    "error": f"phase1_thread_exception: {exc}",
                    "survivors": [],
                    "rejected": [],
                }
            phase1_results_by_camera.setdefault(camera_name, {})[ch_name] = ch_result
            status = "OK" if ch_result.get("success") else "FAIL"
            print(f"[llm] Phase-1 {camera_name}/{ch_name}: {status}", flush=True)

        for camera, row in eligible_cameras:
            future = executor.submit(
                _process_single_camera_llm,
                camera,
                row,
                config,
                output_root,
                phase1_results_by_camera.get(str(camera.get("camera_name") or ""), {}),
            )
            phase2_future_map[future] = camera

        for future in as_completed(phase2_future_map):
            camera = phase2_future_map[future]
            camera_name = str(camera.get("camera_name") or "")
            try:
                result = future.result()
                selection_rows.append(result)
                status = "OK" if result.get("success") else "FAIL"
                print(f"[llm] Phase-2 {camera_name}: {status}", flush=True)
            except Exception as exc:
                selection_rows.append({"camera_name": camera_name, "success": False, "error": f"thread_exception: {exc}"})
                print(f"[llm] Phase-2 {camera_name}: EXCEPTION {exc}", flush=True)

    # ── Story consistency judge (parallel) ──────────────────────────
    pre_story_rows: list[dict[str, Any]] = []

    def _judge_one(cam: dict[str, Any]) -> dict[str, Any]:
        result = llm_story_consistency_judge(camera_package=cam, config=config)
        cam["story_consistency"] = result
        persist_camera_package(cam)
        return {
            "camera_name": cam.get("camera_name"),
            "consistency_score": result.get("consistency_score"),
            "needs_reshoot": result.get("needs_reshoot"),
            "reason": result.get("reason"),
            "preview_path": result.get("preview_path") or "",
            "review_source": result.get("review_source") or "",
        }

    if config.run_pre_continuity_story_judge:
        print(f"[llm] Running pre-continuity story consistency judge on {len(all_cameras)} cameras...", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            story_futures = {executor.submit(_judge_one, cam): cam for cam in all_cameras}
            for future in as_completed(story_futures):
                try:
                    pre_story_rows.append(future.result())
                except Exception as exc:
                    cam = story_futures[future]
                    pre_story_rows.append({"camera_name": cam.get("camera_name"), "error": str(exc)})
    else:
        print("[llm] Pre-continuity story judge skipped; final preview review is the release gate.", flush=True)

    continuity_repairs = repair_scene_continuity(
        shot_outputs=shot_outputs,
        rows_by_camera=rows_by_camera,
        config=config,
    )
    same_shot_diversity_repairs = repair_same_shot_diversity(
        shot_outputs=shot_outputs,
        rows_by_camera=rows_by_camera,
        config=config,
    )

    report = {
        "schema_version": "storyblender.llm_selection_report.v2",
        "generated_at": utc_now(),
        "pipeline": "two_phase_channel_filter",
        "max_workers": max_workers,
        "phase1_execution": "global_bounded_parallel",
        "run_pre_continuity_story_judge": bool(config.run_pre_continuity_story_judge),
        "llm_retry_count": config.llm_retry_count,
        "final_review_retry_count": config.final_review_retry_count,
        "success": all(r.get("success") for r in selection_rows),
        "selection_rows": selection_rows,
        "pre_continuity_story_rows": pre_story_rows,
        "final_preview_story_rows": [],
        "story_consistency_rows": [],
        "continuity_repairs": continuity_repairs,
        "same_shot_diversity_repairs": same_shot_diversity_repairs,
    }
    report_path = save_json(report, output_root / "outputs" / "llm_selection_report_v1.json")
    print(f"[llm] Selection report saved to {report_path}", flush=True)
    return report


def build_camera_package(
    *,
    plan: dict[str, Any],
    shot_context: dict[str, Any],
    shot_contract: dict[str, Any],
    scene_details_map: dict[int, dict[str, Any]],
    output_root: Path,
    config: CinematographerConfig,
) -> dict[str, Any]:
    scene_id = int(plan["scene_id"])
    scene_details = scene_details_map.get(scene_id) or {}
    camera_name = str(plan["camera_name"])
    angle_label = str(plan.get("angle") or "eye-level")
    direction_tag = canonical_direction(plan.get("direction"), angle_label)
    authored_movement = canonical_movement(plan.get("movement"))
    description = str(plan.get("description") or "")
    start_contract = ((shot_contract or {}).get("start_frame_contract") or {})
    if not isinstance(start_contract, dict):
        start_contract = {}
    contract_start_ids = _string_list(start_contract.get("start_focus_ids"))
    contract_primary = str(start_contract.get("primary_focus_id") or "").strip()
    contract_secondary = _string_list(start_contract.get("secondary_focus_ids"))
    contract_secondary_semantics = start_contract.get("secondary_semantic_targets") or {}
    if not isinstance(contract_secondary_semantics, dict):
        contract_secondary_semantics = {}
    keyframe_primary_ids: list[str] = []
    keyframe_plan = (shot_contract or {}).get("keyframe_plan") or []
    if isinstance(keyframe_plan, list):
        for keyframe in keyframe_plan:
            if isinstance(keyframe, dict):
                keyframe_primary_ids.extend(_string_list(keyframe.get("primary_focus_id")))
    if contract_start_ids:
        start_primary = contract_start_ids[0]
        keyframe_votes = sum(1 for item in keyframe_primary_ids if item == start_primary)
        has_keyframe_consensus = bool(keyframe_primary_ids) and keyframe_votes >= max(1, len(keyframe_primary_ids) // 2 + 1)
        if contract_primary and contract_primary != start_primary and has_keyframe_consensus:
            old_primary = contract_primary
            contract_primary = start_primary
            if old_primary and old_primary not in contract_secondary:
                contract_secondary = [old_primary] + contract_secondary
    plan_focus_ids = _string_list(plan.get("focus_ids"))
    plan_primary = str(plan.get("primary_focus_id") or "").strip()
    plan_secondary = _string_list(plan.get("secondary_focus_ids"))
    if contract_primary:
        effective_primary_ids = [contract_primary]
        contract_start_secondary = [item for item in contract_start_ids if item != contract_primary]
    elif contract_start_ids:
        effective_primary_ids = contract_start_ids[:1]
        contract_start_secondary = contract_start_ids[1:]
    else:
        effective_primary_ids = []
        contract_start_secondary = []
    effective_primary_ids = effective_primary_ids or ([plan_primary] if plan_primary else []) or plan_focus_ids[:1]
    effective_secondary_ids = []
    for focus_id in contract_secondary + contract_start_secondary:
        if focus_id and focus_id not in effective_secondary_ids and focus_id not in effective_primary_ids:
            effective_secondary_ids.append(focus_id)
    if not effective_secondary_ids:
        effective_secondary_ids = plan_secondary or plan_focus_ids[1:]
    shot_text = " ".join(str(value or "") for value in (description, shot_context.get("shot_description"), shot_context.get("scene_description"))).lower()
    if effective_primary_ids and not likely_human_focus_id(effective_primary_ids[0]):
        human_secondaries = [
            focus_id for focus_id in effective_secondary_ids
            if likely_human_focus_id(focus_id)
            and str(contract_secondary_semantics.get(focus_id) or "").lower() in {"face", "front", "full_body"}
        ]
        if human_secondaries and any(term in shot_text for term in ("look", "reaction", "expression", "face", "stoic")):
            promoted = human_secondaries[0]
            effective_secondary_ids = [effective_primary_ids[0]] + [item for item in effective_secondary_ids if item != promoted]
            effective_primary_ids = [promoted]
    effective_focus_ids: list[str] = []
    for focus_id in effective_primary_ids:
        if focus_id and focus_id not in effective_focus_ids:
            effective_focus_ids.append(focus_id)
    if not effective_focus_ids:
        for focus_id in plan_focus_ids[:1] + effective_secondary_ids[:1]:
            if focus_id and focus_id not in effective_focus_ids:
                effective_focus_ids.append(focus_id)
    effective_primary = effective_primary_ids[0] if effective_primary_ids else (effective_focus_ids[0] if effective_focus_ids else plan_primary)
    effective_secondary_ids = [focus_id for focus_id in effective_secondary_ids if focus_id and focus_id != effective_primary]
    resolved_shot_contract = json.loads(json.dumps(shot_contract or {}))
    resolved_start_contract = dict((resolved_shot_contract.get("start_frame_contract") or {}) if isinstance(resolved_shot_contract, dict) else {})
    original_contract_primary = str(resolved_start_contract.get("primary_focus_id") or "").strip()
    original_contract_start_ids = _string_list(resolved_start_contract.get("start_focus_ids"))
    resolved_start_contract["primary_focus_id"] = effective_primary
    resolved_start_contract["start_focus_ids"] = [effective_primary] if effective_primary else []
    resolved_start_contract["secondary_focus_ids"] = list(effective_secondary_ids)
    secondary_semantic_targets = resolved_start_contract.get("secondary_semantic_targets") or {}
    if not isinstance(secondary_semantic_targets, dict):
        secondary_semantic_targets = {}
    secondary_semantic_targets = {
        str(focus_id): str(target)
        for focus_id, target in secondary_semantic_targets.items()
        if str(focus_id) != effective_primary
    }
    for focus_id in effective_secondary_ids:
        secondary_semantic_targets.setdefault(focus_id, "full_body")
    resolved_start_contract["secondary_semantic_targets"] = secondary_semantic_targets
    if isinstance(resolved_shot_contract, dict):
        resolved_shot_contract["start_frame_contract"] = resolved_start_contract
        repaired_keyframes: list[dict[str, Any]] = []
        for keyframe in resolved_shot_contract.get("keyframe_plan") or []:
            if not isinstance(keyframe, dict):
                continue
            row = dict(keyframe)
            keyframe_original_primary = str(row.get("primary_focus_id") or "").strip()
            row["primary_focus_id"] = effective_primary
            row_secondaries: list[str] = []
            for focus_id in _string_list(row.get("secondary_focus_ids")) + ([keyframe_original_primary] if keyframe_original_primary else []) + effective_secondary_ids:
                if focus_id and focus_id != effective_primary and focus_id not in row_secondaries:
                    row_secondaries.append(focus_id)
            row["secondary_focus_ids"] = row_secondaries
            row_targets = row.get("secondary_semantic_targets") or {}
            if not isinstance(row_targets, dict):
                row_targets = {}
            row_targets = {
                str(focus_id): str(target)
                for focus_id, target in row_targets.items()
                if str(focus_id) != effective_primary
            }
            for focus_id in row_secondaries:
                row_targets.setdefault(focus_id, "full_body")
            row["secondary_semantic_targets"] = row_targets
            repaired_keyframes.append(row)
        if repaired_keyframes:
            resolved_shot_contract["keyframe_plan"] = repaired_keyframes
        if original_contract_primary and original_contract_primary != effective_primary:
            resolved_shot_contract["focus_consistency_repair"] = {
                "applied": True,
                "original_primary_focus_id": original_contract_primary,
                "canonical_primary_focus_id": effective_primary,
                "original_start_focus_ids": original_contract_start_ids,
                "source": "cinematographer_stage_defensive_repair",
            }
    movement_tag = authored_movement
    motion_profile = "authored"
    closeup = is_closeup(str(plan.get("distance") or ""))
    if closeup:
        movement_tag = "push_out"
        magnitude = 0.02
        motion_profile = "closeup_static"
    elif movement_tag in {"static", "locked_off"} and not any(
        phrase in description.lower() for phrase in ("locked", "lock-off", "lock off", "still frame", "fixed frame")
    ):
        movement_tag, magnitude = subtle_motion(plan.get("distance"))
        motion_profile = "semantic_light_dynamic"
    else:
        magnitude = 0.04 if movement_tag in {"push_in", "push_out"} else 0.05
        if movement_tag in {"static", "locked_off"}:
            magnitude = 0.0
    start_window = frame_window(
        focus_ids=effective_focus_ids,
        scene_details=scene_details,
        distance_label=str(plan.get("distance") or "medium shot"),
    )
    movement_direction = direction_tag
    if movement_tag in {"pan", "truck", "orbit"}:
        movement_direction = direction_tag if direction_tag in {"left", "right"} else "left"
    end_window = apply_movement_to_window(start_window, movement_tag, magnitude, movement_direction)
    start_transform = camera_transform(
        focus_ids=effective_focus_ids,
        scene_details=scene_details,
        distance_label=str(plan.get("distance") or "medium shot"),
        angle_label=angle_label,
        direction_tag=direction_tag,
        frame=start_window,
    )
    end_transform = camera_transform(
        focus_ids=effective_focus_ids,
        scene_details=scene_details,
        distance_label=str(plan.get("distance") or "medium shot"),
        angle_label=angle_label,
        direction_tag=direction_tag,
        frame=end_window,
    )
    trajectory_preset = preset_for_motion(movement_tag, movement_direction, motion_profile)
    duration_seconds = trajectory_duration_seconds(trajectory_preset)
    frame_count = max(24, int(round(duration_seconds * config.fps)))
    shot_dir = ensure_directory(output_root / "camera_packages" / f"scene_{scene_id}_shot_{plan['shot_id']}")
    plan_payload = {
        "scene_id": scene_id,
        "shot_id": int(plan["shot_id"]),
        "camera_name": camera_name,
        "camera_index": int(plan.get("camera_index") or 0),
        "camera_role": plan.get("camera_role"),
        "cut_reason": plan.get("cut_reason"),
        "scene_description": shot_context.get("scene_description"),
        "shot_description": shot_context.get("shot_description"),
        "primary_focus_id": effective_primary,
        "secondary_focus_ids": list(effective_secondary_ids),
        "focus_ids": effective_focus_ids,
        "original_primary_focus_id": plan.get("primary_focus_id"),
        "original_focus_ids": list(plan.get("focus_ids") or []),
        "focus_conflict_resolved": bool(effective_focus_ids != plan_focus_ids),
        "distance_label": plan.get("distance"),
        "angle_label": angle_label,
        "authored_movement": authored_movement,
        "movement_tag": movement_tag,
        "direction_tag": direction_tag,
        "motion_profile": motion_profile,
        "motion_magnitude": magnitude,
        "lens_mm": plan.get("lens_mm") or lens_mm_for_distance(plan.get("distance")),
        "target_duration_seconds": round(duration_seconds, 3),
        "target_frame_count": frame_count,
        "start_transform": start_transform,
        "end_transform": end_transform,
        "shot_contract": resolved_shot_contract,
        "primary_semantic_target": (resolved_start_contract or {}).get("primary_semantic_target"),
        "secondary_semantic_targets": (resolved_start_contract or {}).get("secondary_semantic_targets") or {},
        "selected_candidate": plan.get("selected_candidate") or {},
        "top_candidates": plan.get("top_candidates") or [],
        "render_brief": {
            "frame_width": config.frame_width,
            "frame_height": config.frame_height,
            "plate_width": config.plate_width,
            "plate_height": config.plate_height,
            "start_frame_window": start_window,
            "end_frame_window": end_window,
            "easing": "ease_in_out",
            "reference_image_paths": [
                value
                for value in ((plan.get("selected_candidate") or {}).values())
                if isinstance(value, str) and value.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
        },
    }
    plan_payload["trajectory_keyframes"] = build_trajectory_keyframes(
        start_transform=start_transform,
        end_transform=end_transform,
        movement_tag=movement_tag,
        motion_profile=motion_profile,
        frame_count=frame_count,
        fps=config.fps,
        preset_name=trajectory_preset,
    )
    if plan_payload["trajectory_keyframes"]:
        plan_payload["end_transform"] = dict(plan_payload["trajectory_keyframes"][-1].get("transform") or end_transform)
    plan_payload["trajectory_plan"] = build_camera_trajectory_plan(
        camera=plan_payload,
        preset_name=trajectory_preset,
        start_transform=start_transform,
        end_transform=plan_payload["end_transform"],
        start_candidate=plan.get("selected_candidate") or {},
        endpoint_candidate=None,
        selection_source="authored_camera_plan",
        reason=f"authored movement {movement_tag}",
        fps=config.fps,
    )
    plan_payload["trajectory_selection_source"] = "authored_camera_plan"
    plan_payload["trajectory_safety_report"] = plan_payload["trajectory_plan"].get("safety_report") or {}
    plan_payload["editor_handles_seconds"] = {"head": 0.25, "tail": 0.25}
    plan_payload["source_plate_path"] = ""
    plan_payload["preview_frame_path"] = ""
    plan_payload["final_preview_path"] = ""
    plan_payload["llm_input_board_path"] = ""
    plan_payload["llm_selected_candidate_preview_path"] = ""
    plan_payload["final_preview_llm_review"] = {}
    plan_payload["final_preview_repair_trace"] = []
    plan_payload["same_shot_diversity_review"] = {}
    plan_payload["same_shot_diversity_status"] = ""
    plan_payload["similar_to_camera_name"] = ""
    plan_payload["diversity_reselection_trace"] = []
    plan_payload["editor_recommended_omit"] = False
    plan_payload["downstream_eligible"] = True
    package_path = save_json(plan_payload, shot_dir / f"{camera_name}.json")
    plan_payload["camera_package_path"] = str(package_path)
    save_json(plan_payload, package_path)
    return plan_payload


def run_cinematographer(config: CinematographerConfig) -> dict[str, Any]:
    _set_ablation_flags(config)
    print(
        f"[ablation] flags: vlm_reflection={'OFF' if config.disable_vlm_reflection else 'ON'} "
        f"trajectory_grounding={'OFF' if config.disable_trajectory_grounding else 'ON'} "
        f"semantic_height_adjust={'OFF' if config.disable_semantic_height_adjust else 'ON'} "
        f"pre_continuity_story_judge={'ON' if config.run_pre_continuity_story_judge else 'OFF'} "
        f"camera_quality={config.camera_quality}",
        flush=True,
    )
    output_root = ensure_directory(Path(config.output_root).resolve())
    outputs_dir = ensure_directory(output_root / "outputs")
    director_handoff_path = Path(config.director_handoff_path).resolve()
    handoff, blocking_plans = load_director_context(director_handoff_path)
    shot_contracts_path = Path(str((handoff.get("files") or {}).get("shot_contracts_path") or ""))
    shot_contracts = load_json(shot_contracts_path) if shot_contracts_path.exists() else []
    contracts_by_camera = contract_lookup(shot_contracts)
    scene_details_map = {int(key): value for key, value in (handoff.get("scene_details") or {}).items()} if isinstance(handoff.get("scene_details"), dict) else {}
    if not scene_details_map:
        scene_details_map = {
            int(item.get("scene_id")): item
            for item in handoff.get("scene_index") or []
            if item.get("scene_id") is not None
        }
    shot_sequence = handoff.get("shot_sequence") or []
    plans_by_camera = {str(item.get("camera_name") or ""): item for item in blocking_plans}
    shot_outputs: list[dict[str, Any]] = []
    camera_rows: list[dict[str, Any]] = []

    for shot_context in shot_sequence:
        cameras: list[dict[str, Any]] = []
        for camera_name in shot_context.get("camera_names") or []:
            plan = plans_by_camera.get(str(camera_name))
            if not plan:
                continue
            camera_package = build_camera_package(
                plan=plan,
                shot_context=shot_context,
                shot_contract=contracts_by_camera.get(
                    (int(plan.get("scene_id") or 0), int(plan.get("shot_id") or 0), str(plan.get("camera_name") or ""))
                )
                or {},
                scene_details_map=scene_details_map,
                output_root=output_root,
                config=config,
            )
            cameras.append(camera_package)
            camera_rows.append(
                {
                    "scene_id": camera_package["scene_id"],
                    "shot_id": camera_package["shot_id"],
                    "camera_name": camera_package["camera_name"],
                    "movement_tag": camera_package["movement_tag"],
                    "motion_profile": camera_package["motion_profile"],
                    "source_plate_path": camera_package["source_plate_path"],
                    "preview_frame_path": camera_package["preview_frame_path"],
                }
            )
        bridges = [build_bridge(cameras[index], cameras[index + 1]) for index in range(len(cameras) - 1)]
        for index, camera in enumerate(cameras):
            camera["bridge_to_next"] = bridges[index] if index < len(bridges) else None
        shot_outputs.append(
            {
                "scene_id": shot_context["scene_id"],
                "shot_id": shot_context["shot_id"],
                "scene_description": shot_context.get("scene_description"),
                "shot_description": shot_context.get("shot_description"),
                "duration_target_seconds": shot_context.get("duration_target_seconds"),
                "bridge_trajectories": bridges,
                "cameras": cameras,
            }
        )

    quality_result: dict[str, Any] = {}
    quality_report_payload: dict[str, Any] = {}
    llm_selection_result: dict[str, Any] = {}
    if config.camera_quality == "quality":
        preliminary_handoff_path = save_json(
            {
                "schema_version": "storyblender.camera_handoff.v1",
                "generated_at": utc_now(),
                "run_id": config.run_id or output_root.name,
                "director_handoff_path": str(director_handoff_path),
                "fps": config.fps,
                "frame_size": {"width": config.frame_width, "height": config.frame_height},
                "plate_size": {"width": config.plate_width, "height": config.plate_height},
                "camera_quality": config.camera_quality,
                "shots": shot_outputs,
                "cameras": [camera for shot in shot_outputs for camera in shot["cameras"]],
            },
            outputs_dir / "camera_handoff_quality_input_v1.json",
        )
        existing_quality_report_path = outputs_dir / "camera_quality_report_v1.json"
        if config.resume_existing and existing_quality_report_path.exists():
            quality_result = {
                "success": True,
                "quality_report_path": str(existing_quality_report_path),
                "resumed_existing_quality_report": True,
            }
            print(f"[quality] Resuming from existing report: {existing_quality_report_path}", flush=True)
        else:
            quality_result = run_quality_candidate_worker(
                config=config,
                director_handoff=handoff,
                camera_handoff_path=preliminary_handoff_path,
                output_root=output_root,
            )
        quality_report_path = Path(str(quality_result.get("quality_report_path") or ""))
        if not quality_result.get("success") or not quality_report_path.exists():
            raise RuntimeError(
                "Cinematographer quality candidate worker failed. "
                f"See {quality_result.get('stderr_path') or (output_root / 'camera_quality_stderr.log')}"
            )
        quality_report_payload = load_json(quality_report_path)
        apply_quality_results(shot_outputs, quality_report_payload, config)

        llm_selection_result = run_llm_selection_pipeline(
            quality_report_payload=quality_report_payload,
            shot_outputs=shot_outputs,
            config=config,
            output_root=output_root,
        )
        quality_result["llm_selection_result"] = llm_selection_result

    preview_result = render_and_apply_previews(
        shot_outputs=shot_outputs,
        camera_rows=camera_rows,
        config=config,
        director_handoff=handoff,
        director_handoff_path=director_handoff_path,
        output_root=output_root,
        outputs_dir=outputs_dir,
        handoff_filename="camera_handoff_preview_input_v1.json",
    )
    quality_result["preview_result"] = preview_result

    rows_by_camera = {
        str(row.get("camera_name") or ""): row
        for row in quality_report_payload.get("rows") or []
        if row.get("camera_name")
    }
    final_preview_review_result = run_final_preview_review_pipeline(
        shot_outputs=shot_outputs,
        rows_by_camera=rows_by_camera,
        camera_rows=camera_rows,
        config=config,
        director_handoff=handoff,
        director_handoff_path=director_handoff_path,
        output_root=output_root,
        outputs_dir=outputs_dir,
    )
    quality_result["final_preview_review_result"] = final_preview_review_result

    selection_rows = llm_selection_result.get("selection_rows") or []
    selection_rows_by_camera = {str(row.get("camera_name") or ""): row for row in selection_rows if row.get("camera_name")}
    for shot in shot_outputs:
        for camera in shot.get("cameras") or []:
            camera_name = str(camera.get("camera_name") or "")
            row = selection_rows_by_camera.get(camera_name)
            if row is None:
                continue
            row["board_selection_decision"] = row.get("selection_decision") or {}
            row["llm_board_history"] = camera.get("llm_board_history") or row.get("llm_board_history") or []
            row["llm_input_board_path"] = camera.get("llm_input_board_path") or row.get("llm_input_board_path") or ""
            row["board_selected_candidate_id"] = row.get("selected_candidate_id") or ""
            row["board_selected_candidate_preview_path"] = row.get("selected_candidate_preview_path") or ""
            row["selected_candidate_preview_path"] = camera.get("llm_selected_candidate_preview_path") or row.get("selected_candidate_preview_path") or ""
            row["final_preview_path"] = camera.get("final_preview_path") or camera.get("preview_frame_path") or ""
            row["final_preview_review"] = camera.get("final_preview_llm_review") or {}
            row["final_preview_review_source"] = (camera.get("final_preview_llm_review") or {}).get("review_source") or ""
            row["final_preview_repair_trace"] = camera.get("final_preview_repair_trace") or []
            final_candidate = camera.get("selected_candidate") or {}
            row["final_selected_candidate_id"] = final_candidate.get("candidate_id")
            row["final_selected_candidate_preview_path"] = final_candidate.get("preview_image_path") or ""
            row["final_render_source_candidate_id"] = camera.get("final_render_source_candidate_id") or final_candidate.get("candidate_id") or ""
            row["final_render_source_candidate_preview_path"] = (
                camera.get("final_render_source_candidate_preview_path")
                or final_candidate.get("preview_image_path")
                or ""
            )
            row["protected_semantic_seed_active"] = bool(camera.get("protected_semantic_seed_active"))
            row["protected_semantic_seed_candidate_id"] = camera.get("protected_semantic_seed_candidate_id") or ""
            row["protected_semantic_seed_candidate_preview_path"] = camera.get("protected_semantic_seed_candidate_preview_path") or ""
            row["downstream_eligible"] = bool(camera.get("downstream_eligible", True))
            row["same_shot_diversity_review"] = camera.get("same_shot_diversity_review") or {}
            row["same_shot_diversity_status"] = camera.get("same_shot_diversity_status") or ""
            row["similar_to_camera_name"] = camera.get("similar_to_camera_name") or ""
            row["diversity_reselection_trace"] = camera.get("diversity_reselection_trace") or []
            row["editor_recommended_omit"] = bool(camera.get("editor_recommended_omit"))

    if llm_selection_result:
        llm_selection_result["final_preview_story_rows"] = final_preview_review_result.get("rows") or []
        llm_selection_result["story_consistency_rows"] = final_preview_review_result.get("rows") or []
        llm_selection_result["final_preview_repaired_cameras"] = final_preview_review_result.get("repaired_camera_names") or []
        llm_selection_result["final_preview_blocked_cameras"] = final_preview_review_result.get("blocked_camera_names") or []

    same_shot_handoff_motion_result = repair_same_shot_handoff_motion(shot_outputs, fps=config.fps)
    quality_result["same_shot_handoff_motion_result"] = same_shot_handoff_motion_result
    if llm_selection_result:
        llm_selection_result["same_shot_handoff_motion_result"] = same_shot_handoff_motion_result
        llm_report_path = save_json(llm_selection_result, outputs_dir / "llm_selection_report_v1.json")
        quality_result["llm_selection_report_path"] = str(llm_report_path)

    downstream_shots = build_downstream_shots(shot_outputs)
    downstream_cameras = [camera for shot in downstream_shots for camera in shot["cameras"]]

    ablation_flags_payload = {
        "disable_vlm_reflection": bool(config.disable_vlm_reflection),
        "disable_trajectory_grounding": bool(config.disable_trajectory_grounding),
        "disable_semantic_height_adjust": bool(config.disable_semantic_height_adjust),
        "run_pre_continuity_story_judge": bool(config.run_pre_continuity_story_judge),
        "camera_quality": config.camera_quality,
    }
    camera_handoff_path = save_json(
        {
            "schema_version": "storyblender.camera_handoff.v1",
            "generated_at": utc_now(),
            "run_id": config.run_id or output_root.name,
            "director_handoff_path": str(director_handoff_path),
            "fps": config.fps,
            "frame_size": {"width": config.frame_width, "height": config.frame_height},
            "plate_size": {"width": config.plate_width, "height": config.plate_height},
            "camera_quality": config.camera_quality,
            "ablation_flags": ablation_flags_payload,
            "quality_report_path": str(quality_result.get("quality_report_path") or ""),
            "quality_summary": {
                "candidate_count_raw_min": min((int(row.get("candidate_count_raw") or 0) for row in quality_report_payload.get("rows") or []), default=0),
                "candidate_count_retained_min": min((int(row.get("candidate_count_retained") or 0) for row in quality_report_payload.get("rows") or []), default=0),
                "board_count": sum(1 for row in quality_report_payload.get("rows") or [] if Path(str(row.get("candidate_board_path") or "")).exists()),
            },
            "downstream_blocked_camera_names": final_preview_review_result.get("blocked_camera_names") or [],
            "shots": downstream_shots,
            "cameras": downstream_cameras,
        },
        outputs_dir / "camera_handoff_v1.json",
    )
    report_path = save_json(
        {
            "schema_version": "storyblender.camera_shot_report.v1",
            "generated_at": utc_now(),
            "rows": camera_rows,
        },
        outputs_dir / "camera_shot_report_v1.json",
    )
    manifest_path = save_json(
        {
            "schema_version": "storyblender.cinematographer_manifest.v1",
            "generated_at": utc_now(),
            "success": True,
            "run_id": config.run_id or output_root.name,
            "director_handoff_path": str(director_handoff_path),
            "output_root": str(output_root),
            "camera_handoff_path": str(camera_handoff_path),
            "camera_shot_report_path": str(report_path),
            "camera_quality": config.camera_quality,
            "ablation_flags": ablation_flags_payload,
            "quality_result": quality_result,
        },
        output_root / "manifest.json",
    )
    return {
        "success": True,
        "manifest_path": str(manifest_path),
        "camera_handoff_path": str(camera_handoff_path),
        "camera_shot_report_path": str(report_path),
    }


def main() -> int:
    config = parse_args()
    result = run_cinematographer(config)
    print("Cinematographer completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
