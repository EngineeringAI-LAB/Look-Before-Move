from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from director_engine_blender import run_blender_python_script
from director_engine_blocking import build_scene_blocking_plans
from director_scene_context_builder import build_scene_context_request_payload
from director_scene_dossier import load_scene_dossier_manifest
from director_scene_understanding import generate_scene_understanding
from director_shot_contract import generate_scene_shot_contracts


STAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = STAGE_DIR.parent
CODE_DIR = WORKSPACE_DIR.parent
PIPELINE_DIR = CODE_DIR.parent
CAMERA_DIR = PIPELINE_DIR.parent
DEFAULT_DEMO_ROOT = CAMERA_DIR / "scripts" / "demo"
DEFAULT_BLENDER_EXE = Path(os.getenv("STORYBLENDER_BLENDER_EXE", "blender"))
CONFIG_DIR = WORKSPACE_DIR / "config"
CONFIG_CANDIDATE_PATHS = [
    CONFIG_DIR / "runtime_config.json",
    CONFIG_DIR / "storyblender_runtime_config.json",
    CONFIG_DIR / "local_config.json",
]
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "his",
    "her",
    "into",
    "its",
    "of",
    "on",
    "out",
    "the",
    "their",
    "them",
    "then",
    "to",
    "with",
}
WEAK_OBJECT_ALIAS_TOKENS = {
    "area",
    "ceremony",
    "ceremonies",
    "church",
    "city",
    "large",
    "ornate",
    "outside",
    "room",
    "scene",
    "stone",
    "street",
    "used",
}
NONHUMAN_FOCUS_TOKENS = (
    "altar",
    "bed",
    "car",
    "cart",
    "desk",
    "door",
    "font",
    "gun",
    "lamp",
    "pistol",
    "shotgun",
    "table",
    "wall",
    "weapon",
    "window",
)
HUMAN_PRIMARY_CUE_PHRASES = (
    "close-up",
    "close up",
    "expression",
    "face",
    "looks on",
    "reaction",
    "stares",
)
HUMAN_PRIMARY_CUE_TOKENS = {
    "calm",
    "eyes",
    "glares",
    "look",
    "looks",
    "recoils",
    "screams",
    "stoic",
    "weeping",
    "whispers",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def save_json(payload: Any, path: Path) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def default_output_root(run_id: str | None = None) -> Path:
    suffix = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return STAGE_DIR / "output" / suffix


def parse_scene_ids(scene_ids: str) -> list[int]:
    result: list[int] = []
    for raw in str(scene_ids or "").replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value not in result:
            result.append(value)
    return result


def tokenize(text: Any) -> list[str]:
    return WORD_RE.findall(str(text or "").lower())


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, list) else [value]
    rows: list[str] = []
    for item in raw_values:
        text = str(item or "").strip()
        if text and text not in rows:
            rows.append(text)
    return rows


def likely_human_focus_id(asset_id: str, asset_entry: dict[str, Any] | None = None) -> bool:
    asset_type = str((asset_entry or {}).get("asset_type") or "").lower()
    if asset_type:
        return asset_type == "character"
    text = str(asset_id or "").lower()
    return bool(text) and not any(token in text for token in NONHUMAN_FOCUS_TOKENS)


def human_primary_requested(shot_description: str) -> bool:
    text = str(shot_description or "").lower()
    tokens = set(tokenize(text))
    return any(phrase in text for phrase in HUMAN_PRIMARY_CUE_PHRASES) or bool(tokens & HUMAN_PRIMARY_CUE_TOKENS)


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
    if "front" in text:
        return "front"
    return "front"


def canonical_movement(raw_movement: Any) -> str:
    text = str(raw_movement or "").strip().lower().replace("-", " ").replace("_", " ")
    if not text:
        return "static"
    if "lock" in text or "still" in text:
        return "locked_off"
    if "push in" in text or "dolly in" in text or text == "push":
        return "push_in"
    if "push out" in text or "dolly out" in text:
        return "push_out"
    if "zoom in" in text:
        return "push_in"
    if "zoom out" in text:
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


def find_latest_versioned_json(directory: Path, prefix: str) -> Path | None:
    matches: list[tuple[int, Path]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.json$")
    for path in directory.glob(f"{prefix}_v*.json"):
        hit = pattern.match(path.name)
        if hit:
            matches.append((int(hit.group(1)), path))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1].name))
    return matches[-1][1].resolve()


def select_latest_versioned_json(directory: Path, prefix: str) -> Path:
    selected = find_latest_versioned_json(directory, prefix)
    if selected is None:
        raise FileNotFoundError(f"No versioned JSON matching {prefix}_v*.json found in {directory}")
    return selected


def select_latest_versioned_json_with_fallback(directory: Path, prefix: str, fallback_prefix: str) -> tuple[Path, str]:
    selected = find_latest_versioned_json(directory, prefix)
    if selected is not None:
        return selected, prefix
    selected = find_latest_versioned_json(directory, fallback_prefix)
    if selected is not None:
        return selected, fallback_prefix
    raise FileNotFoundError(
        f"No versioned JSON matching {prefix}_v*.json or {fallback_prefix}_v*.json found in {directory}"
    )


def resolve_demo_sources(demo_root: Path) -> dict[str, Any]:
    animated_models_dir = demo_root / "animated_models"
    layout_script_dir = demo_root / "layout_script"
    formatted_model_dir = demo_root / "formatted_model"

    story_source_path = select_latest_versioned_json(animated_models_dir, "animated_models")
    layout_source_path = select_latest_versioned_json(layout_script_dir, "layout_script")
    formatted_model_path = (formatted_model_dir / "formatted_model.json").resolve()
    dimension_estimation_path = (formatted_model_dir / "dimension_estimation.json").resolve()
    selected_animation_path, selected_animation_source_kind = select_latest_versioned_json_with_fallback(
        animated_models_dir,
        "selected_animation",
        "animation_plan",
    )
    blend_paths = sorted((path.resolve() for path in demo_root.glob("*.blend*")), key=lambda item: item.name)

    return {
        "demo_root": str(demo_root),
        "approved_story_source_path": str(story_source_path),
        "layout_source_path": str(layout_source_path),
        "formatted_model_source_path": str(formatted_model_path) if formatted_model_path.exists() else "",
        "dimension_estimation_source_path": str(dimension_estimation_path) if dimension_estimation_path.exists() else "",
        "selected_animation_source_path": str(selected_animation_path),
        "selected_animation_source_kind": selected_animation_source_kind,
        "blend_paths": [str(path) for path in blend_paths],
    }


def deep_merge_truthy(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_truthy(result[key], value)
            continue
        if value in (None, "", [], {}):
            result.setdefault(key, value)
            continue
        result[key] = value
    return result


def merge_asset_sheets(*asset_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for asset_list in asset_lists:
        for asset in asset_list or []:
            asset_id = str(asset.get("asset_id") or "").strip()
            if not asset_id:
                continue
            if asset_id not in merged:
                merged[asset_id] = {}
                order.append(asset_id)
            merged[asset_id] = deep_merge_truthy(merged[asset_id], asset)
    return [merged[asset_id] for asset_id in order]


def resolve_asset_file(
    *,
    raw_value: Any,
    demo_root: Path,
    asset_id: str,
    view_label: str,
) -> str:
    search_roots = [
        demo_root,
        demo_root / "formatted_model",
        demo_root / "animated_models",
        demo_root / "layout_script",
    ]
    candidates: list[Path] = []
    raw_text = str(raw_value or "").strip()
    if raw_text:
        raw_path = Path(raw_text)
        candidates.append(raw_path)
        candidates.append(Path(raw_path.name))
        if raw_path.is_absolute():
            candidates.append(demo_root / raw_path.name)
        else:
            candidates.append(demo_root / raw_path)
            candidates.append(demo_root / raw_path.name)
        for root in search_roots:
            candidates.append(root / raw_path)
            candidates.append(root / raw_path.name)
    suffixes = {
        "front": [f"{asset_id}_front_view.png", f"{asset_id}_front_view.jpg", f"{asset_id}.png", f"{asset_id}.jpg"],
        "top": [f"{asset_id}_top_view.png", f"{asset_id}_top_view.jpg", f"{asset_id}.png", f"{asset_id}.jpg"],
        "left": [f"{asset_id}_left_view.png", f"{asset_id}_left_view.jpg", f"{asset_id}.png", f"{asset_id}.jpg"],
        "thumbnail": [f"{asset_id}.png", f"{asset_id}.jpg", f"{asset_id}_front_view.png", f"{asset_id}_front_view.jpg"],
    }
    for filename in suffixes.get(view_label, []):
        for root in search_roots:
            candidates.append(root / filename)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def build_asset_index(asset_sheet: list[dict[str, Any]], demo_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for asset in asset_sheet:
        asset_id = str(asset.get("asset_id") or "").strip()
        if not asset_id:
            continue
        image_paths = {
            "front": resolve_asset_file(raw_value=asset.get("front_view_url"), demo_root=demo_root, asset_id=asset_id, view_label="front"),
            "top": resolve_asset_file(raw_value=asset.get("top_view_url"), demo_root=demo_root, asset_id=asset_id, view_label="top"),
            "left": resolve_asset_file(raw_value=asset.get("left_view_url"), demo_root=demo_root, asset_id=asset_id, view_label="left"),
            "thumbnail": resolve_asset_file(raw_value=asset.get("thumbnail_url"), demo_root=demo_root, asset_id=asset_id, view_label="thumbnail"),
        }
        index[asset_id] = {
            "asset_id": asset_id,
            "asset_type": asset.get("asset_type"),
            "description": asset.get("description"),
            "dimensions": {
                "width": asset.get("width"),
                "depth": asset.get("depth"),
                "height": asset.get("height"),
            },
            "image_paths": image_paths,
            "source_fields": {
                "main_file_path": str(asset.get("main_file_path") or ""),
                "rigged_file_path": str(asset.get("rigged_file_path") or ""),
                "rigged_running_file_path": str(asset.get("rigged_running_file_path") or ""),
            },
        }
    return index


def contract_focus_ids(shot_contract: dict[str, Any] | None) -> list[str]:
    if not isinstance(shot_contract, dict):
        return []
    start_contract = shot_contract.get("start_frame_contract") or {}
    if not isinstance(start_contract, dict):
        return []
    primary = str(start_contract.get("primary_focus_id") or "").strip()
    start_ids = string_list(start_contract.get("start_focus_ids"))
    secondary_ids = string_list(start_contract.get("secondary_focus_ids"))
    rows: list[str] = []
    for focus_id in ([primary] if primary else []) + start_ids + secondary_ids:
        if focus_id and focus_id not in rows:
            rows.append(focus_id)
    return rows


def camera_instruction_with_contract_focus(
    camera_instruction: dict[str, Any],
    shot_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    focus_ids = contract_focus_ids(shot_contract)
    if not focus_ids:
        return dict(camera_instruction)
    row = dict(camera_instruction)
    original_focus_ids = string_list(row.get("focus_on_ids"))
    row["focus_on_ids"] = focus_ids
    row["primary_focus_id"] = focus_ids[0]
    row["secondary_focus_ids"] = focus_ids[1:]
    if original_focus_ids != focus_ids:
        row["focus_contract_override"] = {
            "applied": True,
            "original_focus_on_ids": original_focus_ids,
            "contract_focus_ids": focus_ids,
            "contract_focus_consistency_repair": (shot_contract or {}).get("focus_consistency_repair") or {},
        }
    return row


def build_storyboard_maps(payload: dict[str, Any]) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    scene_map: dict[int, str] = {}
    shot_map: dict[tuple[int, int], str] = {}
    for scene in payload.get("storyboard_outline") or []:
        try:
            scene_id = int(scene.get("scene_id"))
        except (TypeError, ValueError):
            continue
        scene_map[scene_id] = str(scene.get("scene_description") or "").strip()
        for shot in scene.get("shots") or []:
            try:
                shot_id = int(shot.get("shot_id"))
            except (TypeError, ValueError):
                continue
            shot_map[(scene_id, shot_id)] = str(shot.get("shot_description") or "").strip()
    return scene_map, shot_map


def build_scene_maps(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    scene_index: dict[int, dict[str, Any]] = {}
    for scene in payload.get("scene_details") or []:
        try:
            scene_id = int(scene.get("scene_id"))
        except (TypeError, ValueError):
            continue
        scene_setup = scene.get("scene_setup") or {}
        layout = scene_setup.get("layout_description") or {}
        scene_index[scene_id] = {
            "scene_id": scene_id,
            "scene_type": scene_setup.get("scene_type"),
            "asset_ids": list(scene_setup.get("asset_ids") or []),
            "layout_description": layout.get("description"),
            "scene_size": layout.get("scene_size") or {},
            "layout_assets": list(layout.get("assets") or []),
            "wall_constraints": {
                key: layout.get(key)
                for key in (
                    "wall_description",
                    "wall_asset_id",
                    "wall_categories_limitation",
                    "wall_x",
                    "wall_x_negative",
                    "wall_y",
                    "wall_y_negative",
                    "wall_z",
                )
                if key in layout
            },
            "layout_metadata": {
                key: value
                for key, value in layout.items()
                if key not in {"description", "scene_size", "assets"}
            },
        }
    return scene_index


def shot_size_contract(distance: str) -> dict[str, Any]:
    label = str(distance or "medium shot").strip().lower()
    contracts = {
        "extreme close-up": (0.26, 0.32, 0.56, 0.92, "primary_first"),
        "close-up": (0.18, 0.22, 0.38, 0.90, "primary_first"),
        "medium close-up": (0.14, 0.18, 0.30, 0.88, "primary_first"),
        "medium shot": (0.08, 0.12, 0.24, 0.84, "balanced"),
        "full shot": (0.05, 0.08, 0.18, 0.80, "balanced"),
        "wide shot": (0.03, 0.05, 0.14, 0.76, "environment_first"),
        "long shot": (0.03, 0.05, 0.14, 0.76, "environment_first"),
    }
    hard_floor, desired_min, desired_max, visibility_ratio_target, seed_priority = contracts.get(
        label,
        contracts["medium shot"],
    )
    return {
        "label": label,
        "hard_floor": hard_floor,
        "desired_min": desired_min,
        "desired_max": desired_max,
        "re_search_trigger": hard_floor,
        "visibility_ratio_target": visibility_ratio_target,
        "seed_priority": seed_priority,
    }


def lens_mm_for_distance(distance: str) -> float:
    label = str(distance or "").strip().lower()
    mapping = {
        "extreme close-up": 85.0,
        "close-up": 65.0,
        "medium close-up": 50.0,
        "medium shot": 35.0,
        "full shot": 32.0,
        "wide shot": 28.0,
        "long shot": 24.0,
    }
    return mapping.get(label, 35.0)


def duration_for_movement(movement: str) -> float:
    if movement in {"orbit", "truck", "pedestal"}:
        return 2.6
    if movement in {"push_in", "push_out", "pan"}:
        return 2.2
    if movement == "locked_off":
        return 1.5
    if movement == "static":
        return 1.8
    return 2.0


def choose_reference_image(
    *,
    asset_index: dict[str, dict[str, Any]],
    focus_ids: list[str],
    angle: str,
    direction: str,
) -> tuple[str, str]:
    view_preference = "front"
    if "top" in angle.lower() or "bird" in angle.lower():
        view_preference = "top"
    elif direction == "left":
        view_preference = "left"
    for focus_id in focus_ids:
        asset = asset_index.get(focus_id) or {}
        image_paths = asset.get("image_paths") or {}
        for candidate_view in (view_preference, "front", "thumbnail", "left", "top"):
            path = str(image_paths.get(candidate_view) or "")
            if path and Path(path).exists():
                return path, candidate_view
    return "", view_preference


def build_layout_index(scene_entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for asset in scene_entry.get("layout_assets") or []:
        asset_id = str(asset.get("asset_id") or "").strip()
        if asset_id:
            result[asset_id] = asset
    return result


def build_keyframe_plan(
    *,
    focus_ids: list[str],
    distance: str,
    contract: dict[str, Any],
    movement: str,
) -> list[dict[str, Any]]:
    primary = focus_ids[0] if focus_ids else None
    secondaries = focus_ids[1:]
    start = contract["desired_min"]
    end = contract["desired_max"] if movement == "push_out" else contract["desired_min"]
    if movement == "push_in":
        end = min(contract["desired_max"], contract["desired_min"] * 1.12)
    if movement in {"static", "locked_off"}:
        end = start
    middle = (start + end) / 2.0
    return [
        {
            "keyframe_id": "kf0",
            "primary_focus_id": primary,
            "secondary_focus_ids": secondaries,
            "shot_size": distance,
            "primary_screen_area_target": round(start, 4),
            "primary_screen_area_floor": contract["hard_floor"],
            "visibility_ratio_target": contract["visibility_ratio_target"],
            "notes": ["Establish the primary story subject cleanly."],
        },
        {
            "keyframe_id": "kf1",
            "primary_focus_id": primary,
            "secondary_focus_ids": secondaries,
            "shot_size": distance,
            "primary_screen_area_target": round(middle, 4),
            "primary_screen_area_floor": contract["hard_floor"],
            "visibility_ratio_target": max(contract["visibility_ratio_target"] - 0.04, 0.7),
            "notes": ["Carry the motion while keeping the primary subject readable."],
        },
        {
            "keyframe_id": "kf2",
            "primary_focus_id": primary,
            "secondary_focus_ids": secondaries,
            "shot_size": distance,
            "primary_screen_area_target": round(end, 4),
            "primary_screen_area_floor": max(contract["hard_floor"] - 0.02, 0.03),
            "visibility_ratio_target": max(contract["visibility_ratio_target"] - 0.06, 0.68),
            "notes": ["Land the framing without losing the narrative anchor."],
        },
    ]


def build_blocking_plan(
    *,
    shot: dict[str, Any],
    scene_entry: dict[str, Any],
    asset_index: dict[str, dict[str, Any]],
    scene_description: str,
    shot_description: str,
    camera_instruction: dict[str, Any],
    camera_index: int,
) -> dict[str, Any]:
    focus_ids = [str(item).strip() for item in camera_instruction.get("focus_on_ids") or [] if str(item).strip()]
    primary_focus_id = focus_ids[0] if focus_ids else None
    secondary_focus_ids = focus_ids[1:]
    angle = str(camera_instruction.get("angle") or "eye-level").strip()
    distance = str(camera_instruction.get("distance") or "medium shot").strip()
    direction = canonical_direction(camera_instruction.get("direction"), angle)
    movement = canonical_movement(camera_instruction.get("movement"))
    contract = shot_size_contract(distance)
    reference_image_path, reference_view = choose_reference_image(
        asset_index=asset_index,
        focus_ids=focus_ids,
        angle=angle,
        direction=direction,
    )
    layout_index = build_layout_index(scene_entry)
    layout_assets = {
        asset_id: {
            "location": asset.get("location") or {},
            "rotation": asset.get("rotation") or {},
            "dimensions": asset.get("dimensions") or {},
        }
        for asset_id, asset in layout_index.items()
    }
    return {
        "scene_id": shot.get("scene_id"),
        "shot_id": shot.get("shot_id"),
        "camera_name": camera_instruction.get("camera_name"),
        "camera_index": camera_index,
        "camera_role": camera_instruction.get("camera_role"),
        "cut_reason": camera_instruction.get("cut_reason"),
        "scene_description": scene_description,
        "shot_description": shot_description,
        "primary_focus_id": primary_focus_id,
        "secondary_focus_ids": secondary_focus_ids,
        "focus_ids": focus_ids,
        "angle": angle,
        "distance": distance,
        "direction": direction,
        "movement": movement,
        "description": camera_instruction.get("description"),
        "shot_size_contract": contract,
        "start_frame_contract": {
            "primary_focus_id": primary_focus_id,
            "start_focus_ids": focus_ids,
            "distance": distance,
            "angle": angle,
            "screen_region": "center",
            "primary_screen_area_target": contract["desired_min"],
            "primary_screen_area_floor": contract["hard_floor"],
            "visibility_ratio_target": contract["visibility_ratio_target"],
            "shot_size_contract": contract,
            "seed_priority": contract["seed_priority"],
        },
        "keyframe_plan": build_keyframe_plan(
            focus_ids=focus_ids,
            distance=distance,
            contract=contract,
            movement=movement,
        ),
        "motion_contract": {
            "movement": movement,
            "direction": None if direction == "front" else direction,
            "allow_semantic_light_dynamic": movement in {"static", "locked_off"},
            "must_preserve_primary_anchor": True,
            "sync_to_action_duration": False,
            "story_priority": "story_first",
            "keyframe_count": 3,
        },
        "selected_candidate": {
            "candidate_id": f"{camera_instruction.get('camera_name')}_{reference_view}_seed",
            "camera_name": camera_instruction.get("camera_name"),
            "scene_id": shot.get("scene_id"),
            "shot_id": shot.get("shot_id"),
            "source": "asset_view_seed",
            "primary_focus_id": primary_focus_id,
            "secondary_focus_ids": secondary_focus_ids,
            "direction": direction,
            "reference_image_path": reference_image_path,
            "reference_view": reference_view,
            "normalized_score": 0.82,
            "estimated_metrics": {
                "candidate_primary_area_ratio": contract["desired_min"],
                "candidate_visible_ratio": contract["visibility_ratio_target"],
                "candidate_region_distance": 0.0,
            },
            "notes": [
                "Seed candidate chosen from the resolved focus asset reference view.",
                "This candidate acts as the camera stage handoff anchor.",
            ],
        },
        "top_candidates": [],
        "candidate_count": 1,
        "duration_target_seconds": duration_for_movement(movement),
        "lens_mm": lens_mm_for_distance(distance),
        "scene_layout_assets": layout_assets,
        "generated_at": utc_now(),
        "blocking_engine_version": "storyblender.director.blocking.v1",
    }


def asset_alias_tokens(asset_id: str, asset_entry: dict[str, Any]) -> set[str]:
    asset_type = str(asset_entry.get("asset_type") or "").lower()
    id_tokens = [token for token in tokenize(asset_id.replace("_", " ")) if len(token) > 1 and token not in STOPWORDS]
    if asset_type == "character":
        tokens = set(id_tokens)
    else:
        tokens = {id_tokens[-1]} if id_tokens else set()
    description_tokens = {
        token
        for token in tokenize(asset_entry.get("description"))
        if len(token) > 3 and token not in STOPWORDS
    }
    if asset_type != "character":
        description_tokens = {
            token
            for token in description_tokens
            if token not in WEAK_OBJECT_ALIAS_TOKENS
        }
    tokens.update(description_tokens)
    description = str(asset_entry.get("description") or "").lower()
    if "don vito" in description or "vito corleone" in description:
        tokens.add("vito")
    if "michael" in description:
        tokens.add("michael")
    if "kay" in description:
        tokens.add("kay")
    return tokens


def earliest_alias_index(asset_id: str, asset_entry: dict[str, Any], text: str) -> int | None:
    lower_text = text.lower()
    positions: list[int] = []
    asset_name = asset_id.lower()
    if asset_name in lower_text:
        positions.append(lower_text.index(asset_name))
    for token in asset_alias_tokens(asset_id, asset_entry):
        hit = re.search(rf"\b{re.escape(token)}\b", lower_text)
        if hit:
            positions.append(hit.start())
    if not positions:
        return None
    return min(positions)


def mentioned_character_focus_ids(
    *,
    shot_description: str,
    scene_asset_ids: list[str],
    asset_index: dict[str, dict[str, Any]],
) -> list[str]:
    rows: list[tuple[int, int, str]] = []
    for asset_order, asset_id in enumerate(scene_asset_ids):
        asset_entry = asset_index.get(asset_id) or {}
        if not likely_human_focus_id(asset_id, asset_entry):
            continue
        mention_index = earliest_alias_index(asset_id, asset_entry, shot_description)
        if mention_index is None:
            continue
        rows.append((mention_index, asset_order, asset_id))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    result: list[str] = []
    for _, _, asset_id in rows:
        if asset_id not in result:
            result.append(asset_id)
    return result


def repair_human_primary_focus(
    *,
    focus_ids: list[str],
    shot_description: str,
    scene_asset_ids: list[str],
    asset_index: dict[str, dict[str, Any]],
    shot_tokens: set[str],
) -> list[str]:
    if not focus_ids or not human_primary_requested(shot_description):
        return focus_ids
    primary_entry = asset_index.get(focus_ids[0]) or {}
    if likely_human_focus_id(focus_ids[0], primary_entry):
        return focus_ids
    mentioned_humans = mentioned_character_focus_ids(
        shot_description=shot_description,
        scene_asset_ids=scene_asset_ids,
        asset_index=asset_index,
    )
    if not mentioned_humans:
        return focus_ids
    primary = mentioned_humans[0]
    if "and" in shot_tokens:
        merged = mentioned_humans + [asset_id for asset_id in focus_ids if asset_id not in mentioned_humans]
        return merged[:3]
    return [primary]


def infer_focus_ids(
    *,
    shot_description: str,
    scene_description: str,
    scene_entry: dict[str, Any],
    asset_index: dict[str, dict[str, Any]],
) -> list[str]:
    shot_text = str(shot_description or "").lower()
    scene_text = str(scene_description or "").lower()
    shot_tokens = set(tokenize(shot_text))
    scene_tokens = set(tokenize(scene_text))
    scene_asset_ids = list(scene_entry.get("asset_ids") or [])
    scored: list[tuple[float, int, int, str]] = []
    mentioned_characters: list[tuple[int, str]] = []
    for asset_order, asset_id in enumerate(scene_asset_ids):
        asset_entry = asset_index.get(asset_id) or {}
        mention_index = earliest_alias_index(asset_id, asset_entry, shot_text)
        score = 0.0
        if asset_id.lower() in shot_text:
            score += 3.5
        if asset_id.lower() in scene_text:
            score += 1.0
        aliases = asset_alias_tokens(asset_id, asset_entry)
        for token in aliases:
            if token in shot_tokens:
                score += 2.4 if asset_entry.get("asset_type") == "character" else 1.8
            elif token in scene_tokens:
                score += 0.5
        if mention_index is not None:
            score += max(0.8, 3.0 - min(mention_index, 180) / 90.0)
            if asset_entry.get("asset_type") == "character":
                mentioned_characters.append((mention_index, asset_id))
        if "assassin" in shot_tokens and asset_id.startswith("assassin_"):
            score += 4.0
        if "car" in shot_tokens and asset_entry.get("asset_type") == "object":
            description = str(asset_entry.get("description") or "").lower()
            if "car" in description or "sedan" in description:
                score += 2.5
        if "bed" in shot_tokens and "bed" in str(asset_entry.get("description") or "").lower():
            score += 2.5
        if "cat" in shot_tokens and "cat" in str(asset_entry.get("description") or "").lower():
            score += 2.5
        if score > 0.0:
            character_bias = 0 if asset_entry.get("asset_type") == "character" else 1
            mention_sort = mention_index if mention_index is not None else 10**6
            scored.append((score, mention_sort, character_bias * 100 + asset_order, asset_id))

    if mentioned_characters:
        earliest_character_index = min(item[0] for item in mentioned_characters)
        boosted: list[tuple[float, int, int, str]] = []
        for score, mention_sort, bias_order, asset_id in scored:
            if mention_sort == earliest_character_index and (asset_index.get(asset_id) or {}).get("asset_type") == "character":
                score += 2.5
            boosted.append((score, mention_sort, bias_order, asset_id))
        scored = boosted

    mentioned_humans = mentioned_character_focus_ids(
        shot_description=shot_description,
        scene_asset_ids=scene_asset_ids,
        asset_index=asset_index,
    )
    if human_primary_requested(shot_description) and mentioned_humans:
        return mentioned_humans[:3] if "and" in shot_tokens else mentioned_humans[:1]

    if scored:
        scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        if "and" not in shot_tokens:
            mentioned = [item for item in scored if item[1] < 10**6]
            if mentioned:
                mentioned.sort(key=lambda item: (item[1], item[2], item[3]))
                earliest_index = mentioned[0][1]
                earliest_assets = [asset_id for _, mention_sort, _, asset_id in mentioned if mention_sort == earliest_index]
                if len(earliest_assets) == 1:
                    return repair_human_primary_focus(
                        focus_ids=earliest_assets,
                        shot_description=shot_description,
                        scene_asset_ids=scene_asset_ids,
                        asset_index=asset_index,
                        shot_tokens=shot_tokens,
                    )
                return repair_human_primary_focus(
                    focus_ids=earliest_assets[:3],
                    shot_description=shot_description,
                    scene_asset_ids=scene_asset_ids,
                    asset_index=asset_index,
                    shot_tokens=shot_tokens,
                )
        top_score = scored[0][0]
        selected = [asset_id for score, _, _, asset_id in scored if score >= max(top_score - 1.0, 2.0)]
        if "and" not in shot_tokens and len(selected) > 1:
            selected = selected[:1]
        return repair_human_primary_focus(
            focus_ids=selected[:3],
            shot_description=shot_description,
            scene_asset_ids=scene_asset_ids,
            asset_index=asset_index,
            shot_tokens=shot_tokens,
        )

    character_ids = [
        asset_id
        for asset_id in scene_asset_ids
        if (asset_index.get(asset_id) or {}).get("asset_type") == "character"
    ]
    if character_ids:
        return character_ids[:2] if "and" in shot_tokens else character_ids[:1]
    return scene_asset_ids[:1]


def infer_distance_label(shot_description: str, scene_description: str) -> str:
    text = f"{shot_description} {scene_description}".lower()
    if "extreme close-up" in text or "extreme close up" in text:
        return "extreme close-up"
    if "medium close-up" in text or "medium close up" in text:
        return "medium close-up"
    if "close-up" in text or "close up" in text:
        return "close-up"
    if any(keyword in text for keyword in ("face", "eyes", "expression", "whispers")):
        return "close-up"
    if any(keyword in text for keyword in ("crowd", "bedroom", "street", "garden", "hospital", "highway", "outside")):
        return "wide shot"
    if any(keyword in text for keyword in ("chases", "collapses", "falls", "stumbles", "walks out", "stands at the top")):
        return "full shot"
    return "medium shot"


def infer_movement_intent(shot_description: str) -> str:
    text = shot_description.lower()
    if "pan" in text:
        return "pan"
    if any(keyword in text for keyword in ("pulling back", "pulls back", "reveal", "reveals")):
        return "push_out"
    if any(keyword in text for keyword in ("rush toward", "runs toward", "walks around", "walks out", "chases", "drives away")):
        return "truck"
    if any(keyword in text for keyword in ("approaches", "emerge", "stands guard")):
        return "push_in"
    if any(keyword in text for keyword in ("stares", "glares", "looks on", "whispers", "wakes up", "recoils")):
        return "static"
    return "static"


def infer_angle_label(shot_description: str) -> str:
    text = shot_description.lower()
    if any(keyword in text for keyword in ("top-down", "top down", "overhead", "bird")):
        return "bird's-eye"
    if any(keyword in text for keyword in ("lies", "collapses", "body on the ground", "stands over")):
        return "high-angle"
    if any(keyword in text for keyword in ("looming", "authority", "from below")):
        return "low-angle"
    return "eye-level"


def infer_direction_label(shot_description: str) -> str:
    text = shot_description.lower()
    if any(keyword in text for keyword in ("profile", "side view")):
        return "left"
    if any(keyword in text for keyword in ("from behind", "back view")):
        return "back"
    return "front"


def infer_shot_goal(shot_description: str) -> str:
    text = shot_description.lower()
    if any(keyword in text for keyword in ("reveal", "recognized", "find", "notices")):
        return "reveal"
    if any(keyword in text for keyword in ("screams", "recoils", "glares", "stares", "weeping")):
        return "reaction"
    if any(keyword in text for keyword in ("shoots", "ambushed", "rush", "runs", "collapses", "falls", "chases")):
        return "action"
    if any(keyword in text for keyword in ("stands", "sits", "looks on", "holds", "whispers")):
        return "character_beat"
    return "story_beat"


def infer_sound_effect(scene_entry: dict[str, Any], scene_description: str, shot_description: str) -> str:
    scene_type = str(scene_entry.get("scene_type") or "").lower()
    text = f"{scene_description} {shot_description}".lower()
    if "wedding" in text:
        return "muted party music in the distance"
    if any(keyword in text for keyword in ("street", "car", "highway", "crowd")):
        return "city ambience with vehicle noise"
    if any(keyword in text for keyword in ("garden", "tomato", "baker")):
        return "soft outdoor ambience"
    if "church" in text or "baptism" in text:
        return "reverberant church ambience"
    if scene_type == "outdoor":
        return "outdoor room tone"
    return "low indoor room tone"


def build_camera_requirement(
    *,
    distance: str,
    angle: str,
    movement: str,
    focus_ids: list[str],
    shot_goal: str,
    shot_description: str,
    camera_count: int,
    camera_roles: list[str],
) -> dict[str, Any]:
    notes: list[str] = [f"Prioritize {distance} readability on the primary story subject."]
    if movement in {"static", "locked_off"}:
        notes.append("Allow a subtle semantic push only if the shot is not explicitly locked.")
    else:
        notes.append(f"Use {movement} only to reinforce the story beat, not for spectacle.")
    if len(focus_ids) > 1:
        notes.append("Keep secondary subjects legible while preserving the primary anchor.")
    if shot_goal == "reveal":
        notes.append("Land the shot on the reveal with readable spatial context.")
    return {
        "camera_count": max(int(camera_count), 1),
        "camera_roles": list(camera_roles),
        "primary_distance": distance,
        "primary_angle": angle,
        "primary_movement": movement,
        "framing_priority": "primary_first" if distance in {"close-up", "medium close-up", "extreme close-up"} else "balanced",
        "shot_goal": shot_goal,
        "notes": notes,
        "summary": shot_description,
    }


def build_character_actions(focus_ids: list[str], shot_description: str) -> list[dict[str, Any]]:
    return [
        {
            "asset_id": asset_id,
            "action_description": shot_description,
            "action_name": "story_action",
            "action_id": f"{asset_id}_story_action",
        }
        for asset_id in focus_ids
    ]


def rotate_direction(direction: str, offset: int = 1) -> str:
    directions = ["front", "right", "back", "left"]
    text = str(direction or "front").strip().lower()
    if text not in directions:
        text = "front"
    return directions[(directions.index(text) + offset) % len(directions)]


def single_camera_detail_shot(shot_description: str, distance: str, focus_ids: list[str]) -> bool:
    text = str(shot_description or "").lower()
    detail_words = {"letter", "gun", "newspaper", "cat", "tomato", "bed", "door", "window"}
    if any(word in text for word in detail_words) and len(focus_ids) <= 1:
        return True
    return str(distance or "").lower() in {"extreme close-up"} and len(focus_ids) <= 1


def camera_count_for_shot(shot_goal: str, shot_description: str, distance: str, focus_ids: list[str]) -> int:
    text = str(shot_description or "").lower()
    if single_camera_detail_shot(text, distance, focus_ids):
        return 1
    if shot_goal in {"action", "reveal"}:
        return 3 if len(focus_ids) >= 2 or any(word in text for word in ("crowd", "chases", "ambushed", "shoots")) else 2
    if shot_goal in {"reaction", "character_beat"} and (len(focus_ids) >= 2 or any(word in text for word in ("looks on", "stares", "glares", "whispers"))):
        return 2
    return 1


def camera_role_specs(
    *,
    shot_goal: str,
    shot_description: str,
    focus_ids: list[str],
    distance: str,
    angle: str,
    movement: str,
    direction: str,
) -> list[dict[str, Any]]:
    count = camera_count_for_shot(shot_goal, shot_description, distance, focus_ids)
    primary = focus_ids[:1]
    secondary = focus_ids[1:2]
    focus_pair = focus_ids[:2] if len(focus_ids) >= 2 else focus_ids[:1]
    if count == 1:
        return [
            {
                "camera_role": "primary_story_angle",
                "focus_on_ids": focus_ids,
                "distance": distance,
                "angle": angle,
                "movement": movement,
                "direction": direction,
                "cut_reason": "single readable angle is enough for this beat",
            }
        ]
    if shot_goal in {"action", "reveal"}:
        specs = [
            {
                "camera_role": "spatial_establishing",
                "focus_on_ids": focus_pair,
                "distance": "wide shot" if distance not in {"wide shot", "long shot"} else distance,
                "angle": angle,
                "movement": "push_in" if shot_goal == "reveal" else "truck",
                "direction": direction,
                "cut_reason": "establish geography before the action beat",
            },
            {
                "camera_role": "action_follow",
                "focus_on_ids": primary or focus_pair,
                "distance": "medium shot" if distance in {"wide shot", "long shot"} else distance,
                "angle": angle,
                "movement": movement if movement not in {"static", "locked_off"} else "truck",
                "direction": rotate_direction(direction, 1),
                "cut_reason": "follow the active story subject",
            },
        ]
        if count >= 3:
            specs.append(
                {
                    "camera_role": "reaction_or_detail",
                    "focus_on_ids": secondary or primary or focus_pair,
                    "distance": "close-up" if distance != "extreme close-up" else distance,
                    "angle": "eye-level",
                    "movement": "static",
                    "direction": rotate_direction(direction, -1),
                    "cut_reason": "land on the emotional or consequence beat",
                }
            )
        return specs
    return [
        {
            "camera_role": "primary_reaction",
            "focus_on_ids": primary or focus_pair,
            "distance": distance,
            "angle": angle,
            "movement": movement,
            "direction": direction,
            "cut_reason": "hold the primary reaction clearly",
        },
        {
            "camera_role": "reverse_or_listener",
            "focus_on_ids": secondary or primary or focus_pair,
            "distance": "medium close-up" if distance in {"close-up", "medium close-up"} else "medium shot",
            "angle": "eye-level",
            "movement": "static",
            "direction": rotate_direction(direction, 2),
            "cut_reason": "provide a connected reverse/listener angle",
        },
    ]


def build_camera_specs(
    *,
    scene_id: int,
    shot_id: int,
    shot_description: str,
    scene_description: str,
    scene_entry: dict[str, Any],
    asset_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    focus_ids = infer_focus_ids(
        shot_description=shot_description,
        scene_description=scene_description,
        scene_entry=scene_entry,
        asset_index=asset_index,
    )
    distance = infer_distance_label(shot_description, scene_description)
    movement = infer_movement_intent(shot_description)
    angle = infer_angle_label(shot_description)
    direction = infer_direction_label(shot_description)
    shot_goal = infer_shot_goal(shot_description)
    role_specs = camera_role_specs(
        shot_goal=shot_goal,
        shot_description=shot_description,
        focus_ids=focus_ids,
        distance=distance,
        angle=angle,
        movement=movement,
        direction=direction,
    )
    cameras: list[dict[str, Any]] = []
    for index, spec in enumerate(role_specs, start=1):
        cameras.append(
            {
                "camera_name": f"scene_{scene_id}_shot_{shot_id}_cam{index}",
                "camera_role": spec["camera_role"],
                "focus_on_ids": list(spec.get("focus_on_ids") or focus_ids),
                "angle": spec.get("angle") or angle,
                "distance": spec.get("distance") or distance,
                "movement": spec.get("movement") or movement,
                "direction": spec.get("direction") or direction,
                "description": shot_description,
                "cut_reason": spec.get("cut_reason") or "",
            }
        )
    return cameras


def build_director_script(
    *,
    story_payload: dict[str, Any],
    scene_map: dict[int, dict[str, Any]],
    asset_index: dict[str, dict[str, Any]],
    requested_scene_ids: list[int],
    config: "DirectorConfig",
    source_inventory_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenes_payload: list[dict[str, Any]] = []
    flat_shots: list[dict[str, Any]] = []
    shot_order = 0
    for storyboard_scene in story_payload.get("storyboard_outline") or []:
        try:
            scene_id = int(storyboard_scene.get("scene_id"))
        except (TypeError, ValueError):
            continue
        if requested_scene_ids and scene_id not in requested_scene_ids:
            continue
        scene_description = str(storyboard_scene.get("scene_description") or "").strip()
        scene_entry = scene_map.get(scene_id) or {
            "scene_id": scene_id,
            "scene_type": None,
            "asset_ids": [],
            "layout_description": scene_description,
            "scene_size": {},
            "layout_assets": [],
        }
        script_scene = {
            "scene_id": scene_id,
            "scene_description": scene_description,
            "scene_type": scene_entry.get("scene_type"),
            "asset_ids": list(scene_entry.get("asset_ids") or []),
            "shots": [],
        }
        for storyboard_shot in storyboard_scene.get("shots") or []:
            try:
                shot_id = int(storyboard_shot.get("shot_id"))
            except (TypeError, ValueError):
                continue
            shot_description = str(storyboard_shot.get("shot_description") or "").strip()
            cameras = build_camera_specs(
                scene_id=scene_id,
                shot_id=shot_id,
                shot_description=shot_description,
                scene_description=scene_description,
                scene_entry=scene_entry,
                asset_index=asset_index,
            )
            focus_ids = list(cameras[0].get("focus_on_ids") or []) if cameras else []
            shot_goal = infer_shot_goal(shot_description)
            shot_order += 1
            script_shot = {
                "scene_id": scene_id,
                "shot_id": shot_id,
                "shot_order": shot_order,
                "scene_description": scene_description,
                "shot_description": shot_description,
                "shot_type": cameras[0]["distance"] if cameras else "medium shot",
                "shot_goal": shot_goal,
                "focus_ids": focus_ids,
                "supporting_focus_ids": focus_ids[1:],
                "movement_intent": canonical_movement(cameras[0]["movement"]) if cameras else "static",
                "camera_requirement": build_camera_requirement(
                    distance=cameras[0]["distance"] if cameras else "medium shot",
                    angle=cameras[0]["angle"] if cameras else "eye-level",
                    movement=canonical_movement(cameras[0]["movement"]) if cameras else "static",
                    focus_ids=focus_ids,
                    shot_goal=shot_goal,
                    shot_description=shot_description,
                    camera_count=len(cameras),
                    camera_roles=[str(camera.get("camera_role") or "") for camera in cameras],
                ),
                "character_actions": build_character_actions(focus_ids, shot_description),
                "sound_effect": infer_sound_effect(scene_entry, scene_description, shot_description),
                "cameras": cameras,
            }
            flat_shots.append(script_shot)
            script_scene["shots"].append(script_shot)
        scenes_payload.append(script_scene)
    director_script_payload = {
        "schema_version": "storyblender.director_script.v1",
        "generated_at": utc_now(),
        "run_id": config.run_id or Path(config.output_root).name,
        "demo_root": config.demo_root,
        "approved_story_source_path": config.approved_story_source_path,
        "source_inventory_path": str(source_inventory_path),
        "story_summary": story_payload.get("story_summary"),
        "scene_count": len(scenes_payload),
        "shot_count": len(flat_shots),
        "camera_count": sum(len(shot.get("cameras") or []) for shot in flat_shots),
        "scenes": scenes_payload,
    }
    return director_script_payload, flat_shots


@dataclass(slots=True)
class DirectorConfig:
    demo_root: str
    output_root: str
    scene_ids: str = ""
    run_id: str = ""
    phase: str = "director_isolated_pipeline"
    approved_story_source_path: str = ""
    blender_exe: str = ""
    blend_file: str = ""
    vision_model: str = "gemini-3-flash-preview"
    anyllm_api_key: str = ""
    anyllm_api_base: str = "https://yunwu.ai"
    anyllm_provider: str = "gemini"
    timeout_seconds: int = 1800
    resolution_x: int = 960
    resolution_y: int = 540
    context_margin_factor: float = 1.7
    shot_contract_keyframe_count: int = 3


def deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def resolved_scene_ids(requested_scene_ids: list[int], flat_shots: list[dict[str, Any]]) -> list[int]:
    if requested_scene_ids:
        return requested_scene_ids
    ordered: list[int] = []
    for shot in flat_shots:
        try:
            scene_id = int(shot.get("scene_id"))
        except (TypeError, ValueError):
            continue
        if scene_id not in ordered:
            ordered.append(scene_id)
    return ordered


def build_scene_context_input(flat_shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for shot in flat_shots:
        cameras = list(shot.get("cameras") or [])
        if not cameras:
            continue
        payload.append(
            {
                "scene_id": int(shot.get("scene_id") or 0),
                "shot_id": int(shot.get("shot_id") or 0),
                "shot_description": shot.get("shot_description"),
                "scene_description": shot.get("scene_description"),
                "shot_goal": shot.get("shot_goal"),
                "sound_effect": shot.get("sound_effect"),
                "character_actions": deep_copy_json(shot.get("character_actions") or []),
                "asset_modifications": [],
                "camera_instruction": deep_copy_json(cameras[0]),
                "additional_camera_instructions": deep_copy_json(cameras[1:]),
            }
        )
    return payload


def default_blender_exe(configured_path: str) -> Path:
    if configured_path:
        return Path(configured_path).resolve()
    local_blender = CAMERA_DIR / "blender" / "blender.exe"
    if local_blender.exists():
        return local_blender.resolve()
    return DEFAULT_BLENDER_EXE


def default_blend_file(configured_path: str, source_inventory: dict[str, Any], demo_root: Path) -> Path:
    if configured_path:
        return Path(configured_path).resolve()
    for candidate in source_inventory.get("blend_paths") or []:
        candidate_path = Path(str(candidate))
        if candidate_path.exists():
            return candidate_path.resolve()
    fallback = demo_root / "The Godfather.blend"
    return fallback.resolve()


def _scene_character_ids_from_layout(
    *,
    scene_ids: list[int],
    scene_map: dict[int, dict[str, Any]],
    asset_index: dict[str, dict[str, Any]],
) -> dict[int, list[str]]:
    """For each scene, list asset ids whose asset_type is 'character'.

    The list comes from ``scene_map[scene_id]['asset_ids']`` (the authoritative
    layout asset list), filtered by ``asset_index[asset_id]['asset_type']``.
    Order follows the layout asset order so that primary anchors stay first.
    Used to keep statically-placed characters present in the scene dossier
    even when no shot action references them as actors.
    """

    result: dict[int, list[str]] = {}
    for scene_id in scene_ids:
        scene_entry = scene_map.get(scene_id) or {}
        ordered: list[str] = []
        for asset_id in scene_entry.get("asset_ids") or []:
            asset_id_str = str(asset_id or "").strip()
            if not asset_id_str or asset_id_str in ordered:
                continue
            entry = asset_index.get(asset_id_str) or {}
            if entry.get("asset_type") == "character":
                ordered.append(asset_id_str)
        if ordered:
            result[scene_id] = ordered
    return result


def run_scene_context_builder(
    *,
    config: DirectorConfig,
    scene_input_path: Path,
    output_root: Path,
    scene_ids: list[int],
    blend_file: Path,
    blender_exe: Path,
    scene_map: dict[int, dict[str, Any]] | None = None,
    asset_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scene_character_ids = (
        _scene_character_ids_from_layout(
            scene_ids=scene_ids,
            scene_map=scene_map or {},
            asset_index=asset_index or {},
        )
        if scene_map and asset_index
        else None
    )
    request_payload = build_scene_context_request_payload(
        filtered_input_path=scene_input_path,
        output_root=output_root,
        project_dir=config.demo_root,
        scene_ids=scene_ids,
        resolution_x=config.resolution_x,
        resolution_y=config.resolution_y,
        margin_factor=config.context_margin_factor,
        scene_character_ids=scene_character_ids,
    )
    request_path = save_json(request_payload, output_root / "outputs" / "scene_context_build_request_v1.json")
    stdout_path = output_root / "scene_context_builder_stdout.log"
    stderr_path = output_root / "scene_context_builder_stderr.log"
    manifest_path = output_root / "scene_dossiers" / "scene_dossier_manifest_v1.json"
    if manifest_path.exists():
        try:
            existing_manifest = load_json(manifest_path)
        except Exception:
            existing_manifest = {}
        existing_scene_ids = {
            int(item.get("scene_id"))
            for item in (existing_manifest.get("scenes") or [])
            if item.get("scene_id") is not None
        }
        requested_scene_set = set(scene_ids)
        if existing_scene_ids and (not requested_scene_set or requested_scene_set.issubset(existing_scene_ids)):
            return {
                "success": True,
                "returncode": 0,
                "command": [],
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "request_path": str(request_path),
                "manifest_path": str(manifest_path),
                "reused_existing_manifest": True,
            }
    result = run_blender_python_script(
        blender_exe=blender_exe,
        blend_file=blend_file,
        python_script=STAGE_DIR / "director_scene_context_builder.py",
        script_args=["--request-path", str(request_path)],
        workdir=STAGE_DIR,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=config.timeout_seconds,
        background=True,
    )
    result["request_path"] = str(request_path)
    result["manifest_path"] = str(manifest_path)
    return result


def build_director_outputs(config: DirectorConfig) -> dict[str, Any]:
    demo_root = Path(config.demo_root).resolve()
    output_root = ensure_directory(Path(config.output_root).resolve())
    outputs_dir = ensure_directory(output_root / "outputs")
    requested_scene_ids = parse_scene_ids(config.scene_ids)

    source_inventory = resolve_demo_sources(demo_root)
    story_payload = load_json(Path(source_inventory["approved_story_source_path"]))
    layout_payload = load_json(Path(source_inventory["layout_source_path"]))
    formatted_payload = (
        load_json(Path(source_inventory["formatted_model_source_path"]))
        if source_inventory["formatted_model_source_path"]
        else {}
    )
    dimension_payload = (
        load_json(Path(source_inventory["dimension_estimation_source_path"]))
        if source_inventory["dimension_estimation_source_path"]
        else {}
    )

    merged_asset_sheet = merge_asset_sheets(
        story_payload.get("asset_sheet") or [],
        layout_payload.get("asset_sheet") or [],
        formatted_payload.get("asset_sheet") or [],
        dimension_payload.get("asset_sheet") or [],
    )
    asset_index = build_asset_index(merged_asset_sheet, demo_root)
    scene_map = build_scene_maps(layout_payload if layout_payload else story_payload)
    scene_description_map, shot_description_map = build_storyboard_maps(story_payload)

    source_inventory_payload = {
        "schema_version": "storyblender.source_inventory.v1",
        "generated_at": utc_now(),
        "demo_root": str(demo_root),
        "selected_sources": source_inventory,
        "counts": {
            "storyboard_scene_count": len(story_payload.get("storyboard_outline") or []),
            "storyboard_shot_count": sum(len(scene.get("shots") or []) for scene in story_payload.get("storyboard_outline") or []),
            "layout_scene_count": len(layout_payload.get("scene_details") or []),
            "merged_asset_count": len(asset_index),
        },
        "runtime_rules": {
            "approved_story_source": "latest animated_models_vN.json",
            "layout_source": "latest layout_script_vN.json",
            "formatted_model_source": "formatted_model/formatted_model.json when available",
            "runtime_consumes_upstream_shot_details_directly": False,
            "runtime_consumes_upstream_camera_instructions_directly": False,
            "director_responsibility": "expand approved story into a shot-level shooting script before downstream handoff",
        },
    }
    source_inventory_path = save_json(source_inventory_payload, outputs_dir / "source_inventory_v1.json")

    config.approved_story_source_path = source_inventory["approved_story_source_path"]
    director_script_payload, flat_shots = build_director_script(
        story_payload=story_payload,
        scene_map=scene_map,
        asset_index=asset_index,
        requested_scene_ids=requested_scene_ids,
        config=config,
        source_inventory_path=source_inventory_path,
    )
    director_script_path = save_json(director_script_payload, outputs_dir / "director_script_v1.json")
    scene_context_input = build_scene_context_input(flat_shots)
    scene_input_path = save_json(scene_context_input, outputs_dir / "scene_input_v1.json")

    scene_entries: list[dict[str, Any]] = []
    scene_understanding: list[dict[str, Any]] = []
    shot_contracts: list[dict[str, Any]] = []
    blocking_plans: list[dict[str, Any]] = []
    contract_blocking_plans: list[dict[str, Any]] = []
    contract_blocking_by_camera: dict[tuple[int, int, str], dict[str, Any]] = {}
    shot_sequence: list[dict[str, Any]] = []

    shots_by_scene: dict[int, list[dict[str, Any]]] = {}
    for shot in flat_shots:
        shots_by_scene.setdefault(int(shot["scene_id"]), []).append(shot)

    context_shots_by_scene: dict[int, list[dict[str, Any]]] = {}
    for shot in scene_context_input:
        context_shots_by_scene.setdefault(int(shot["scene_id"]), []).append(shot)

    active_scene_ids = resolved_scene_ids(requested_scene_ids, flat_shots)
    blend_file = default_blend_file(config.blend_file, source_inventory, demo_root)
    blender_exe = default_blender_exe(config.blender_exe)
    context_result = run_scene_context_builder(
        config=config,
        scene_input_path=scene_input_path,
        output_root=output_root,
        scene_ids=active_scene_ids,
        blend_file=blend_file,
        blender_exe=blender_exe,
        scene_map=scene_map,
        asset_index=asset_index,
    )
    if not context_result.get("success"):
        raise RuntimeError(
            "Scene preview build failed. "
            f"See {context_result.get('stderr_path') or (output_root / 'scene_context_builder_stderr.log')}"
        )
    scene_dossier_manifest_path = Path(str(context_result["manifest_path"])).resolve()
    dossier_manifest = load_scene_dossier_manifest(scene_dossier_manifest_path)
    scene_dossier_entries = list(dossier_manifest.get("scenes") or [])
    scene_dossiers_by_scene_id: dict[int, dict[str, Any]] = {}
    scene_dossier_paths_by_scene_id: dict[int, str] = {}
    for scene_entry in scene_dossier_entries:
        try:
            scene_id = int(scene_entry.get("scene_id"))
        except (TypeError, ValueError):
            continue
        scene_dossiers_by_scene_id[scene_id] = dict(scene_entry.get("dossier") or {})
        scene_dossier_paths_by_scene_id[scene_id] = str(scene_entry.get("dossier_path") or "")

    def _process_scene_llm(scene_id: int) -> dict[str, Any]:
        scene_shots = shots_by_scene[scene_id]
        context_scene_shots = context_shots_by_scene.get(scene_id, [])
        scene_entry = scene_map.get(scene_id) or {
            "scene_id": scene_id,
            "scene_type": None,
            "asset_ids": [],
            "layout_description": scene_description_map.get(scene_id) or "",
            "scene_size": {},
            "layout_assets": [],
        }
        scene_dossier = scene_dossiers_by_scene_id.get(scene_id) or {}
        character_ids = [
            str(row.get("character_id") or "").strip()
            for row in (scene_dossier.get("character_summaries") or [])
            if str(row.get("character_id") or "").strip()
        ]
        scene_entry_row = {
            "scene_id": scene_id,
            "shot_count": len(scene_shots),
            "camera_count": sum(len(shot.get("cameras") or []) for shot in scene_shots),
            "character_ids": character_ids,
            "asset_ids": list(scene_entry.get("asset_ids") or []),
            "scene_description": scene_description_map.get(scene_id) or scene_entry.get("layout_description") or "",
            "scene_top_view_path": str(scene_dossier.get("scene_top_view_path") or ""),
            "scene_dossier_path": scene_dossier_paths_by_scene_id.get(scene_id, ""),
        }
        understanding = generate_scene_understanding(
            scene_dossier=scene_dossier,
            scene_shots=context_scene_shots,
            vision_model=config.vision_model,
            anyllm_api_key=config.anyllm_api_key,
            anyllm_api_base=config.anyllm_api_base,
            anyllm_provider=config.anyllm_provider,
        )
        understanding["scene_description"] = scene_description_map.get(scene_id) or scene_entry.get("layout_description") or ""
        scene_contracts = generate_scene_shot_contracts(
            scene_dossier=scene_dossier,
            scene_understanding=understanding,
            scene_shots=context_scene_shots,
            vision_model=config.vision_model,
            anyllm_api_key=config.anyllm_api_key,
            anyllm_api_base=config.anyllm_api_base,
            anyllm_provider=config.anyllm_provider,
            keyframe_count=config.shot_contract_keyframe_count,
        )
        scene_contract_blocking_plans = build_scene_blocking_plans(
            scene_dossier=scene_dossier,
            shot_contracts=scene_contracts,
        )
        return {
            "scene_id": scene_id,
            "scene_entry": scene_entry,
            "scene_entry_row": scene_entry_row,
            "understanding": understanding,
            "scene_contracts": scene_contracts,
            "scene_contract_blocking_plans": scene_contract_blocking_plans,
        }

    sorted_scene_ids = sorted(shots_by_scene)
    llm_workers = min(len(sorted_scene_ids), 9)
    scene_results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=llm_workers) as executor:
        futures = {executor.submit(_process_scene_llm, sid): sid for sid in sorted_scene_ids}
        for future in as_completed(futures):
            result = future.result()
            scene_results[result["scene_id"]] = result

    for scene_id in sorted_scene_ids:
        r = scene_results[scene_id]
        scene_entries.append(r["scene_entry_row"])
        scene_understanding.append(r["understanding"])
        shot_contracts.extend(r["scene_contracts"])
        contract_blocking_plans.extend(r["scene_contract_blocking_plans"])
        for contract_plan in r["scene_contract_blocking_plans"]:
            key = (
                int(contract_plan.get("scene_id") or 0),
                int(contract_plan.get("shot_id") or 0),
                str(contract_plan.get("camera_name") or ""),
            )
            contract_blocking_by_camera[key] = contract_plan

        scene_contract_by_camera: dict[tuple[int, int, str], dict[str, Any]] = {}
        for shot_contract in r["scene_contracts"]:
            scene_contract_by_camera[
                (
                    int(shot_contract.get("scene_id") or 0),
                    int(shot_contract.get("shot_id") or 0),
                    str(shot_contract.get("camera_name") or ""),
                )
            ] = shot_contract

        scene_shots = shots_by_scene[scene_id]
        scene_entry = r["scene_entry"]
        for shot in scene_shots:
            cameras = list(shot.get("cameras") or [])
            plan_refs: list[str] = []
            camera_names: list[str] = []
            camera_roles: list[str] = []
            cut_reasons: list[str] = []
            focus_ids: list[str] = []
            shot_duration = 0.0
            for camera_index, camera_instruction in enumerate(cameras, start=1):
                shot_contract = scene_contract_by_camera.get(
                    (scene_id, int(shot.get("shot_id") or 0), str(camera_instruction.get("camera_name") or ""))
                )
                camera_instruction_for_blocking = camera_instruction_with_contract_focus(camera_instruction, shot_contract)
                blocking_plan = build_blocking_plan(
                    shot=shot,
                    scene_entry=scene_entry,
                    asset_index=asset_index,
                    scene_description=scene_description_map.get(scene_id) or scene_entry.get("layout_description") or "",
                    shot_description=str(shot.get("shot_description") or shot_description_map.get((scene_id, int(shot.get("shot_id") or 0))) or ""),
                    camera_instruction=camera_instruction_for_blocking,
                    camera_index=camera_index,
                )
                if camera_instruction_for_blocking.get("focus_contract_override"):
                    blocking_plan["focus_contract_override"] = camera_instruction_for_blocking["focus_contract_override"]
                contract_plan = contract_blocking_by_camera.get(
                    (scene_id, int(shot.get("shot_id") or 0), str(blocking_plan.get("camera_name") or ""))
                )
                if contract_plan:
                    blocking_plan["contract_blocking_plan"] = contract_plan
                    blocking_plan["contract_selected_candidate"] = contract_plan.get("selected_candidate") or {}
                    blocking_plan["contract_top_candidates"] = contract_plan.get("top_candidates") or []
                    blocking_plan["shot_size_contract"] = contract_plan.get("shot_size_contract") or blocking_plan.get("shot_size_contract")
                    blocking_plan["primary_first_seeding"] = contract_plan.get("primary_first_seeding") or {}
                blocking_plan["top_candidates"] = [blocking_plan["selected_candidate"]]
                if blocking_plan.get("contract_top_candidates"):
                    blocking_plan["top_candidates"].extend(blocking_plan["contract_top_candidates"])
                blocking_plans.append(blocking_plan)
                plan_refs.append(str(blocking_plan["camera_name"]))
                camera_names.append(str(blocking_plan["camera_name"]))
                camera_roles.append(str(blocking_plan.get("camera_role") or ""))
                cut_reasons.append(str(blocking_plan.get("cut_reason") or ""))
                shot_duration += float(blocking_plan["duration_target_seconds"])
                for focus_id in blocking_plan.get("focus_ids") or []:
                    if focus_id not in focus_ids:
                        focus_ids.append(focus_id)
            shot_sequence.append(
                {
                    "scene_id": scene_id,
                    "shot_id": int(shot.get("shot_id") or 0),
                    "shot_order": int(shot.get("shot_order") or 0),
                    "scene_description": scene_description_map.get(scene_id) or scene_entry.get("layout_description") or "",
                    "shot_description": shot.get("shot_description"),
                    "shot_type": shot.get("shot_type"),
                    "shot_goal": shot.get("shot_goal"),
                    "sound_effect": shot.get("sound_effect"),
                    "camera_count": len(camera_names),
                    "camera_names": camera_names,
                    "camera_roles": camera_roles,
                    "cut_reasons": cut_reasons,
                    "blocking_plan_refs": plan_refs,
                    "duration_target_seconds": round(shot_duration, 3),
                    "focus_ids": focus_ids,
                    "camera_requirement": shot.get("camera_requirement") or {},
                }
            )

    scene_index_path = save_json(
        {
            "schema_version": "storyblender.scene_index.v1",
            "generated_at": utc_now(),
            "scene_entries": scene_entries,
        },
        outputs_dir / "scene_index_v1.json",
    )
    scene_understanding_path = save_json(scene_understanding, outputs_dir / "scene_understanding_v1.json")
    shot_contracts_path = save_json(shot_contracts, outputs_dir / "shot_contracts_v1.json")
    blocking_plans_path = save_json(blocking_plans, outputs_dir / "blocking_plans_v1.json")
    contract_blocking_plans_path = save_json(contract_blocking_plans, outputs_dir / "contract_blocking_plans_v1.json")
    director_handoff_path = save_json(
        {
            "schema_version": "storyblender.director_handoff.v1",
            "generated_at": utc_now(),
            "run_id": config.run_id or output_root.name,
            "demo_root": str(demo_root),
            "instruction_source_path": str(demo_root),
            "approved_story_source_path": source_inventory["approved_story_source_path"],
            "story_summary": story_payload.get("story_summary"),
            "files": {
                "scene_input_path": str(scene_input_path),
                "scene_index_path": str(scene_index_path),
                "scene_understanding_path": str(scene_understanding_path),
                "shot_contracts_path": str(shot_contracts_path),
                "blocking_plans_path": str(blocking_plans_path),
                "contract_blocking_plans_path": str(contract_blocking_plans_path),
                "scene_dossier_manifest_path": str(scene_dossier_manifest_path),
                "scene_context_request_path": str(context_result.get("request_path") or ""),
                "scene_context_builder_stdout_path": str(context_result.get("stdout_path") or ""),
                "scene_context_builder_stderr_path": str(context_result.get("stderr_path") or ""),
                "source_inventory_path": str(source_inventory_path),
                "director_script_path": str(director_script_path),
            },
            "asset_index": asset_index,
            "scene_index": scene_entries,
            "scene_details": scene_map,
            "scene_context_build": context_result,
            "summary": {
                "scene_count": len(scene_entries),
                "shot_count": len(shot_sequence),
                "camera_count": len(blocking_plans),
                "multi_camera_shot_count": sum(1 for shot in shot_sequence if len(shot.get("camera_names") or []) > 1),
            },
            "shot_sequence": shot_sequence,
        },
        outputs_dir / "director_handoff_v1.json",
    )
    manifest_path = save_json(
        {
            "schema_version": "storyblender.director_manifest.v1",
            "generated_at": utc_now(),
            "success": True,
            "phase": config.phase,
            "run_id": config.run_id or output_root.name,
            "scene_ids": requested_scene_ids,
            "demo_root": str(demo_root),
            "approved_story_source_path": source_inventory["approved_story_source_path"],
            "output_root": str(output_root),
            "scene_input_path": str(scene_input_path),
            "scene_index_path": str(scene_index_path),
            "scene_understanding_path": str(scene_understanding_path),
            "shot_contracts_path": str(shot_contracts_path),
            "blocking_plans_path": str(blocking_plans_path),
            "contract_blocking_plans_path": str(contract_blocking_plans_path),
            "scene_dossier_manifest_path": str(scene_dossier_manifest_path),
            "scene_context_request_path": str(context_result.get("request_path") or ""),
            "scene_context_builder_stdout_path": str(context_result.get("stdout_path") or ""),
            "scene_context_builder_stderr_path": str(context_result.get("stderr_path") or ""),
            "source_inventory_path": str(source_inventory_path),
            "director_script_path": str(director_script_path),
            "director_handoff_path": str(director_handoff_path),
        },
        output_root / "manifest.json",
    )
    return {
        "success": True,
        "manifest_path": str(manifest_path),
        "director_handoff_path": str(director_handoff_path),
        "director_script_path": str(director_script_path),
        "source_inventory_path": str(source_inventory_path),
        "blocking_plans_path": str(blocking_plans_path),
        "contract_blocking_plans_path": str(contract_blocking_plans_path),
        "shot_contracts_path": str(shot_contracts_path),
        "scene_understanding_path": str(scene_understanding_path),
    }


def parse_args() -> DirectorConfig:
    runtime_defaults = load_runtime_defaults("director")
    parser = argparse.ArgumentParser(description="Run the isolated Director stage.")
    parser.add_argument("--demo-root", default=str(DEFAULT_DEMO_ROOT))
    parser.add_argument("--instruction-source-path", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--scene-ids", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--phase", default="director_isolated_pipeline")
    parser.add_argument("--blender-exe", default=str(runtime_defaults.get("blender_exe") or os.getenv("STORYBLENDER_BLENDER_EXE", "")))
    parser.add_argument("--blend-file", default="")
    parser.add_argument("--vision-model", default=str(runtime_defaults.get("vision_model") or os.getenv("STORYBLENDER_VISION_MODEL", "gemini-3-flash-preview")))
    parser.add_argument("--anyllm-api-key", default=str(runtime_defaults.get("anyllm_api_key") or os.getenv("ANYLLM_API_KEY", "")))
    parser.add_argument("--anyllm-api-base", default=str(runtime_defaults.get("anyllm_api_base") or os.getenv("ANYLLM_API_BASE", "https://yunwu.ai")))
    parser.add_argument("--anyllm-provider", default=str(runtime_defaults.get("anyllm_provider") or os.getenv("ANYLLM_PROVIDER", "gemini")))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
    parser.add_argument("--context-margin-factor", type=float, default=1.7)
    parser.add_argument("--shot-contract-keyframe-count", type=int, default=3)
    args = parser.parse_args()
    demo_root = args.demo_root
    if args.instruction_source_path:
        demo_root = args.instruction_source_path
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(args.run_id).resolve()
    return DirectorConfig(
        demo_root=demo_root,
        output_root=str(output_root),
        scene_ids=args.scene_ids,
        run_id=args.run_id,
        phase=args.phase,
        blender_exe=args.blender_exe,
        blend_file=args.blend_file,
        vision_model=args.vision_model,
        anyllm_api_key=args.anyllm_api_key,
        anyllm_api_base=args.anyllm_api_base,
        anyllm_provider=args.anyllm_provider,
        timeout_seconds=args.timeout_seconds,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        context_margin_factor=args.context_margin_factor,
        shot_contract_keyframe_count=args.shot_contract_keyframe_count,
    )


def main() -> int:
    config = parse_args()
    result = build_director_outputs(config)
    print("Director completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
