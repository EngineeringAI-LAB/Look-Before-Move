from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from video_runtime import _as_vector, _look_at_euler, _normalize_preset, _object_world_bounds, _round_vector
from video_runtime import apply_trajectory_plan, build_trajectory_plan

try:  # pragma: no cover - Blender-only runtime
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
except Exception:  # pragma: no cover
    bpy = None
    mathutils = None
    world_to_camera_view = None


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render StoryBlender camera handoff clips inside Blender.")
    parser.add_argument("--camera-handoff-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
    parser.add_argument("--render-engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--render-samples", type=int, default=8)
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    return parser.parse_args(argv)


def _scene_name_candidates(scene_id: int, shot_id: int | None = None) -> list[str]:
    candidates: list[str] = []
    if shot_id is not None:
        candidates.extend(
            [
                f"Scene_{scene_id}_Shot_{shot_id}",
                f"Scene_{scene_id}_shot_{shot_id}",
                f"scene_{scene_id}_shot_{shot_id}",
            ]
        )
    candidates.extend([f"Scene_{scene_id}", f"Scene {scene_id}", f"scene_{scene_id}"])
    return candidates


def _resolve_scene(scene_id: int, shot_id: int | None = None):
    if bpy is None:
        return None, ""
    for scene_name in _scene_name_candidates(scene_id, shot_id):
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            return scene, scene_name
    return None, ""


def _set_active_scene(scene) -> None:
    if bpy is None or scene is None:
        return
    try:
        window = bpy.context.window
    except Exception:
        window = None
    if window is not None:
        try:
            window.scene = scene
        except Exception:
            pass


def _find_object_by_asset_id(scene, asset_id: str):
    normalized = str(asset_id or "").strip()
    if not normalized or scene is None:
        return None
    if normalized in scene.objects:
        return scene.objects[normalized]
    exact_match = None
    dotted_matches = []
    underscored_matches = []
    property_matches = []
    for obj in scene.objects:
        if obj.name == normalized:
            exact_match = obj
            break
        if obj.name.startswith(f"{normalized}."):
            dotted_matches.append(obj)
        elif obj.name.startswith(f"{normalized}_"):
            underscored_matches.append(obj)
        try:
            object_asset_id = str(obj.get("asset_id") or "").strip()
        except Exception:
            object_asset_id = ""
        if object_asset_id == normalized:
            property_matches.append(obj)
    if exact_match is not None:
        return exact_match
    if property_matches:
        return property_matches[0]
    if dotted_matches:
        return dotted_matches[0]
    if underscored_matches:
        return underscored_matches[0]
    return None


def _descendant_objects(root_obj) -> list[Any]:
    descendants: list[Any] = []
    stack = [root_obj]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if current.name in seen:
            continue
        seen.add(current.name)
        descendants.append(current)
        for child in list(getattr(current, "children", []) or []):
            stack.append(child)
    return descendants


def _points_bounds(points: list[Any]) -> tuple[Any, Any]:
    first_point = points[0].copy()
    minimum = first_point.copy()
    maximum = first_point.copy()
    for point in points[1:]:
        minimum.x = min(minimum.x, point.x)
        minimum.y = min(minimum.y, point.y)
        minimum.z = min(minimum.z, point.z)
        maximum.x = max(maximum.x, point.x)
        maximum.y = max(maximum.y, point.y)
        maximum.z = max(maximum.z, point.z)
    return minimum, maximum


def _layout_scene_details(director_handoff: dict[str, Any], scene_id: int) -> dict[str, Any]:
    scene_details = director_handoff.get("scene_details") or {}
    return dict(scene_details.get(str(scene_id)) or scene_details.get(scene_id) or {})


def _layout_focus_center(scene_details: dict[str, Any], focus_ids: list[str]):
    rows = list(scene_details.get("layout_assets") or [])
    matches = []
    for row in rows:
        asset_id = str(row.get("asset_id") or "").strip()
        if asset_id not in focus_ids:
            continue
        location = row.get("location") or {}
        matches.append(
            (
                float(location.get("x", 0.0)),
                float(location.get("y", 0.0)),
                float(location.get("z", 0.8)),
            )
        )
    if not matches:
        return mathutils.Vector((0.0, 0.0, 1.2))
    count = float(len(matches))
    return mathutils.Vector(
        (
            sum(item[0] for item in matches) / count,
            sum(item[1] for item in matches) / count,
            sum(item[2] for item in matches) / count + 0.9,
        )
    )


def _layout_focus_extent(scene_details: dict[str, Any], focus_ids: list[str]):
    rows = list(scene_details.get("layout_assets") or [])
    extents = []
    for row in rows:
        asset_id = str(row.get("asset_id") or "").strip()
        if asset_id not in focus_ids:
            continue
        dimensions = row.get("dimensions") or {}
        extents.append(
            (
                float(dimensions.get("x", 0.8)),
                float(dimensions.get("y", 0.8)),
                float(dimensions.get("z", 1.7)),
            )
        )
    if not extents:
        return mathutils.Vector((0.8, 0.8, 1.7))
    return mathutils.Vector(
        (
            max(item[0] for item in extents),
            max(item[1] for item in extents),
            max(item[2] for item in extents),
        )
    )


def _focus_bounds(scene, director_handoff: dict[str, Any], scene_id: int, focus_ids: list[str]):
    points = []
    for focus_id in focus_ids:
        obj = _find_object_by_asset_id(scene, focus_id)
        if obj is None:
            continue
        for candidate in _descendant_objects(obj):
            try:
                points.extend(_object_world_bounds(candidate))
            except Exception:
                continue
    if points:
        minimum, maximum = _points_bounds(points)
        return (minimum + maximum) * 0.5, maximum - minimum, "scene_geometry"
    scene_details = _layout_scene_details(director_handoff, scene_id)
    return (
        _layout_focus_center(scene_details, focus_ids),
        _layout_focus_extent(scene_details, focus_ids),
        "layout_fallback",
    )


def _semantic_focus_center(center, extent, camera: dict[str, Any]):
    target = center.copy()
    semantic_target = str(camera.get("primary_semantic_target") or "").strip().lower()
    distance_label = str(camera.get("distance_label") or "").strip().lower()
    wants_face = semantic_target in {"face", "eyes", "head", "back_of_head"} or "close" in distance_label
    if wants_face:
        target.z = center.z + max(float(extent.z) * 0.30, 0.35)
    elif semantic_target in {"chest", "torso", "upper_body", "upper body"}:
        target.z = center.z + max(float(extent.z) * 0.12, 0.15)
    elif semantic_target in {"feet", "foot", "shoes"}:
        target.z = center.z - max(float(extent.z) * 0.38, 0.35)
    elif "medium" in distance_label:
        target.z = center.z + max(float(extent.z) * 0.12, 0.15)
    return target


def _camera_instruction(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "movement": camera.get("movement_tag") or camera.get("authored_movement") or "static",
        "direction": camera.get("direction_tag") or "front",
        "motion_profile": camera.get("motion_profile") or "",
        "camera_parameters": {"focal_length": float(camera.get("lens_mm") or 35.0)},
    }


def _prepare_transforms(camera: dict[str, Any], focus_center) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    start_transform = json.loads(json.dumps(camera.get("start_transform") or {}))
    end_transform = json.loads(json.dumps(camera.get("end_transform") or start_transform))
    start_location = _as_vector(start_transform.get("location"), default=(0.0, -3.0, 1.6))
    end_location = _as_vector(end_transform.get("location"), default=tuple(start_location))
    delta = end_location - start_location
    framing_start = dict(start_transform.get("framing") or {})
    framing_end = dict(end_transform.get("framing") or framing_start)
    start_width = float(framing_start.get("width", 0.0) or 0.0)
    end_width = float(framing_end.get("width", start_width) or start_width)
    synthesized = {
        "used_framing_synthesis": False,
        "distance_delta_ratio": 0.0,
    }
    if delta.length <= 1e-4 and start_width > 1e-6 and abs(end_width - start_width) > 1e-4:
        focus_direction = focus_center - start_location
        if focus_direction.length > 1e-6:
            focus_direction.normalize()
            ratio = abs(end_width - start_width) / max(start_width, 1e-6)
            travel = max(min(ratio, 0.08) * max((focus_center - start_location).length, 1.0), 0.06)
            end_location = start_location + focus_direction * travel if end_width < start_width else start_location - focus_direction * travel
            synthesized["used_framing_synthesis"] = True
            synthesized["distance_delta_ratio"] = round(ratio, 6)
    start_transform["location"] = _round_vector(start_location)
    end_transform["location"] = _round_vector(end_location)
    if not start_transform.get("rotation_euler"):
        start_transform["rotation_euler"] = _round_vector(_look_at_euler(start_location, focus_center))
    if synthesized.get("used_framing_synthesis") or not end_transform.get("rotation_euler"):
        end_transform["rotation_euler"] = _round_vector(_look_at_euler(end_location, focus_center))
    return start_transform, end_transform, synthesized


def _trajectory_travel_limit(preset_name: str, camera: dict[str, Any]) -> float:
    preset = _normalize_preset(preset_name)
    distance_label = str(camera.get("distance_label") or "").strip().lower()
    closeup = bool(camera.get("closeup_required")) or distance_label in {"extreme close-up", "close-up", "medium close-up"}
    if preset in {"static_hold", "static_hold_locked", "static_subtle_zoom", "pan_left", "pan_right"}:
        return 0.0
    limits = {
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
    limit = float(limits.get(preset, 1.4))
    return min(limit, 0.25) if closeup else limit


def _smooth_executable_keyframes(
    *,
    preset_name: str,
    keyframes: list[dict[str, Any]],
    camera: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(keyframes) < 2:
        return keyframes, {"speed_limited": False}
    preset = _normalize_preset(preset_name)
    start_location = _as_vector(keyframes[0].get("location"))
    end_location = _as_vector(keyframes[-1].get("location"), default=tuple(start_location))
    original_travel = float((end_location - start_location).length)
    limit = _trajectory_travel_limit(preset, camera)
    speed_limited = False
    smoothed: list[dict[str, Any]] = []
    if preset in {"static_hold", "static_hold_locked", "static_subtle_zoom", "pan_left", "pan_right"}:
        for keyframe in keyframes:
            item = dict(keyframe)
            item["location"] = _round_vector(start_location)
            smoothed.append(item)
        speed_limited = original_travel > 1e-4
    elif limit > 0.0 and original_travel > limit:
        ratio = limit / max(original_travel, 1e-6)
        for keyframe in keyframes:
            item = dict(keyframe)
            location = _as_vector(item.get("location"), default=tuple(start_location))
            item["location"] = _round_vector(start_location + (location - start_location) * ratio)
            smoothed.append(item)
        speed_limited = True
    else:
        smoothed = [dict(keyframe) for keyframe in keyframes]
    limited_end = _as_vector(smoothed[-1].get("location"), default=tuple(start_location))
    return smoothed, {
        "speed_limited": speed_limited,
        "preset_name": preset,
        "max_travel": round(limit, 6),
        "original_travel_distance": round(original_travel, 6),
        "limited_travel_distance": round(float((limited_end - start_location).length), 6),
    }


ROTATION_CONTINUITY_THRESHOLD_RAD = 0.35


def _lens_float(value: Any, fallback: float = 35.0) -> float:
    try:
        lens = float(value)
        if math.isfinite(lens) and lens > 0.0:
            return lens
    except Exception:
        pass
    return float(fallback)


def _plan_lens_policy(trajectory_plan: dict[str, Any]) -> dict[str, Any]:
    policy = trajectory_plan.get("lens_policy") if isinstance(trajectory_plan.get("lens_policy"), dict) else {}
    return dict(policy)


def _plan_base_lens(trajectory_plan: dict[str, Any], camera: dict[str, Any]) -> float:
    policy = _plan_lens_policy(trajectory_plan)
    keyframes = list(trajectory_plan.get("keyframes") or [])
    first_lens = keyframes[0].get("lens_mm") if keyframes else None
    return round(
        _lens_float(
            policy.get("base_lens_mm"),
            _lens_float(first_lens, _lens_float(camera.get("lens_mm"), 35.0)),
        ),
        4,
    )


def _plan_allows_lens_ramp(trajectory_plan: dict[str, Any], preset_name: str | None = None) -> bool:
    policy = _plan_lens_policy(trajectory_plan)
    preset = _normalize_preset(preset_name or trajectory_plan.get("preset_name"))
    return preset == "static_subtle_zoom" and bool(policy.get("uses_subtle_lens_ramp", True))


def _rotation_quaternion_from_keyframe(keyframe: dict[str, Any]) -> Any:
    quat_values = keyframe.get("rotation_quaternion")
    if mathutils is not None and isinstance(quat_values, (list, tuple)) and len(quat_values) == 4:
        try:
            quat = mathutils.Quaternion(tuple(float(value) for value in quat_values[:4]))
            if float(quat.length) > 1e-8:
                quat.normalize()
                return quat
        except Exception:
            pass
    euler_values = _as_vector(keyframe.get("rotation_euler"))
    quat = mathutils.Euler((float(euler_values.x), float(euler_values.y), float(euler_values.z)), "XYZ").to_quaternion()
    quat.normalize()
    return quat


def _continuous_rotation_quaternions(keyframes: list[dict[str, Any]]) -> list[Any]:
    rotations: list[Any] = []
    previous = None
    for keyframe in keyframes:
        quat = _rotation_quaternion_from_keyframe(keyframe)
        if previous is not None:
            dot = sum(float(previous[index]) * float(quat[index]) for index in range(4))
            if dot < 0.0:
                quat = mathutils.Quaternion(tuple(-float(quat[index]) for index in range(4)))
                quat.normalize()
        rotations.append(quat)
        previous = quat
    return rotations


def _quat_to_json(quat: Any) -> list[float]:
    return [round(float(quat[index]), 8) for index in range(4)]


def _quat_angle(a: Any, b: Any) -> float:
    dot = abs(sum(float(a[index]) * float(b[index]) for index in range(4)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _trajectory_motion_qc(trajectory_plan: dict[str, Any]) -> dict[str, Any]:
    keyframes = list(trajectory_plan.get("keyframes") or [])
    preset = _normalize_preset(trajectory_plan.get("preset_name"))
    lens_values = [round(_lens_float(keyframe.get("lens_mm"), 35.0), 4) for keyframe in keyframes]
    has_lens_change = bool(lens_values) and any(abs(value - lens_values[0]) > 1e-4 for value in lens_values[1:])
    allows_lens_ramp = _plan_allows_lens_ramp(trajectory_plan, preset)
    lens_ramp_monotonic = all(lens_values[index + 1] + 1e-4 >= lens_values[index] for index in range(max(len(lens_values) - 1, 0)))
    has_unallowed_lens_change = has_lens_change and not allows_lens_ramp
    rotations = _continuous_rotation_quaternions(keyframes) if keyframes else []
    adjacent_deltas = [
        _quat_angle(rotations[index], rotations[index + 1])
        for index in range(max(len(rotations) - 1, 0))
    ]
    max_adjacent_rotation_delta = max(adjacent_deltas) if adjacent_deltas else 0.0
    rotation_continuity_valid = max_adjacent_rotation_delta <= ROTATION_CONTINUITY_THRESHOLD_RAD + 1e-6
    lens_policy_valid = (not has_unallowed_lens_change) and (lens_ramp_monotonic if allows_lens_ramp else True)
    return {
        "lens_policy_valid": bool(lens_policy_valid),
        "rotation_continuity_valid": bool(rotation_continuity_valid),
        "has_lens_change": bool(has_lens_change),
        "has_unallowed_lens_change": bool(has_unallowed_lens_change),
        "allows_lens_ramp": bool(allows_lens_ramp),
        "lens_values": lens_values,
        "max_adjacent_rotation_delta": round(max_adjacent_rotation_delta, 6),
        "rotation_delta_threshold_rad": ROTATION_CONTINUITY_THRESHOLD_RAD,
    }


def _subdivide_rotation_keyframes(keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(keyframes) < 2:
        return keyframes
    rotations = _continuous_rotation_quaternions(keyframes)
    result: list[dict[str, Any]] = []
    for index in range(len(keyframes) - 1):
        current = dict(keyframes[index])
        next_keyframe = dict(keyframes[index + 1])
        result.append(current)
        delta = _quat_angle(rotations[index], rotations[index + 1])
        segments = max(1, int(math.ceil(delta / max(ROTATION_CONTINUITY_THRESHOLD_RAD, 1e-6))))
        if segments <= 1:
            continue
        start_frame = int(current.get("frame") or 1)
        end_frame = int(next_keyframe.get("frame") or start_frame)
        if end_frame - start_frame <= 1:
            continue
        start_location = _as_vector(current.get("location"))
        end_location = _as_vector(next_keyframe.get("location"), default=tuple(start_location))
        start_lens = _lens_float(current.get("lens_mm"), 35.0)
        end_lens = _lens_float(next_keyframe.get("lens_mm"), start_lens)
        for segment in range(1, segments):
            alpha = segment / float(segments)
            frame = int(round(start_frame + (end_frame - start_frame) * alpha))
            if frame <= start_frame or frame >= end_frame:
                continue
            location = start_location + (end_location - start_location) * alpha
            try:
                quat = rotations[index].slerp(rotations[index + 1], alpha)
                if quat is None:
                    raise ValueError("slerp returned None")
            except Exception:
                quat = rotations[index].copy()
                slerped = quat.slerp(rotations[index + 1], alpha)
                if slerped is not None:
                    quat = slerped
            try:
                quat.normalize()
            except Exception:
                pass
            item = dict(current)
            item["frame"] = frame
            item["location"] = _round_vector(location)
            item["lens_mm"] = round(start_lens + (end_lens - start_lens) * alpha, 4)
            item["rotation_quaternion"] = _quat_to_json(quat)
            try:
                item["rotation_euler"] = _round_vector(quat.to_euler())
            except Exception:
                pass
            item["easing"] = "linear"
            result.append(item)
    result.append(dict(keyframes[-1]))
    deduped: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    for keyframe in sorted(result, key=lambda item: int(item.get("frame") or 1)):
        frame = int(keyframe.get("frame") or 1)
        if frame in seen_frames:
            continue
        seen_frames.add(frame)
        deduped.append(keyframe)
    return deduped


def _normalize_trajectory_plan(trajectory_plan: dict[str, Any], camera: dict[str, Any]) -> dict[str, Any]:
    plan = dict(trajectory_plan)
    keyframes = [dict(keyframe) for keyframe in list(plan.get("keyframes") or [])]
    if not keyframes:
        return plan
    preset = _normalize_preset(plan.get("preset_name"))
    plan["preset_name"] = preset
    base_lens = _plan_base_lens(plan, camera)
    allows_lens_ramp = _plan_allows_lens_ramp(plan, preset)
    if allows_lens_ramp:
        start_frame = int(plan.get("start_frame") or keyframes[0].get("frame") or 1)
        end_frame = int(plan.get("end_frame") or keyframes[-1].get("frame") or start_frame)
        end_lens = _lens_float(keyframes[-1].get("lens_mm"), min(base_lens * 1.04, 85.0))
        if end_lens <= base_lens + 1e-4:
            end_lens = min(base_lens * 1.04, 85.0)
        for keyframe in keyframes:
            frame = int(keyframe.get("frame") or start_frame)
            t = (frame - start_frame) / max(end_frame - start_frame, 1)
            keyframe["lens_mm"] = round(base_lens + (end_lens - base_lens) * max(0.0, min(t, 1.0)), 4)
    else:
        for keyframe in keyframes:
            keyframe["lens_mm"] = base_lens
    keyframes = _subdivide_rotation_keyframes(keyframes)
    rotations = _continuous_rotation_quaternions(keyframes)
    for index, keyframe in enumerate(keyframes):
        keyframe["rotation_quaternion"] = _quat_to_json(rotations[index])
    plan["keyframes"] = keyframes
    lens_policy = _plan_lens_policy(plan)
    lens_policy["base_lens_mm"] = base_lens
    lens_policy["uses_subtle_lens_ramp"] = bool(allows_lens_ramp)
    plan["lens_policy"] = lens_policy
    motion_qc = _trajectory_motion_qc(plan)
    plan["motion_qc"] = motion_qc
    safety = plan.get("safety_report") if isinstance(plan.get("safety_report"), dict) else {}
    safety = dict(safety)
    safety["motion_qc"] = motion_qc
    plan["safety_report"] = safety
    return plan


def _visibility_extent_for_camera(camera: dict[str, Any], focus_extent):
    semantic_target = str(camera.get("primary_semantic_target") or "").strip().lower()
    distance_label = str(camera.get("distance_label") or "").strip().lower()
    extent = focus_extent.copy()
    if semantic_target in {"face", "eyes", "head", "back_of_head"}:
        return mathutils.Vector((min(float(extent.x), 0.45), min(float(extent.y), 0.45), min(float(extent.z), 0.75)))
    if semantic_target in {"chest", "torso", "upper_body", "upper body"}:
        return mathutils.Vector((min(float(extent.x), 0.65), min(float(extent.y), 0.65), min(float(extent.z), 0.6)))
    if semantic_target in {"feet", "foot", "shoes"}:
        return mathutils.Vector((min(float(extent.x), 0.75), min(float(extent.y), 0.75), min(float(extent.z), 0.45)))
    if distance_label in {"extreme close-up", "close-up", "medium close-up"}:
        return mathutils.Vector((min(float(extent.x), 0.7), min(float(extent.y), 0.7), min(float(extent.z), 0.9)))
    return extent


def _focus_visibility_points(center, extent) -> list[Any]:
    half_x = max(float(extent.x) * 0.5, 0.12)
    half_y = max(float(extent.y) * 0.5, 0.12)
    half_z = max(float(extent.z) * 0.5, 0.18)
    offsets = [
        (0.0, 0.0, 0.0),
        (-half_x, -half_y, -half_z),
        (-half_x, -half_y, half_z),
        (-half_x, half_y, -half_z),
        (-half_x, half_y, half_z),
        (half_x, -half_y, -half_z),
        (half_x, -half_y, half_z),
        (half_x, half_y, -half_z),
        (half_x, half_y, half_z),
    ]
    return [center + mathutils.Vector(offset) for offset in offsets]


def _project_focus_visibility(scene, camera_obj, points: list[Any]) -> dict[str, Any]:
    if world_to_camera_view is None or not points:
        return {"available": False, "visible_fraction": 1.0, "center_distance": 0.0, "in_frame": True}
    projected = []
    center_coord = None
    for index, point in enumerate(points):
        coord = world_to_camera_view(scene, camera_obj, point)
        if index == 0:
            center_coord = coord
        if float(getattr(coord, "z", 0.0) or 0.0) > 0.0:
            projected.append(coord)
    if center_coord is None:
        return {"available": True, "visible_fraction": 0.0, "center_distance": 1.0, "in_frame": False}
    center_x = float(center_coord.x)
    center_y = float(center_coord.y)
    center_in_front = float(getattr(center_coord, "z", 0.0) or 0.0) > 0.0
    center_distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
    if not projected:
        return {
            "available": True,
            "visible_fraction": 0.0,
            "center_distance": round(center_distance, 6),
            "in_frame": False,
            "bbox": None,
        }
    min_x = min(float(coord.x) for coord in projected)
    max_x = max(float(coord.x) for coord in projected)
    min_y = min(float(coord.y) for coord in projected)
    max_y = max(float(coord.y) for coord in projected)
    bbox_width = max(max_x - min_x, 1e-6)
    bbox_height = max(max_y - min_y, 1e-6)
    inside_min_x = max(min_x, 0.0)
    inside_max_x = min(max_x, 1.0)
    inside_min_y = max(min_y, 0.0)
    inside_max_y = min(max_y, 1.0)
    visible_area = max(inside_max_x - inside_min_x, 0.0) * max(inside_max_y - inside_min_y, 0.0)
    bbox_area = bbox_width * bbox_height
    center_in_frame = center_in_front and 0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0
    return {
        "available": True,
        "visible_fraction": round(min(visible_area / max(bbox_area, 1e-6), 1.0), 6),
        "center_distance": round(center_distance, 6),
        "in_frame": center_in_frame or visible_area > 0.0,
        "center_in_frame": center_in_frame,
        "bbox": {
            "min_x": round(min_x, 6),
            "max_x": round(max_x, 6),
            "min_y": round(min_y, 6),
            "max_y": round(max_y, 6),
        },
    }


def _trajectory_visibility_report(
    *,
    scene,
    camera_obj,
    trajectory_plan: dict[str, Any],
    camera: dict[str, Any],
    focus_center,
    focus_extent,
) -> dict[str, Any]:
    if world_to_camera_view is None:
        return {"available": False, "valid": True, "reasons": ["projection_api_unavailable"], "samples": []}
    start_frame = int(trajectory_plan.get("start_frame") or 1)
    end_frame = int(trajectory_plan.get("end_frame") or start_frame)
    mid_frame = max(start_frame, min(end_frame, (start_frame + end_frame) // 2))
    sample_frames = list(dict.fromkeys([start_frame, mid_frame, end_frame]))
    target_extent = _visibility_extent_for_camera(camera, focus_extent)
    points = _focus_visibility_points(focus_center, target_extent)
    distance_label = str(camera.get("distance_label") or "").strip().lower()
    semantic_target = str(camera.get("primary_semantic_target") or "").strip().lower()
    closeup = bool(camera.get("closeup_required")) or distance_label in {"extreme close-up", "close-up", "medium close-up"}
    min_visible_fraction = 0.02 if not closeup else 0.05
    max_center_distance = 0.62 if not closeup else 0.52
    samples: list[dict[str, Any]] = []
    reasons: list[str] = []
    original_frame = int(getattr(scene, "frame_current", start_frame) or start_frame)
    try:
        for frame in sample_frames:
            scene.frame_set(frame)
            projection = _project_focus_visibility(scene, camera_obj, points)
            projection["frame"] = frame
            projection["semantic_target"] = semantic_target
            projection["min_visible_fraction"] = min_visible_fraction
            projection["max_center_distance"] = max_center_distance
            sample_reasons = []
            center_in_frame = bool(projection.get("center_in_frame"))
            if not bool(projection.get("in_frame")):
                sample_reasons.append("focus_not_in_frame")
            if float(projection.get("visible_fraction") or 0.0) < min_visible_fraction and not center_in_frame:
                sample_reasons.append("focus_visible_fraction_too_low")
            if float(projection.get("center_distance") or 0.0) > max_center_distance:
                sample_reasons.append("focus_center_too_far_from_frame_center")
            projection["valid"] = not sample_reasons
            projection["reasons"] = sample_reasons
            reasons.extend(sample_reasons)
            samples.append(projection)
    finally:
        try:
            scene.frame_set(original_frame)
        except Exception:
            pass
    return {
        "available": True,
        "valid": not reasons,
        "min_visible_fraction": min_visible_fraction,
        "max_center_distance": max_center_distance,
        "target_extent": _round_vector(target_extent),
        "samples": samples,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _visibility_fallback_plan(
    *,
    trajectory_plan: dict[str, Any],
    frame_count: int,
    focus_center,
    motion_scale: float,
    reason: str,
) -> dict[str, Any]:
    keyframes = list(trajectory_plan.get("keyframes") or [])
    if not keyframes:
        return trajectory_plan
    # Fix F: do not collapse the trajectory into a fully static frozen shot
    # when visibility validation fails. Even at the lowest fallback step we
    # keep at least 20% of the planned motion so the camera continues to
    # move toward the focus rather than holding on whatever empty patch the
    # initial frame happened to land on.
    scale = max(0.2, min(float(motion_scale), 1.0))
    start_location = _as_vector(keyframes[0].get("location"))
    scaled_keyframes: list[dict[str, Any]] = []
    for index, keyframe in enumerate(keyframes):
        item = dict(keyframe)
        location = _as_vector(item.get("location"), default=tuple(start_location))
        scaled_location = start_location + (location - start_location) * scale
        item["location"] = _round_vector(scaled_location)
        item["rotation_euler"] = _round_vector(_look_at_euler(scaled_location, focus_center))
        if index == 0:
            item["frame"] = 1
        elif index == len(keyframes) - 1:
            item["frame"] = frame_count
        scaled_keyframes.append(item)
    if scaled_keyframes[-1].get("frame") != frame_count:
        last = dict(scaled_keyframes[-1])
        last["frame"] = frame_count
        scaled_keyframes.append(last)
    fallback = dict(trajectory_plan)
    if scale <= 1e-6:
        fallback["preset_name"] = "static_hold"
    fallback["selection_reason"] = f"{trajectory_plan.get('selection_reason') or 'trajectory'}; visibility_guard_scaled_fallback"
    fallback["start_frame"] = 1
    fallback["end_frame"] = frame_count
    fallback["keyframes"] = scaled_keyframes
    timing_policy = fallback.get("timing_policy") if isinstance(fallback.get("timing_policy"), dict) else {}
    timing_policy = dict(timing_policy)
    motion_policy = timing_policy.get("motion_speed_policy") if isinstance(timing_policy.get("motion_speed_policy"), dict) else {}
    motion_policy = dict(motion_policy)
    motion_policy["visibility_guard_applied"] = True
    motion_policy["visibility_guard_reason"] = reason
    motion_policy["visibility_guard_motion_scale"] = round(scale, 3)
    try:
        end_location = _as_vector(scaled_keyframes[-1].get("location"), default=tuple(start_location))
        motion_policy["limited_travel_distance"] = round(float((end_location - start_location).length), 6)
    except Exception:
        motion_policy["limited_travel_distance"] = 0.0
    timing_policy["motion_speed_policy"] = motion_policy
    fallback["timing_policy"] = timing_policy
    safety = fallback.get("safety_report") if isinstance(fallback.get("safety_report"), dict) else {}
    safety = dict(safety)
    safety["visibility_guard_applied"] = True
    safety["visibility_guard_reason"] = reason
    safety["visibility_guard_motion_scale"] = round(scale, 3)
    fallback["safety_report"] = safety
    return fallback


def _snapshot_scene(scene) -> dict[str, Any]:
    original = {
        "camera": scene.camera,
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "filepath": str(scene.render.filepath),
        "resolution_x": int(scene.render.resolution_x),
        "resolution_y": int(scene.render.resolution_y),
        "fps": int(scene.render.fps),
        "file_format": str(scene.render.image_settings.file_format),
        "engine": str(scene.render.engine),
        "cycles_samples": None,
        "cycles_use_denoising": None,
    }
    if hasattr(scene, "cycles"):
        try:
            original["cycles_samples"] = int(scene.cycles.samples)
            original["cycles_use_denoising"] = bool(scene.cycles.use_denoising)
        except Exception:
            pass
    return original


def _restore_scene(scene, original: dict[str, Any]) -> None:
    scene.camera = original.get("camera")
    scene.frame_start = int(original.get("frame_start", scene.frame_start))
    scene.frame_end = int(original.get("frame_end", scene.frame_end))
    scene.render.filepath = str(original.get("filepath", scene.render.filepath))
    scene.render.resolution_x = int(original.get("resolution_x", scene.render.resolution_x))
    scene.render.resolution_y = int(original.get("resolution_y", scene.render.resolution_y))
    scene.render.fps = int(original.get("fps", scene.render.fps))
    scene.render.image_settings.file_format = str(original.get("file_format", scene.render.image_settings.file_format))
    scene.render.engine = str(original.get("engine", scene.render.engine))
    if hasattr(scene, "cycles"):
        try:
            if original.get("cycles_samples") is not None:
                scene.cycles.samples = int(original["cycles_samples"])
            if original.get("cycles_use_denoising") is not None:
                scene.cycles.use_denoising = bool(original["cycles_use_denoising"])
        except Exception:
            pass


def _create_temp_camera(scene, name: str, lens_mm: float):
    camera_data = bpy.data.cameras.new(name=f"{name}_data")
    camera_data.lens = float(lens_mm)
    camera_obj = bpy.data.objects.new(name=name, object_data=camera_data)
    scene.collection.objects.link(camera_obj)
    return camera_obj


def _remove_temp_camera(camera_obj) -> None:
    if camera_obj is None:
        return
    camera_data = getattr(camera_obj, "data", None)
    try:
        bpy.data.objects.remove(camera_obj, do_unlink=True)
    except Exception:
        pass
    if camera_data is not None:
        try:
            bpy.data.cameras.remove(camera_data, do_unlink=True)
        except Exception:
            pass


def _build_plan_from_upstream_keyframes(
    *,
    upstream_keyframes: list[dict[str, Any]],
    camera: dict[str, Any],
    instruction: dict[str, Any],
    start_transform: dict[str, Any],
    end_transform: dict[str, Any],
    frame_count: int,
    focus_center: Any,
    focus_extent: Any,
) -> dict[str, Any]:
    """Convert Cinematographer trajectory_keyframes into a VideoEngineer trajectory plan."""
    preset_name = str(camera.get("motion_profile") or "")
    if preset_name == "closeup_static":
        preset_name = "static_subtle_zoom"
    elif not preset_name or preset_name in {"authored", "semantic_light_dynamic"}:
        from video_runtime import _preset_for_instruction
        preset_name = _preset_for_instruction(instruction, None)
    blender_keyframes = []
    lens = float(camera.get("lens_mm") or 35.0)
    inherited_lens = lens
    for kf in upstream_keyframes:
        t = float(kf.get("t", 0.0))
        frame_number = int(kf.get("frame", max(1, round(t * (frame_count - 1)) + 1)))
        frame_number = max(1, min(frame_number + 1, frame_count))
        kf_transform = kf.get("transform") or {}
        location = kf_transform.get("location") or start_transform.get("location") or [0.0, -3.0, 1.6]
        rotation = kf_transform.get("rotation_euler") or start_transform.get("rotation_euler") or [0.0, 0.0, 0.0]
        inherited_lens = _lens_float(kf_transform.get("lens_mm"), inherited_lens)
        blender_keyframes.append({
            "frame": frame_number,
            "location": _round_vector(_as_vector(location)),
            "rotation_euler": _round_vector(_as_vector(rotation)),
            "lens_mm": round(inherited_lens, 4),
        })
    if not blender_keyframes:
        blender_keyframes = [
            {"frame": 1, "location": _round_vector(_as_vector(start_transform.get("location"))), "rotation_euler": _round_vector(_as_vector(start_transform.get("rotation_euler"))), "lens_mm": round(lens, 4)},
            {"frame": frame_count, "location": _round_vector(_as_vector(end_transform.get("location"))), "rotation_euler": _round_vector(_as_vector(end_transform.get("rotation_euler"))), "lens_mm": round(lens, 4)},
        ]
    blender_keyframes, speed_policy = _smooth_executable_keyframes(
        preset_name=preset_name,
        keyframes=blender_keyframes,
        camera=camera,
    )
    return {
        "schema_version": "plan_a.video_trajectory.v1",
        "preset_name": preset_name,
        "selection_reason": "upstream_cinematographer_keyframes",
        "start_frame": 1,
        "end_frame": frame_count,
        "timing_policy": {"motion_speed_policy": speed_policy},
        "focus_center": _round_vector(focus_center),
        "focus_extent": _round_vector(focus_extent),
        "lens_policy": {
            "base_lens_mm": round(lens, 4),
            "uses_subtle_lens_ramp": preset_name == "static_subtle_zoom",
        },
        "keyframes": blender_keyframes,
    }


def _build_plan_from_explicit_trajectory(
    *,
    explicit_plan: dict[str, Any],
    camera: dict[str, Any],
    instruction: dict[str, Any],
    start_transform: dict[str, Any],
    end_transform: dict[str, Any],
    frame_count: int,
    focus_center: Any,
    focus_extent: Any,
) -> dict[str, Any]:
    """Normalize a Cinematographer trajectory_plan into executable Blender keyframes."""
    from video_runtime import _preset_for_instruction

    preset_name = _normalize_preset(explicit_plan.get("preset_name")) or _preset_for_instruction(instruction, None)
    keyframes = list(explicit_plan.get("keyframes") or [])
    if not keyframes:
        local_plan = build_trajectory_plan(
            camera_instruction=instruction,
            start_transform=start_transform,
            end_transform=end_transform,
            start_frame=1,
            end_frame=frame_count,
            focus_center=focus_center,
            focus_extent=focus_extent,
            preset_override=preset_name,
            timing_policy=explicit_plan.get("timing_policy") if isinstance(explicit_plan.get("timing_policy"), dict) else None,
        )
        smoothed_keyframes, speed_policy = _smooth_executable_keyframes(
            preset_name=preset_name,
            keyframes=list(local_plan.get("keyframes") or []),
            camera=camera,
        )
        timing_policy = local_plan.get("timing_policy") if isinstance(local_plan.get("timing_policy"), dict) else {}
        timing_policy = dict(timing_policy)
        timing_policy["motion_speed_policy"] = speed_policy
        local_plan["timing_policy"] = timing_policy
        local_plan["keyframes"] = smoothed_keyframes
        return local_plan

    lens = float(camera.get("lens_mm") or 35.0)
    frame_numbers = [int(kf.get("frame") or 0) for kf in keyframes]
    zero_based = bool(frame_numbers and min(frame_numbers) <= 0)
    blender_keyframes = []
    inherited_lens = lens
    for keyframe in keyframes:
        frame_number = int(keyframe.get("frame") or 1)
        if zero_based:
            frame_number += 1
        frame_number = max(1, min(frame_number, frame_count))
        inherited_lens = _lens_float(keyframe.get("lens_mm"), inherited_lens)
        blender_keyframes.append(
            {
                "frame": frame_number,
                "location": _round_vector(_as_vector(keyframe.get("location") or start_transform.get("location"))),
                "rotation_euler": _round_vector(_as_vector(keyframe.get("rotation_euler") or start_transform.get("rotation_euler"))),
                "lens_mm": round(inherited_lens, 4),
            }
        )
    blender_keyframes.sort(key=lambda item: int(item.get("frame") or 1))
    if blender_keyframes[0]["frame"] != 1:
        first = dict(blender_keyframes[0])
        first["frame"] = 1
        blender_keyframes.insert(0, first)
    if blender_keyframes[-1]["frame"] != frame_count:
        last = dict(blender_keyframes[-1])
        last["frame"] = frame_count
        blender_keyframes.append(last)
    blender_keyframes, speed_policy = _smooth_executable_keyframes(
        preset_name=preset_name,
        keyframes=blender_keyframes,
        camera=camera,
    )
    timing_policy = explicit_plan.get("timing_policy") if isinstance(explicit_plan.get("timing_policy"), dict) else {}
    timing_policy = dict(timing_policy)
    timing_policy["motion_speed_policy"] = speed_policy
    safety_report = explicit_plan.get("safety_report") or camera.get("trajectory_safety_report") or {}
    if isinstance(safety_report, dict):
        safety_report = dict(safety_report)
        safety_report["motion_speed_policy"] = speed_policy
    return {
        "schema_version": "plan_a.video_trajectory.v1",
        "preset_name": preset_name,
        "selection_reason": explicit_plan.get("selection_reason") or "upstream_explicit_trajectory_plan",
        "trajectory_selection_source": explicit_plan.get("trajectory_selection_source") or camera.get("trajectory_selection_source") or "",
        "start_candidate_id": explicit_plan.get("start_candidate_id") or "",
        "endpoint_candidate_id": explicit_plan.get("endpoint_candidate_id") or "",
        "start_frame": 1,
        "end_frame": frame_count,
        "timing_policy": timing_policy,
        "focus_center": _round_vector(focus_center),
        "focus_extent": _round_vector(focus_extent),
        "lens_policy": explicit_plan.get("lens_policy") if isinstance(explicit_plan.get("lens_policy"), dict) else {
            "base_lens_mm": round(lens, 4),
            "uses_subtle_lens_ramp": preset_name == "static_subtle_zoom",
        },
        "safety_report": safety_report,
        "keyframes": blender_keyframes,
    }


def _render_camera(scene, scene_name: str, camera: dict[str, Any], director_handoff: dict[str, Any], args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    _set_active_scene(scene)
    focus_ids = list(camera.get("focus_ids") or [camera.get("primary_focus_id")] or [])
    focus_ids = [str(item).strip() for item in focus_ids if str(item).strip()]
    focus_center, focus_extent, focus_source = _focus_bounds(scene, director_handoff, int(camera.get("scene_id") or 0), focus_ids)
    focus_center = _semantic_focus_center(focus_center, focus_extent, camera)
    start_transform, end_transform, synthesis = _prepare_transforms(camera, focus_center)
    frame_count = int(camera.get("target_frame_count") or max(int(float(camera.get("target_duration_seconds") or 1.5) * args.fps), 24))
    frame_count = max(frame_count, 2)

    clip_dir = output_root / "renders" / f"scene_{camera['scene_id']}_shot_{camera['shot_id']}" / str(camera["camera_name"])
    frame_dir = clip_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    instruction = _camera_instruction(camera)
    explicit_plan = camera.get("trajectory_plan") if isinstance(camera.get("trajectory_plan"), dict) else {}
    upstream_keyframes = camera.get("trajectory_keyframes") or []
    if explicit_plan:
        trajectory_plan = _build_plan_from_explicit_trajectory(
            explicit_plan=explicit_plan,
            camera=camera,
            instruction=instruction,
            start_transform=start_transform,
            end_transform=end_transform,
            frame_count=frame_count,
            focus_center=focus_center,
            focus_extent=focus_extent,
        )
    elif upstream_keyframes:
        trajectory_plan = _build_plan_from_upstream_keyframes(
            upstream_keyframes=upstream_keyframes,
            camera=camera,
            instruction=instruction,
            start_transform=start_transform,
            end_transform=end_transform,
            frame_count=frame_count,
            focus_center=focus_center,
            focus_extent=focus_extent,
        )
    else:
        trajectory_plan = build_trajectory_plan(
            camera_instruction=instruction,
            start_transform=start_transform,
            end_transform=end_transform,
            start_frame=1,
            end_frame=frame_count,
            focus_center=focus_center,
            focus_extent=focus_extent,
        )
    trajectory_plan = _normalize_trajectory_plan(trajectory_plan, camera)

    original_scene = _snapshot_scene(scene)
    camera_obj = _create_temp_camera(scene, f"video_render_{camera['camera_name']}", lens_mm=float(camera.get("lens_mm") or 35.0))
    try:
        apply_trajectory_plan(camera_obj, trajectory_plan)
        visibility_report = _trajectory_visibility_report(
            scene=scene,
            camera_obj=camera_obj,
            trajectory_plan=trajectory_plan,
            camera=camera,
            focus_center=focus_center,
            focus_extent=focus_extent,
        )
        if visibility_report.get("available") and not bool(visibility_report.get("valid")):
            initial_report = visibility_report
            fallback_trace = []
            selected_plan = trajectory_plan
            selected_report = visibility_report
            # Fix F: drop the `0.0` step so the fallback never produces a
            # dead static shot when visibility validation fails. The lowest
            # remaining step (0.2) is also clamped inside `_visibility_fallback_plan`.
            for motion_scale in (0.75, 0.5, 0.35, 0.2):
                candidate_plan = _visibility_fallback_plan(
                    trajectory_plan=trajectory_plan,
                    frame_count=frame_count,
                    focus_center=focus_center,
                    motion_scale=motion_scale,
                    reason="trajectory_focus_visibility_failed",
                )
                candidate_plan = _normalize_trajectory_plan(candidate_plan, camera)
                apply_trajectory_plan(camera_obj, candidate_plan)
                candidate_report = _trajectory_visibility_report(
                    scene=scene,
                    camera_obj=camera_obj,
                    trajectory_plan=candidate_plan,
                    camera=camera,
                    focus_center=focus_center,
                    focus_extent=focus_extent,
                )
                fallback_trace.append(
                    {
                        "motion_scale": motion_scale,
                        "valid": bool(candidate_report.get("valid")),
                        "reasons": candidate_report.get("reasons") or [],
                    }
                )
                selected_plan = candidate_plan
                selected_report = candidate_report
                if candidate_report.get("valid"):
                    break
            trajectory_plan = selected_plan
            visibility_report = {
                "available": True,
                "valid": bool(selected_report.get("valid")),
                "initial_report": initial_report,
                "fallback_report": selected_report,
                "fallback_applied": True,
                "fallback_trace": fallback_trace,
                "reasons": list(dict.fromkeys(list(initial_report.get("reasons") or []) + list(selected_report.get("reasons") or []))),
            }
        else:
            visibility_report["fallback_applied"] = False
        safety_report = trajectory_plan.get("safety_report") if isinstance(trajectory_plan.get("safety_report"), dict) else {}
        safety_report = dict(safety_report)
        safety_report["trajectory_visibility_report"] = visibility_report
        trajectory_plan["safety_report"] = safety_report
        trajectory_plan_path = _save_json(trajectory_plan, clip_dir / "trajectory_plan.json")
        scene.camera = camera_obj
        scene.frame_start = 1
        scene.frame_end = frame_count
        scene.render.resolution_x = int(args.resolution_x)
        scene.render.resolution_y = int(args.resolution_y)
        scene.render.fps = int(args.fps)
        scene.render.image_settings.file_format = "PNG"
        try:
            scene.render.engine = str(args.render_engine)
        except Exception:
            try:
                scene.render.engine = "BLENDER_EEVEE_NEXT"
            except Exception:
                pass
        if hasattr(scene, "eevee"):
            try:
                scene.eevee.taa_render_samples = max(1, int(args.render_samples))
            except Exception:
                pass
        if hasattr(scene, "cycles"):
            try:
                scene.cycles.samples = max(1, int(args.render_samples))
                scene.cycles.use_denoising = True
            except Exception:
                pass
        scene.render.filepath = str(frame_dir / "frame_")
        bpy.ops.render.render(animation=True, scene=scene.name)
    finally:
        _restore_scene(scene, original_scene)
        _remove_temp_camera(camera_obj)

    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    return {
        "scene_id": int(camera.get("scene_id") or 0),
        "shot_id": int(camera.get("shot_id") or 0),
        "camera_name": str(camera.get("camera_name") or ""),
        "scene_name": scene_name,
        "frame_dir": str(frame_dir.resolve()),
        "frame_count": len(frame_paths),
        "focus_ids": focus_ids,
        "focus_source": focus_source,
        "focus_center": _round_vector(focus_center),
        "focus_extent": _round_vector(focus_extent),
        "trajectory_plan_path": str(trajectory_plan_path),
        "trajectory_plan": trajectory_plan,
        "start_transform": start_transform,
        "end_transform": end_transform,
        "motion_synthesis": synthesis,
        "success": bool(frame_paths),
    }


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    camera_handoff = _load_json(args.camera_handoff_path)
    director_handoff = _load_json(camera_handoff["director_handoff_path"])

    render_report: list[dict[str, Any]] = []
    for shot in camera_handoff.get("shots") or []:
        scene_id = int(shot.get("scene_id") or 0)
        shot_id = int(shot.get("shot_id") or 0)
        scene, scene_name = _resolve_scene(scene_id, shot_id)
        if scene is None:
            scene, scene_name = _resolve_scene(scene_id)
        if scene is None:
            for camera in shot.get("cameras") or []:
                print(f"[video-render] scene={scene_id} shot={shot_id} camera={camera.get('camera_name')} missing scene", flush=True)
                render_report.append(
                    {
                        "scene_id": scene_id,
                        "shot_id": shot_id,
                        "camera_name": camera.get("camera_name"),
                        "success": False,
                        "error": f"scene_not_found_for_scene_{scene_id}_shot_{shot_id}",
                    }
                )
            continue
        for camera in shot.get("cameras") or []:
            print(f"[video-render] scene={scene_id} shot={shot_id} camera={camera.get('camera_name')} start", flush=True)
            try:
                row = _render_camera(scene, scene_name, camera, director_handoff, args, output_root)
                print(
                    f"[video-render] scene={scene_id} shot={shot_id} camera={camera.get('camera_name')} done frames={row.get('frame_count')}",
                    flush=True,
                )
                render_report.append(row)
            except Exception as exc:
                print(f"[video-render] scene={scene_id} shot={shot_id} camera={camera.get('camera_name')} failed: {exc}", flush=True)
                render_report.append(
                    {
                        "scene_id": scene_id,
                        "shot_id": shot_id,
                        "camera_name": camera.get("camera_name"),
                        "success": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    report_path = _save_json(
        {
            "schema_version": "storyblender.video_blender_render_report.v1",
            "generated_at": bpy.app.version_string if bpy is not None else "",
            "camera_handoff_path": str(Path(args.camera_handoff_path).resolve()),
            "render_report": render_report,
        },
        output_root / "outputs" / "blender_render_report_v1.json",
    )
    print(
        json.dumps(
            {
                "success": all(bool(item.get("success")) for item in render_report),
                "render_report_path": str(report_path),
                "render_count": len(render_report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if render_report and all(bool(item.get("success")) for item in render_report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
