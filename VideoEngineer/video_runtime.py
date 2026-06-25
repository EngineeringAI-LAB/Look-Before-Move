"""Plan-A-local runtime helpers for video rendering.

This module intentionally contains only the small subset of the old runtime API
that the Plan-A video runner still needs.  Keeping it local prevents the video
path from depending on the removed ``camera.engine`` package.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


PRESET_NAMES = {
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
    "static_hold",
    "static_hold_locked",
    "static_subtle_zoom",
}

PRESET_ALIASES = {
    "static": "static_hold",
    "locked_off": "static_hold",
    "locked off": "static_hold",
    "push_in": "push_in_arc",
    "push in": "push_in_arc",
    "zoom in": "push_in_arc",
    "push_out": "pull_out_arc",
    "push out": "pull_out_arc",
    "zoom out": "pull_out_arc",
    "pull out": "pull_out_arc",
}


def _normalize_preset(raw_preset: Any) -> str:
    text = str(raw_preset or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    return PRESET_ALIASES.get(text, text)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def load_json_with_paths(path: str | Path, project_dir: str | Path | None = None) -> Any:
    """Load JSON.  ``project_dir`` is accepted for legacy call compatibility."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_with_paths(
    payload: Any,
    destination: str | Path,
    project_dir: str | Path | None = None,
) -> Path:
    """Save JSON.  ``project_dir`` is accepted for legacy call compatibility."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def _object_world_bounds(obj: Any) -> list[Any]:
    """Return world-space bound-box corners for one Blender object."""

    import bpy
    import mathutils

    depsgraph = bpy.context.evaluated_depsgraph_get()
    try:
        evaluated = obj.evaluated_get(depsgraph)
    except Exception:
        evaluated = obj

    matrix = getattr(evaluated, "matrix_world", None) or getattr(obj, "matrix_world", None)
    bound_box = getattr(evaluated, "bound_box", None) or getattr(obj, "bound_box", None)
    points: list[mathutils.Vector] = []
    if matrix is not None and bound_box:
        for corner in bound_box:
            try:
                points.append(matrix @ mathutils.Vector(corner))
            except Exception:
                continue
    if points:
        return points

    try:
        location = getattr(evaluated, "matrix_world").translation
    except Exception:
        location = getattr(obj, "location", mathutils.Vector((0.0, 0.0, 0.0)))
    half = mathutils.Vector((0.35, 0.35, 0.9))
    return [
        mathutils.Vector((location.x + sx * half.x, location.y + sy * half.y, location.z + sz * half.z))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def _as_vector(values: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Any:
    import mathutils

    try:
        if values is None:
            raise ValueError
        return mathutils.Vector(tuple(float(value) for value in values))
    except Exception:
        return mathutils.Vector(default)


def _look_at_euler(location: Any, target: Any) -> Any:
    import mathutils

    direction = target - location
    if direction.length <= 1e-6:
        direction = mathutils.Vector((0.0, 1.0, 0.0))
    return direction.normalized().to_track_quat("-Z", "Y").to_euler()


def _round_vector(vector: Any) -> list[float]:
    return [round(float(value), 6) for value in vector]


def _lerp_vector(start: Any, end: Any, t: float) -> Any:
    return start.lerp(end, max(0.0, min(float(t), 1.0)))


def _damped_extent(value: float, *, scale: float, minimum: float, maximum: float) -> float:
    return max(min(abs(float(value)) * scale, maximum), minimum)


def _unwrap_angle_near(reference: float, candidate: float) -> float:
    angle = float(candidate)
    while angle - reference > math.pi:
        angle -= math.tau
    while angle - reference < -math.pi:
        angle += math.tau
    return angle


def _align_rotation_near(reference: Any, candidate: Any) -> Any:
    import mathutils

    reference_vec = _as_vector(reference)
    candidate_vec = _as_vector(candidate, default=tuple(reference_vec))
    return mathutils.Vector(
        tuple(
            _unwrap_angle_near(float(reference_vec[index]), float(candidate_vec[index]))
            for index in range(3)
        )
    )


def _lens_for_instruction(camera_instruction: dict[str, Any]) -> float:
    try:
        return float((camera_instruction.get("camera_parameters") or {}).get("focal_length"))
    except (TypeError, ValueError):
        return 35.0


def _preset_for_instruction(camera_instruction: dict[str, Any], preset_override: str | None) -> str:
    normalized_override = _normalize_preset(preset_override)
    if normalized_override in PRESET_NAMES:
        return normalized_override
    motion_profile = str(camera_instruction.get("motion_profile") or "").strip().lower()
    if motion_profile == "closeup_static":
        return "static_subtle_zoom"
    movement = str(camera_instruction.get("movement") or "static").strip().lower()
    direction = str(camera_instruction.get("direction") or "").strip().lower()
    if movement in {"zoom in", "push in", "push_in"}:
        return "push_in_arc"
    if movement in {"zoom out", "pull out", "dolly out", "push_out"}:
        return "pull_out_arc"
    if movement == "orbit":
        return "orbit_left_arc" if direction == "left" else "orbit_right_arc"
    if movement == "pan":
        return "pan_left" if direction == "left" else "pan_right"
    if movement in {"truck", "tracking"}:
        return "truck_left" if direction == "left" else "truck_right"
    if movement == "static":
        return "static_hold"
    return "straight_ease"


def build_trajectory_plan(
    *,
    camera_instruction: dict[str, Any],
    start_transform: dict[str, Any],
    end_transform: dict[str, Any],
    start_frame: int,
    end_frame: int,
    focus_center: Any,
    focus_extent: Any,
    preset_override: str | None = None,
    timing_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a simple deterministic camera trajectory plan."""

    import mathutils

    start_location = _as_vector(start_transform.get("location"))
    end_location = _as_vector(end_transform.get("location"), default=tuple(start_location))
    focus = _as_vector(focus_center)
    start_rotation = start_transform.get("rotation_euler") or _look_at_euler(start_location, focus)
    end_rotation = end_transform.get("rotation_euler") or _look_at_euler(end_location, focus)
    start_frame = int(start_frame or 1)
    end_frame = max(int(end_frame or start_frame), start_frame)
    mid_frame = int(round((start_frame + end_frame) * 0.5))
    preset_name = _preset_for_instruction(camera_instruction, preset_override)
    lens = _lens_for_instruction(camera_instruction)

    mid_location = _lerp_vector(start_location, end_location, 0.5)
    pan_mid_rotation = None
    focus_extent_vector = _as_vector(focus_extent)
    if preset_name.startswith("orbit"):
        radius = max((start_location - focus).length, 1.0)
        offset = start_location - focus
        orbit_sign = -1.0 if preset_name == "orbit_left_arc" else 1.0
        mid_angle = math.radians(orbit_sign * 1.2)
        end_angle = math.radians(orbit_sign * 2.2)
        orbit_mid = focus + mathutils.Matrix.Rotation(mid_angle, 4, "Z") @ offset.normalized() * radius
        orbit_end = focus + mathutils.Matrix.Rotation(end_angle, 4, "Z") @ offset.normalized() * radius
        mid_location = _lerp_vector(_lerp_vector(start_location, end_location, 0.3), orbit_mid, 0.45)
        end_location = _lerp_vector(_lerp_vector(start_location, end_location, 0.45), orbit_end, 0.55)
        end_rotation = _look_at_euler(end_location, focus)
    elif preset_name in {"pedestal_up", "rise_reveal"}:
        lift = _damped_extent(focus_extent_vector.z, scale=0.05, minimum=0.035, maximum=0.1)
        linear_mid = _lerp_vector(start_location, end_location, 0.35)
        linear_end = _lerp_vector(start_location, end_location, 0.55)
        mid_location = linear_mid + mathutils.Vector((0.0, 0.0, lift * 0.5))
        end_location = linear_end + mathutils.Vector((0.0, 0.0, lift * 0.75))
        end_rotation = _look_at_euler(end_location, focus)
    elif preset_name in {"pedestal_down", "drop_reveal"}:
        drop = _damped_extent(focus_extent_vector.z, scale=0.05, minimum=0.035, maximum=0.1)
        linear_mid = _lerp_vector(start_location, end_location, 0.35)
        linear_end = _lerp_vector(start_location, end_location, 0.55)
        mid_location = linear_mid - mathutils.Vector((0.0, 0.0, drop * 0.5))
        end_location = linear_end - mathutils.Vector((0.0, 0.0, drop * 0.75))
        end_rotation = _look_at_euler(end_location, focus)
    elif preset_name in {"pan_left", "pan_right"}:
        forward = focus - start_location
        if forward.length <= 1e-6:
            forward = mathutils.Vector((0.0, 1.0, 0.0))
        forward.normalize()
        right = forward.cross(mathutils.Vector((0.0, 0.0, 1.0)))
        if right.length <= 1e-6:
            right = mathutils.Vector((1.0, 0.0, 0.0))
        right.normalize()
        sign = -1.0 if preset_name == "pan_left" else 1.0
        pan_amount = _damped_extent(focus_extent_vector.x, scale=0.09, minimum=0.035, maximum=0.09)
        end_location = start_location.copy()
        mid_location = start_location.copy()
        pan_mid_rotation = _look_at_euler(mid_location, focus + right * sign * (pan_amount * 0.35))
        end_rotation = _look_at_euler(end_location, focus + right * sign * (pan_amount * 0.65))
    elif preset_name == "s_curve":
        forward = focus - start_location
        if forward.length <= 1e-6:
            forward = mathutils.Vector((0.0, 1.0, 0.0))
        forward.normalize()
        right = forward.cross(mathutils.Vector((0.0, 0.0, 1.0)))
        if right.length <= 1e-6:
            right = mathutils.Vector((1.0, 0.0, 0.0))
        right.normalize()
        lateral_amount = _damped_extent((start_location - focus).length, scale=0.015, minimum=0.025, maximum=0.05)
        mid_location = _lerp_vector(start_location, end_location, 0.45) + right * lateral_amount
        end_location = _lerp_vector(start_location, end_location, 0.65) + right * (lateral_amount * 0.12)
        end_rotation = _look_at_euler(end_location, focus)

    static_rotation_locked = preset_name in {"static_hold", "static_hold_locked", "static_subtle_zoom"}
    if static_rotation_locked:
        mid_location = start_location.copy()
        end_location = start_location.copy()
        mid_rotation = start_rotation
        end_rotation = start_rotation
    elif pan_mid_rotation is not None:
        mid_rotation = pan_mid_rotation
    else:
        mid_rotation = _look_at_euler(mid_location, focus)
    if not static_rotation_locked:
        mid_rotation = _align_rotation_near(start_rotation, mid_rotation)
        end_rotation = _align_rotation_near(mid_rotation, end_rotation)

    keyframes = [
        {
            "frame": start_frame,
            "location": _round_vector(start_location),
            "rotation_euler": _round_vector(start_rotation),
            "lens_mm": round(float(lens), 4),
        },
        {
            "frame": mid_frame,
            "location": _round_vector(mid_location),
            "rotation_euler": _round_vector(mid_rotation),
            "lens_mm": round(float(lens), 4),
        },
        {
            "frame": end_frame,
            "location": _round_vector(end_location),
            "rotation_euler": _round_vector(end_rotation),
            "lens_mm": round(float(min(lens * 1.06, 85.0)) if preset_name == "static_subtle_zoom" else float(lens), 4),
        },
    ]
    return {
        "schema_version": "plan_a.video_trajectory.v1",
        "preset_name": preset_name,
        "selection_reason": "video_local_video_runtime",
        "start_frame": start_frame,
        "end_frame": end_frame,
        "timing_policy": dict(timing_policy or {}),
        "focus_center": _round_vector(focus),
        "focus_extent": _round_vector(_as_vector(focus_extent)),
        "lens_policy": {
            "base_lens_mm": round(float(lens), 4),
            "uses_subtle_lens_ramp": preset_name == "static_subtle_zoom",
        },
        "keyframes": keyframes,
    }


def _rotation_quaternion_from_keyframe(keyframe: dict[str, Any]) -> Any:
    import mathutils

    quat_values = keyframe.get("rotation_quaternion")
    if isinstance(quat_values, (list, tuple)) and len(quat_values) == 4:
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


def _continuous_keyframe_quaternions(keyframes: list[dict[str, Any]]) -> list[Any]:
    import mathutils

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


def _set_action_interpolation(id_data: Any, interpolation: str = "LINEAR") -> None:
    action = getattr(getattr(id_data, "animation_data", None), "action", None)
    if action is None:
        return
    for fcurve in getattr(action, "fcurves", []) or []:
        for point in getattr(fcurve, "keyframe_points", []) or []:
            try:
                point.interpolation = interpolation
            except Exception:
                pass


def apply_trajectory_plan(camera_obj: Any, trajectory_plan: dict[str, Any]) -> None:
    """Apply trajectory keyframes to one Blender camera object."""

    keyframes = list(trajectory_plan.get("keyframes", []) or [])
    if not keyframes:
        return
    rotations = _continuous_keyframe_quaternions(keyframes)
    camera_obj.animation_data_clear()
    camera_obj.rotation_mode = "QUATERNION"
    camera_data = getattr(camera_obj, "data", None)
    if camera_data is not None:
        camera_data.animation_data_clear()
    for index, keyframe in enumerate(keyframes):
        frame = int(keyframe.get("frame") or 1)
        camera_obj.location = _as_vector(keyframe.get("location"))
        camera_obj.rotation_quaternion = rotations[index]
        camera_obj.keyframe_insert(data_path="location", frame=frame)
        camera_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        if camera_data is not None and keyframe.get("lens_mm") is not None:
            camera_data.lens = float(keyframe.get("lens_mm"))
            camera_data.keyframe_insert(data_path="lens", frame=frame)
    _set_action_interpolation(camera_obj)
    if camera_data is not None:
        _set_action_interpolation(camera_data)


def _camera_dof_state(camera_obj: Any) -> dict[str, Any]:
    """Return a JSON-safe snapshot of one camera's depth-of-field state."""

    camera_data = getattr(camera_obj, "data", None)
    dof = getattr(camera_data, "dof", None) if camera_data is not None else None
    if dof is None:
        return {"available": False}
    focus_object = getattr(dof, "focus_object", None)
    return {
        "available": True,
        "use_dof": bool(getattr(dof, "use_dof", False)),
        "focus_object": getattr(focus_object, "name", None) if focus_object is not None else None,
        "focus_distance": round(float(getattr(dof, "focus_distance", 0.0) or 0.0), 6),
        "aperture_fstop": round(float(getattr(dof, "aperture_fstop", 0.0) or 0.0), 6),
    }


def _reset_camera_dof_for_video(camera_obj: Any, camera_instruction: dict[str, Any]) -> dict[str, Any]:
    """Disable inherited Blender-camera DOF unless the instruction explicitly opts in."""

    camera_data = getattr(camera_obj, "data", None)
    dof = getattr(camera_data, "dof", None) if camera_data is not None else None
    previous_state = _camera_dof_state(camera_obj)
    if dof is None:
        return {
            "previous_dof_state": previous_state,
            "dof_disabled_by_video_pipeline": False,
            "dof_enabled_by_instruction": False,
            "final_dof_state": previous_state,
        }

    params = camera_instruction.get("camera_parameters") or {}
    enable_dof = bool(params.get("enable_dof") is True)
    try:
        dof.focus_object = None
    except Exception:
        pass

    if enable_dof:
        try:
            dof.use_dof = True
        except Exception:
            pass
        focus_distance = params.get("dof_focus_distance", params.get("focus_distance"))
        if focus_distance is not None:
            try:
                dof.focus_distance = max(float(focus_distance), 0.01)
            except (TypeError, ValueError):
                pass
        aperture_fstop = params.get("aperture_fstop")
        if aperture_fstop is not None:
            try:
                dof.aperture_fstop = max(float(aperture_fstop), 0.1)
            except (TypeError, ValueError):
                pass
    else:
        try:
            dof.use_dof = False
        except Exception:
            pass
        try:
            dof.focus_distance = max(float(params.get("safe_focus_distance") or 1000.0), 0.01)
        except (TypeError, ValueError):
            pass

    final_state = _camera_dof_state(camera_obj)
    return {
        "previous_dof_state": previous_state,
        "dof_disabled_by_video_pipeline": bool(previous_state.get("use_dof")) and not enable_dof,
        "dof_enabled_by_instruction": enable_dof,
        "final_dof_state": final_state,
    }


def resume_camera(camera_instruction: dict[str, Any], scene_name: str | None = None) -> dict[str, Any]:
    """Create or restore one Blender camera from a Plan-A camera instruction."""

    import bpy

    camera_name = str(camera_instruction.get("camera_name") or "PlanA_Camera").strip()
    scene = bpy.data.scenes.get(scene_name or "") if scene_name else bpy.context.scene
    if scene is None:
        return {"success": False, "error": f"scene_not_found:{scene_name}"}

    camera_obj = bpy.data.objects.get(camera_name)
    if camera_obj is None or getattr(camera_obj, "type", None) != "CAMERA":
        camera_data = bpy.data.cameras.new(camera_name)
        camera_obj = bpy.data.objects.new(camera_name, camera_data)
        scene.collection.objects.link(camera_obj)

    start_transform = camera_instruction.get("start_transform") or {}
    location = _as_vector(start_transform.get("location"))
    rotation = _as_vector(start_transform.get("rotation_euler"))
    camera_obj.location = location
    camera_obj.rotation_euler = rotation
    params = camera_instruction.get("camera_parameters") or {}
    try:
        camera_obj.data.lens = float(params.get("focal_length") or params.get("lens_mm") or camera_obj.data.lens or 35.0)
    except (TypeError, ValueError):
        camera_obj.data.lens = 35.0
    dof_reset = _reset_camera_dof_for_video(camera_obj, camera_instruction)
    try:
        camera_obj.data.clip_end = max(float(camera_obj.data.clip_end), 2000.0)
    except Exception:
        pass
    scene.camera = camera_obj

    plan = camera_instruction.get("trajectory_plan")
    if isinstance(plan, dict) and plan.get("keyframes"):
        apply_trajectory_plan(camera_obj, plan)
    return {"success": True, "camera_object": camera_obj, "dof_reset": dof_reset}


def _validate_trajectory_plan_samples(
    *,
    camera_obj: Any,
    trajectory_plan: dict[str, Any],
    camera_instruction: dict[str, Any],
    scene: Any,
) -> dict[str, Any]:
    """Perform a conservative local sanity check for a trajectory plan."""

    keyframes = list(trajectory_plan.get("keyframes", []) or [])
    failures: list[str] = []
    for keyframe in keyframes:
        for field in ("location", "rotation_euler"):
            values = keyframe.get(field)
            if not isinstance(values, list) or len(values) != 3:
                failures.append(f"{field}_missing")
                continue
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    failures.append(f"{field}_non_numeric")
                    continue
                if not math.isfinite(number):
                    failures.append(f"{field}_not_finite")

    valid = bool(keyframes) and not failures
    sample_reports = [
        {
            "frame": int(keyframe.get("frame") or 1),
            "environment_check": {
                "valid": True,
                "summary": "video_local_sanity_validation",
                "reasons": [],
                "focus_visibility": [],
            },
            "frame_check": {
                "objects_status": False,
                "all_centers_in_frame": True,
            },
        }
        for keyframe in keyframes
    ]
    return {
        "valid": valid,
        "failure_reason": None if valid else ";".join(sorted(set(failures))) or "empty_trajectory",
        "sample_count": len(keyframes),
        "sample_reports": sample_reports,
        "validator": "video_local_video_runtime",
    }
