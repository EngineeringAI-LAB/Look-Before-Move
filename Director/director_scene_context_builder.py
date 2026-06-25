"""Standalone scene-context builder for Plan-A scene dossier scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

CODE_DIR = Path(__file__).resolve().parent

from director_engine_paths import load_json
from director_scene_dossier import (
    SceneDossier,
    load_scene_dossier_json,
    save_scene_dossier_json,
    save_scene_dossier_manifest,
)

try:  # pragma: no cover - Blender-only runtime
    import bpy
    import mathutils
except Exception:  # pragma: no cover
    bpy = None
    mathutils = None


DEFAULT_OBJECT_FRONT_AXIS = "-Y"
TURNAROUND_ANCHOR_TOLERANCE_MIN = 0.15
TURNAROUND_ANCHOR_TOLERANCE_SCALE = 0.6
LOCAL_FRONT_AXIS_VECTORS = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
}


def _as_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def collect_scene_character_ids(
    scene_shots: list[dict[str, Any]],
    extra_character_ids: list[str] | None = None,
) -> list[str]:
    """Collect unique character ids from shot payloads.

    Actors referenced via ``character_actions`` are returned first to preserve
    the existing primary-anchor priority. ``extra_character_ids`` is appended
    last (deduplicated) so that statically-placed characters — e.g. a subject
    present in the scene layout but not driving any shot action — are still
    recognised as characters by the dossier builder. This avoids downstream
    LLM stages concluding that such characters are "not modeled".
    """

    ordered_ids: list[str] = []
    for shot in scene_shots:
        for action in _as_list(shot.get("character_actions")):
            if not isinstance(action, dict):
                continue
            asset_id = _as_str(action.get("asset_id"))
            if asset_id and asset_id not in ordered_ids:
                ordered_ids.append(asset_id)
    for asset_id in extra_character_ids or []:
        normalized = _as_str(asset_id)
        if normalized and normalized not in ordered_ids:
            ordered_ids.append(normalized)
    return ordered_ids


def collect_scene_focus_asset_ids(scene_shots: list[dict[str, Any]]) -> list[str]:
    """Collect focus and modified asset ids referenced by one scene."""

    ordered_ids: list[str] = []
    for shot in scene_shots:
        for camera_instruction in [shot.get("camera_instruction"), *(_as_list(shot.get("additional_camera_instructions")))]:
            if not isinstance(camera_instruction, dict):
                continue
            for asset_id in _as_list(camera_instruction.get("focus_on_ids")):
                normalized = _as_str(asset_id)
                if normalized and normalized not in ordered_ids:
                    ordered_ids.append(normalized)
        for modification in _as_list(shot.get("asset_modifications")):
            if not isinstance(modification, dict):
                continue
            for key in ("asset_id", "anchor_asset_id"):
                asset_id = _as_str(modification.get(key))
                if asset_id and asset_id not in ordered_ids:
                    ordered_ids.append(asset_id)
    return ordered_ids


def build_scene_context_request_payload(
    *,
    filtered_input_path: str | Path,
    output_root: str | Path,
    project_dir: str | Path,
    scene_ids: list[int],
    resolution_x: int,
    resolution_y: int,
    margin_factor: float,
    scene_character_ids: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    """Build one JSON-serializable request for the Blender context worker.

    ``scene_character_ids`` lets the caller pre-compute the authoritative list
    of character asset ids per scene (e.g. by intersecting the scene layout
    with the asset index). The worker uses it to ensure statically-placed
    characters are included in the dossier even if no shot action references
    them as actors.
    """

    serialized_scene_character_ids: dict[str, list[str]] = {}
    for scene_id, asset_ids in (scene_character_ids or {}).items():
        try:
            key = str(int(scene_id))
        except (TypeError, ValueError):
            continue
        normalized = [str(asset_id) for asset_id in asset_ids or [] if str(asset_id or "").strip()]
        if normalized:
            serialized_scene_character_ids[key] = normalized

    return {
        "schema_version": "plan_a.scene_context_build_request.v1",
        "filtered_input_path": str(Path(filtered_input_path).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "project_dir": str(Path(project_dir).resolve()),
        "scene_ids": [int(scene_id) for scene_id in scene_ids],
        "resolution_x": int(resolution_x),
        "resolution_y": int(resolution_y),
        "margin_factor": float(margin_factor),
        "scene_character_ids": serialized_scene_character_ids,
    }


def _iter_scene_shots(filtered_shots: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for shot in filtered_shots:
        try:
            scene_id = int(shot.get("scene_id"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(scene_id, []).append(shot)
    return grouped


def build_scene_dossier(
    *,
    scene_id: str,
    scene_top_view_path: str = "",
    layout_data: Mapping[str, Any] | None = None,
    character_summaries: list[Mapping[str, Any]] | None = None,
    asset_summaries: list[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SceneDossier:
    """Build one normalized scene dossier."""

    payload = {
        "scene_id": _as_str(scene_id),
        "scene_top_view_path": _as_str(scene_top_view_path),
        "layout_data": _as_dict(layout_data),
        "character_summaries": [row for row in _as_list(character_summaries) if isinstance(row, dict)],
        "asset_summaries": [row for row in _as_list(asset_summaries) if isinstance(row, dict)],
        "metadata": _as_dict(metadata),
    }
    return SceneDossier.from_dict(payload)


def build_scene_dossier_from_payload(payload: Mapping[str, Any]) -> SceneDossier:
    """Build a scene dossier from one loose payload dictionary."""

    return build_scene_dossier(
        scene_id=_as_str(payload.get("scene_id")),
        scene_top_view_path=_as_str(payload.get("scene_top_view_path")),
        layout_data=_as_dict(payload.get("layout_data")),
        character_summaries=[row for row in _as_list(payload.get("character_summaries")) if isinstance(row, dict)],
        asset_summaries=[row for row in _as_list(payload.get("asset_summaries")) if isinstance(row, dict)],
        metadata=_as_dict(payload.get("metadata")),
    )


def _scene_for_worker():
    if bpy is None or mathutils is None:  # pragma: no cover
        raise RuntimeError("Blender scene worker requires bpy and mathutils.")
    return bpy.context.scene


def _scene_name_candidates(scene_id: int, shot_id: int | None = None) -> list[str]:
    candidates: list[str] = []
    if shot_id is not None:
        candidates.extend(
            [
                f"Scene_{scene_id}_Shot_{shot_id}",
                f"Scene_{scene_id}_shot_{shot_id}",
            ]
        )
    candidates.extend(
        [
            f"Scene_{scene_id}",
            f"Scene {scene_id}",
        ]
    )
    return candidates


def _resolve_scene(scene_id: int, shot_id: int | None = None):
    if bpy is None:
        return None
    for scene_name in _scene_name_candidates(scene_id, shot_id):
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            return scene
    return None


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
    """Best-effort local asset-id lookup without shared helpers."""

    normalized = _as_str(asset_id)
    if not normalized or scene is None:
        return None
    if normalized in scene.objects:
        return scene.objects[normalized]

    exact_match = None
    dot_matches = []
    underscore_matches = []
    custom_property_matches = []
    for obj in scene.objects:
        if obj.name == normalized:
            exact_match = obj
            break
        if obj.name.startswith(f"{normalized}."):
            dot_matches.append(obj)
        elif obj.name.startswith(f"{normalized}_"):
            underscore_matches.append(obj)
        try:
            object_asset_id = str(obj.get("asset_id") or "").strip()
        except Exception:
            object_asset_id = ""
        if object_asset_id == normalized:
            custom_property_matches.append(obj)
    if exact_match is not None:
        return exact_match
    if custom_property_matches:
        return custom_property_matches[0]
    if dot_matches:
        return dot_matches[0]
    if underscore_matches:
        return underscore_matches[0]
    return None


def _resolve_character_scene(scene_id: int, scene_shots: list[dict[str, Any]], asset_id: str):
    for shot in scene_shots:
        shot_id = shot.get("shot_id")
        try:
            shot_id_int = int(shot_id)
        except (TypeError, ValueError):
            continue
        for action in _as_list(shot.get("character_actions")):
            if not isinstance(action, dict):
                continue
            if _as_str(action.get("asset_id")) != asset_id:
                continue
            shot_scene = _resolve_scene(scene_id, shot_id_int)
            if shot_scene is not None:
                return shot_scene
    return _resolve_scene(scene_id)


def _iter_candidate_scenes_for_asset(scene_id: int, scene_shots: list[dict[str, Any]], asset_id: str) -> list[Any]:
    candidates: list[Any] = []
    for shot in scene_shots:
        shot_id = shot.get("shot_id")
        try:
            shot_id_int = int(shot_id)
        except (TypeError, ValueError):
            continue
        referenced = False
        for action in _as_list(shot.get("character_actions")):
            if isinstance(action, dict) and _as_str(action.get("asset_id")) == asset_id:
                referenced = True
                break
        if not referenced:
            for camera_instruction in [shot.get("camera_instruction"), *(_as_list(shot.get("additional_camera_instructions")))]:
                if not isinstance(camera_instruction, dict):
                    continue
                if asset_id in _as_list(camera_instruction.get("focus_on_ids")):
                    referenced = True
                    break
        if referenced:
            shot_scene = _resolve_scene(scene_id, shot_id_int)
            if shot_scene is not None and shot_scene not in candidates:
                candidates.append(shot_scene)
    base_scene = _resolve_scene(scene_id)
    if base_scene is not None and base_scene not in candidates:
        candidates.append(base_scene)
    return candidates


def _points_bounds(points: list[Any]) -> tuple[Any, Any]:
    first_point = points[0].copy()
    min_corner = first_point.copy()
    max_corner = first_point.copy()
    for point in points[1:]:
        min_corner.x = min(min_corner.x, point.x)
        min_corner.y = min(min_corner.y, point.y)
        min_corner.z = min(min_corner.z, point.z)
        max_corner.x = max(max_corner.x, point.x)
        max_corner.y = max(max_corner.y, point.y)
        max_corner.z = max(max_corner.z, point.z)
    return min_corner, max_corner


def _points_center_and_extent(points: list[Any]) -> tuple[Any, Any]:
    min_corner, max_corner = _points_bounds(points)
    return (min_corner + max_corner) * 0.5, max_corner - min_corner


def _object_local_bbox_points(obj) -> list[Any]:
    corners: list[Any] = []
    try:
        for corner in obj.bound_box:
            corners.append(mathutils.Vector(corner))
    except Exception:
        corners = []
    return corners or [mathutils.Vector((0.0, 0.0, 0.0))]


def _world_points_from_matrix(matrix, local_points: list[Any]) -> list[Any]:
    return [matrix @ point for point in local_points]


def _location_offset_points(obj, local_points: list[Any]) -> list[Any]:
    try:
        location = mathutils.Vector((float(obj.location.x), float(obj.location.y), float(obj.location.z)))
    except Exception:
        location = mathutils.Vector((0.0, 0.0, 0.0))
    return [location + point for point in local_points]


def _world_bounds_candidate(
    *,
    source: str,
    points: list[Any],
    expected_center,
    local_extent,
) -> dict[str, Any]:
    min_corner, max_corner = _points_bounds(points)
    center = (min_corner + max_corner) * 0.5
    distance = 0.0
    consistent = True
    if expected_center is not None:
        distance = float((center - expected_center).length)
        tolerance = max(
            float(local_extent.x),
            float(local_extent.y),
            float(local_extent.z),
            TURNAROUND_ANCHOR_TOLERANCE_MIN,
        ) * TURNAROUND_ANCHOR_TOLERANCE_SCALE
        tolerance = max(tolerance, TURNAROUND_ANCHOR_TOLERANCE_MIN)
        consistent = distance <= tolerance
    return {
        "source": source,
        "min": min_corner,
        "max": max_corner,
        "center": center,
        "distance_to_expected": distance,
        "consistent": consistent,
    }


def _reliable_object_world_bounds(obj) -> tuple[Any, Any, str]:
    local_points = _object_local_bbox_points(obj)
    local_center, local_extent = _points_center_and_extent(local_points)
    try:
        object_location = mathutils.Vector((float(obj.location.x), float(obj.location.y), float(obj.location.z)))
        expected_center = object_location + local_center
    except Exception:
        expected_center = None

    candidates: dict[str, dict[str, Any]] = {}
    if bpy is not None:
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated_obj = obj.evaluated_get(depsgraph)
            candidates["evaluated_world"] = _world_bounds_candidate(
                source="evaluated_world",
                points=_world_points_from_matrix(evaluated_obj.matrix_world, local_points),
                expected_center=expected_center,
                local_extent=local_extent,
            )
        except Exception:
            pass
    try:
        candidates["matrix_world"] = _world_bounds_candidate(
            source="matrix_world",
            points=_world_points_from_matrix(obj.matrix_world, local_points),
            expected_center=expected_center,
            local_extent=local_extent,
        )
    except Exception:
        pass
    candidates["location_fallback"] = _world_bounds_candidate(
        source="location_fallback",
        points=_location_offset_points(obj, local_points),
        expected_center=expected_center,
        local_extent=local_extent,
    )

    for source_name in ("evaluated_world", "matrix_world"):
        candidate = candidates.get(source_name)
        if candidate and candidate.get("consistent"):
            return candidate["min"], candidate["max"], candidate["source"]

    fallback_candidate = candidates["location_fallback"]
    if expected_center is not None:
        return fallback_candidate["min"], fallback_candidate["max"], fallback_candidate["source"]

    for source_name in ("evaluated_world", "matrix_world"):
        candidate = candidates.get(source_name)
        if candidate:
            return candidate["min"], candidate["max"], candidate["source"]
    return fallback_candidate["min"], fallback_candidate["max"], fallback_candidate["source"]


def _reliable_group_bounds(objects: list[Any]) -> tuple[Any, Any, str]:
    first_min, first_max, first_source = _reliable_object_world_bounds(objects[0])
    min_corner = first_min.copy()
    max_corner = first_max.copy()
    sources = {first_source}
    for obj in objects[1:]:
        obj_min, obj_max, obj_source = _reliable_object_world_bounds(obj)
        sources.add(obj_source)
        min_corner.x = min(min_corner.x, obj_min.x)
        min_corner.y = min(min_corner.y, obj_min.y)
        min_corner.z = min(min_corner.z, obj_min.z)
        max_corner.x = max(max_corner.x, obj_max.x)
        max_corner.y = max(max_corner.y, obj_max.y)
        max_corner.z = max(max_corner.z, obj_max.z)
    if len(sources) == 1:
        return min_corner, max_corner, next(iter(sources))
    return min_corner, max_corner, f"mixed:{','.join(sorted(sources))}"


def _object_world_bounds(obj) -> tuple[Any, Any]:
    corners = []
    try:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ mathutils.Vector(corner))
    except Exception:
        location = obj.matrix_world.translation
        corners.append(mathutils.Vector((location.x, location.y, location.z)))
    min_corner = corners[0].copy()
    max_corner = corners[0].copy()
    for corner in corners[1:]:
        min_corner.x = min(min_corner.x, corner.x)
        min_corner.y = min(min_corner.y, corner.y)
        min_corner.z = min(min_corner.z, corner.z)
        max_corner.x = max(max_corner.x, corner.x)
        max_corner.y = max(max_corner.y, corner.y)
        max_corner.z = max(max_corner.z, corner.z)
    return min_corner, max_corner


def _group_bounds(objects: list[Any]) -> tuple[Any, Any]:
    first_min, first_max = _object_world_bounds(objects[0])
    min_corner = first_min.copy()
    max_corner = first_max.copy()
    for obj in objects[1:]:
        obj_min, obj_max = _object_world_bounds(obj)
        min_corner.x = min(min_corner.x, obj_min.x)
        min_corner.y = min(min_corner.y, obj_min.y)
        min_corner.z = min(min_corner.z, obj_min.z)
        max_corner.x = max(max_corner.x, obj_max.x)
        max_corner.y = max(max_corner.y, obj_max.y)
        max_corner.z = max(max_corner.z, obj_max.z)
    return min_corner, max_corner


def _object_center_and_extent(obj) -> tuple[Any, Any]:
    min_corner, max_corner = _object_world_bounds(obj)
    return (min_corner + max_corner) * 0.5, max_corner - min_corner


def _group_center_and_extent(objects: list[Any]) -> tuple[Any, Any]:
    min_corner, max_corner = _group_bounds(objects)
    return (min_corner + max_corner) * 0.5, max_corner - min_corner


def _max_extent_component(objects: list[Any]) -> float:
    _, extent = _group_center_and_extent(_preferred_bounds_objects(objects))
    return max(float(extent.x), float(extent.y), float(extent.z), 0.0)


def _descendant_objects(root_obj) -> list[Any]:
    descendants: list[Any] = []
    stack = [root_obj]
    seen = set()
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


def _preferred_bounds_objects(objects: list[Any]) -> list[Any]:
    mesh_like = [obj for obj in objects if getattr(obj, "type", "") in {"MESH", "CURVE", "SURFACE", "META", "FONT"}]
    return mesh_like or objects


def _preferred_render_objects(objects: list[Any]) -> list[Any]:
    mesh_like = [obj for obj in objects if getattr(obj, "type", "") in {"MESH", "CURVE", "SURFACE", "META", "FONT"}]
    armatures = [obj for obj in objects if getattr(obj, "type", "") == "ARMATURE"]
    preferred = armatures + mesh_like
    return preferred or objects


def _ensure_scene_render_settings(scene, *, resolution_x: int, resolution_y: int, transparent: bool) -> dict[str, Any]:
    original = {
        "engine": str(scene.render.engine),
        "resolution_x": int(scene.render.resolution_x),
        "resolution_y": int(scene.render.resolution_y),
        "transparent": bool(scene.render.film_transparent),
        "view_transform": "",
        "look": "",
        "exposure": 0.0,
        "gamma": 1.0,
        "cycles_samples": None,
        "cycles_use_denoising": None,
    }
    try:
        original["view_transform"] = str(scene.view_settings.view_transform)
        original["look"] = str(scene.view_settings.look)
        original["exposure"] = float(scene.view_settings.exposure)
        original["gamma"] = float(scene.view_settings.gamma)
    except Exception:
        pass
    if hasattr(scene, "cycles"):
        try:
            original["cycles_samples"] = int(scene.cycles.samples)
            original["cycles_use_denoising"] = bool(scene.cycles.use_denoising)
        except Exception:
            pass

    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = transparent
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass
    if hasattr(scene, "cycles"):
        try:
            scene.cycles.samples = 16
            scene.cycles.use_denoising = True
        except Exception:
            pass
    return original


def _restore_scene_render_settings(scene, original: dict[str, Any]) -> None:
    scene.render.engine = original.get("engine", scene.render.engine)
    scene.render.resolution_x = int(original.get("resolution_x", scene.render.resolution_x))
    scene.render.resolution_y = int(original.get("resolution_y", scene.render.resolution_y))
    scene.render.film_transparent = bool(original.get("transparent", scene.render.film_transparent))
    try:
        scene.view_settings.view_transform = str(original.get("view_transform") or scene.view_settings.view_transform)
        scene.view_settings.look = str(original.get("look") or scene.view_settings.look)
        scene.view_settings.exposure = float(original.get("exposure", scene.view_settings.exposure))
        scene.view_settings.gamma = float(original.get("gamma", scene.view_settings.gamma))
    except Exception:
        pass
    if hasattr(scene, "cycles"):
        try:
            if original.get("cycles_samples") is not None:
                scene.cycles.samples = int(original["cycles_samples"])
            if original.get("cycles_use_denoising") is not None:
                scene.cycles.use_denoising = bool(original["cycles_use_denoising"])
        except Exception:
            pass


def _create_preview_world(name: str, *, color: tuple[float, float, float, float], strength: float):
    world = bpy.data.worlds.new(name=name)
    world.use_nodes = True
    node_tree = world.node_tree
    if node_tree is None:
        return world
    background = node_tree.nodes.get("Background")
    if background is None:
        background = node_tree.nodes.new(type="ShaderNodeBackground")
    background.inputs[0].default_value = color
    background.inputs[1].default_value = strength
    return world


def _remove_preview_world(world) -> None:
    if world is None:
        return
    try:
        bpy.data.worlds.remove(world, do_unlink=True)
    except Exception:
        pass


def _create_temp_light(scene, name: str, light_type: str, *, energy: float, size: float = 1.0):
    light_data = bpy.data.lights.new(name=f"{name}_data", type=light_type)
    light_data.energy = energy
    if hasattr(light_data, "size"):
        light_data.size = size
    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    scene.collection.objects.link(light_obj)
    return light_obj


def _remove_temp_light(light_obj) -> None:
    if light_obj is None:
        return
    light_data = getattr(light_obj, "data", None)
    try:
        bpy.data.objects.remove(light_obj, do_unlink=True)
    except Exception:
        pass
    if light_data is not None:
        try:
            bpy.data.lights.remove(light_data, do_unlink=True)
        except Exception:
            pass


def _create_preview_light_rig(scene, *, center, extent) -> list[Any]:
    scale = max(float(extent.x), float(extent.y), float(extent.z), 1.0)
    lights: list[Any] = []

    key = _create_temp_light(scene, "director_preview_key", "AREA", energy=1400.0, size=max(scale * 1.6, 1.4))
    key.location = center + mathutils.Vector((scale * 1.6, -scale * 1.8, scale * 1.8))
    key.rotation_euler = _look_at_rotation(key.location, center)
    lights.append(key)

    fill = _create_temp_light(scene, "director_preview_fill", "AREA", energy=600.0, size=max(scale * 1.8, 1.4))
    fill.location = center + mathutils.Vector((-scale * 1.4, scale * 1.1, scale * 1.3))
    fill.rotation_euler = _look_at_rotation(fill.location, center)
    lights.append(fill)

    sun = _create_temp_light(scene, "director_preview_sun", "SUN", energy=0.7)
    sun.location = center + mathutils.Vector((scale * 0.4, -scale * 0.8, scale * 3.5))
    sun.rotation_euler = _look_at_rotation(sun.location, center)
    lights.append(sun)

    return lights


def _render_camera_to_path(scene, camera_obj, destination: Path) -> str:
    original_camera = scene.camera
    original_filepath = scene.render.filepath
    scene.camera = camera_obj
    scene.render.filepath = str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    scene.camera = original_camera
    scene.render.filepath = original_filepath
    return str(destination.resolve())


def _set_render_visibility(scene, visible_objects: list[Any] | None) -> list[tuple[Any, bool]]:
    original_states: list[tuple[Any, bool]] = []
    if visible_objects is None or scene is None:
        return original_states
    visible_set = set(visible_objects)
    for obj in scene.objects:
        original_states.append((obj, bool(obj.hide_render)))
        obj.hide_render = obj not in visible_set
    return original_states


def _restore_render_visibility(original_states: list[tuple[Any, bool]]) -> None:
    for obj, state in original_states:
        try:
            obj.hide_render = state
        except Exception:
            continue


def _create_temp_camera(scene, name: str, lens_mm: float = 50.0):
    camera_data = bpy.data.cameras.new(name=f"{name}_data")
    camera_data.lens = lens_mm
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


def _look_at_rotation(camera_position, target_position):
    direction = target_position - camera_position
    if direction.length <= 1e-6:
        direction = mathutils.Vector((0.0, 1.0, 0.0))
    return direction.normalized().to_track_quat("-Z", "Y").to_euler()


def _subject_front_axis(obj) -> str:
    if obj is not None and hasattr(obj, "get"):
        for key in ("director_front_axis", "front_axis", "asset_front_axis"):
            try:
                value = str(obj.get(key) or "").strip().upper()
            except Exception:
                value = ""
            if value in LOCAL_FRONT_AXIS_VECTORS:
                return value
    return DEFAULT_OBJECT_FRONT_AXIS


def _subject_world_rotation(obj):
    """Return the evaluated world rotation so animated/placed scene transforms count."""

    if bpy is not None:
        try:
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            return obj.evaluated_get(depsgraph).matrix_world.to_quaternion()
        except Exception:
            pass
    return obj.matrix_world.to_quaternion()


def _subject_basis_vectors(obj) -> dict[str, Any]:
    """Return camera-side subject axes derived from one object's local orientation."""

    local_front = mathutils.Vector(LOCAL_FRONT_AXIS_VECTORS[_subject_front_axis(obj)])
    local_up = mathutils.Vector((0.0, 0.0, 1.0))
    local_right = local_front.cross(local_up).normalized()
    rotation = _subject_world_rotation(obj)
    front = rotation @ local_front
    right = rotation @ local_right
    up = rotation @ local_up
    front.normalize()
    right.normalize()
    up.normalize()
    return {"front": front, "right": right, "up": up}


def _preferred_preview_primary_obj(group_objects: list[Any], fallback_obj):
    """Pick one stable render-facing object for preview basis and framing."""

    render_objects = _preferred_render_objects(group_objects)
    for obj in render_objects:
        if getattr(obj, "type", "") == "MESH":
            return obj
    for obj in group_objects:
        if getattr(obj, "type", "") == "MESH":
            return obj
    return fallback_obj


def _preferred_orientation_basis_obj(group_objects: list[Any], fallback_obj):
    """Pick one stable object whose local axes represent the character facing."""

    if getattr(fallback_obj, "type", "") in {"ARMATURE", "EMPTY", "MESH"}:
        return fallback_obj
    for obj in group_objects:
        if getattr(obj, "type", "") == "ARMATURE":
            return obj
    for obj in group_objects:
        if getattr(obj, "type", "") == "EMPTY":
            return obj
    return _preferred_preview_primary_obj(group_objects, fallback_obj)


def _looks_like_proxy_preview_object(obj) -> bool:
    """Return whether one object likely behaves like a staging proxy/rig root."""

    return getattr(obj, "type", "") in {"ARMATURE", "EMPTY"}


def _capture_character_turnaround(
    *,
    scene,
    character_id: str,
    primary_obj,
    basis_obj,
    group_objects: list[Any],
    bounds_group_objects: list[Any] | None,
    output_dir: Path,
    filename_prefix: str,
    resolution_x: int,
    resolution_y: int,
    margin_factor: float,
    hide_environment: bool,
    visible_scene_objects: list[Any] | None = None,
    preview_frame: float | None = None,
) -> dict[str, str]:
    _set_active_scene(scene)
    render_objects = _preferred_render_objects(group_objects)

    bounds_objects = _preferred_bounds_objects(bounds_group_objects or group_objects)
    min_corner, max_corner, _anchor_source = _reliable_group_bounds(bounds_objects)
    center = (min_corner + max_corner) * 0.5
    extent = max_corner - min_corner
    basis = _subject_basis_vectors(basis_obj or primary_obj)
    horizontal_radius = max(extent.x, extent.y, extent.z, 0.6) * margin_factor
    top_radius = max(horizontal_radius * 1.1, 1.2)
    vertical_offset = max(extent.z * 0.02, 0.03)
    target = center + basis["up"] * vertical_offset
    semantic_directions = {
        "front": basis["front"],
        "front_right": (basis["front"] + basis["right"]).normalized(),
        "right": basis["right"],
        "back_right": (-basis["front"] + basis["right"]).normalized(),
        "back": -basis["front"],
        "back_left": (-basis["front"] - basis["right"]).normalized(),
        "left": -basis["right"],
        "front_left": (basis["front"] - basis["right"]).normalized(),
    }
    image_paths: dict[str, str] = {}
    original_settings = _ensure_scene_render_settings(
        scene,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        transparent=False,
    )
    if hide_environment:
        visibility_states = _set_render_visibility(scene, render_objects)
    elif visible_scene_objects is not None:
        visibility_states = _set_render_visibility(scene, visible_scene_objects)
    else:
        visibility_states = []
    original_world = scene.world
    original_frame = int(scene.frame_current)
    preview_world = None
    light_rig: list[Any] = []
    if hide_environment:
        preview_world = _create_preview_world(
            f"director_preview_world_{character_id}",
            color=(0.985, 0.985, 0.985, 1.0),
            strength=0.08,
        )
        scene.world = preview_world
        light_rig = _create_preview_light_rig(scene, center=center, extent=extent)
    try:
        if preview_frame is not None:
            scene.frame_set(int(round(preview_frame)))
        for direction_name, direction_vector in semantic_directions.items():
            camera_obj = _create_temp_camera(scene, f"director_turnaround_{character_id}_{direction_name}", lens_mm=35.0)
            try:
                camera_obj.location = target + direction_vector * horizontal_radius
                camera_obj.rotation_euler = _look_at_rotation(camera_obj.location, target)
                destination = output_dir / f"{filename_prefix}_{direction_name}.png"
                image_paths[direction_name] = _render_camera_to_path(scene, camera_obj, destination)
            finally:
                _remove_temp_camera(camera_obj)

        top_camera = _create_temp_camera(scene, f"director_turnaround_{character_id}_top", lens_mm=32.0)
        try:
            top_camera.location = center + basis["up"] * (top_radius + extent.z * 0.8)
            top_camera.rotation_euler = _look_at_rotation(top_camera.location, center)
            image_paths["top"] = _render_camera_to_path(
                scene,
                top_camera,
                output_dir / f"{filename_prefix}_top.png",
            )
        finally:
            _remove_temp_camera(top_camera)
    finally:
        for light_obj in light_rig:
            _remove_temp_light(light_obj)
        if preview_world is not None:
            scene.world = original_world
            _remove_preview_world(preview_world)
        if preview_frame is not None:
            scene.frame_set(original_frame)
        _restore_render_visibility(visibility_states)
        _restore_scene_render_settings(scene, original_settings)
    return image_paths


def _capture_character_portrait(
    *,
    scene,
    character_id: str,
    primary_obj,
    group_objects: list[Any],
    bounds_group_objects: list[Any] | None,
    output_dir: Path,
    filename_prefix: str,
    resolution_x: int,
    resolution_y: int,
    margin_factor: float,
    preview_frame: float | None = None,
) -> str:
    _set_active_scene(scene)
    bounds_objects = _preferred_bounds_objects(bounds_group_objects or group_objects)
    render_objects = _preferred_render_objects(group_objects)
    center, extent = _group_center_and_extent(bounds_objects)
    basis = _subject_basis_vectors(primary_obj)
    horizontal_radius = max(extent.x, extent.y, extent.z, 0.6) * margin_factor * 0.92
    vertical_offset = max(extent.z * 0.08, 0.08)
    target = center + basis["up"] * vertical_offset

    original_settings = _ensure_scene_render_settings(
        scene,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        transparent=False,
    )
    original_world = scene.world
    original_frame = int(scene.frame_current)
    visibility_states = _set_render_visibility(scene, render_objects)
    preview_world = _create_preview_world(
        f"director_portrait_world_{character_id}",
        color=(0.985, 0.985, 0.985, 1.0),
        strength=0.08,
    )
    scene.world = preview_world
    light_rig = _create_preview_light_rig(scene, center=center, extent=extent)
    camera_obj = _create_temp_camera(scene, f"director_portrait_{character_id}", lens_mm=50.0)
    try:
        if preview_frame is not None:
            scene.frame_set(int(round(preview_frame)))
        camera_obj.location = target + basis["front"] * horizontal_radius
        camera_obj.rotation_euler = _look_at_rotation(camera_obj.location, target)
        return _render_camera_to_path(
            scene,
            camera_obj,
            output_dir / f"{filename_prefix}.png",
        )
    finally:
        _remove_temp_camera(camera_obj)
        for light_obj in light_rig:
            _remove_temp_light(light_obj)
        scene.world = original_world
        _remove_preview_world(preview_world)
        if preview_frame is not None:
            scene.frame_set(original_frame)
        _restore_render_visibility(visibility_states)
        _restore_scene_render_settings(scene, original_settings)


def _capture_scene_top_view(
    *,
    scene,
    scene_id: int,
    focus_objects: list[Any],
    output_dir: Path,
    resolution_x: int,
    resolution_y: int,
    margin_factor: float,
) -> str:
    _set_active_scene(scene)
    relevant_objects = list(focus_objects) if focus_objects else list(scene.objects)
    center, extent = _group_center_and_extent(relevant_objects)
    vertical_radius = max(extent.x, extent.y, extent.z, 1.0) * margin_factor * 1.6
    camera_obj = _create_temp_camera(scene, f"director_scene_{scene_id}_top", lens_mm=36.0)
    original_settings = _ensure_scene_render_settings(
        scene,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        transparent=False,
    )
    original_world = scene.world
    preview_world = _create_preview_world(
        f"director_scene_preview_world_{scene_id}",
        color=(0.88, 0.89, 0.91, 1.0),
        strength=0.25,
    )
    scene.world = preview_world
    light_rig = _create_preview_light_rig(scene, center=center, extent=extent)
    try:
        camera_obj.location = center + mathutils.Vector((0.0, 0.0, vertical_radius))
        camera_obj.rotation_euler = _look_at_rotation(camera_obj.location, center)
        return _render_camera_to_path(scene, camera_obj, output_dir / f"scene_{scene_id}_top_view.png")
    finally:
        for light_obj in light_rig:
            _remove_temp_light(light_obj)
        scene.world = original_world
        _remove_preview_world(preview_world)
        _restore_scene_render_settings(scene, original_settings)
        _remove_temp_camera(camera_obj)


def _object_layout_summary(obj) -> dict[str, Any]:
    center, extent = _object_center_and_extent(obj)
    return {
        "object_name": obj.name,
        "location": [round(float(value), 6) for value in obj.matrix_world.translation],
        "center": [round(float(value), 6) for value in center],
        "extent": [round(float(value), 6) for value in extent],
        "rotation_euler": [round(float(value), 6) for value in obj.matrix_world.to_euler()],
    }


def _group_layout_summary(objects: list[Any], object_name: str) -> dict[str, Any]:
    center, extent = _group_center_and_extent(_preferred_bounds_objects(objects))
    return {
        "object_name": object_name,
        "location": [round(float(value), 6) for value in center],
        "center": [round(float(value), 6) for value in center],
        "extent": [round(float(value), 6) for value in extent],
    }


def _scene_layout_data(scene, focus_objects: list[Any]) -> dict[str, Any]:
    layout = {
        "scene_name": scene.name,
        "object_count": len(scene.objects),
        "focus_object_names": [obj.name for obj in focus_objects],
        "focus_object_count": len(focus_objects),
    }
    if focus_objects:
        min_corner, max_corner = _group_bounds(focus_objects)
        layout["focus_bounds"] = {
            "min": [round(float(value), 6) for value in min_corner],
            "max": [round(float(value), 6) for value in max_corner],
        }
        layout["focus_center"] = [round(float(value), 6) for value in ((min_corner + max_corner) * 0.5)]
    wall_like_names = []
    for obj in scene.objects:
        name_text = obj.name.lower()
        if any(token in name_text for token in ("wall", "room_shell", "floor", "desk", "bed", "door")):
            wall_like_names.append(obj.name)
    layout["notable_scene_objects"] = wall_like_names[:48]
    return layout


def _scene_objects_for_character_preview(
    scene,
    *,
    target_group_objects: list[Any],
    target_asset_id: str,
    character_ids: list[str],
    suppressed_objects: list[Any] | None = None,
) -> list[Any]:
    hidden_names: set[str] = set()
    for obj in suppressed_objects or []:
        hidden_names.add(obj.name)
    return [obj for obj in scene.objects if obj.name not in hidden_names]


def _build_scene_dossier_for_scene(
    *,
    scene_id: int,
    scene_shots: list[dict[str, Any]],
    output_root: Path,
    resolution_x: int,
    resolution_y: int,
    margin_factor: float,
    project_dir: Path,
    extra_character_ids: list[str] | None = None,
) -> dict[str, Any]:
    base_scene = _resolve_scene(scene_id) or _scene_for_worker()
    dossier_dir = output_root / "scene_dossiers" / f"scene_{scene_id}"
    previews_dir = dossier_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    character_ids = collect_scene_character_ids(scene_shots, extra_character_ids=extra_character_ids)
    focus_asset_ids = collect_scene_focus_asset_ids(scene_shots)
    referenced_ids: list[str] = []
    for asset_id in character_ids + focus_asset_ids:
        if asset_id and asset_id not in referenced_ids:
            referenced_ids.append(asset_id)

    focus_objects = []
    character_rows = []
    asset_rows = []

    for asset_id in referenced_ids:
        candidate_scenes = _iter_candidate_scenes_for_asset(scene_id, scene_shots, asset_id)
        obj = None
        obj_scene = None
        for candidate_scene in candidate_scenes:
            obj = _find_object_by_asset_id(candidate_scene, asset_id)
            if obj is not None:
                obj_scene = candidate_scene
                break
        if obj is None:
            obj = _find_object_by_asset_id(base_scene, asset_id)
            obj_scene = base_scene if obj is not None else None

        base_scene_obj = _find_object_by_asset_id(base_scene, asset_id)
        if base_scene_obj is not None and base_scene_obj not in focus_objects:
            focus_objects.append(base_scene_obj)
        elif obj is not None and obj_scene is base_scene and obj not in focus_objects:
            focus_objects.append(obj)

        if obj is None:
            if asset_id in character_ids:
                character_rows.append(
                    {
                        "character_id": asset_id,
                        "display_name": asset_id,
                        "summary": "Referenced by the scene but not found in the loaded Blender scene.",
                        "layout_data": {},
                        "metadata": {"found_in_blender": False},
                    }
                )
            else:
                asset_rows.append(
                    {
                        "asset_id": asset_id,
                        "asset_type": "scene_asset",
                        "summary": "Referenced by scene camera directions or actions.",
                        "metadata": {"found_in_blender": False},
                    }
                )
            continue

        if asset_id in character_ids:
            base_scene_preview_obj = _find_object_by_asset_id(base_scene, asset_id)
            preview_scene = obj_scene or base_scene
            preview_root_obj = obj
            preview_group_objects = _descendant_objects(preview_root_obj)
            has_renderable_preview_objects = any(
                getattr(candidate_obj, "type", "") in {"MESH", "CURVE", "SURFACE", "META", "FONT"}
                for candidate_obj in preview_group_objects
            )
            if (
                preview_scene is not base_scene
                and base_scene_preview_obj is not None
                and (_looks_like_proxy_preview_object(preview_root_obj) or not has_renderable_preview_objects)
            ):
                preview_scene = base_scene
                preview_root_obj = base_scene_preview_obj
                preview_group_objects = _descendant_objects(preview_root_obj)
            preview_obj = _preferred_preview_primary_obj(preview_group_objects, preview_root_obj)
            preview_basis_obj = _preferred_orientation_basis_obj(preview_group_objects, preview_root_obj)
            bounds_proxy_obj = preview_obj
            bounds_group_objects = preview_group_objects
            preview_frame: float | None = None
            contextual_turnaround_paths = _capture_character_turnaround(
                scene=preview_scene,
                character_id=asset_id,
                primary_obj=preview_obj,
                basis_obj=preview_basis_obj,
                group_objects=preview_group_objects,
                bounds_group_objects=bounds_group_objects,
                output_dir=previews_dir,
                filename_prefix=f"character_{asset_id}_context_turnaround",
                resolution_x=resolution_x,
                resolution_y=resolution_y,
                margin_factor=margin_factor,
                hide_environment=False,
                visible_scene_objects=_scene_objects_for_character_preview(
                    preview_scene,
                    target_group_objects=preview_group_objects,
                    target_asset_id=asset_id,
                    character_ids=character_ids,
                    suppressed_objects=None,
                ),
                preview_frame=preview_frame,
            )
            character_rows.append(
                {
                    "character_id": asset_id,
                    "display_name": obj.name,
                    "summary": "Character referenced by the scene and available for semantic shot planning.",
                    "portrait_path": "",
                    "turnaround_paths": {},
                    "context_turnaround_paths": contextual_turnaround_paths,
                    "layout_data": _group_layout_summary(bounds_group_objects, bounds_proxy_obj.name),
                    "metadata": {
                        "found_in_blender": True,
                        "source_scene_name": str(preview_scene.name),
                        "bounds_source_scene_name": str(preview_scene.name),
                        "context_turnaround_mode": "contextual_scene",
                        "turnaround_semantics": "subject_side",
                        "preview_pose_frame": preview_frame,
                    },
                }
            )
        else:
            asset_rows.append(
                {
                    "asset_id": asset_id,
                    "asset_type": "scene_asset",
                    "summary": "Referenced by scene camera directions or actions.",
                    "metadata": {
                        "found_in_blender": True,
                        "layout_data": _object_layout_summary(obj),
                    },
                }
            )

    scene_top_view_path = _capture_scene_top_view(
        scene=base_scene,
        scene_id=scene_id,
        focus_objects=focus_objects,
        output_dir=previews_dir,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        margin_factor=margin_factor,
    )

    dossier = build_scene_dossier(
        scene_id=str(scene_id),
        scene_top_view_path=scene_top_view_path,
        layout_data=_scene_layout_data(base_scene, focus_objects),
        character_summaries=character_rows,
        asset_summaries=asset_rows,
        metadata={
            "shot_count": len(scene_shots),
            "character_ids": character_ids,
            "focus_asset_ids": focus_asset_ids,
        },
    )
    dossier_path = save_scene_dossier_json(dossier, dossier_dir / "scene_dossier.json")
    return {
        "scene_id": scene_id,
        "dossier_path": str(dossier_path),
        "dossier": dossier.to_dict(),
    }


def _run_blender_scene_context_builder(request_path: Path) -> dict[str, Any]:
    request_payload = load_json(request_path)
    filtered_input_path = Path(request_payload["filtered_input_path"])
    output_root = Path(request_payload["output_root"])
    project_dir = Path(request_payload["project_dir"])
    scene_ids = [int(scene_id) for scene_id in request_payload["scene_ids"]]
    resolution_x = int(request_payload["resolution_x"])
    resolution_y = int(request_payload["resolution_y"])
    margin_factor = float(request_payload["margin_factor"])

    filtered_shots = load_json(filtered_input_path)
    grouped = _iter_scene_shots(filtered_shots)

    raw_scene_character_ids = request_payload.get("scene_character_ids") or {}
    scene_character_ids: dict[int, list[str]] = {}
    if isinstance(raw_scene_character_ids, dict):
        for raw_key, raw_value in raw_scene_character_ids.items():
            try:
                scene_key = int(raw_key)
            except (TypeError, ValueError):
                continue
            if isinstance(raw_value, list):
                scene_character_ids[scene_key] = [str(item) for item in raw_value if str(item or "").strip()]

    scene_entries = []
    for scene_id in scene_ids:
        scene_entries.append(
            _build_scene_dossier_for_scene(
                scene_id=scene_id,
                scene_shots=grouped.get(scene_id, []),
                output_root=output_root,
                resolution_x=resolution_x,
                resolution_y=resolution_y,
                margin_factor=margin_factor,
                project_dir=project_dir,
                extra_character_ids=scene_character_ids.get(scene_id),
            )
        )

    manifest_path = save_scene_dossier_manifest(
        scene_entries=scene_entries,
        output_path=output_root / "scene_dossiers" / "scene_dossier_manifest_v1.json",
    )
    return {
        "success": True,
        "scene_count": len(scene_entries),
        "manifest_path": str(manifest_path),
        "scenes": scene_entries,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standalone Plan-A scene dossiers inside Blender.")
    parser.add_argument("--request-path", required=True)
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    result = _run_blender_scene_context_builder(Path(args.request_path))
    print(result)
    return 0 if result.get("success") else 1


__all__ = [
    "build_scene_dossier",
    "build_scene_dossier_from_payload",
    "collect_scene_character_ids",
    "collect_scene_focus_asset_ids",
    "build_scene_context_request_payload",
    "save_scene_dossier_json",
    "load_scene_dossier_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())


