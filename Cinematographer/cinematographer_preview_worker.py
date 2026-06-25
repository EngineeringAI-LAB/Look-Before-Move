"""Blender worker: render one clear preview frame per camera with proper look-at.

Computes focus target from real scene geometry (or layout fallback), applies
semantic height adjustment (face / hands / medium), then calculates look-at
rotation so the camera actually frames the subject.

Run inside Blender via:
    blender -b <blend_file> --python cinematographer_preview_worker.py -- \
        --camera-handoff-path <path> --output-root <path> \
        --resolution-x 480 --resolution-y 270 --render-samples 8
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils
from mathutils import Euler, Matrix, Vector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-handoff-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resolution-x", type=int, default=480)
    parser.add_argument("--resolution-y", type=int, default=270)
    parser.add_argument("--render-engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--render-samples", type=int, default=8)
    parser.add_argument("--report-filename", default="camera_preview_report_v1.json")
    return parser.parse_args(argv)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _vector3_or_none(values) -> Vector | None:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    try:
        return Vector(tuple(float(v) for v in values[:3]))
    except (TypeError, ValueError):
        return None


def _euler3_or_none(values) -> Euler | None:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    try:
        return Euler(tuple(float(v) for v in values[:3]), "XYZ")
    except (TypeError, ValueError):
        return None


def _as_vector(values, default=(0.0, 0.0, 0.0)) -> Vector:
    parsed = _vector3_or_none(values)
    return parsed if parsed is not None else Vector(default)


# ---------------------------------------------------------------------------
# Scene / Object helpers (aligned with blender_render_worker)
# ---------------------------------------------------------------------------

def _find_scene(scene_id: int, shot_id: int):
    preferred = f"Scene_{scene_id}_Shot_{shot_id}"
    if preferred in bpy.data.scenes:
        return bpy.data.scenes[preferred]
    fallback = f"Scene_{scene_id}"
    if fallback in bpy.data.scenes:
        return bpy.data.scenes[fallback]
    for scene in bpy.data.scenes:
        if f"_{scene_id}" in scene.name:
            return scene
    return bpy.data.scenes[0] if bpy.data.scenes else None


def _set_active_scene(scene):
    bpy.context.window.scene = scene


def _normalize_id(asset_id: str) -> str:
    return asset_id.strip().lower().replace(" ", "_").replace("-", "_")


def _find_object(scene, asset_id: str):
    norm = _normalize_id(asset_id)
    exact = None
    dotted: list = []
    underscored: list = []
    prop_match: list = []
    for obj in scene.objects:
        low = obj.name.lower()
        if low == norm:
            exact = obj
            break
        if low.startswith(f"{norm}."):
            dotted.append(obj)
        elif low.startswith(f"{norm}_"):
            underscored.append(obj)
        try:
            if str(obj.get("asset_id") or "").strip().lower() == norm:
                prop_match.append(obj)
        except Exception:
            pass
    return exact or (prop_match[0] if prop_match else None) or (dotted[0] if dotted else None) or (underscored[0] if underscored else None)


def _descendants(root) -> list:
    out: list = []
    stack = [root]
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur is None or cur.name in seen:
            continue
        seen.add(cur.name)
        out.append(cur)
        stack.extend(list(getattr(cur, "children", []) or []))
    return out


def _world_bounds(obj) -> list[Vector]:
    if not hasattr(obj, "bound_box") or not obj.bound_box:
        return [obj.matrix_world @ Vector((0, 0, 0))]
    return [obj.matrix_world @ Vector(c) for c in obj.bound_box]


def _points_aabb(points: list[Vector]):
    mn = points[0].copy()
    mx = points[0].copy()
    for p in points[1:]:
        mn.x, mn.y, mn.z = min(mn.x, p.x), min(mn.y, p.y), min(mn.z, p.z)
        mx.x, mx.y, mx.z = max(mx.x, p.x), max(mx.y, p.y), max(mx.z, p.z)
    return mn, mx


# ---------------------------------------------------------------------------
# Focus target computation
# ---------------------------------------------------------------------------

def _focus_bounds_from_geometry(scene, focus_ids: list[str]):
    """Get focus center and extent from actual Blender geometry.

    Detects centimeter-scale scenes (extent > 10 in any axis) and
    auto-converts to meters so camera positions stay consistent.
    """
    pts: list[Vector] = []
    for fid in focus_ids:
        obj = _find_object(scene, fid)
        if obj is None:
            continue
        for d in _descendants(obj):
            if getattr(d, "type", "") not in {"MESH", "ARMATURE", "EMPTY", "CURVE", "SURFACE", ""}:
                continue
            try:
                pts.extend(_world_bounds(d))
            except Exception:
                continue
    if not pts:
        return None, None
    mn, mx = _points_aabb(pts)
    center = (mn + mx) * 0.5
    extent = mx - mn
    max_dim = max(abs(extent.x), abs(extent.y), abs(extent.z))
    if max_dim > 10.0:
        scale = 0.01
        center = center * scale
        extent = extent * scale
    return center, extent


def _focus_from_layout(director_handoff: dict, scene_id: int, focus_ids: list[str]):
    """Fallback: compute focus from Director layout data."""
    details = (director_handoff.get("scene_details") or {}).get(str(scene_id)) or {}
    rows = details.get("layout_assets") or []
    centers: list[tuple[float, float, float]] = []
    for row in rows:
        aid = str(row.get("asset_id") or "").strip()
        if aid not in focus_ids:
            continue
        loc = row.get("location") or {}
        centers.append((float(loc.get("x", 0)), float(loc.get("y", 0)), float(loc.get("z", 0.8))))
    if not centers:
        return Vector((0, 0, 1.2)), Vector((0.8, 0.8, 1.7))
    n = len(centers)
    c = Vector((sum(t[0] for t in centers) / n, sum(t[1] for t in centers) / n, sum(t[2] for t in centers) / n + 0.9))
    return c, Vector((0.8, 0.8, 1.7))


def _semantic_adjust(center: Vector, extent: Vector, camera: dict) -> Vector:
    """Adjust look-at height for face / hands / medium shots."""
    target = center.copy()
    sem = str(camera.get("primary_semantic_target") or "").strip().lower()
    dist = str(camera.get("distance_label") or "").strip().lower()
    desc = str(camera.get("shot_description") or "").strip().lower()
    if sem in {"face", "eyes", "head", "back_of_head"} or "close" in dist:
        target.z = center.z + max(float(extent.z) * 0.30, 0.35)
    elif sem == "hands" or "hand" in desc:
        target.z = center.z - max(float(extent.z) * 0.12, 0.12)
    elif "medium" in dist:
        target.z = center.z + max(float(extent.z) * 0.12, 0.15)
    return target


def _look_at_rotation(cam_location: Vector, target: Vector) -> Euler:
    """Compute Euler rotation so camera at cam_location looks at target."""
    direction = target - cam_location
    if direction.length < 1e-6:
        return Euler((math.pi / 2, 0, 0), "XYZ")
    rot_quat = direction.to_track_quat("-Z", "Y")
    return rot_quat.to_euler("XYZ")


def _preview_rotation_from_transform(start_transform: dict, location: Vector, fallback_target: Vector) -> tuple[Euler, Vector, str]:
    """Prefer the selected candidate pose; only synthesize look-at as a fallback."""
    explicit_target = _vector3_or_none(start_transform.get("target"))
    explicit_rotation = _euler3_or_none(start_transform.get("rotation_euler"))
    if explicit_rotation is not None:
        return explicit_rotation, explicit_target or fallback_target, "start_transform_rotation"
    if explicit_target is not None:
        return _look_at_rotation(location, explicit_target), explicit_target, "start_transform_target"
    return _look_at_rotation(location, fallback_target), fallback_target, "geometry_look_at"


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return slug.strip("_") or "unknown"


def _preview_output_path(camera: dict[str, Any], args: argparse.Namespace, previews_dir: Path) -> Path:
    camera_name = _safe_slug(str(camera.get("camera_name") or "unknown"))
    report_stem = Path(str(args.report_filename or "")).stem
    render_label = _safe_slug(str(camera.get("preview_render_label") or ""))
    candidate_id = _safe_slug(str((camera.get("selected_candidate") or {}).get("candidate_id") or ""))
    if not render_label and report_stem == "camera_preview_repair_report_v1":
        render_label = "repair"
    if render_label:
        source = candidate_id if candidate_id != "unknown" else camera_name
        return previews_dir / f"{source}_{render_label}_preview.png"
    return previews_dir / f"{camera_name}_preview.png"


# ---------------------------------------------------------------------------
# Preview renderer
# ---------------------------------------------------------------------------

def _render_preview(
    camera: dict[str, Any],
    director_handoff: dict[str, Any],
    args: argparse.Namespace,
    previews_dir: Path,
) -> dict[str, Any]:
    scene_id = int(camera.get("scene_id") or 0)
    shot_id = int(camera.get("shot_id") or 0)
    camera_name = str(camera.get("camera_name") or "unknown")

    scene = _find_scene(scene_id, shot_id)
    if scene is None:
        return {"camera_name": camera_name, "success": False, "error": "scene_not_found", "preview_path": ""}

    _set_active_scene(scene)

    focus_ids = list(camera.get("focus_ids") or [camera.get("primary_focus_id")] or [])
    focus_ids = [str(f).strip() for f in focus_ids if str(f).strip()]

    focus_center, focus_extent = _focus_bounds_from_geometry(scene, focus_ids)
    focus_source = "geometry"
    if focus_center is None:
        focus_center, focus_extent = _focus_from_layout(director_handoff, scene_id, focus_ids)
        focus_source = "layout"

    geometry_look_target = _semantic_adjust(focus_center, focus_extent, camera)

    start_transform = camera.get("start_transform") or {}
    location = _as_vector(start_transform.get("location"), default=(0.0, -3.0, 1.6))
    lens_mm = float(start_transform.get("lens_mm") or camera.get("lens_mm") or 35.0)

    rotation, look_target, rotation_source = _preview_rotation_from_transform(start_transform, location, geometry_look_target)

    cam_data = bpy.data.cameras.new(name=f"preview_{camera_name}")
    cam_data.lens = lens_mm
    cam_data.clip_start = 0.1
    cam_data.clip_end = 1000.0
    cam_obj = bpy.data.objects.new(name=f"preview_{camera_name}", object_data=cam_data)
    scene.collection.objects.link(cam_obj)

    cam_obj.location = location
    cam_obj.rotation_euler = rotation
    scene.camera = cam_obj
    scene.frame_set(1)

    scene.render.resolution_x = int(args.resolution_x)
    scene.render.resolution_y = int(args.resolution_y)
    scene.render.resolution_percentage = 100
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
        except Exception:
            pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    preview_path = _preview_output_path(camera, args, previews_dir)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(preview_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)

    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.cameras.remove(cam_data, do_unlink=True)

    exists = preview_path.exists()
    return {
        "camera_name": camera_name,
        "scene_id": scene_id,
        "shot_id": shot_id,
        "success": exists,
        "preview_path": str(preview_path.resolve()) if exists else "",
        "focus_source": focus_source,
        "focus_center": [round(float(v), 4) for v in focus_center],
        "look_target": [round(float(v), 4) for v in look_target],
        "geometry_look_target": [round(float(v), 4) for v in geometry_look_target],
        "rotation_source": rotation_source,
        "camera_location": [round(float(v), 4) for v in location],
        "camera_rotation_euler": [round(float(v), 6) for v in rotation],
        "selected_candidate_id": (camera.get("selected_candidate") or {}).get("candidate_id") or camera.get("selected_candidate_id") or "",
        "selected_candidate_preview_path": ((camera.get("selected_candidate") or {}).get("preview_image_path") or "")
        or camera.get("llm_selected_candidate_preview_path")
        or camera.get("selected_candidate_preview_path")
        or "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    camera_handoff = _load_json(Path(args.camera_handoff_path))
    output_root = Path(args.output_root)
    previews_dir = output_root / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    director_handoff_path = Path(str(camera_handoff.get("director_handoff_path") or ""))
    director_handoff = _load_json(director_handoff_path) if director_handoff_path.exists() else {}

    cameras = camera_handoff.get("cameras") or []
    if not cameras:
        for shot in camera_handoff.get("shots") or []:
            cameras.extend(shot.get("cameras") or [])

    results = []
    for camera in cameras:
        result = _render_preview(camera, director_handoff, args, previews_dir)
        results.append(result)
        status = "OK" if result["success"] else "FAIL"
        src = result.get("focus_source", "?")
        print(f"[preview] {result['camera_name']}: {status} (focus: {src})")

    report_path = _save_json(
        {
            "schema_version": "storyblender.camera_preview_report.v1",
            "total": len(results),
            "success_count": sum(1 for r in results if r["success"]),
            "results": results,
        },
        output_root / "outputs" / str(args.report_filename or "camera_preview_report_v1.json"),
    )
    print(f"Preview report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
