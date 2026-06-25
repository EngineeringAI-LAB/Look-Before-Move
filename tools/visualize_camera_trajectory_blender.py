"""Visualize StoryBlender camera trajectories inside Blender.

Run from Blender:
  blender scene.blend --python tools/visualize_camera_trajectory_blender.py -- \
    --handoff Cinematographer/output/.../camera_handoff_v1.json --label Ours \
    --handoff Cinematographer/output/.../camera_handoff_v1.json --label Fast \
    --scene-id 1 --make-view-camera
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


DEFAULT_COLORS = [
    (0.05, 0.36, 1.0, 1.0),   # Ours: blue
    (1.0, 0.45, 0.05, 1.0),   # Fast: orange
    (0.95, 0.08, 0.18, 1.0),  # w/o TG: red
    (0.0, 0.62, 0.34, 1.0),   # extra: green
    (0.55, 0.16, 0.9, 1.0),   # extra: purple
]


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Draw camera trajectories from camera_handoff_v1.json.")
    parser.add_argument("--handoff", action="append", required=True, help="Path to camera_handoff_v1.json.")
    parser.add_argument("--label", action="append", default=[], help="Legend label for the matching --handoff.")
    parser.add_argument("--scene-id", type=int, default=0, help="Only draw this scene_id. 0 means all scenes.")
    parser.add_argument("--shot-id", type=int, default=0, help="Only draw this shot_id. 0 means all shots.")
    parser.add_argument("--z-offset", type=float, default=0.03, help="Vertical offset per method to avoid overlap.")
    parser.add_argument("--method-spacing", type=float, default=0.0, help="Shift each method on X axis to create side-by-side panels.")
    parser.add_argument("--bevel-depth", type=float, default=0.035, help="Curve thickness.")
    parser.add_argument("--marker-radius", type=float, default=0.08, help="Start/end marker radius.")
    parser.add_argument("--min-travel", type=float, default=0.0, help="Skip shots with less camera travel than this distance.")
    parser.add_argument("--label-mode", choices=("shot", "method", "none"), default="shot", help="How much text to draw.")
    parser.add_argument("--label-size", type=float, default=0.14, help="Text label size.")
    parser.add_argument("--show-cut-jumps", action="store_true", help="Draw line from one shot end to next shot start.")
    parser.add_argument("--make-view-camera", action="store_true", help="Create a top-down orthographic camera framing the paths.")
    parser.add_argument("--save-blend", default="", help="Optional .blend output path.")
    return parser.parse_args(argv)


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def ensure_collection(name: str) -> bpy.types.Collection:
    existing = bpy.data.collections.get(name)
    if existing:
        return existing
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def clear_collection(col: bpy.types.Collection) -> None:
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def vec3(value: Any) -> Vector | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return Vector((float(value[0]), float(value[1]), float(value[2])))
    except Exception:
        return None


def camera_points(camera: dict[str, Any]) -> list[Vector]:
    plan = camera.get("trajectory_plan") or {}
    keyframes = plan.get("keyframes") or camera.get("trajectory_keyframes") or []
    points = [vec3(frame.get("location")) for frame in keyframes if isinstance(frame, dict)]
    points = [point for point in points if point is not None]
    if points:
        return points
    start = vec3((camera.get("start_transform") or {}).get("location"))
    end = vec3((camera.get("end_transform") or {}).get("location"))
    points = [point for point in (start, end) if point is not None]
    if len(points) == 1:
        points.append(points[0].copy())
    return points


def travel_length(points: list[Vector]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum((points[index + 1] - points[index]).length for index in range(len(points) - 1)))


def load_cameras(path: Path, scene_id: int, shot_id: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras = list(data.get("cameras") or [])
    if scene_id:
        cameras = [cam for cam in cameras if int(cam.get("scene_id") or 0) == scene_id]
    if shot_id:
        cameras = [cam for cam in cameras if int(cam.get("shot_id") or 0) == shot_id]
    cameras.sort(key=lambda cam: (int(cam.get("scene_id") or 0), int(cam.get("shot_id") or 0), str(cam.get("camera_name") or "")))
    return cameras


def make_polyline(name: str, points: list[Vector], mat: bpy.types.Material, bevel_depth: float, col: bpy.types.Collection) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, co in zip(spline.points, points):
        point.co = (co.x, co.y, co.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def add_uv_sphere(name: str, location: Vector, radius: float, mat: bpy.types.Material, col: bpy.types.Collection) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    for parent_col in list(obj.users_collection):
        parent_col.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def add_text(name: str, text: str, location: Vector, mat: bpy.types.Material, col: bpy.types.Collection, size: float = 0.22) -> bpy.types.Object:
    bpy.ops.object.text_add(location=location, rotation=(math.radians(90.0), 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.materials.append(mat)
    for parent_col in list(obj.users_collection):
        parent_col.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def add_arrow(name: str, start: Vector, end: Vector, mat: bpy.types.Material, col: bpy.types.Collection) -> None:
    direction = end - start
    if direction.length < 1e-4:
        return
    loc = start + direction * 0.82
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.08, radius2=0.0, depth=0.22, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    for parent_col in list(obj.users_collection):
        parent_col.objects.unlink(obj)
    col.objects.link(obj)


def make_top_camera(points: list[Vector]) -> None:
    if not points:
        return
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    center = Vector(((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, max(zs) + 25.0))
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 1.25
    cam_data = bpy.data.cameras.new("TrajectoryViz_TopCamera_data")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = span
    cam_obj = bpy.data.objects.new("TrajectoryViz_TopCamera", cam_data)
    cam_obj.location = center
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj


def main() -> None:
    args = parse_args()
    col = ensure_collection("TrajectoryViz")
    clear_collection(col)
    cut_mat = material("TrajectoryViz_cut_jump", (0.05, 0.05, 0.05, 0.45))
    all_points: list[Vector] = []

    labels = list(args.label or [])
    while len(labels) < len(args.handoff):
        labels.append(Path(args.handoff[len(labels)]).parents[1].name)

    for method_index, (handoff_str, label) in enumerate(zip(args.handoff, labels)):
        handoff = Path(handoff_str).expanduser().resolve()
        cameras = load_cameras(handoff, args.scene_id, args.shot_id)
        color = DEFAULT_COLORS[method_index % len(DEFAULT_COLORS)]
        mat = material(f"TrajectoryViz_{label}", color)
        offset = Vector((args.method_spacing * method_index, 0.0, args.z_offset * method_index))
        previous_end: Vector | None = None
        method_points: list[Vector] = []

        for order, cam in enumerate(cameras, start=1):
            points = [point + offset for point in camera_points(cam)]
            if len(points) < 2:
                continue
            if travel_length(points) < float(args.min_travel):
                continue
            all_points.extend(points)
            method_points.extend(points)
            cam_name = str(cam.get("camera_name") or f"cam_{order}")
            scene_id = int(cam.get("scene_id") or 0)
            shot_id = int(cam.get("shot_id") or 0)
            prefix = f"{label}_s{scene_id:02d}_sh{shot_id:02d}_{cam_name}"
            make_polyline(prefix, points, mat, args.bevel_depth, col)
            add_uv_sphere(prefix + "_start", points[0], args.marker_radius, mat, col)
            add_uv_sphere(prefix + "_end", points[-1], args.marker_radius * 0.75, mat, col)
            add_arrow(prefix + "_arrow", points[0], points[-1], mat, col)
            if args.label_mode == "shot":
                add_text(prefix + "_label", f"{label}\\nS{scene_id}.{shot_id}", points[0] + Vector((0.0, 0.0, 0.22)), mat, col, size=args.label_size)
            if args.show_cut_jumps and previous_end is not None:
                make_polyline(prefix + "_cut_jump", [previous_end, points[0]], cut_mat, args.bevel_depth * 0.45, col)
            previous_end = points[-1]

        if args.label_mode == "method" and method_points:
            xs = [point.x for point in method_points]
            ys = [point.y for point in method_points]
            zs = [point.z for point in method_points]
            add_text(
                f"{label}_method_label",
                label,
                Vector(((min(xs) + max(xs)) * 0.5, min(ys) - 0.8, max(zs) + 0.35)),
                mat,
                col,
                size=max(float(args.label_size), 0.22),
            )

    if args.make_view_camera:
        make_top_camera(all_points)
    if args.save_blend:
        out = Path(args.save_blend).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"[TrajectoryViz] drew {len(args.handoff)} handoff(s), points={len(all_points)}")


if __name__ == "__main__":
    main()
