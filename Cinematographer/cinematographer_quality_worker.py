from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
except Exception:  # pragma: no cover
    bpy = None
    mathutils = None
    world_to_camera_view = None

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None


DIRECTIONS = ("front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left")
PUBLIC_DIRECTION_NAMES = {
    "front": "front",
    "front_right": "front_right",
    "right": "right",
    "back_right": "back_right",
    "back": "back",
    "back_left": "back_left",
    "left": "left",
    "front_left": "front_left",
}
OPERATIONS = (
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
    "orbit_left",
    "orbit_right",
    "orbit_up",
    "orbit_down",
    "truck_left",
    "truck_right",
    "dolly_in",
    "dolly_out",
    "zoom_in",
    "zoom_out",
    "pedestal_up",
    "pedestal_down",
)
PRESETS = (
    # Close-up presets
    (0.65, 0.00, 85.0, "extreme_close_normal"),
    (0.70, 0.15, 75.0, "extreme_close_high"),
    (0.70, -0.15, 75.0, "extreme_close_low"),
    (0.75, 0.00, 65.0, "close_normal"),
    (0.75, 0.25, 50.0, "close_high"),
    (0.75, -0.20, 50.0, "close_low"),
    (0.80, 0.10, 55.0, "close_over_shoulder"),
    (0.85, 0.05, 60.0, "close_dutch_left"),
    (0.85, 0.05, 60.0, "close_dutch_right"),
    (0.60, 0.00, 75.0, "tight_tele"),
    
    # Medium presets
    (0.90, 0.00, 42.0, "medium_close_normal"),
    (1.00, -0.12, 42.0, "medium_low"),
    (1.00, 0.18, 35.0, "medium_high"),
    (1.05, 0.00, 35.0, "medium_normal"),
    (1.10, 0.25, 35.0, "medium_over_shoulder"),
    (1.15, -0.25, 28.0, "medium_wide_low"),
    (1.15, 0.20, 28.0, "medium_wide_high"),
    (1.20, 0.00, 35.0, "medium_two_shot"),
    (1.20, 0.15, 42.0, "medium_dutch_angle"),
    (0.95, 0.40, 35.0, "medium_top_down"),
    
    # Wide/Far presets
    (1.30, 0.00, 35.0, "far_wide"),
    (1.30, 0.30, 28.0, "far_high_wide"),
    (1.40, -0.20, 24.0, "far_low_wide"),
    (1.50, 0.00, 24.0, "extreme_wide"),
    (1.60, 0.10, 30.0, "establishing"),
    (1.70, 0.40, 20.0, "establishing_high"),
    (1.80, -0.15, 18.0, "establishing_low"),
    (2.00, 0.50, 16.0, "drone_shot"),
    (2.20, 0.00, 20.0, "ultra_wide_panoramic"),
    (1.50, 0.60, 24.0, "bird_eye_view"),
)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


_DISABLE_SEMANTIC_HEIGHT_ADJUST: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quality candidates for isolated Cinematographer.")
    parser.add_argument("--camera-handoff-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resolution-x", type=int, default=480)
    parser.add_argument("--resolution-y", type=int, default=270)
    parser.add_argument("--render-engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--render-samples", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--per-channel-top-k", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--frontier-limit", type=int, default=24)
    parser.add_argument("--enable-candidate-validation", action="store_true")
    parser.add_argument("--validation-failed-top-per-channel", type=int, default=20)
    parser.add_argument("--validation-success-near-threshold-per-channel", type=int, default=20)
    parser.add_argument("--validation-reason-samples", type=int, default=5)
    parser.add_argument("--validation-score-threshold", type=float, default=0.18)
    parser.add_argument(
        "--disable-semantic-height-adjust",
        action="store_true",
        help="Ablation: skip semantic Z offset / face anchor / enforce_camera_height; aim at geometric center.",
    )
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    global _DISABLE_SEMANTIC_HEIGHT_ADJUST
    _DISABLE_SEMANTIC_HEIGHT_ADJUST = args.disable_semantic_height_adjust
    return args


def scene_candidates(scene_id: int, shot_id: int) -> list[str]:
    return [
        f"Scene_{scene_id}_Shot_{shot_id}",
        f"Scene_{scene_id}_shot_{shot_id}",
        f"scene_{scene_id}_shot_{shot_id}",
        f"Scene_{scene_id}",
        f"Scene {scene_id}",
        f"scene_{scene_id}",
    ]


def resolve_scene(scene_id: int, shot_id: int):
    for name in scene_candidates(scene_id, shot_id):
        scene = bpy.data.scenes.get(name)
        if scene is not None:
            return scene, name
    return None, ""


def set_active_scene(scene) -> None:
    """Make ray casts use the same depsgraph as the scene being evaluated."""
    if scene is None:
        return
    try:
        window = getattr(bpy.context, "window", None)
        if window is not None:
            window.scene = scene
    except Exception:
        pass
    try:
        scene.frame_set(scene.frame_current)
    except Exception:
        pass


def round_vector(vector: Any) -> list[float]:
    return [round(float(value), 6) for value in vector]


def public_direction_name(direction_name: str) -> str:
    text = str(direction_name or "").strip()
    if text.startswith("hand_"):
        return f"hand_{public_direction_name(text[5:])}"
    if "_" in text:
        base, suffix = _split_direction_suffix(text)
        if base in PUBLIC_DIRECTION_NAMES:
            return f"{PUBLIC_DIRECTION_NAMES[base]}{suffix}"
    for internal_name in sorted(PUBLIC_DIRECTION_NAMES, key=len, reverse=True):
        public_name = PUBLIC_DIRECTION_NAMES[internal_name]
        if text == internal_name:
            return public_name
        if text.startswith(f"{internal_name}_"):
            return f"{public_name}{text[len(internal_name):]}"
    return text


def _split_direction_suffix(value: str) -> tuple[str, str]:
    text = str(value or "").strip().lower()
    for name in sorted(DIRECTIONS, key=len, reverse=True):
        if text == name:
            return name, ""
        if text.startswith(f"{name}_"):
            return name, text[len(name):]
    return text, ""


def normalize_direction(value: Any, fallback: str = "front") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("hand_"):
        text = text[5:]
    base, _suffix = _split_direction_suffix(text)
    return base if base in DIRECTIONS else fallback


def requested_direction(camera: dict[str, Any]) -> str:
    contract = start_frame_contract(camera)
    motion = (camera.get("shot_contract") or {}).get("motion_contract") or {}
    return normalize_direction(
        camera.get("direction_tag")
        or contract.get("direction")
        or motion.get("direction")
        or "front"
    )


_ADJACENT_DIRECTIONS = {
    "front": {"front_left", "front_right"},
    "front_right": {"front", "right"},
    "right": {"front_right", "back_right"},
    "back_right": {"right", "back"},
    "back": {"back_right", "back_left"},
    "back_left": {"back", "left"},
    "left": {"back_left", "front_left"},
    "front_left": {"left", "front"},
}


def allowed_directions(camera: dict[str, Any], *, include_neighbors: bool = True) -> list[str]:
    wanted = requested_direction(camera)
    if not include_neighbors:
        return [wanted]
    return [direction for direction in DIRECTIONS if direction == wanted or direction in _ADJACENT_DIRECTIONS.get(wanted, set())]


def direction_alignment(camera: dict[str, Any], direction_name: Any) -> float:
    wanted = requested_direction(camera)
    actual = normalize_direction(direction_name, fallback=wanted)
    if actual == wanted:
        return 1.0
    if actual in _ADJACENT_DIRECTIONS.get(wanted, set()):
        return 0.65
    return 0.0


def _vector_or_none(values: Any) -> Any | None:
    try:
        if values is None:
            return None
        vector = mathutils.Vector(tuple(float(value) for value in list(values)[:3]))
        if vector.length <= 1e-6:
            return None
        return vector
    except Exception:
        return None


def semantic_direction_from_transform(candidate: dict[str, Any]) -> str:
    location = _vector_or_none(candidate.get("location"))
    target = _vector_or_none(candidate.get("target"))
    front = _vector_or_none(candidate.get("subject_front_vector"))
    right = _vector_or_none(candidate.get("subject_right_vector"))
    if location is None or target is None or front is None or right is None:
        return normalize_direction(candidate.get("semantic_direction") or candidate.get("direction"))
    vector = location - target
    vector.z = 0.0
    if vector.length <= 1e-6:
        return normalize_direction(candidate.get("semantic_direction") or candidate.get("direction"))
    vector.normalize()
    front.z = 0.0
    right.z = 0.0
    if front.length <= 1e-6 or right.length <= 1e-6:
        return normalize_direction(candidate.get("semantic_direction") or candidate.get("direction"))
    front.normalize()
    right.normalize()
    angle = math.degrees(math.atan2(vector.dot(right), vector.dot(front)))
    if -22.5 <= angle < 22.5:
        return "front"
    if 22.5 <= angle < 67.5:
        return "front_right"
    if 67.5 <= angle < 112.5:
        return "right"
    if 112.5 <= angle < 157.5:
        return "back_right"
    if angle >= 157.5 or angle < -157.5:
        return "back"
    if -157.5 <= angle < -112.5:
        return "back_left"
    if -112.5 <= angle < -67.5:
        return "left"
    return "front_left"


def look_at(location: Any, target: Any) -> Any:
    direction = target - location
    if direction.length <= 1e-6:
        direction = mathutils.Vector((0.0, 1.0, 0.0))
    return direction.normalized().to_track_quat("-Z", "Y").to_euler()


def lens_clamp(value: float) -> float:
    return max(24.0, min(float(value), 200.0))


def find_object(scene, asset_id: str):
    wanted = str(asset_id or "").strip()
    if not wanted:
        return None
    if wanted in scene.objects:
        return scene.objects[wanted]
    dotted = []
    underscored = []
    property_matches = []
    for obj in scene.objects:
        if obj.name == wanted:
            return obj
        if obj.name.startswith(f"{wanted}."):
            dotted.append(obj)
        elif obj.name.startswith(f"{wanted}_"):
            underscored.append(obj)
        try:
            if str(obj.get("asset_id") or "").strip() == wanted:
                property_matches.append(obj)
        except Exception:
            pass
    return (property_matches or dotted or underscored or [None])[0]


def descendants(root) -> list[Any]:
    rows = []
    stack = [root]
    seen = set()
    while stack:
        obj = stack.pop()
        if obj is None or obj.name in seen:
            continue
        seen.add(obj.name)
        rows.append(obj)
        stack.extend(list(getattr(obj, "children", []) or []))
    return rows


def object_identity_names(obj) -> set[str]:
    names: set[str] = set()
    stack = [obj]
    original = getattr(obj, "original", None)
    if original is not None and original is not obj:
        stack.append(original)
    while stack:
        current = stack.pop()
        if current is None:
            continue
        name = str(getattr(current, "name", "") or "")
        if not name or name in names:
            continue
        names.add(name)
        parent = getattr(current, "parent", None)
        if parent is not None:
            stack.append(parent)
        current_original = getattr(current, "original", None)
        if current_original is not None and current_original is not current:
            stack.append(current_original)
    return names


def hit_is_focus_object(hit_obj, focus_names: set[str]) -> bool:
    if hit_obj is None:
        return False
    return bool(object_identity_names(hit_obj) & set(focus_names or set()))


def object_bounds(obj) -> list[Any]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    try:
        evaluated = obj.evaluated_get(depsgraph)
    except Exception:
        evaluated = obj
    matrix = getattr(evaluated, "matrix_world", None) or getattr(obj, "matrix_world", None)
    box = getattr(evaluated, "bound_box", None) or getattr(obj, "bound_box", None)
    points = []
    if matrix is not None and box:
        for corner in box:
            try:
                points.append(matrix @ mathutils.Vector(corner))
            except Exception:
                pass
    if points:
        return points
    location = getattr(obj, "location", mathutils.Vector((0.0, 0.0, 0.0)))
    half = mathutils.Vector((0.35, 0.35, 0.9))
    return [
        mathutils.Vector((location.x + sx * half.x, location.y + sy * half.y, location.z + sz * half.z))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def points_bounds(points: list[Any]) -> tuple[Any, Any]:
    minimum = points[0].copy()
    maximum = points[0].copy()
    for point in points[1:]:
        minimum.x = min(minimum.x, point.x)
        minimum.y = min(minimum.y, point.y)
        minimum.z = min(minimum.z, point.z)
        maximum.x = max(maximum.x, point.x)
        maximum.y = max(maximum.y, point.y)
        maximum.z = max(maximum.z, point.z)
    return minimum, maximum


def scene_details(director_handoff: dict[str, Any], scene_id: int) -> dict[str, Any]:
    rows = director_handoff.get("scene_details") or {}
    return dict(rows.get(str(scene_id)) or rows.get(scene_id) or {})


def layout_center(details: dict[str, Any], focus_ids: list[str]):
    matches = []
    for row in details.get("layout_assets") or []:
        if str(row.get("asset_id") or "").strip() not in focus_ids:
            continue
        loc = row.get("location") or {}
        matches.append((float(loc.get("x", 0.0) or 0.0), float(loc.get("y", 0.0) or 0.0), float(loc.get("z", 0.0) or 0.0) + 0.9))
    if not matches:
        return mathutils.Vector((0.0, 0.0, 1.2))
    count = float(len(matches))
    return mathutils.Vector((sum(x for x, _, _ in matches) / count, sum(y for _, y, _ in matches) / count, sum(z for _, _, z in matches) / count))


def start_subject_layout(camera: dict[str, Any]) -> dict[str, Any]:
    start_contract = camera.get("start_frame_contract") or {}
    if not isinstance(start_contract, dict) or not start_contract:
        contract = camera.get("shot_contract") or {}
        start_contract = contract.get("start_frame_contract") or {}
    layout = start_contract.get("start_subject_layout") or {}
    return layout if isinstance(layout, dict) else {}


def hint_extent(layout: dict[str, Any]) -> Any:
    extent = layout.get("extent") or {}
    if isinstance(extent, dict):
        values = (
            float(extent.get("x", 0.8) or 0.8),
            float(extent.get("y", 0.35) or 0.35),
            float(extent.get("z", 1.75) or 1.75),
        )
    else:
        values = tuple(float(value or 0.0) for value in list(extent)[:3]) if isinstance(extent, (list, tuple)) else ()
    if len(values) != 3 or max(values) <= 0.0:
        values = (0.8, 0.35, 1.75)
    return mathutils.Vector(values)


def projection_points_from_center(center: Any, extent: Any) -> list[Any]:
    half_x = max(min(float(extent.x) * 0.28, 0.6), 0.18)
    half_y = max(min(float(extent.y) * 0.22, 0.4), 0.12)
    half_z = max(min(float(extent.z) * 0.50, 1.05), 0.45)
    return [
        center + mathutils.Vector((sx * half_x, sy * half_y, sz * half_z))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def serialize_points(points: list[Any]) -> list[list[float]]:
    return [round_vector(point) for point in points]


def deserialize_points(points: Any) -> list[Any]:
    rows: list[Any] = []
    for point in points or []:
        try:
            rows.append(mathutils.Vector(tuple(float(value or 0.0) for value in list(point)[:3])))
        except Exception:
            continue
    return rows


def horizontal_vector(vector: Any, fallback: Any) -> Any:
    result = vector.copy()
    result.z = 0.0
    if result.length <= 1e-5:
        result = fallback.copy()
        result.z = 0.0
    if result.length <= 1e-5:
        result = mathutils.Vector((0.0, -1.0, 0.0))
    result.normalize()
    return result


def orthogonal_right(front: Any, hint: Any) -> Any:
    right = hint - front * hint.dot(front)
    right = horizontal_vector(right, mathutils.Vector((front.y, -front.x, 0.0)))
    if abs(front.dot(right)) >= 0.98:
        right = horizontal_vector(mathutils.Vector((front.y, -front.x, 0.0)), mathutils.Vector((1.0, 0.0, 0.0)))
    return right


def basis_from_axes(primary_root, front_axis: Any, right_axis: Any, *, local_space: bool) -> tuple[Any, Any]:
    front_vec = mathutils.Vector(front_axis)
    right_vec = mathutils.Vector(right_axis)
    if primary_root is not None and local_space:
        try:
            quat = primary_root.matrix_world.to_quaternion()
            front_vec = quat @ front_vec
            right_vec = quat @ right_vec
        except Exception:
            pass
    front = horizontal_vector(front_vec, mathutils.Vector((0.0, -1.0, 0.0)))
    right = orthogonal_right(front, right_vec)
    return front, right


def basis_variants(primary_root) -> list[dict[str, Any]]:
    rows = [
        {"basis_source": "local_minus_y_plus_x", "front_axis": (0.0, -1.0, 0.0), "right_axis": (1.0, 0.0, 0.0), "local_space": True},
        {"basis_source": "local_plus_y_minus_x", "front_axis": (0.0, 1.0, 0.0), "right_axis": (-1.0, 0.0, 0.0), "local_space": True},
        {"basis_source": "local_plus_x_plus_y", "front_axis": (1.0, 0.0, 0.0), "right_axis": (0.0, 1.0, 0.0), "local_space": True},
        {"basis_source": "local_minus_x_minus_y", "front_axis": (-1.0, 0.0, 0.0), "right_axis": (0.0, -1.0, 0.0), "local_space": True},
    ]
    if primary_root is None:
        rows = []
    rows.extend(
        [
            {"basis_source": "world_minus_y_plus_x", "front_axis": (0.0, -1.0, 0.0), "right_axis": (1.0, 0.0, 0.0), "local_space": False},
            {"basis_source": "world_plus_y_minus_x", "front_axis": (0.0, 1.0, 0.0), "right_axis": (-1.0, 0.0, 0.0), "local_space": False},
        ]
    )
    return rows


def pose_bones(root) -> list[Any]:
    pose = getattr(root, "pose", None)
    return list(getattr(pose, "bones", []) or []) if pose is not None else []


def bone_world_center(root, bone) -> Any:
    try:
        head = root.matrix_world @ bone.head
        tail = root.matrix_world @ bone.tail
        return (head + tail) * 0.5
    except Exception:
        return mathutils.Vector((0.0, 0.0, 0.0))


def bone_name_matches(name: str, token_groups: tuple[tuple[str, ...], ...], *, side: str | None = None) -> bool:
    text = str(name or "").lower()
    side_tokens = {
        "left": (".l", "_l", "-l", "left", "hand_l", "wrist_l", "forearm_l"),
        "right": (".r", "_r", "-r", "right", "hand_r", "wrist_r", "forearm_r"),
    }
    if side and not any(token in text for token in side_tokens.get(side, ())):
        return False
    return any(all(token in text for token in group) for group in token_groups)


def average_points(points: list[Any]) -> Any | None:
    if not points:
        return None
    total = mathutils.Vector((0.0, 0.0, 0.0))
    for point in points:
        total += point
    return total / float(len(points))


def anchor_near_root(primary_root, point: Any, extent: Any) -> bool:
    if primary_root is None or point is None:
        return False
    root_location = getattr(primary_root, "location", mathutils.Vector((0.0, 0.0, 0.0)))
    max_distance = max(float(extent.z) * 2.2, 3.5)
    try:
        return bool((point - root_location).length <= max_distance)
    except Exception:
        return False


def anchor_near_geometry(point: Any, center: Any, extent: Any) -> bool:
    if point is None or center is None or extent is None:
        return False
    try:
        delta = point - center
        xy_distance = math.sqrt(float(delta.x) ** 2 + float(delta.y) ** 2)
        max_xy = max(float(extent.x), float(extent.y), 0.5) * 1.8 + 0.35
        max_z = max(float(extent.z) * 0.9, 0.9)
        return bool(xy_distance <= max_xy and abs(float(delta.z)) <= max_z)
    except Exception:
        return False


def face_anchor_from_bones(primary_root, extent: Any) -> Any | None:
    if primary_root is None or getattr(primary_root, "type", "") != "ARMATURE":
        return None
    bone_groups = (
        (("head",),),
        (("face",),),
        (("neck",),),
        (("eye",),),
    )
    for groups in bone_groups:
        matches = [bone_world_center(primary_root, bone) for bone in pose_bones(primary_root) if bone_name_matches(bone.name, groups)]
        anchor = average_points(matches)
        if anchor is not None and anchor_near_root(primary_root, anchor, extent):
            return anchor
    return None


def armature_face_forward(primary_root) -> Any | None:
    """Extract a subject-facing forward vector from rig bones.

    Tries three increasingly aggressive paths so we never silently fall back
    to the area-based probe (which cannot distinguish front vs. back of head):

    1. Horizontal offset from ``head`` bone centre to ``eye``/``face``/jaw
       bone centre. Most reliable when present.
    2. Horizontal projection of ``head`` bone's local axes (``y_axis``,
       ``z_axis`` for rigs whose head bone points forward in those frames).
    3. Last-resort fallback: armature's local ``-Y`` direction in world
       space, matching Blender's default armature ``rest forward``. This
       mirrors what `basis_variants[0]` ("local_minus_y_plus_x") returns
       and ensures every cam on the same character uses a *consistent*
       front direction instead of having the area probe oscillate
       between local +Y / -Y / +X / -X across shots.
    """
    if primary_root is None or getattr(primary_root, "type", "") != "ARMATURE":
        return None
    bones = pose_bones(primary_root)
    if not bones:
        return None

    def _match(*token_lists: tuple[str, ...]) -> list[Any]:
        for tokens in token_lists:
            matches = [b for b in bones if bone_name_matches(b.name, (tokens,))]
            if matches:
                return matches
        return []

    def _avg_centre(blist: list[Any]) -> Any | None:
        return average_points([bone_world_center(primary_root, bone) for bone in blist])

    # Path 1: eye/face offset relative to head/neck centre.
    head_bones = _match(("head",), ("skull",), ("cranium",))
    neck_bones = _match(("neck",), ("throat",))
    head_centre = _avg_centre(head_bones) if head_bones else _avg_centre(neck_bones)
    face_bones = _match(("eye",), ("face",), ("brow",), ("nose",), ("jaw",), ("mouth",))
    face_probe = _avg_centre(face_bones) if face_bones else None
    if head_centre is not None and face_probe is not None:
        try:
            dx = float(face_probe.x) - float(head_centre.x)
            dy = float(face_probe.y) - float(head_centre.y)
        except (TypeError, ValueError):
            dx, dy = 0.0, 0.0
        delta = mathutils.Vector((dx, dy, 0.0))
        if delta.length >= 1e-3:
            delta.normalize()
            return delta

    # Path 2: head bone's bone-local axes projected onto the horizontal plane.
    # Many rigs encode head forward in either Y or Z of the head bone frame.
    if head_bones:
        bone = head_bones[0]
        for axis_name in ("y_axis", "z_axis"):
            try:
                axis = getattr(bone, axis_name, None)
                if axis is None:
                    continue
                horiz = mathutils.Vector((float(axis.x), float(axis.y), 0.0))
                if horiz.length >= 0.4:
                    horiz.normalize()
                    return horiz
            except Exception:
                continue

    # Path 3: armature object's local -Y axis in world space. This is the
    # canonical Blender "armature rest forward" direction and matches the
    # `local_minus_y_plus_x` basis variant. Using it as a last resort means
    # every cam on the same character collapses onto the same front
    # direction, eliminating the area-probe oscillation across shots.
    try:
        quat = primary_root.matrix_world.to_quaternion()
        forward_world = quat @ mathutils.Vector((0.0, -1.0, 0.0))
        horiz = mathutils.Vector((float(forward_world.x), float(forward_world.y), 0.0))
        if horiz.length >= 1e-4:
            horiz.normalize()
            return horiz
    except Exception:
        pass
    return None


def hand_anchors_from_bones(primary_root, extent: Any) -> dict[str, Any]:
    anchors: dict[str, Any] = {}
    if primary_root is None or getattr(primary_root, "type", "") != "ARMATURE":
        return anchors
    token_groups = (
        (("hand",),),
        (("wrist",),),
        (("forearm",),),
    )
    for side in ("left", "right"):
        for groups in token_groups:
            matches = [bone_world_center(primary_root, bone) for bone in pose_bones(primary_root) if bone_name_matches(bone.name, groups, side=side)]
            anchor = average_points(matches)
            if anchor is not None and anchor_near_root(primary_root, anchor, extent):
                anchors[side] = anchor
                break
    return anchors


def _as_string_list(value: Any) -> list[str]:
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
    if not text:
        return False
    if any(token in text for token in _NONHUMAN_FOCUS_TOKENS):
        return False
    return True


def _keyframe_primary_ids(camera: dict[str, Any]) -> list[str]:
    contract_payload = camera.get("shot_contract") or {}
    if not isinstance(contract_payload, dict):
        return []
    keyframe_plan = contract_payload.get("keyframe_plan") or []
    if not isinstance(keyframe_plan, list):
        return []
    rows: list[str] = []
    for keyframe in keyframe_plan:
        if isinstance(keyframe, dict):
            rows.extend(_as_string_list(keyframe.get("primary_focus_id")))
    return rows


def _focus_role_ids(camera: dict[str, Any]) -> dict[str, Any]:
    contract = start_frame_contract(camera)
    contract_start_ids = _as_string_list(contract.get("start_focus_ids"))
    contract_primary = str(contract.get("primary_focus_id") or "").strip()
    contract_secondary = _as_string_list(contract.get("secondary_focus_ids"))
    package_focus = _as_string_list(camera.get("focus_ids"))
    package_primary = str(camera.get("primary_focus_id") or "").strip()
    package_secondary = _as_string_list(camera.get("secondary_focus_ids"))
    keyframe_primary_ids = _keyframe_primary_ids(camera)

    if contract_start_ids:
        start_primary = contract_start_ids[0]
        keyframe_votes = sum(1 for item in keyframe_primary_ids if item == start_primary)
        has_keyframe_consensus = bool(keyframe_primary_ids) and keyframe_votes >= max(1, len(keyframe_primary_ids) // 2 + 1)
        if contract_primary and contract_primary != start_primary and has_keyframe_consensus:
            old_primary = contract_primary
            contract_primary = start_primary
            if old_primary and old_primary not in contract_secondary:
                contract_secondary = [old_primary] + contract_secondary

    if contract_primary:
        primary_ids = [contract_primary]
        contract_start_secondary = [item for item in contract_start_ids if item != contract_primary]
    elif contract_start_ids:
        primary_ids = contract_start_ids[:1]
        contract_start_secondary = contract_start_ids[1:]
    else:
        primary_ids = []
        contract_start_secondary = []
    if not primary_ids:
        primary_ids = ([package_primary] if package_primary else []) or package_focus[:1]
    secondary_ids = []
    for item in contract_secondary + contract_start_secondary:
        if item and item not in secondary_ids and item not in primary_ids:
            secondary_ids.append(item)
    if not secondary_ids:
        secondary_ids = package_secondary
    if not secondary_ids and len(package_focus) > 1:
        secondary_ids = package_focus[1:]

    description = " ".join(str(camera.get(key) or "") for key in ("shot_description", "scene_description")).lower()
    secondary_semantics = contract.get("secondary_semantic_targets") or camera.get("secondary_semantic_targets") or {}
    if primary_ids and not likely_human_focus_id(primary_ids[0]):
        human_secondaries = [
            focus_id for focus_id in secondary_ids
            if likely_human_focus_id(focus_id)
            and str((secondary_semantics or {}).get(focus_id) or "").lower() in FACE_SEMANTICS | {"front", "full_body"}
        ]
        if human_secondaries and _contains_any_phrase(description, FACE_DETAIL_PHRASES + ("looks", "reaction", "expression")):
            promoted = human_secondaries[0]
            secondary_ids = [primary_ids[0]] + [item for item in secondary_ids if item != promoted]
            primary_ids = [promoted]

    focus_ids = []
    for item in primary_ids:
        if item and item not in focus_ids:
            focus_ids.append(item)
    if not focus_ids:
        for item in secondary_ids[:1] + package_focus[:1]:
            if item and item not in focus_ids:
                focus_ids.append(item)
    source = "shot_contract" if (contract_start_ids or contract_primary or contract_secondary) else "camera_package"
    package_seed = package_focus or ([package_primary] if package_primary else [])
    conflict = bool(package_seed and focus_ids and package_seed[: len(primary_ids)] != primary_ids)
    return {
        "primary_ids": primary_ids,
        "secondary_ids": [item for item in secondary_ids if item not in primary_ids],
        "focus_ids": focus_ids,
        "source": source,
        "focus_conflict_resolved": conflict,
        "package_focus_ids": package_seed,
    }


def _collect_focus_geometry(root) -> tuple[list[Any], set[str], Any | None]:
    points: list[Any] = []
    names: set[str] = set()
    geometry_types = {"MESH", "CURVE", "SURFACE", "FONT"}
    if root is None:
        return points, names, None
        
    largest_mesh_points = []
    max_mesh_volume = -1.0
    
    for obj in descendants(root):
        names.add(str(obj.name))
        if getattr(obj, "type", "") not in geometry_types:
            continue
        try:
            obj_points = object_bounds(obj)
            if obj_points:
                min_p, max_p = points_bounds(obj_points)
                dims = max_p - min_p
                # Filter out abnormally large meshes (e.g. floor attached to character)
                if max(abs(dims.x), abs(dims.y)) > 10.0:
                    continue
                points.extend(obj_points)
                
                # Keep track of the largest reasonable mesh to act as a fallback center
                volume = abs(dims.x * dims.y * dims.z)
                if volume > max_mesh_volume and volume < 5.0: # Ignore anything bigger than 5 cubic meters for a character
                    max_mesh_volume = volume
                    largest_mesh_points = obj_points
        except Exception:
            pass
            
    primary_mesh_center = None
    if largest_mesh_points:
        min_p, max_p = points_bounds(largest_mesh_points)
        primary_mesh_center = (min_p + max_p) * 0.5
        
    return points, names, primary_mesh_center


def anchor_box_points(anchor: Any, extent: Any, kind: str) -> list[Any]:
    if kind == "face_anchor":
        half_x = max(min(float(extent.x) * 0.16, 0.22), 0.08)
        half_y = max(min(float(extent.y) * 0.12, 0.18), 0.06)
        half_z = max(min(float(extent.z) * 0.10, 0.18), 0.08)
    elif kind == "face_motion_anchor":
        half_x = max(min(float(extent.x) * 0.32, 0.38), 0.12)
        half_y = max(min(float(extent.y) * 0.18, 0.25), 0.07)
        up_z = max(min(float(extent.z) * 0.12, 0.20), 0.08)
        down_z = max(min(float(extent.z) * 0.28, 0.42), 0.16)
        return [
            anchor + mathutils.Vector((sx * half_x, sy * half_y, z_offset))
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for z_offset in (-down_z, up_z)
        ]
    elif kind == "chest_anchor":
        half_x = max(min(float(extent.x) * 0.24, 0.30), 0.10)
        half_y = max(min(float(extent.y) * 0.16, 0.22), 0.06)
        half_z = max(min(float(extent.z) * 0.12, 0.22), 0.08)
    elif kind == "feet_anchor":
        half_x = max(min(float(extent.x) * 0.24, 0.32), 0.10)
        half_y = max(min(float(extent.y) * 0.18, 0.24), 0.06)
        half_z = max(min(float(extent.z) * 0.08, 0.14), 0.05)
    else:
        return projection_points_from_center(anchor, extent)
    return [
        anchor + mathutils.Vector((sx * half_x, sy * half_y, sz * half_z))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def upper_points_for_anchor(anchor: Any, extent: Any, kind: str) -> list[Any]:
    if kind == "face_anchor":
        offset_x = max(min(float(extent.x) * 0.12, 0.16), 0.06)
        offset_z = max(min(float(extent.z) * 0.07, 0.12), 0.05)
        return [
            anchor.copy(),
            anchor + mathutils.Vector((offset_x, 0.0, 0.0)),
            anchor + mathutils.Vector((-offset_x, 0.0, 0.0)),
            anchor + mathutils.Vector((0.0, 0.0, offset_z)),
            anchor + mathutils.Vector((0.0, 0.0, -offset_z)),
        ]
    if kind == "face_motion_anchor":
        shoulder_x = max(min(float(extent.x) * 0.22, 0.28), 0.10)
        upper_z = max(min(float(extent.z) * 0.07, 0.12), 0.05)
        lower_z = max(min(float(extent.z) * 0.22, 0.34), 0.12)
        return [
            anchor.copy(),
            anchor + mathutils.Vector((shoulder_x, 0.0, -lower_z)),
            anchor + mathutils.Vector((-shoulder_x, 0.0, -lower_z)),
            anchor + mathutils.Vector((0.0, 0.0, upper_z)),
            anchor + mathutils.Vector((0.0, 0.0, -lower_z)),
        ]
    if kind in {"chest_anchor", "feet_anchor"}:
        offset_x = max(min(float(extent.x) * 0.14, 0.18), 0.06)
        offset_z = max(min(float(extent.z) * 0.06, 0.12), 0.04)
        return [
            anchor.copy(),
            anchor + mathutils.Vector((offset_x, 0.0, 0.0)),
            anchor + mathutils.Vector((-offset_x, 0.0, 0.0)),
            anchor + mathutils.Vector((0.0, 0.0, offset_z)),
            anchor + mathutils.Vector((0.0, 0.0, -offset_z)),
        ]
    shoulder = max(float(extent.x) * 0.22, 0.12)
    return [
        anchor + mathutils.Vector((0.0, 0.0, max(float(extent.z) * 0.25, 0.22))),
        anchor.copy(),
        anchor + mathutils.Vector((shoulder, 0.0, max(float(extent.z) * 0.05, 0.05))),
        anchor + mathutils.Vector((-shoulder, 0.0, max(float(extent.z) * 0.05, 0.05))),
    ]


def focus_bundle(scene, director_handoff: dict[str, Any], camera: dict[str, Any]) -> dict[str, Any]:
    roles = _focus_role_ids(camera)
    focus_ids = list(roles["focus_ids"])
    primary_ids = list(roles["primary_ids"])
    roots_by_id = {focus_id: find_object(scene, focus_id) for focus_id in focus_ids}
    roots = [root for root in roots_by_id.values() if root is not None]
    primary_root = roots_by_id.get(primary_ids[0]) if primary_ids else (roots[0] if roots else None)
    points: list[Any] = []
    primary_points: list[Any] = []
    focus_names: set[str] = set()
    primary_names: set[str] = set()
    fallback_mesh_center = None
    
    for focus_id in focus_ids:
        root = roots_by_id.get(focus_id)
        focus_points, focus_object_names, mesh_center = _collect_focus_geometry(root)
        points.extend(focus_points)
        focus_names.update(focus_object_names)
        if focus_id in primary_ids[:1]:
            primary_points.extend(focus_points)
            primary_names.update(focus_object_names)
            if mesh_center is not None:
                fallback_mesh_center = mesh_center
                
    layout_hint = start_subject_layout(camera)

    def _center_extent_from_points(source_points: list[Any]) -> tuple[Any | None, Any | None]:
        if not source_points:
            return None, None
        minimum, maximum = points_bounds(source_points)
        source_center = (minimum + maximum) * 0.5
        source_extent = maximum - minimum
        max_dim = max(abs(source_extent.x), abs(source_extent.y), abs(source_extent.z))
        if max_dim > 10.0:
            scale = 0.01
            source_center = source_center * scale
            source_extent = source_extent * scale
        return source_center, source_extent

    geometry_center, geometry_extent = _center_extent_from_points(points)
    primary_geometry_center, primary_geometry_extent = _center_extent_from_points(primary_points)
    layout_center_value = None
    layout_extent_value = None
    if layout_hint:
        layout_values = layout_hint.get("center") or layout_hint.get("location")
        if layout_values is not None:
            try:
                if isinstance(layout_values, dict):
                    layout_center_value = mathutils.Vector(
                        (
                            float(layout_values.get("x", 0.0) or 0.0),
                            float(layout_values.get("y", 0.0) or 0.0),
                            float(layout_values.get("z", 0.9) or 0.9),
                        )
                    )
                else:
                    values = list(layout_values)[:3]
                    if len(values) == 3:
                        layout_center_value = mathutils.Vector(tuple(float(value or 0.0) for value in values))
            except Exception:
                layout_center_value = None
        if layout_center_value is not None:
            layout_extent_value = hint_extent(layout_hint)
    if layout_center_value is None or layout_extent_value is None:
        details = scene_details(director_handoff, int(camera.get("scene_id") or 0))
        layout_center_value = layout_center(details, focus_ids)
        layout_extent_value = mathutils.Vector((0.8, 0.8, 1.7))
    center_source = "layout"
    if primary_root is not None and getattr(primary_root, "type", "") == "ARMATURE":
        extent = primary_geometry_extent.copy() if primary_geometry_extent is not None else hint_extent(layout_hint)
        root_location = getattr(primary_root, "location", mathutils.Vector((0.0, 0.0, 0.0)))
        center = mathutils.Vector((float(root_location.x), float(root_location.y), float(root_location.z) + max(float(extent.z) * 0.5, 0.9)))
        center_source = "armature_root"
    elif fallback_mesh_center is not None:
        extent = primary_geometry_extent.copy() if primary_geometry_extent is not None else hint_extent(layout_hint)
        center = fallback_mesh_center.copy()
        center_source = "largest_mesh_center"
    elif primary_geometry_center is not None and primary_geometry_extent is not None:
        center = primary_geometry_center.copy()
        extent = primary_geometry_extent.copy()
        center_source = "primary_geometry_center"
    elif geometry_center is not None and geometry_extent is not None:
        center = geometry_center.copy()
        extent = geometry_extent.copy()
        center_source = "geometry_center"
    else:
        center = layout_center_value.copy()
        extent = layout_extent_value.copy()
    raw_center = center.copy()
    group_center = geometry_center.copy() if geometry_center is not None else center.copy()
    group_extent = geometry_extent.copy() if geometry_extent is not None else extent.copy()
    contract_info = semantic_contract(camera)
    semantic = str(contract_info.get("primary_semantic_target") or "").lower()
    if (
        bool(contract_info.get("closeup_required"))
        and semantic in {"face", "eyes", "head", "back_of_head"}
        and not _DISABLE_SEMANTIC_HEIGHT_ADJUST
    ):
        center.z += max(float(extent.z) * 0.25, 0.25)
    shoulder = max(float(extent.x) * 0.22, 0.12)
    upper = [
        center + mathutils.Vector((0.0, 0.0, max(float(extent.z) * 0.25, 0.22))),
        center.copy(),
        center + mathutils.Vector((shoulder, 0.0, max(float(extent.z) * 0.05, 0.05))),
        center + mathutils.Vector((-shoulder, 0.0, max(float(extent.z) * 0.05, 0.05))),
    ]
    projection_points = projection_points_from_center(center, extent)
    face_anchor = face_anchor_from_bones(primary_root, extent)
    hand_anchors = hand_anchors_from_bones(primary_root, extent)
    if face_anchor is not None and primary_geometry_center is not None and not anchor_near_geometry(face_anchor, center, extent):
        face_anchor = None
    if primary_geometry_center is not None:
        hand_anchors = {
            side: anchor
            for side, anchor in hand_anchors.items()
            if anchor_near_geometry(anchor, center, extent)
        }
    primary_projection_points = projection_points_from_center(center, extent)
    group_projection_points = projection_points_from_center(group_center, group_extent)
    return {
        "focus_ids": focus_ids,
        "primary_focus_id": primary_ids[0] if primary_ids else (focus_ids[0] if focus_ids else ""),
        "secondary_focus_ids": list(roles["secondary_ids"]),
        "focus_resolution_source": roles["source"],
        "focus_conflict_resolved": bool(roles["focus_conflict_resolved"]),
        "package_focus_ids": list(roles["package_focus_ids"]),
        "roots": roots,
        "primary_root": primary_root,
        "points": projection_points,
        "raw_points": points or upper,
        "primary_points": primary_projection_points,
        "primary_raw_points": primary_points or primary_projection_points,
        "group_points": group_projection_points,
        "group_raw_points": points or group_projection_points,
        "upper_points": upper,
        "focus_names": focus_names,
        "primary_focus_names": primary_names or focus_names,
        "center": center,
        "center_source": center_source,
        "raw_center": raw_center,
        "extent": extent,
        "geometry_center": geometry_center,
        "geometry_extent": geometry_extent,
        "primary_geometry_center": primary_geometry_center,
        "primary_geometry_extent": primary_geometry_extent,
        "group_center": group_center,
        "group_extent": group_extent,
        "layout_center": layout_center_value,
        "layout_extent": layout_extent_value,
        "face_anchor": face_anchor,
        "left_hand_anchor": hand_anchors.get("left"),
        "right_hand_anchor": hand_anchors.get("right"),
    }


FACE_SEMANTICS = {"face", "eyes", "head", "back_of_head"}
CHEST_SEMANTICS = {"chest", "torso", "upper_body", "upper body"}
FEET_SEMANTICS = {"feet", "foot", "shoe", "shoes"}
HAND_SEMANTICS = {"hands", "hand"}
CLOSEUP_STRATEGY_BY_PART = {
    "face": ("S3_ExtremeMacro",),
    "chest": (),
    "feet": ("S1_TopHat", "S4_GoldenRatio"),
    "hands": (),
}
CLOSEUP_STRATEGY_PARAMS = {
    "S3_ExtremeMacro": {
        "distance": 1.8,
        "lens_mm": 200.0,
        "target_z_offset": -0.08,
        "angle_deg": 20.0,
        "height_offset": 0.0,
        "source": "semantic_face_s3_extreme_macro",
    },
    "S3_ExtremeMacro_Dynamic": {
        "distance": 2.6,
        "lens_mm": 135.0,
        "target_z_offset": -0.04,
        "angle_deg": 20.0,
        "height_offset": 0.0,
        "source": "semantic_face_s3_extreme_macro_dynamic",
    },
    "S4_GoldenRatio": {
        "distance": 0.4,
        "lens_mm": 65.0,
        "target_z_offset": -0.04,
        "angle_deg": 15.0,
        "height_offset": 0.0,
        "source": "",
    },
    "S1_TopHat": {
        "distance": 0.6,
        "lens_mm": 85.0,
        "target_z_offset": -0.05,
        "angle_deg": 15.0,
        "height_offset": 0.0,
        "source": "",
    },
}
WORD_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)?")
DYNAMIC_FACE_ACTION_TERMS = {
    "approach",
    "approaches",
    "approaching",
    "collapse",
    "collapses",
    "collapsing",
    "enter",
    "enters",
    "entering",
    "fall",
    "falls",
    "falling",
    "fight",
    "fights",
    "fighting",
    "fire",
    "fires",
    "firing",
    "leave",
    "leaves",
    "leaving",
    "move",
    "moves",
    "moving",
    "return",
    "returns",
    "returning",
    "run",
    "runs",
    "running",
    "shoot",
    "shoots",
    "shooting",
    "sit",
    "sits",
    "sitting",
    "stand",
    "stands",
    "standing",
    "struggle",
    "struggles",
    "struggling",
    "turn",
    "turns",
    "turning",
    "walk",
    "walks",
    "walking",
}
DYNAMIC_FACE_ACTION_PHRASES = (
    "moves across",
    "large motion",
    "action follow",
    "gets up",
    "sits down",
    "stands up",
    "turns around",
    "turns back",
    "walks away",
    "walks toward",
    "runs away",
    "runs toward",
    "返回",
    "靠近",
    "走",
    "跑",
    "移动",
    "转身",
    "倒下",
    "起身",
    "坐下",
    "进入",
    "离开",
)
CLOSEUP_DYNAMIC_FACE_MIN_VISIBLE = 0.12
CLOSEUP_PHRASES = (
    "close up",
    "close-up",
    "closeup",
    "medium close-up",
    "medium close up",
    "tight close-up",
    "tight close up",
    "extreme close-up",
    "extreme close up",
    "insert shot",
    "detail shot",
    "object detail",
    "reaction shot",
    "reaction close-up",
    "reaction close up",
)
FACE_DETAIL_PHRASES = (
    "on his face",
    "on her face",
    "pleading face",
    "facial expression",
    "close on face",
    "eyes",
    "eye line",
    "eye-line",
    "back of head",
)
HAND_DETAIL_PHRASES = (
    "hand",
    "hands",
    "handing",
    "handed",
    "in his pocket",
    "in her pocket",
    "holding a weapon",
    "holds a lighter",
    "shaking hands",
    "steady hands",
)


def start_frame_contract(camera: dict[str, Any]) -> dict[str, Any]:
    start_contract = camera.get("start_frame_contract") or {}
    if not isinstance(start_contract, dict) or not start_contract:
        contract = camera.get("shot_contract") or {}
        start_contract = contract.get("start_frame_contract") or {}
    return start_contract if isinstance(start_contract, dict) else {}


def semantic_contract(camera: dict[str, Any]) -> dict[str, Any]:
    contract = start_frame_contract(camera)
    shot_size_contract = contract.get("shot_size_contract") or {}
    shot_size = ""
    if isinstance(shot_size_contract, dict):
        shot_size = str(shot_size_contract.get("shot_size") or shot_size_contract.get("size") or "").strip().lower()
    shot_size = shot_size or str(contract.get("shot_size") or camera.get("shot_size") or "").strip().lower()
    primary_semantic = str(
        contract.get("primary_semantic_target")
        or camera.get("primary_semantic_target")
        or ""
    ).strip().lower()
    if primary_semantic in HAND_SEMANTICS:
        primary_semantic = "hands"
    if primary_semantic in CHEST_SEMANTICS:
        primary_semantic = "full_body"
    distance_label = str(contract.get("distance") or camera.get("distance_label") or "").strip().lower()
    role_text = " ".join(
        str(value or "")
        for value in (
            camera.get("camera_role"),
            camera.get("coverage_role"),
            camera.get("coverage_type"),
            camera.get("movement_tag"),
        )
    ).lower()
    description = " ".join(
        str(value or "")
        for value in (
            camera.get("shot_description"),
            camera.get("scene_description"),
        )
    ).lower()
    contract_text = " ".join(str(value or "") for value in (contract.get("framing"), contract.get("shot_size"), contract.get("distance"))).lower()
    package_text = " ".join((distance_label, shot_size, role_text))
    closeup_from_contract = bool(contract.get("must_be_closeup") or contract.get("requires_closeup") or contract.get("must_show_face"))
    closeup_from_contract = closeup_from_contract or _contains_any_phrase(contract_text, CLOSEUP_PHRASES)
    closeup_from_package = _contains_any_phrase(package_text, CLOSEUP_PHRASES)
    closeup_from_description = _contains_any_phrase(description, CLOSEUP_PHRASES)
    closeup_required = bool(closeup_from_contract or closeup_from_package or closeup_from_description)
    must_show_face = bool(contract.get("must_show_face")) or primary_semantic in FACE_SEMANTICS or _contains_any_phrase(description, FACE_DETAIL_PHRASES)
    must_show_hands = False
    if closeup_from_contract:
        semantic_weight_reason = "contract_requires_closeup"
    elif closeup_from_package:
        semantic_weight_reason = "camera_package_closeup"
    elif closeup_from_description:
        semantic_weight_reason = "script_mentions_closeup_or_detail"
    else:
        semantic_weight_reason = "semantic_candidate_only"
    return {
        "primary_semantic_target": primary_semantic,
        "distance_label": distance_label,
        "shot_size": shot_size,
        "shot_size_contract": shot_size_contract if isinstance(shot_size_contract, dict) else {},
        "must_show_face": must_show_face,
        "must_show_hands": must_show_hands,
        "closeup_required": closeup_required,
        "semantic_weight_reason": semantic_weight_reason,
        "is_closeup": closeup_required,
    }


def _text_tokens(text: str) -> list[str]:
    return WORD_RE.findall(str(text or "").lower())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = " " + " ".join(_text_tokens(text)) + " "
    target = " " + " ".join(_text_tokens(phrase)) + " "
    return bool(target.strip()) and target in normalized


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _camera_text_blob(camera: dict[str, Any]) -> str:
    contract = start_frame_contract(camera)
    shot_contract = camera.get("shot_contract") or {}
    pieces: list[str] = [
        str(camera.get("movement_tag") or ""),
        str(camera.get("shot_description") or ""),
        str(camera.get("scene_description") or ""),
        str(camera.get("camera_role") or ""),
        str(camera.get("coverage_role") or ""),
        str(camera.get("coverage_type") or ""),
    ]
    for value in (contract, shot_contract):
        if isinstance(value, (dict, list)):
            pieces.append(json.dumps(value, ensure_ascii=False))
        else:
            pieces.append(str(value or ""))
    return " ".join(pieces).lower()


def is_dynamic_face_closeup(camera: dict[str, Any]) -> bool:
    if semantic_part_for_camera(camera) != "face":
        return False
    text = _camera_text_blob(camera)
    tokens = set(_text_tokens(text))
    if tokens.intersection(DYNAMIC_FACE_ACTION_TERMS):
        return True
    return any(phrase and phrase.lower() in text for phrase in DYNAMIC_FACE_ACTION_PHRASES)


def candidate_projection_visible(candidate: dict[str, Any]) -> float:
    projection = candidate.get("projection") or {}
    try:
        value = float(projection.get("visible_fraction") or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, value)


def preview_qc_failed(candidate: dict[str, Any]) -> bool:
    qc = candidate.get("preview_image_qc")
    if not isinstance(qc, dict):
        return False
    return str(qc.get("status") or "") == "failed"


BACK_DIRECTIONS = {"back", "back_left", "back_right"}
BACK_VIEW_PHRASES = (
    "back view",
    "back-facing",
    "back facing",
    "from behind",
    "behind him",
    "behind her",
    "over shoulder",
    "over-the-shoulder",
    "ots",
    "back of head",
    "背影",
    "背面",
    "后背",
    "身后",
)


def candidate_face_direction(candidate: dict[str, Any]) -> str:
    for key in ("semantic_direction", "direction", "internal_direction"):
        value = candidate.get(key)
        if value:
            return normalize_direction(value)
    return "front"


def face_back_view_allowed(camera: dict[str, Any]) -> bool:
    contract = semantic_contract(camera)
    primary = str(contract.get("primary_semantic_target") or camera.get("primary_semantic_target") or "").lower().replace(" ", "_")
    if primary == "back_of_head":
        return True
    if requested_direction(camera) in BACK_DIRECTIONS:
        return True
    text = _camera_text_blob(camera)
    return any(phrase in text for phrase in BACK_VIEW_PHRASES)


def face_candidate_back_facing(candidate: dict[str, Any]) -> bool:
    return candidate_face_direction(candidate) in BACK_DIRECTIONS


def readability_threshold(camera: dict[str, Any], candidate: dict[str, Any]) -> float:
    try:
        explicit = float(candidate.get("readability_threshold"))
        if math.isfinite(explicit) and explicit > 0.0:
            return explicit
    except (TypeError, ValueError):
        pass
    if semantic_closeup_source(candidate):
        return 0.08
    return 0.02


def readability_grade(camera: dict[str, Any], candidate: dict[str, Any]) -> str:
    visible = candidate_projection_visible(candidate)
    threshold = readability_threshold(camera, candidate)
    if visible <= 0.0:
        return "blocked"
    if visible < threshold:
        return "marginal"
    if visible < threshold * 1.8:
        return "readable"
    return "strong"


def candidate_readability_reasons(candidate: dict[str, Any], camera: dict[str, Any]) -> list[str]:
    projection = candidate.get("projection") or {}
    scores = candidate.get("scores") or {}
    visible_fraction = candidate_projection_visible(candidate)
    reasons: list[str] = []
    threshold = readability_threshold(camera, candidate)
    if visible_fraction <= 0.0:
        reasons.append("visible_fraction_zero")
    if not bool(projection.get("valid", True)):
        reasons.append("projection_invalid")
    if float(projection.get("area_ratio") or 0.0) <= 0.0:
        reasons.append("projection_area_zero")
    if preview_qc_failed(candidate):
        reasons.append(str(((candidate.get("preview_image_qc") or {}).get("reason") or "preview_qc_failed")))
    source = str(candidate.get("source") or "").lower()
    primary_visible = float(scores.get("primary_visible_fraction") or 0.0)
    primary_area = float(scores.get("primary_area_ratio") or 0.0)
    group_area = float(scores.get("group_subject_area_ratio") or 0.0)
    direction_score = float(scores.get("direction_alignment") or 0.0)
    is_semantic_closeup = semantic_closeup_source(candidate)
    if source.startswith("semantic_chest"):
        reasons.append("semantic_chest_disabled")
    if direction_score <= 0.0 and not is_semantic_closeup:
        reasons.append(f"direction_mismatch:{requested_direction(camera)}!={normalize_direction(candidate.get('semantic_direction') or candidate.get('direction'))}")
    if float(scores.get("camera_height_ok") or 0.0) < 0.5:
        reasons.append("camera_height_too_low")
    if primary_visible <= 0.0:
        reasons.append("primary_subject_not_visible")
    if primary_area < required_primary_area(camera):
        reasons.append(f"primary_area={primary_area:.3f} < floor={required_primary_area(camera):.3f}")
    group_floor = float(scores.get("required_group_area_floor") or 0.0)
    if group_floor > 0.0 and group_area < group_floor:
        reasons.append(f"group_area={group_area:.3f} < floor={group_floor:.3f}")
    if source.startswith("semantic_face") and face_candidate_back_facing(candidate) and not face_back_view_allowed(camera):
        reasons.append("face_target_back_facing")
    if source.startswith("semantic_face") and visible_fraction < threshold:
        reasons.append("face_detail_unreadable")
    if bool(candidate.get("dynamic_face_closeup")) and visible_fraction < CLOSEUP_DYNAMIC_FACE_MIN_VISIBLE:
        reasons.append("dynamic_face_action_visibility_below_threshold")
    if source.startswith("semantic_feet") and visible_fraction < threshold:
        reasons.append("semantic_detail_unreadable")
    if not source.startswith("semantic") and visible_fraction < threshold:
        reasons.append("action_visibility_below_threshold")
    return reasons


def candidate_readable_bonus(candidate: dict[str, Any], camera: dict[str, Any]) -> bool:
    projection = candidate.get("projection") or {}
    scores = candidate.get("scores") or {}
    occlusion = candidate.get("occlusion_check") or {}
    if candidate_readability_reasons(candidate, camera):
        return False
    return (
        bool(projection.get("valid", True))
        and candidate_projection_visible(candidate) > 0.0
        and float(scores.get("line_of_sight") or 0.0) >= 0.35
        and not bool(occlusion.get("severely_occluded"))
        and bool((candidate.get("wall_check") or {}).get("inside_scene_bounds", True))
    )


def semantic_priority(camera: dict[str, Any]) -> bool:
    contract = semantic_contract(camera)
    if bool(contract["closeup_required"]):
        return True
    return semantic_part_for_camera(camera) in {"face", "feet"}


def semantic_target_mode(camera: dict[str, Any]) -> str:
    part = semantic_part_for_camera(camera)
    return part if part in {"face", "feet"} else "body"


def semantic_closeup_source(candidate: dict[str, Any]) -> bool:
    source = str(candidate.get("source") or "")
    return source.startswith("semantic_face") or source.startswith("semantic_feet")


def likely_human_occluder(name: str) -> bool:
    text = str(name or "").lower()
    return any(token in text for token in ("char", "person", "actor", "human", "man", "woman"))


def basis_source(primary_root) -> str:
    return "primary_root_local_minus_y_plus_x" if primary_root is not None else "fallback_world_minus_y_plus_x"


def subject_basis(primary_root) -> tuple[Any, Any]:
    return basis_from_axes(primary_root, (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), local_space=primary_root is not None)


def evaluated_obj(obj):
    if obj is None:
        return None
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        return obj.evaluated_get(depsgraph)
    except Exception:
        return obj


def evaluated_world_bbox_points(obj) -> list[Any]:
    eval_obj = evaluated_obj(obj)
    matrix = getattr(eval_obj, "matrix_world", None) or getattr(obj, "matrix_world", None)
    box = getattr(eval_obj, "bound_box", None) or getattr(obj, "bound_box", None)
    points: list[Any] = []
    if matrix is not None and box:
        for corner in box:
            try:
                points.append(matrix @ mathutils.Vector(corner))
            except Exception:
                continue
    return points


def _primary_armature(bundle: dict[str, Any]):
    root = bundle.get("primary_root")
    if root is not None and getattr(root, "type", "") == "ARMATURE":
        return root
    for obj in descendants(root) if root is not None else []:
        if getattr(obj, "type", "") == "ARMATURE":
            return obj
    return None


def evaluated_pose_bone_points(bundle: dict[str, Any], search_terms: tuple[str, ...]) -> list[Any]:
    armature = _primary_armature(bundle)
    if armature is None:
        return []
    eval_armature = evaluated_obj(armature)
    rows: list[Any] = []
    for bone in pose_bones(eval_armature):
        name = str(getattr(bone, "name", "") or "").lower()
        if not any(term in name for term in search_terms):
            continue
        try:
            head = eval_armature.matrix_world @ bone.head
            tail = eval_armature.matrix_world @ bone.tail
            rows.append((head + tail) * 0.5)
        except Exception:
            continue
    return rows


def semantic_bbox_points(bundle: dict[str, Any]) -> list[Any]:
    root = bundle.get("primary_root")
    points: list[Any] = []
    for obj in descendants(root) if root is not None else []:
        if getattr(obj, "type", "") in {"MESH", "CURVE", "SURFACE", "FONT"}:
            points.extend(evaluated_world_bbox_points(obj))
    if not points and root is not None:
        points.extend(evaluated_world_bbox_points(root))
    return points or list(bundle.get("primary_raw_points") or bundle.get("raw_points") or [])


def bbox_ratio_point(bundle: dict[str, Any], center: Any, extent: Any, ratio: float) -> Any:
    points = semantic_bbox_points(bundle)
    if points:
        z_min = min(float(point.z) for point in points)
        z_max = max(float(point.z) for point in points)
        x = sum(float(point.x) for point in points) / float(len(points))
        y = sum(float(point.y) for point in points) / float(len(points))
        bbox_center = mathutils.Vector((x, y, (z_min + z_max) * 0.5))
        bbox_extent = mathutils.Vector(
            (
                max(float(point.x) for point in points) - min(float(point.x) for point in points),
                max(float(point.y) for point in points) - min(float(point.y) for point in points),
                z_max - z_min,
            )
        )
        max_dim = max(abs(float(bbox_extent.x)), abs(float(bbox_extent.y)), abs(float(bbox_extent.z)))
        center_delta = (bbox_center - center).length
        allowed_delta = max(float(extent.x), float(extent.y), float(extent.z), 0.5) * 3.0 + 0.5
        if max_dim <= 10.0 and center_delta <= allowed_delta:
            return mathutils.Vector((x, y, z_min + (z_max - z_min) * float(ratio)))
    target = center.copy()
    target.z = float(center.z) - float(extent.z) * 0.5 + float(extent.z) * float(ratio)
    return target


def valid_semantic_target(bundle: dict[str, Any], target: Any, center: Any, extent: Any) -> bool:
    try:
        values = (float(target.x), float(target.y), float(target.z))
    except Exception:
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    height = max(float(extent.z), 1e-6)
    z_margin = max(0.2, height * 0.25)
    xy_margin = max(0.35, max(float(extent.x), float(extent.y), 0.5) * 0.75)
    target_xy = mathutils.Vector((float(target.x), float(target.y), 0.0))
    center_xy = mathutils.Vector((float(center.x), float(center.y), 0.0))
    if not (
        float(center.z) - height * 0.5 - z_margin <= float(target.z) <= float(center.z) + height * 0.5 + z_margin
        and (target_xy - center_xy).length <= max(float(extent.x), float(extent.y), 0.5) + xy_margin
    ):
        return False
    points = semantic_bbox_points(bundle)
    if not points:
        return True
    minimum, maximum = points_bounds(points)
    raw_height = max(float(maximum.z - minimum.z), 1e-6)
    if raw_height > 10.0:
        return True
    z_margin = max(0.25, raw_height * 0.25)
    xy_margin = max(0.5, raw_height * 0.5)
    xy_center = mathutils.Vector((sum(float(p.x) for p in points) / len(points), sum(float(p.y) for p in points) / len(points), 0.0))
    return bool(
        float(minimum.z) - z_margin <= float(target.z) <= float(maximum.z) + z_margin
        and (target_xy - xy_center).length <= max(float(extent.x), float(extent.y), 0.5) + xy_margin
    )


def semantic_target_point(part: str, strategy_name: str, bundle: dict[str, Any], center: Any, extent: Any) -> Any:
    if _DISABLE_SEMANTIC_HEIGHT_ADJUST:
        # Ablation: always aim at geometric center; skip face/feet/hand bone lookup.
        try:
            return center.copy()
        except AttributeError:
            return mathutils.Vector((float(center.x), float(center.y), float(center.z)))
    part = str(part or "").lower()
    if part == "face":
        bone_points = evaluated_pose_bone_points(bundle, ("head", "face", "eye"))
        target = average_points(bone_points) if bone_points else None
        fallback = bbox_ratio_point(bundle, center, extent, 0.88)
    elif part == "chest":
        bone_points = evaluated_pose_bone_points(bundle, ("spine", "chest", "breast", "ribs"))
        bone_points.sort(key=lambda point: float(point.z))
        top_points = bone_points[-2:] if len(bone_points) >= 2 else bone_points
        target = average_points(top_points) if top_points else None
        if target is not None:
            target.z += 0.05
        fallback = bbox_ratio_point(bundle, center, extent, 0.70)
    elif part == "feet":
        bone_points = evaluated_pose_bone_points(bundle, ("foot", "toe", "heel", "shoe"))
        bone_points.sort(key=lambda point: float(point.z))
        bottom_points = bone_points[:2] if len(bone_points) >= 2 else bone_points
        target = average_points(bottom_points) if bottom_points else None
        fallback = bbox_ratio_point(bundle, center, extent, 0.10)
    else:
        return center.copy()
    if target is None or not valid_semantic_target(bundle, target, center, extent):
        return fallback
    return target


def anchor_metadata(anchor_kind: str, anchor_point: Any, extent: Any) -> dict[str, Any]:
    projection_points = anchor_box_points(anchor_point, extent, anchor_kind)
    upper_points = upper_points_for_anchor(anchor_point, extent, anchor_kind)
    return {
        "anchor_kind": anchor_kind,
        "anchor_point": round_vector(anchor_point),
        "projection_points": serialize_points(projection_points),
        "upper_points": serialize_points(upper_points),
    }


def semantic_part_for_camera(camera: dict[str, Any]) -> str:
    contract = semantic_contract(camera)
    semantic = str(contract.get("primary_semantic_target") or "").lower().replace("-", "_")
    desc = " ".join(str(value or "") for value in (camera.get("shot_description"), camera.get("scene_description"))).lower()
    if semantic in FACE_SEMANTICS or bool(contract.get("must_show_face")):
        return "face"
    if semantic in FEET_SEMANTICS:
        return "feet"
    if semantic in HAND_SEMANTICS:
        return "hands"
    if _contains_any_phrase(desc, ("feet", "foot", "shoes")):
        return "feet"
    if _contains_any_phrase(desc, FACE_DETAIL_PHRASES):
        return "face"
    return "body"


def closeup_strategy_source(part: str, strategy_name: str) -> str:
    if part == "face" and strategy_name == "S3_ExtremeMacro":
        return "semantic_face_s3_extreme_macro"
    if part == "face" and strategy_name == "S3_ExtremeMacro_Dynamic":
        return "semantic_face_s3_extreme_macro_dynamic"
    if part == "feet" and strategy_name == "S1_TopHat":
        return "semantic_feet_s1_tophat"
    if part == "feet" and strategy_name == "S4_GoldenRatio":
        return "semantic_feet_s4_golden_ratio"
    return f"semantic_{part}_{strategy_name.lower()}"


def closeup_anchor_kind(part: str) -> str:
    return {
        "face": "face_anchor",
        "chest": "chest_anchor",
        "feet": "feet_anchor",
    }.get(part, "body_anchor")


def make_closeup_seed(
    camera: dict[str, Any],
    bundle: dict[str, Any],
    center: Any,
    extent: Any,
    front: Any,
    right: Any,
    *,
    part: str,
    strategy_name: str,
    semantic_direction: str,
) -> dict[str, Any]:
    contract = semantic_contract(camera)
    params = dict(CLOSEUP_STRATEGY_PARAMS[strategy_name])
    target = semantic_target_point(part, strategy_name, bundle, center, extent)
    framing_target = target.copy()
    if not _DISABLE_SEMANTIC_HEIGHT_ADJUST:
        framing_target.z += float(params["target_z_offset"])
    direction = direction_vector(semantic_direction, front, right)
    direction = horizontal_vector(direction, front)
    angle = math.radians(float(params["angle_deg"]))
    cinematic_dir = mathutils.Matrix.Rotation(angle, 4, "Z") @ direction
    cinematic_dir = horizontal_vector(cinematic_dir, direction)
    location = framing_target + cinematic_dir * float(params["distance"])
    if part == "feet":
        location.z += 0.4
    elif part == "chest":
        location.z += 0.1
    location = enforce_camera_height(camera, location, center, extent, framing_target)
    anchor_kind = closeup_anchor_kind(part)
    dynamic_face = part == "face" and strategy_name == "S3_ExtremeMacro_Dynamic"
    if dynamic_face:
        anchor_kind = "face_motion_anchor"
    seed = {
        "location": round_vector(location),
        "rotation_euler": round_vector(look_at(location, framing_target)),
        "target": round_vector(framing_target),
        "lens_mm": round(float(params["lens_mm"]), 4),
        "operation_sequence": [],
        "operation_depth": 0,
        "source": closeup_strategy_source(part, strategy_name),
        "semantic_contract": contract,
        "semantic_part": part,
        "closeup_strategy": strategy_name,
        "readability_threshold": 0.06 if dynamic_face else 0.08,
        **anchor_metadata(anchor_kind, target, extent),
    }
    if dynamic_face:
        seed["motion_variant"] = "dynamic_face_closeup"
        seed["dynamic_face_closeup"] = True
        seed["min_motion_visible_fraction"] = CLOSEUP_DYNAMIC_FACE_MIN_VISIBLE
    return _seed_subject_vectors(seed, front, right, semantic_direction)


def mirrored_face_seed(seed: dict[str, Any], bundle: dict[str, Any], front: Any, right: Any) -> dict[str, Any]:
    target = mathutils.Vector(tuple(float(value) for value in seed.get("target", (0.0, 0.0, 1.0))))
    location = mathutils.Vector(tuple(float(value) for value in seed.get("location", (0.0, 0.0, 1.0))))
    root = bundle.get("primary_root")
    mirrored_location = None
    if root is not None:
        try:
            local_position = root.matrix_world.inverted() @ location
            local_position.y *= -1.0
            mirrored_location = root.matrix_world @ local_position
        except Exception:
            mirrored_location = None
    if mirrored_location is None:
        offset = location - target
        front_component = front * offset.dot(front)
        mirrored_location = location - front_component * 2.0
    mirrored = dict(seed)
    mirrored["location"] = round_vector(mirrored_location)
    mirrored["rotation_euler"] = round_vector(look_at(mirrored_location, target))
    mirrored["operation_sequence"] = list(seed.get("operation_sequence") or []) + ["frontback_mirror"]
    mirrored["operation_depth"] = 0
    mirrored["source"] = f"{seed.get('source')}_mirror"
    mirrored["frontback_mirror_candidate"] = True
    return _seed_subject_vectors(mirrored, front, right, semantic_direction_from_transform(mirrored))


def semantic_closeup_seeds(
    camera: dict[str, Any],
    bundle: dict[str, Any],
    center: Any,
    extent: Any,
    front: Any,
    right: Any,
    part: str,
) -> list[dict[str, Any]]:
    strategy_names = CLOSEUP_STRATEGY_BY_PART.get(part, ())
    if part == "face" and is_dynamic_face_closeup(camera):
        strategy_names = ("S3_ExtremeMacro_Dynamic", "S3_ExtremeMacro")
    if not strategy_names:
        return []
    primary_focus_id = str(bundle.get("primary_focus_id") or "").strip()
    primary_root = bundle.get("primary_root")
    if part == "face" and not likely_human_focus_id(primary_focus_id) and getattr(primary_root, "type", "") != "ARMATURE":
        return []
    rows: list[dict[str, Any]] = []
    for strategy_name in strategy_names:
        for semantic_direction in allowed_directions(camera, include_neighbors=True):
            seed = make_closeup_seed(
                camera,
                bundle,
                center,
                extent,
                front,
                right,
                part=part,
                strategy_name=strategy_name,
                semantic_direction=semantic_direction,
            )
            rows.append(seed)
            if part == "face" and strategy_name in {"S3_ExtremeMacro", "S3_ExtremeMacro_Dynamic"}:
                rows.append(mirrored_face_seed(seed, bundle, front, right))
    return rows


def semantic_face_seeds(camera: dict[str, Any], bundle: dict[str, Any], center: Any, extent: Any, front: Any, right: Any) -> list[dict[str, Any]]:
    return semantic_closeup_seeds(camera, bundle, center, extent, front, right, "face") if semantic_part_for_camera(camera) == "face" else []


def semantic_chest_seeds(camera: dict[str, Any], bundle: dict[str, Any], center: Any, extent: Any, front: Any, right: Any) -> list[dict[str, Any]]:
    return []


def semantic_feet_seeds(camera: dict[str, Any], bundle: dict[str, Any], center: Any, extent: Any, front: Any, right: Any) -> list[dict[str, Any]]:
    return semantic_closeup_seeds(camera, bundle, center, extent, front, right, "feet") if semantic_part_for_camera(camera) == "feet" else []


def probe_candidates_for_basis(camera: dict[str, Any], bundle: dict[str, Any], center: Any, extent: Any, front: Any, right: Any) -> list[dict[str, Any]]:
    mode = semantic_target_mode(camera)
    if mode == "face":
        return semantic_face_seeds(camera, bundle, center, extent, front, right)[:3]
    if mode == "feet":
        return semantic_feet_seeds(camera, bundle, center, extent, front, right)[:3]
    rows: list[dict[str, Any]] = []
    for direction in ("front", "front_left", "front_right", "right"):
        seed = base_transform(camera, center, extent, front, right, direction)
        seed["channel"] = "direction"
        seed["readability_threshold"] = 0.02
        seed.update(anchor_metadata("body_anchor", center, extent))
        rows.append(seed)
    return rows


def required_primary_area(camera: dict[str, Any]) -> float:
    contract = start_frame_contract(camera)
    try:
        floor = float(contract.get("primary_screen_area_floor") or 0.0)
    except (TypeError, ValueError):
        floor = 0.0
    label = str(contract.get("distance") or camera.get("distance_label") or "").lower()
    if "close" in label:
        return max(floor, 0.10)
    if "wide" in label or "long" in label:
        return max(min(floor, 0.022), 0.016)
    return max(floor, 0.055)


def required_group_area(camera: dict[str, Any], bundle: dict[str, Any]) -> float:
    if len(bundle.get("focus_ids") or []) <= 1:
        return 0.0
    label = str(camera.get("distance_label") or "").lower()
    if "wide" in label or "long" in label:
        return 0.055
    if "close" in label:
        return 0.0
    return 0.075


def camera_height_ok(camera: dict[str, Any], candidate: dict[str, Any], bundle: dict[str, Any]) -> bool:
    location = candidate.get("location") or []
    target = candidate.get("target") or []
    if len(location) < 3 or len(target) < 3:
        return True
    angle = str(camera.get("angle_label") or "").lower()
    if "low" in angle:
        return True
    try:
        z = float(location[2])
        target_z = float(target[2])
    except (TypeError, ValueError):
        return True
    center = bundle.get("center")
    extent = bundle.get("extent")
    if center is None or extent is None:
        return z >= target_z + 0.12
    return z >= minimum_camera_height(camera, center, extent, mathutils.Vector(tuple(float(value) for value in target[:3]))) - 1e-4


def probe_basis(scene, camera_obj, camera: dict[str, Any], bundle: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    _ = details
    primary_root = bundle.get("primary_root")
    center_choices: list[tuple[str, Any, Any]] = []
    if (
        len(bundle.get("focus_ids") or []) > 1
        and not bool(semantic_contract(camera).get("is_closeup"))
        and bundle.get("group_center") is not None
        and bundle.get("group_extent") is not None
    ):
        center_choices.append(("group_center", bundle["group_center"].copy(), bundle["group_extent"].copy()))
    if bundle.get("geometry_center") is not None and bundle.get("geometry_extent") is not None:
        center_choices.append(("geometry_center", bundle["geometry_center"].copy(), bundle["geometry_extent"].copy()))
    center_choices.append((str(bundle.get("center_source") or "geometry_center"), bundle["center"].copy(), bundle["extent"].copy()))
    unique_centers: list[tuple[str, Any, Any]] = []
    seen_centers = set()
    for source_name, center, extent in center_choices:
        key = tuple(int(round(float(value) * 1000.0)) for value in (center.x, center.y, center.z, extent.x, extent.y, extent.z))
        if key in seen_centers:
            continue
        seen_centers.add(key)
        unique_centers.append((source_name, center, extent))

    # Fix A: prefer rig-derived face forward over projected-area probing.
    # Probing by projection area cannot distinguish front vs. back of head
    # because both directions see roughly equal mesh area. When the armature
    # exposes head + eye/face bones, derive the forward vector geometrically
    # and only run the per-center probe to pick the best center; basis_source
    # is locked to ``armature_face_forward``.
    forced_front = armature_face_forward(primary_root)
    forced_right = None
    if forced_front is not None:
        forced_front = horizontal_vector(forced_front, mathutils.Vector((0.0, -1.0, 0.0)))
        forced_right = orthogonal_right(forced_front, mathutils.Vector((1.0, 0.0, 0.0)))

    best: dict[str, Any] | None = None
    for center_source, center, extent in unique_centers:
        if forced_front is not None:
            basis_iter: list[dict[str, Any]] = [{
                "basis_source": "armature_face_forward",
                "_forced_front": forced_front,
                "_forced_right": forced_right,
            }]
        else:
            basis_iter = basis_variants(primary_root)
        for basis in basis_iter:
            if "_forced_front" in basis:
                front = basis["_forced_front"]
                right = basis["_forced_right"]
            else:
                front, right = basis_from_axes(
                    primary_root,
                    basis["front_axis"],
                    basis["right_axis"],
                    local_space=bool(basis.get("local_space")),
                )
            probe_score = 0.0
            for candidate in probe_candidates_for_basis(camera, bundle, center.copy(), extent.copy(), front, right):
                apply_transform(camera_obj, candidate)
                update_scene_view_layer(scene)
                projection_points = deserialize_points(candidate.get("projection_points")) or projection_points_from_center(center, extent)
                projection = project_bounds(scene, camera_obj, projection_points)
                upper = point_visibility(scene, camera_obj, deserialize_points(candidate.get("upper_points")) or projection_points)
                score = (
                    float(projection.get("visible_fraction") or 0.0)
                    + float(projection.get("area_ratio") or 0.0) * 4.0
                    + float(upper.get("visible_ratio") or 0.0) * 0.35
                    + (0.2 if projection.get("valid") else 0.0)
                )
                probe_score = max(probe_score, score)
            if best is None or probe_score > float(best["probe_score"]):
                best = {
                    "front": front,
                    "right": right,
                    "basis_source": str(basis.get("basis_source") or basis_source(primary_root)),
                    "basis_confidence": "high" if probe_score > 0.0 else "fallback",
                    "basis_front_vector": round_vector(front),
                    "center": center,
                    "extent": extent,
                    "center_source": center_source,
                    "probe_score": probe_score,
                }
    if best is None:
        front, right = subject_basis(primary_root)
        best = {
            "front": front,
            "right": right,
            "basis_source": basis_source(primary_root),
            "basis_confidence": "fallback",
            "basis_front_vector": round_vector(front),
            "center": bundle["center"].copy(),
            "extent": bundle["extent"].copy(),
            "center_source": str(bundle.get("center_source") or "geometry_center"),
            "probe_score": 0.0,
        }
    return best


def direction_vector(name: str, front: Any, right: Any) -> Any:
    mapping = {
        "front": front,
        "front_right": front + right,
        "right": right,
        "back_right": -front + right,
        "back": -front,
        "back_left": -front - right,
        "left": -right,
        "front_left": front - right,
    }
    vector = mapping.get(name, front).copy()
    if vector.length <= 1e-5:
        vector = mathutils.Vector((0.0, -1.0, 0.0))
    vector.normalize()
    return vector


def shot_profile(camera: dict[str, Any], extent: Any) -> dict[str, float]:
    label = str(camera.get("distance_label") or "").lower()
    height = max(float(extent.z), 0.9)
    if "extreme" in label or "macro" in label:
        return {"distance": max(height * 0.25, 0.35), "area_target": 0.35, "target_z_bias": 0.35, "recommended_lens_mm": 200.0}
    if "close" in label:
        return {"distance": max(height * 0.55, 0.7), "area_target": 0.20, "target_z_bias": 0.28, "recommended_lens_mm": 135.0}
    if "medium" in label:
        return {"distance": max(height * 1.6, 2.0), "area_target": 0.12, "target_z_bias": 0.16, "recommended_lens_mm": 50.0}
    if "full" in label:
        return {"distance": max(height * 2.8, 3.6), "area_target": 0.08, "target_z_bias": 0.10, "recommended_lens_mm": 35.0}
    if "wide" in label or "long" in label:
        return {"distance": max(height * 3.6, 4.6), "area_target": 0.05, "target_z_bias": 0.05, "recommended_lens_mm": 24.0}
    return {"distance": max(height * 2.1, 2.6), "area_target": 0.12, "target_z_bias": 0.16, "recommended_lens_mm": 35.0}


def minimum_camera_height(camera: dict[str, Any], center: Any, extent: Any, target: Any) -> float:
    angle = str(camera.get("angle_label") or "").lower()
    if "low" in angle:
        return float(target.z) + 0.08
    if "high" in angle:
        return float(target.z) + max(float(extent.z) * 0.22, 0.35)
    return max(float(center.z) + max(float(extent.z) * 0.15, 0.35), float(target.z) + 0.18, 0.65)


def enforce_camera_height(camera: dict[str, Any], location: Any, center: Any, extent: Any, target: Any) -> Any:
    if _DISABLE_SEMANTIC_HEIGHT_ADJUST:
        # Ablation: do not bump camera height; leave the geometric placement
        # alone so we can attribute IC2 changes to semantic Z adjustment.
        try:
            return location.copy()
        except AttributeError:
            return mathutils.Vector((float(location.x), float(location.y), float(location.z)))
    adjusted = location.copy()
    adjusted.z = max(float(adjusted.z), minimum_camera_height(camera, center, extent, target))
    return adjusted


def _seed_subject_vectors(seed: dict[str, Any], front: Any, right: Any, semantic_direction: str) -> dict[str, Any]:
    seed["semantic_direction"] = normalize_direction(semantic_direction)
    seed["direction"] = seed["semantic_direction"]
    seed["internal_direction"] = seed["semantic_direction"]
    seed["subject_front_vector"] = round_vector(front)
    seed["subject_right_vector"] = round_vector(right)
    return seed


def base_transform(camera: dict[str, Any], center: Any, extent: Any, front: Any, right: Any, direction_name: str) -> dict[str, Any]:
    profile = shot_profile(camera, extent)
    direction = direction_vector(direction_name, front, right)
    target = center.copy()
    target.z += max(float(extent.z) * profile["target_z_bias"], 0.12)
    location = target + direction * float(profile["distance"])
    location.z = max(location.z + max(float(extent.z) * 0.18, 0.25), target.z + 0.18)
    location = enforce_camera_height(camera, location, center, extent, target)
    lens = lens_clamp(float(profile.get("recommended_lens_mm") or camera.get("lens_mm") or 35.0))
    seed = {
        "location": round_vector(location),
        "rotation_euler": round_vector(look_at(location, target)),
        "target": round_vector(target),
        "lens_mm": round(lens, 4),
        "operation_sequence": [],
        "operation_depth": 0,
        "source": "direction_seed",
    }
    return _seed_subject_vectors(seed, front, right, direction_name)


def preset_transforms(camera: dict[str, Any], center: Any, extent: Any, front: Any, right: Any) -> list[dict[str, Any]]:
    import random
    profile = shot_profile(camera, extent)
    rows = []
    direction_name = requested_direction(camera)
    
    # Monte Carlo Sampling: generate 50 random cinematic camera setups
    num_samples = 50
    # Standard cinematic prime lenses
    prime_lenses = [14.0, 18.0, 24.0, 28.0, 35.0, 42.0, 50.0, 65.0, 85.0, 105.0, 135.0]
    
    for i in range(num_samples):
        # Distance multiplier: from tight close-up (0.5) to extreme wide (2.5)
        distance_mult = random.uniform(0.5, 2.5)
        # Elevation bias: from low angle (-0.3) to high/drone angle (0.6)
        elevation_bias = random.uniform(-0.3, 0.6)
        lens_mm = random.choice(prime_lenses)
        
        label = f"mc_sample_{i:02d}"
        direction = direction_vector(direction_name, front, right)
        target = center.copy()
        target.z += max(float(extent.z) * profile["target_z_bias"], 0.12)
        location = target + direction * float(profile["distance"]) * float(distance_mult)
        location.z += max(float(extent.z) * float(elevation_bias), -0.2)
        location.z = max(location.z, target.z + 0.18)
        location = enforce_camera_height(camera, location, center, extent, target)
        
        seed = {
            "location": round_vector(location),
            "rotation_euler": round_vector(look_at(location, target)),
            "target": round_vector(target),
            "lens_mm": round(lens_clamp(float(lens_mm)), 4),
            "operation_sequence": [],
            "operation_depth": 0,
            "source": f"preset_seed_{label}",
        }
        rows.append(_seed_subject_vectors(seed, front, right, direction_name))
    return rows


def camera_vectors(transform: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    location = mathutils.Vector(tuple(float(value) for value in transform.get("location", (0.0, 0.0, 0.0))))
    target = mathutils.Vector(tuple(float(value) for value in transform.get("target", (0.0, 0.0, 1.0))))
    forward = target - location
    if forward.length <= 1e-6:
        forward = mathutils.Vector((0.0, 1.0, 0.0))
    forward.normalize()
    right = forward.cross(mathutils.Vector((0.0, 0.0, 1.0)))
    if right.length <= 1e-6:
        right = mathutils.Vector((1.0, 0.0, 0.0))
    right.normalize()
    up = right.cross(forward)
    if up.length <= 1e-6:
        up = mathutils.Vector((0.0, 0.0, 1.0))
    up.normalize()
    return location, target, forward, right, up


def apply_operation(transform: dict[str, Any], operation: str) -> dict[str, Any]:
    location, target, forward, right, up = camera_vectors(transform)
    distance = max(float((target - location).length), 0.5)
    lens = lens_clamp(float(transform.get("lens_mm") or 35.0))
    fov = 2.0 * math.atan(36.0 / (2.0 * max(lens, 1.0)))
    view_height = 2.0 * math.tan(fov * 0.5) * distance
    pan_step = max(view_height * 0.09, 0.06)
    truck_step = max(view_height * 0.075, 0.05)
    dolly_step = max(distance * 0.10, 0.08)
    pedestal_step = max(view_height * 0.07, 0.05)
    if operation == "pan_left":
        location -= right * pan_step
        target -= right * pan_step
    elif operation == "pan_right":
        location += right * pan_step
        target += right * pan_step
    elif operation == "pan_up":
        location += up * pan_step
        target += up * pan_step
    elif operation == "pan_down":
        location -= up * pan_step
        target -= up * pan_step
    elif operation == "truck_left":
        location -= right * truck_step
    elif operation == "truck_right":
        location += right * truck_step
    elif operation == "pedestal_up":
        location.z += pedestal_step
        target.z += pedestal_step
    elif operation == "pedestal_down":
        location.z -= pedestal_step
        target.z -= pedestal_step
    elif operation == "dolly_in":
        location += forward * min(dolly_step, distance * 0.35)
    elif operation == "dolly_out":
        location -= forward * dolly_step
    elif operation == "zoom_in":
        lens = lens_clamp(lens * 1.10)
    elif operation == "zoom_out":
        lens = lens_clamp(lens / 1.10)
    elif operation in {"orbit_left", "orbit_right"}:
        location = target + mathutils.Matrix.Rotation(math.radians(6.0 if operation == "orbit_right" else -6.0), 4, "Z") @ (location - target)
    elif operation in {"orbit_up", "orbit_down"}:
        location = target + mathutils.Matrix.Rotation(math.radians(4.5 if operation == "orbit_up" else -4.5), 4, right) @ (location - target)
    updated = dict(transform)
    updated["location"] = round_vector(location)
    updated["target"] = round_vector(target)
    updated["rotation_euler"] = round_vector(look_at(location, target))
    updated["lens_mm"] = round(lens, 4)
    updated["operation_sequence"] = list(transform.get("operation_sequence") or []) + [operation]
    updated["operation_depth"] = len(updated["operation_sequence"])
    updated["source"] = "operation_expansion"
    semantic_direction = semantic_direction_from_transform(updated)
    updated["semantic_direction"] = semantic_direction
    updated["direction"] = semantic_direction
    updated["internal_direction"] = semantic_direction
    return updated


def transform_key(transform: dict[str, Any]) -> tuple[int, ...]:
    values = list(transform.get("location") or []) + list(transform.get("rotation_euler") or []) + [transform.get("lens_mm")]
    return tuple(int(round(float(value) * 1000.0)) for value in values if value is not None)


def apply_transform(camera_obj, transform: dict[str, Any]) -> None:
    camera_obj.location = mathutils.Vector(tuple(float(value) for value in transform.get("location", (0.0, 0.0, 0.0))))
    camera_obj.rotation_euler = mathutils.Euler(tuple(float(value) for value in transform.get("rotation_euler", (0.0, 0.0, 0.0))))
    camera_obj.data.lens = lens_clamp(float(transform.get("lens_mm") or 35.0))
    # Expand camera clipping bounds to prevent subjects from disappearing when too close or too far
    camera_obj.data.clip_start = 0.01
    camera_obj.data.clip_end = 1000.0


def update_scene_view_layer(scene) -> None:
    try:
        view_layers = list(getattr(scene, "view_layers", []) or [])
        if view_layers:
            view_layers[0].update()
            return
    except Exception:
        pass
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass


def point_in_front(camera_obj, point: Any) -> bool:
    try:
        location = camera_obj.location
        forward = camera_obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
        delta = point - location
        min_depth = max(float(getattr(camera_obj.data, "clip_start", 0.01) or 0.01) * 0.5, 1e-4)
        return bool(float(delta.dot(forward)) > min_depth)
    except Exception:
        try:
            local_point = camera_obj.matrix_world.inverted() @ point
            return bool(float(local_point.z) < -1e-4)
        except Exception:
            return False


def _finite_camera_coord(coord: Any) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in (coord.x, coord.y, coord.z))
    except Exception:
        return False


def project_bounds(scene, camera_obj, points: list[Any]) -> dict[str, Any]:
    if not points:
        return {
            "valid": False,
            "area_ratio": 0.0,
            "visible_fraction": 0.0,
            "center_distance": 1.0,
            "point_count": 0,
            "in_front_count": 0,
            "in_frame_count": 0,
        }
    coords = []
    for point in points:
        if not point_in_front(camera_obj, point):
            continue
        try:
            coord = world_to_camera_view(scene, camera_obj, point)
        except Exception:
            continue
        if not _finite_camera_coord(coord):
            continue
        coords.append(coord)
    if not coords:
        return {
            "valid": False,
            "area_ratio": 0.0,
            "visible_fraction": 0.0,
            "center_distance": 1.0,
            "point_count": len(points),
            "in_front_count": 0,
            "in_frame_count": 0,
        }
    xs = [float(coord.x) for coord in coords]
    ys = [float(coord.y) for coord in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    inter_w = max(0.0, min(max_x, 1.0) - max(min_x, 0.0))
    inter_h = max(0.0, min(max_y, 1.0) - max(min_y, 0.0))
    visible_fraction = (inter_w * inter_h) / max(width * height, 1e-6)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    in_frame_count = sum(1 for coord in coords if 0.0 <= float(coord.x) <= 1.0 and 0.0 <= float(coord.y) <= 1.0)
    valid = inter_w > 0.0 and inter_h > 0.0
    return {
        "valid": bool(valid),
        "bbox": [round(min_x, 5), round(min_y, 5), round(max_x, 5), round(max_y, 5)],
        "area_ratio": round(float(inter_w * inter_h if valid else 0.0), 6),
        "visible_fraction": round(float(visible_fraction if valid else 0.0), 6),
        "center_distance": round(float(math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)), 6),
        "point_count": len(points),
        "in_front_count": len(coords),
        "in_frame_count": in_frame_count,
        "in_front_fraction": round(float(len(coords) / max(len(points), 1)), 6),
        "in_frame_fraction": round(float(in_frame_count / max(len(points), 1)), 6),
    }


def point_visibility(scene, camera_obj, points: list[Any]) -> dict[str, Any]:
    if not points:
        return {"required": False, "visible_ratio": 1.0, "points_in_frame": 0, "point_count": 0}
    visible = 0
    for point in points:
        coord = world_to_camera_view(scene, camera_obj, point)
        if point_in_front(camera_obj, point) and 0.02 <= float(coord.x) <= 0.98 and 0.04 <= float(coord.y) <= 0.98:
            visible += 1
    return {
        "required": True,
        "visible_ratio": round(float(visible / max(len(points), 1)), 6),
        "points_in_frame": visible,
        "point_count": len(points),
    }


def line_of_sight(scene, camera_obj, points: list[Any], focus_names: set[str]) -> dict[str, Any]:
    if not points:
        return {"required": False, "clear_ratio": 1.0, "clear_count": 0, "point_count": 0}
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None
    origin = camera_obj.location.copy()
    clear_count = 0
    rays = []
    for point in points:
        delta = point - origin
        distance = float(delta.length)
        if distance <= 1e-5:
            clear_count += 1
            rays.append({"clear": True, "reason": "point_at_camera"})
            continue
        clear = True
        hit_name = ""
        try:
            hit, _location, _normal, _face_index, hit_object, _matrix = scene.ray_cast(
                depsgraph,
                origin,
                delta.normalized(),
                distance=max(distance - 0.03, 0.0),
            )
        except Exception:
            hit = False
            hit_object = None
        if hit and hit_object is not None:
            hit_name = str(hit_object.name)
            if not hit_is_focus_object(hit_object, focus_names):
                clear = False
        if clear:
            clear_count += 1
        rays.append({"clear": clear, "hit_object": hit_name or None})
    return {
        "required": True,
        "clear_ratio": round(float(clear_count / max(len(points), 1)), 6),
        "clear_count": clear_count,
        "point_count": len(points),
        "rays": rays[:8],
    }


def occlusion_check(scene, camera_obj, bundle: dict[str, Any]) -> dict[str, Any]:
    """Dense ray-cast occlusion test from camera to a grid of sample points on the focus subject."""
    center = bundle["center"]
    extent = bundle["extent"]
    focus_names = bundle["focus_names"]
    half_x = max(float(extent.x) * 0.5, 0.1)
    half_y = max(float(extent.y) * 0.5, 0.1)
    half_z = max(float(extent.z) * 0.5, 0.1)
    sample_points = []
    for dz in (-0.4, -0.1, 0.15, 0.35, 0.5):
        for dx in (-0.3, 0.0, 0.3):
            for dy in (-0.3, 0.0, 0.3):
                sample_points.append(
                    center + mathutils.Vector((dx * half_x, dy * half_y, dz * half_z * 2.0))
                )
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None
    origin = camera_obj.location.copy()
    total = len(sample_points)
    occluded = 0
    occluders: dict[str, int] = {}
    for point in sample_points:
        delta = point - origin
        distance = float(delta.length)
        if distance <= 1e-5:
            continue
        try:
            hit, _loc, _normal, _face, hit_obj, _mat = scene.ray_cast(
                depsgraph, origin, delta.normalized(), distance=max(distance - 0.02, 0.0)
            )
        except Exception:
            hit = False
            hit_obj = None
        if hit and hit_obj is not None:
            name = str(hit_obj.name)
            if not hit_is_focus_object(hit_obj, focus_names):
                occluded += 1
                occluders[name] = occluders.get(name, 0) + 1
    occlusion_ratio = round(float(occluded) / max(total, 1), 6)
    top_occluder = max(occluders, key=occluders.get) if occluders else None
    human_occluder = likely_human_occluder(top_occluder or "")
    return {
        "occlusion_ratio": occlusion_ratio,
        "sample_count": total,
        "occluded_count": occluded,
        "top_occluder": top_occluder,
        "occluder_counts": dict(sorted(occluders.items(), key=lambda kv: kv[1], reverse=True)[:5]),
        "human_occluder": human_occluder,
        "severely_occluded": occlusion_ratio > (0.10 if human_occluder else 0.35),
    }


_BOUNDARY_WALL_PREFIXES = ("wall_", "boundary_")
_BOUNDARY_WALL_KEYWORDS = ("room_wall", "scene_wall", "boundary")
_NON_BOUNDARY_WALL_KEYWORDS = ("retaining_wall", "stone_retaining", "retaining")
_MIN_BOUNDARY_WALL_SPAN = 3.0
_WALL_BOUNDS_CACHE: dict[str, tuple[float, float, float, float] | None] = {}


def boundary_wall_name(name: str) -> bool:
    text = str(name or "").lower()
    if any(token in text for token in _NON_BOUNDARY_WALL_KEYWORDS):
        return False
    return text == "wall" or text.startswith(_BOUNDARY_WALL_PREFIXES) or any(token in text for token in _BOUNDARY_WALL_KEYWORDS)


def wall_bounds_from_scene(blender_scene) -> tuple[float, float, float, float] | None:
    """Scan Blender scene for wall objects and return their combined XY AABB (meters)."""
    if blender_scene is None:
        return None
    cache_key = blender_scene.name
    if cache_key in _WALL_BOUNDS_CACHE:
        return _WALL_BOUNDS_CACHE[cache_key]
    wall_points: list[Any] = []
    wall_names: list[str] = []
    for obj in blender_scene.objects:
        if not boundary_wall_name(obj.name):
            continue
        if obj.type != "MESH":
            continue
        wall_names.append(obj.name)
        try:
            wall_points.extend(object_bounds(obj))
        except Exception:
            continue
    if not wall_points:
        _WALL_BOUNDS_CACHE[cache_key] = None
        return None
    minimum, maximum = points_bounds(wall_points)
    x_min, x_max = float(minimum.x), float(maximum.x)
    y_min, y_max = float(minimum.y), float(maximum.y)
    span = max(x_max - x_min, y_max - y_min)
    if span > 100.0:
        scale = 0.01
        x_min, x_max, y_min, y_max = x_min * scale, x_max * scale, y_min * scale, y_max * scale
        span = max(x_max - x_min, y_max - y_min)
    if span < _MIN_BOUNDARY_WALL_SPAN:
        print(f"[wall_bounds] scene={cache_key} ignored_small_wall_bounds walls={wall_names} span={span:.2f}", flush=True)
        _WALL_BOUNDS_CACHE[cache_key] = None
        return None
    result = (x_min, x_max, y_min, y_max)
    print(f"[wall_bounds] scene={cache_key} walls={wall_names} x=[{result[0]:.3f},{result[1]:.3f}] y=[{result[2]:.3f},{result[3]:.3f}] span={span:.2f}", flush=True)
    _WALL_BOUNDS_CACHE[cache_key] = result
    return result


def scene_bounds(details: dict[str, Any], blender_scene=None) -> tuple[float, float, float, float]:
    wb = wall_bounds_from_scene(blender_scene)
    if wb is not None:
        return wb
    size = details.get("scene_size") or {}
    return (
        float(size.get("x_negative", -10.0) or -10.0),
        float(size.get("x", 10.0) or 10.0),
        float(size.get("y_negative", -10.0) or -10.0),
        float(size.get("y", 10.0) or 10.0),
    )


def inside_scene_bounds(transform: dict[str, Any], details: dict[str, Any], blender_scene=None) -> bool:
    location = transform.get("location") or [0.0, 0.0, 0.0]
    x_min, x_max, y_min, y_max = scene_bounds(details, blender_scene)
    try:
        x = float(location[0])
        y = float(location[1])
    except Exception:
        return True
    return bool(x_min - 0.25 <= x <= x_max + 0.25 and y_min - 0.25 <= y <= y_max + 0.25)


def required_sight(camera: dict[str, Any]) -> float:
    contract = semantic_contract(camera)
    if contract["is_closeup"]:
        return 0.75
    label = str(camera.get("distance_label") or "").lower()
    if "wide" in label or "long" in label:
        return 0.25
    if "medium" in label or "full" in label:
        return 0.40
    return 0.50


def required_upper(camera: dict[str, Any]) -> float:
    return 0.75 if semantic_contract(camera)["is_closeup"] else 0.5


def required_visibility(camera: dict[str, Any], candidate: dict[str, Any]) -> float:
    if bool(candidate.get("dynamic_face_closeup")):
        return max(readability_threshold(camera, candidate), CLOSEUP_DYNAMIC_FACE_MIN_VISIBLE)
    return readability_threshold(camera, candidate)


def score_candidate(scene, camera_obj, candidate: dict[str, Any], camera: dict[str, Any], bundle: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    apply_transform(camera_obj, candidate)
    update_scene_view_layer(scene)
    projection_points = deserialize_points(candidate.get("projection_points")) or bundle["points"]
    upper_points = deserialize_points(candidate.get("upper_points")) or bundle["upper_points"] or projection_points
    is_semantic_closeup = semantic_closeup_source(candidate)
    projection = project_bounds(scene, camera_obj, projection_points)
    primary_projection = project_bounds(
        scene,
        camera_obj,
        projection_points if is_semantic_closeup else (bundle.get("primary_points") or projection_points),
    )
    group_projection = project_bounds(
        scene,
        camera_obj,
        projection_points if is_semantic_closeup else (bundle.get("group_points") or projection_points),
    )
    upper = point_visibility(scene, camera_obj, upper_points)
    sight = line_of_sight(scene, camera_obj, upper_points or projection_points, bundle["focus_names"])
    if is_semantic_closeup:
        los_clear_for_occlusion = float(sight.get("clear_ratio") or 0.0)
        occlusion = {
            "occlusion_ratio": round(float(max(0.0, 1.0 - los_clear_for_occlusion)), 6),
            "sample_count": int(sight.get("point_count") or 0),
            "occluded_count": int(max(0, int(sight.get("point_count") or 0) - int(sight.get("clear_count") or 0))),
            "top_occluder": None,
            "occluder_counts": {},
            "human_occluder": False,
            "severely_occluded": los_clear_for_occlusion < 0.45,
            "semantic_anchor_occlusion": True,
        }
    else:
        occlusion = occlusion_check(scene, camera_obj, bundle)
    area = float(projection.get("area_ratio") or 0.0)
    primary_area = float(primary_projection.get("area_ratio") or 0.0)
    primary_visible = float(primary_projection.get("visible_fraction") or 0.0)
    group_area = float(group_projection.get("area_ratio") or 0.0)
    occlusion_ratio = float(occlusion.get("occlusion_ratio") or 0.0)
    label = str(camera.get("distance_label") or start_frame_contract(camera).get("distance") or "").lower()
    if (
        not is_semantic_closeup
        and bool(occlusion.get("severely_occluded"))
        and ("wide" in label or "long" in label or "full" in label)
        and primary_visible >= 0.65
        and occlusion_ratio <= 0.30
    ):
        occlusion = dict(occlusion)
        occlusion["severely_occluded"] = False
        occlusion["relaxed_for_wide_visible_primary"] = True
    target_area = shot_profile(camera, bundle["extent"])["area_target"]
    area_for_score = primary_area if primary_area > 0.0 else area
    area_score = max(0.0, 1.0 - abs(area_for_score - target_area) / max(target_area, 0.02))
    center_score = max(0.0, 1.0 - float(projection.get("center_distance") or 1.0) * 1.8)
    raw_visible = float(projection.get("visible_fraction") or 0.0)
    visible = raw_visible
    upper_visible = float(upper.get("visible_ratio") or 0.0)
    los_clear = float(sight.get("clear_ratio") or 0.0)
    occlusion_penalty = min(occlusion_ratio * 1.5, 0.5)
    inside_bounds = inside_scene_bounds(candidate, details, blender_scene=scene)
    semantic_direction = semantic_direction_from_transform(candidate)
    direction_score = direction_alignment(camera, semantic_direction)
    if is_semantic_closeup:
        direction_score = max(direction_score, 0.5)
    height_ok = camera_height_ok(camera, candidate, bundle)
    primary_area_floor = required_primary_area(camera)
    group_area_floor = 0.0 if is_semantic_closeup else required_group_area(camera, bundle)
    final_score = max(
        0.0,
        min(
            1.0,
            0.16 * visible
            + 0.16 * primary_visible
            + 0.18 * upper_visible
            + 0.18 * los_clear
            + 0.14 * area_score
            + 0.06 * center_score
            + 0.08 * max(0.0, 1.0 - occlusion_ratio * 3.0)
            + 0.04 * direction_score
            - min(float(candidate.get("operation_depth") or 0) * 0.018, 0.12)
            - occlusion_penalty,
        ),
    )
    if not projection.get("valid"):
        final_score = min(final_score, 0.08)
    if visible <= 1e-6:
        final_score = min(final_score, 0.12)
    if raw_visible <= 1e-6 and not is_semantic_closeup:
        label = str(camera.get("distance_label") or "").lower()
        if "wide" in label or "long" in label:
            final_score = min(final_score, 0.20)
        else:
            final_score = min(final_score, 0.10)
    if primary_visible <= 1e-6:
        final_score = min(final_score, 0.08)
    if primary_area < primary_area_floor:
        final_score = min(final_score, 0.16)
    if group_area_floor > 0.0 and group_area < group_area_floor:
        final_score = min(final_score, 0.16)
    if upper.get("required") and upper_visible < required_upper(camera):
        final_score = min(final_score, 0.18)
    if sight.get("required") and los_clear < required_sight(camera):
        deficit = required_sight(camera) - los_clear
        final_score = max(final_score - deficit * 0.30, 0.05)
    if not inside_bounds:
        final_score = min(final_score, 0.05)
    if occlusion.get("severely_occluded"):
        final_score = min(final_score, 0.10)
    if direction_score <= 0.0 and not is_semantic_closeup:
        final_score = min(final_score, 0.05)
    if not height_ok:
        final_score = min(final_score, 0.08)
    row = dict(candidate)
    row["semantic_direction"] = semantic_direction
    row["direction"] = semantic_direction
    row["internal_direction"] = semantic_direction
    row["scores"] = {
        "final": round(float(final_score), 6),
        "visibility": round(visible, 6),
        "primary_visible_fraction": round(primary_visible, 6),
        "upper_body_visibility": round(upper_visible, 6),
        "line_of_sight": round(los_clear, 6),
        "occlusion": round(1.0 - occlusion_ratio, 6),
        "human_occluder": 1.0 if bool(occlusion.get("human_occluder")) else 0.0,
        "area": round(area_score, 6),
        "primary_area_ratio": round(primary_area, 6),
        "group_subject_area_ratio": round(group_area, 6),
        "required_primary_area_floor": round(primary_area_floor, 6),
        "required_group_area_floor": round(group_area_floor, 6),
        "direction_alignment": round(direction_score, 6),
        "camera_height_ok": 1.0 if height_ok else 0.0,
        "center": round(center_score, 6),
        "inside_scene_bounds": 1.0 if inside_bounds else 0.0,
        "raw_visibility": round(raw_visible, 6),
    }
    row["projection"] = projection
    row["primary_projection"] = primary_projection
    row["group_projection"] = group_projection
    row["upper_body_visibility"] = upper
    row["line_of_sight"] = sight
    row["occlusion_check"] = occlusion
    row["wall_check"] = {"inside_scene_bounds": inside_bounds}
    row["readability_threshold"] = round(readability_threshold(camera, row), 4)
    row["readability_grade"] = readability_grade(camera, row)
    return row


def candidate_eligible(candidate: dict[str, Any], camera: dict[str, Any]) -> bool:
    scores = candidate.get("scores") or {}
    occlusion = candidate.get("occlusion_check") or {}
    if candidate_projection_visible(candidate) <= 0.0:
        return False
    if candidate_readability_reasons(candidate, camera):
        return False
    is_semantic_closeup = semantic_closeup_source(candidate)
    upper_threshold = 0.15 if is_semantic_closeup else required_upper(camera)
    visible_threshold = required_visibility(camera, candidate)
    return (
        float(scores.get("final") or 0.0) >= 0.18
        and float(scores.get("visibility") or 0.0) >= visible_threshold
        and float(scores.get("primary_visible_fraction") or 0.0) >= visible_threshold
        and float(scores.get("primary_area_ratio") or 0.0) >= required_primary_area(camera)
        and (
            float(scores.get("required_group_area_floor") or 0.0) <= 0.0
            or float(scores.get("group_subject_area_ratio") or 0.0) >= float(scores.get("required_group_area_floor") or 0.0)
        )
        and (is_semantic_closeup or float(scores.get("direction_alignment") or 0.0) > 0.0)
        and float(scores.get("camera_height_ok") or 0.0) >= 0.5
        and float(scores.get("upper_body_visibility") or 0.0) >= upper_threshold
        and float(scores.get("line_of_sight") or 0.0) >= required_sight(camera)
        and bool((candidate.get("wall_check") or {}).get("inside_scene_bounds", True))
        and not bool(occlusion.get("severely_occluded"))
    )


def candidate_fallback_acceptable(candidate: dict[str, Any], camera: dict[str, Any]) -> bool:
    scores = candidate.get("scores") or {}
    projection = candidate.get("projection") or {}
    occlusion = candidate.get("occlusion_check") or {}
    if not bool(projection.get("valid", True)):
        return False
    min_visible = max(min(readability_threshold(camera, candidate) * 0.5, 0.04), 0.01)
    if candidate_projection_visible(candidate) < min_visible:
        return False
    if float(scores.get("primary_visible_fraction") or 0.0) < min_visible:
        return False
    min_area = max(min(required_primary_area(camera) * 0.4, 0.05), 0.01)
    if float(scores.get("primary_area_ratio") or 0.0) < min_area:
        return False
    if float(scores.get("line_of_sight") or 0.0) < max(required_sight(camera) * 0.45, 0.25):
        return False
    if bool(occlusion.get("severely_occluded")):
        return False
    if preview_qc_failed(candidate):
        return False
    # 坚决拦截撞墙或出界的机位
    if not bool((candidate.get("wall_check") or {}).get("inside_scene_bounds", True)):
        return False
    # 我们不再强制方向对齐和严重遮挡，只要能看见人、没撞墙，就允许进入兜底池
    return True


def fallback_candidate_pool(candidates: list[dict[str, Any]], camera: dict[str, Any]) -> list[dict[str, Any]]:
    source = [row for row in candidates if candidate_fallback_acceptable(row, camera)]
    source.sort(key=lambda item: candidate_rank_score(item, camera), reverse=True)
    source = deduplicate_channel_candidates(source)
    for row in source:
        row["fallback_unqualified"] = True
    return source


def channel_quota(camera: dict[str, Any], top_k: int) -> dict[str, int]:
    top_k = max(int(top_k), 1)
    if semantic_priority(camera):
        # Fix G2: when the shot is a semantic-priority (close-up) shot,
        # give the semantic channel roughly half of top_k. Previously the
        # quota was ``max(3, top_k // 3)`` which capped semantic at 6 of
        # 20 slots and squeezed out otherwise-good board candidates in
        # favour of less-relevant direction/preset shots.
        semantic_slots = min(max(top_k // 2, 8), top_k)
        remaining = max(top_k - semantic_slots, 0)
        direction_slots = min(max(1, remaining // 2), remaining)
        preset_slots = max(remaining - direction_slots, 0)
        return {"semantic": semantic_slots, "direction": direction_slots, "preset": preset_slots}
    direction_slots = min(max(4, top_k // 2), top_k)
    preset_slots = min(max(2, (top_k - direction_slots) // 2), max(top_k - direction_slots, 0))
    semantic_slots = max(top_k - direction_slots - preset_slots, 0)
    return {"semantic": semantic_slots, "direction": direction_slots, "preset": preset_slots}


def selection_reason(camera: dict[str, Any], selected: dict[str, Any], eligible_count: int) -> str:
    if not selected:
        return "no_candidate_retained"
    channel = str(selected.get("channel") or "unknown")
    source = str(selected.get("source") or "unknown")
    if semantic_priority(camera) and channel == "semantic":
        return f"deterministic_fallback_closeup_semantic_weighted:{source}"
    if bool(selected.get("fallback_unqualified")):
        return f"fallback_selected_from_unqualified_pool:{source}"
    if eligible_count <= 0:
        return f"selection_invariant_violation_no_eligible:{source}"
    return f"score_selected:{source}"


def candidate_rank_score(candidate: dict[str, Any], camera: dict[str, Any]) -> float:
    scores = candidate.get("scores") or {}
    occlusion = candidate.get("occlusion_check") or {}
    score = float(scores.get("final") or 0.0)
    score += 0.04 * float(scores.get("upper_body_visibility") or 0.0)
    score += 0.04 * float(scores.get("line_of_sight") or 0.0)
    score += 0.03 * float(scores.get("occlusion") or 0.0)
    score += 0.02 * float(scores.get("inside_scene_bounds") or 0.0)
    if bool(occlusion.get("severely_occluded")):
        score -= 0.35
    if bool(occlusion.get("human_occluder")):
        score -= 0.10
    readability_reasons = candidate_readability_reasons(candidate, camera)
    if readability_reasons:
        score -= 0.45 + 0.05 * len(readability_reasons)
    elif candidate_readable_bonus(candidate, camera):
        score += 0.08
    if semantic_priority(camera):
        channel = str(candidate.get("channel") or "")
        if channel == "semantic":
            score += 0.35
        if semantic_closeup_source(candidate) and int(candidate.get("operation_depth") or 0) == 0:
            score += 0.25
        source = str(candidate.get("source") or "")
        if source.startswith("semantic_face"):
            is_mirror = bool(candidate.get("frontback_mirror_candidate"))
            actual_direction = candidate_face_direction(candidate)
            if actual_direction in BACK_DIRECTIONS and not face_back_view_allowed(camera):
                score -= 0.32
            else:
                alignment = direction_alignment(camera, actual_direction)
                score += 0.04 * alignment
                if is_mirror and actual_direction not in BACK_DIRECTIONS:
                    score += 0.04
            if is_dynamic_face_closeup(camera):
                if source.startswith("semantic_face_s3_extreme_macro_dynamic"):
                    score += 0.18
                elif source.startswith("semantic_face_s3_extreme_macro"):
                    score -= 0.04
        if str(candidate.get("source") or "") == "operation_expansion":
            score -= min(int(candidate.get("operation_depth") or 0) * 0.03, 0.12)
    return score


def select_candidates(candidates: list[dict[str, Any]], camera: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], int]:
    eligible = [row for row in candidates if candidate_eligible(row, camera)]
    source = eligible
    if not source:
        source = fallback_candidate_pool(candidates, camera)
    quota = channel_quota(camera, top_k)
    by_channel: dict[str, list[dict[str, Any]]] = {"semantic": [], "direction": [], "preset": []}
    for row in source:
        by_channel.setdefault(str(row.get("channel") or "direction"), []).append(row)
    for rows in by_channel.values():
        rows.sort(key=lambda item: candidate_rank_score(item, camera), reverse=True)
    selected: list[dict[str, Any]] = []
    seen = set()
    if semantic_priority(camera):
        order = ("semantic", "direction", "preset")
    else:
        order = ("direction", "preset", "semantic")
    for channel_name in order:
        limit = int(quota.get(channel_name) or 0)
        for row in by_channel.get(channel_name) or []:
            if len(selected) >= top_k or limit <= 0:
                break
            if id(row) in seen:
                continue
            selected.append(row)
            seen.add(id(row))
            limit -= 1
    remaining = sorted(source, key=lambda item: candidate_rank_score(item, camera), reverse=True)
    for row in remaining:
        if len(selected) >= top_k:
            break
        if id(row) in seen:
            continue
        selected.append(row)
        seen.add(id(row))
    return selected, len(eligible)


def render_preview(scene, camera_obj, candidate: dict[str, Any], path: Path, args: argparse.Namespace) -> str:
    apply_transform(camera_obj, candidate)
    update_scene_view_layer(scene)
    original_camera = scene.camera
    original_filepath = str(scene.render.filepath)
    original_x = int(scene.render.resolution_x)
    original_y = int(scene.render.resolution_y)
    original_engine = str(scene.render.engine)
    try:
        scene.camera = camera_obj
        scene.render.filepath = str(path)
        scene.render.resolution_x = int(args.resolution_x)
        scene.render.resolution_y = int(args.resolution_y)
        try:
            scene.render.engine = str(args.render_engine)
        except Exception:
            pass
        if hasattr(scene, "eevee"):
            try:
                scene.eevee.taa_render_samples = max(1, int(args.render_samples))
            except Exception:
                pass
        bpy.ops.render.render(write_still=True, scene=scene.name)
    finally:
        scene.camera = original_camera
        scene.render.filepath = original_filepath
        scene.render.resolution_x = original_x
        scene.render.resolution_y = original_y
        try:
            scene.render.engine = original_engine
        except Exception:
            pass
    return str(path)


def build_board(candidates: list[dict[str, Any]], board_path: Path) -> str:
    """Render a tile-grid board image of the per-channel candidate previews.

    Fix G1: the board PNG is intentionally **text-free** so the downstream
    LLM must judge purely from the rendered images. Candidate identifiers
    and scores are still passed to the LLM via the per-channel JSON payload
    (see ``llm_filter_channel_board``); the LLM resolves spatial position
    on the board to candidate_id by row-major order matching the JSON list.
    """
    if Image is None or not candidates:
        return ""
    thumb_w, thumb_h, padding = 240, 135, 12
    columns = 4 if len(candidates) > 6 else 3
    rows = int(math.ceil(len(candidates) / columns))
    board = Image.new(
        "RGB",
        (columns * thumb_w + (columns + 1) * padding, rows * thumb_h + (rows + 1) * padding),
        (244, 242, 238),
    )
    for index, candidate in enumerate(candidates):
        row = index // columns
        col = index % columns
        left = padding + col * (thumb_w + padding)
        top = padding + row * (thumb_h + padding)
        path = Path(str(candidate.get("preview_image_path") or ""))
        if path.exists():
            image = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            board.paste(image, (left, top))
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(board_path)
    return str(board_path)


def rejection_reason_for_candidate(candidate: dict[str, Any], camera: dict[str, Any]) -> list[str]:
    """Return a list of human-readable reasons why this candidate would be rejected."""
    reasons: list[str] = []
    scores = candidate.get("scores") or {}
    occlusion = candidate.get("occlusion_check") or {}
    projection = candidate.get("projection") or {}
    wall = candidate.get("wall_check") or {}
    final = float(scores.get("final") or 0.0)
    vis = float(scores.get("visibility") or 0.0)
    upper = float(scores.get("upper_body_visibility") or 0.0)
    los = float(scores.get("line_of_sight") or 0.0)
    occ_ratio = float(occlusion.get("occlusion_ratio") or 0.0)
    req_upper = required_upper(camera)
    req_sight = required_sight(camera)
    req_visibility = required_visibility(camera, candidate)
    if final < 0.18:
        reasons.append(f"final_score={final:.3f} < threshold=0.18")
    if vis < req_visibility:
        reasons.append(f"visibility={vis:.3f} < threshold={req_visibility:.2f}")
    if upper < req_upper:
        reasons.append(f"upper_body_visibility={upper:.3f} < threshold={req_upper:.2f}")
    if los < req_sight:
        reasons.append(f"line_of_sight={los:.3f} < threshold={req_sight:.2f}")
    if not projection.get("valid"):
        reasons.append("subject_behind_camera (projection invalid)")
    if float(projection.get("visible_fraction") or 0.0) < req_visibility:
        reasons.append(f"visible_fraction={float(projection.get('visible_fraction') or 0.0):.3f} < {req_visibility:.2f}")
    if bool(occlusion.get("severely_occluded")):
        top_occ = occlusion.get("top_occluder") or "unknown"
        reasons.append(f"severely_occluded: ratio={occ_ratio:.3f}, top_occluder={top_occ}")
    if not wall.get("inside_scene_bounds", True):
        reasons.append("camera_outside_scene_bounds")
    if candidate.get("preview_error"):
        reasons.append(f"preview_render_failed:{candidate.get('preview_error')}")
    if candidate.get("preview_qc_status") and candidate.get("preview_qc_status") != "passed":
        reasons.append(str(candidate.get("preview_qc_status")))
    reasons.extend(candidate_readability_reasons(candidate, camera))
    return reasons


def build_rejection_report(candidates: list[dict[str, Any]], retained_ids: set[str], camera: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a report listing every candidate and its rejection reasons (empty for retained ones)."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        cid = str(candidate.get("candidate_id") or "")
        is_retained = cid in retained_ids
        reasons = rejection_reason_for_candidate(candidate, camera) if not is_retained else []
        scores = candidate.get("scores") or {}
        rows.append({
            "candidate_id": cid,
            "channel": candidate.get("channel") or "unknown",
            "source": candidate.get("source") or "unknown",
            "direction": candidate.get("direction") or "unknown",
            "semantic_direction": candidate.get("semantic_direction") or candidate.get("direction") or "unknown",
            "operation_depth": candidate.get("operation_depth") or 0,
            "final_score": round(float(scores.get("final") or 0.0), 4),
            "primary_area_ratio": round(float(scores.get("primary_area_ratio") or 0.0), 4),
            "primary_visible_fraction": round(float(scores.get("primary_visible_fraction") or 0.0), 4),
            "direction_alignment": round(float(scores.get("direction_alignment") or 0.0), 4),
            "camera_height_ok": bool(float(scores.get("camera_height_ok") or 0.0) >= 0.5),
            "retained": is_retained,
            "rejection_reasons": reasons,
        })
    rows.sort(key=lambda r: r["final_score"], reverse=True)
    return rows


VALIDATION_CHANNELS = {"direction", "preset"}


def validation_paths(output_root: Path) -> dict[str, Path]:
    outputs_dir = output_root / "outputs"
    return {
        "manifest": outputs_dir / "candidate_validation_manifest_v1.jsonl",
        "blind_input": outputs_dir / "candidate_blind_review_input_v1.jsonl",
        "blind_result": outputs_dir / "candidate_blind_review_result_v1.jsonl",
        "blind_schema": outputs_dir / "candidate_blind_review_schema_v1.json",
        "confusion_report": outputs_dir / "candidate_filter_confusion_report_v1.json",
    }


def prepare_candidate_validation_outputs(output_root: Path) -> None:
    paths = validation_paths(output_root)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text("", encoding="utf-8")
    paths["blind_input"].write_text("", encoding="utf-8")
    if not paths["blind_result"].exists():
        paths["blind_result"].write_text("", encoding="utf-8")
    save_json(
        {
            "schema_version": "storyblender.candidate_blind_review_schema.v1",
            "join_key": "validation_id",
            "usable": ["good", "borderline", "bad"],
            "primary_visible": ["true", "false"],
            "secondary_visible": ["true", "false", "not_required"],
            "semantic_satisfied": ["true", "false"],
            "framing_matches_intent": ["true", "false"],
            "direction_matches": ["true", "false"],
            "failure_reason": [
                "occlusion",
                "outside_scene_bounds",
                "wrong_subject",
                "too_far",
                "too_close",
                "wrong_direction",
                "semantic_mismatch",
                "other",
            ],
            "notes": "Review candidate_blind_review_input_v1.jsonl rows without using algorithm scores or rejection reasons.",
        },
        paths["blind_schema"],
    )
    save_json(
        {
            "schema_version": "storyblender.candidate_filter_confusion_report.v1",
            "status": "pending_review_results",
            "manifest_path": str(paths["manifest"]),
            "blind_review_input_path": str(paths["blind_input"]),
            "blind_review_result_path": str(paths["blind_result"]),
        },
        paths["confusion_report"],
    )


def candidate_score(candidate: dict[str, Any]) -> float:
    return float((candidate.get("scores") or {}).get("final") or 0.0)


def validation_channel(candidate: dict[str, Any]) -> str:
    return str(candidate.get("channel") or "direction").strip().lower()


def validation_reason_category(reason: str) -> str:
    text = str(reason or "")
    if text.startswith("severely_occluded"):
        return "severe_occlusion"
    if text.startswith("line_of_sight="):
        return "line_of_sight_low"
    if text == "camera_outside_scene_bounds":
        return "outside_scene_bounds"
    if text.startswith("primary_area="):
        return "primary_area_low"
    if text.startswith("direction_mismatch"):
        return "direction_mismatch"
    if text.startswith("upper_body_visibility="):
        return "upper_body_visibility_low"
    if text.startswith("visibility=") or text.startswith("visible_fraction="):
        return "visibility_low"
    if text.startswith("preview_render_failed") or text.startswith("preview_") or text == "failed_post_render_candidate_gate":
        return "preview_qc_failed"
    return "other"


def validation_context(camera: dict[str, Any]) -> dict[str, Any]:
    contract = start_frame_contract(camera)
    sem = semantic_contract(camera)
    roles = _focus_role_ids(camera)
    primary_ids = roles.get("primary_ids") or []
    primary_focus = str(primary_ids[0] if primary_ids else "").strip()
    secondary_ids = _as_string_list(roles.get("secondary_ids")) or _as_string_list(contract.get("secondary_focus_ids")) or _as_string_list(camera.get("secondary_focus_ids"))
    return {
        "scene_id": int(camera.get("scene_id") or 0),
        "shot_id": int(camera.get("shot_id") or 0),
        "camera_name": str(camera.get("camera_name") or ""),
        "shot_description": camera.get("shot_description") or "",
        "scene_description": camera.get("scene_description") or "",
        "primary_focus_id": primary_focus,
        "secondary_focus_ids": secondary_ids,
        "primary_semantic_target": sem.get("primary_semantic_target") or contract.get("primary_semantic_target") or "",
        "distance_label": sem.get("distance_label") or camera.get("distance_label") or "",
        "angle_label": camera.get("angle_label") or "",
        "movement_tag": camera.get("movement_tag") or "",
        "semantic_contract": sem,
    }


def add_validation_sample(
    samples: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    sample_reason: str,
    reason_category: str = "",
) -> None:
    cid = str(candidate.get("candidate_id") or "")
    if not cid:
        return
    entry = samples.setdefault(
        cid,
        {
            "candidate": candidate,
            "sample_reasons": [],
            "sample_reason_categories": [],
        },
    )
    if sample_reason and sample_reason not in entry["sample_reasons"]:
        entry["sample_reasons"].append(sample_reason)
    if reason_category and reason_category not in entry["sample_reason_categories"]:
        entry["sample_reason_categories"].append(reason_category)


def sample_validation_candidates(
    candidates: list[dict[str, Any]],
    retained_ids: set[str],
    camera: dict[str, Any],
    quality_success: bool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    validation_candidates = [
        candidate
        for candidate in candidates
        if validation_channel(candidate) in VALIDATION_CHANNELS
    ]
    rejected = [
        candidate
        for candidate in validation_candidates
        if str(candidate.get("candidate_id") or "") not in retained_ids
    ]
    retained = [
        candidate
        for candidate in validation_candidates
        if str(candidate.get("candidate_id") or "") in retained_ids
    ]
    reason_cache: dict[str, list[str]] = {}
    category_cache: dict[str, list[str]] = {}

    def reasons_for(candidate: dict[str, Any]) -> list[str]:
        cid = str(candidate.get("candidate_id") or "")
        if cid not in reason_cache:
            reason_cache[cid] = rejection_reason_for_candidate(candidate, camera)
            categories: list[str] = []
            for reason in reason_cache[cid]:
                category = validation_reason_category(reason)
                if category not in categories:
                    categories.append(category)
            category_cache[cid] = categories
        return reason_cache[cid]

    def categories_for(candidate: dict[str, Any]) -> list[str]:
        reasons_for(candidate)
        return category_cache.get(str(candidate.get("candidate_id") or ""), [])

    if not quality_success:
        per_channel_limit = max(int(args.validation_failed_top_per_channel), 0)
        for channel_name in ("direction", "preset"):
            channel_rejected = [candidate for candidate in rejected if validation_channel(candidate) == channel_name]
            channel_rejected.sort(key=candidate_score, reverse=True)
            for candidate in channel_rejected[:per_channel_limit]:
                add_validation_sample(samples, candidate, f"failed_{channel_name}_top_rejected")
        for category in ("severe_occlusion", "line_of_sight_low", "outside_scene_bounds", "primary_area_low", "direction_mismatch"):
            category_rows = [candidate for candidate in rejected if category in categories_for(candidate)]
            category_rows.sort(key=candidate_score, reverse=True)
            for candidate in category_rows[: max(int(args.validation_reason_samples), 0)]:
                add_validation_sample(samples, candidate, "failed_rejection_reason", category)
        if not retained:
            threshold = float(args.validation_score_threshold)
            for channel_name in ("direction", "preset"):
                channel_rejected = [candidate for candidate in rejected if validation_channel(candidate) == channel_name]
                channel_rejected.sort(key=lambda item: (abs(candidate_score(item) - threshold), -candidate_score(item)))
                for candidate in channel_rejected[:per_channel_limit]:
                    add_validation_sample(samples, candidate, "retained_zero_near_threshold")
    else:
        for candidate in retained:
            add_validation_sample(samples, candidate, "retained_candidate")
        threshold = float(args.validation_score_threshold)
        near_limit = max(int(args.validation_success_near_threshold_per_channel), 0)
        for channel_name in ("direction", "preset"):
            channel_rejected = [candidate for candidate in rejected if validation_channel(candidate) == channel_name]
            channel_rejected.sort(key=lambda item: (abs(candidate_score(item) - threshold), -candidate_score(item)))
            for candidate in channel_rejected[:near_limit]:
                add_validation_sample(samples, candidate, f"success_{channel_name}_near_threshold_rejected")

    rows = list(samples.values())
    rows.sort(key=lambda entry: (-candidate_score(entry["candidate"]), str(entry["candidate"].get("candidate_id") or "")))
    return rows


def existing_preview_path(candidate: dict[str, Any]) -> Path | None:
    for key in ("validation_preview_image_path", "preview_image_path"):
        text = str(candidate.get(key) or "").strip()
        if not text:
            continue
        path = Path(text)
        if path.exists():
            return path
    return None


def ensure_validation_preview(
    scene,
    camera_obj,
    candidate: dict[str, Any],
    validation_preview_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, str]:
    existing = existing_preview_path(candidate)
    if existing is not None:
        return str(existing), ""
    cid = str(candidate.get("candidate_id") or "candidate")
    preview_path = validation_preview_dir / f"{cid}.png"
    if preview_path.exists():
        candidate["validation_preview_image_path"] = str(preview_path)
        return str(preview_path), ""
    try:
        candidate["validation_preview_image_path"] = render_preview(scene, camera_obj, candidate, preview_path, args)
        return str(candidate["validation_preview_image_path"]), ""
    except Exception as exc:
        candidate["validation_preview_error"] = str(exc)
        return "", str(exc)


def candidate_validation_rows(
    *,
    scene,
    camera_obj,
    camera: dict[str, Any],
    candidates: list[dict[str, Any]],
    retained_ids: set[str],
    quality_success: bool,
    candidate_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = sample_validation_candidates(candidates, retained_ids, camera, quality_success, args)
    context = validation_context(camera)
    validation_preview_dir = candidate_dir / "validation_previews"
    validation_preview_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    for entry in samples:
        candidate = entry["candidate"]
        cid = str(candidate.get("candidate_id") or "")
        preview_path, preview_error = ensure_validation_preview(scene, camera_obj, candidate, validation_preview_dir, args)
        if preview_error or not preview_path:
            continue
        scores = candidate.get("scores") or {}
        retained = cid in retained_ids
        rejection_reasons = [] if retained else rejection_reason_for_candidate(candidate, camera)
        rejection_categories: list[str] = []
        for reason in rejection_reasons:
            category = validation_reason_category(reason)
            if category not in rejection_categories:
                rejection_categories.append(category)
        preview_qc = candidate.get("preview_image_qc") or {}
        if not preview_qc:
            qc_path = Path(preview_path)
            if qc_path.exists():
                preview_qc = preview_image_qc(qc_path)
        validation_id = f"{context['camera_name']}:{cid}"
        blind_row = {
            "schema_version": "storyblender.candidate_blind_review_input.v1",
            "validation_id": validation_id,
            "candidate_image_path": preview_path,
            "scene_id": context["scene_id"],
            "shot_id": context["shot_id"],
            "camera_name": context["camera_name"],
            "candidate_id": cid,
            "shot_description": context["shot_description"],
            "scene_description": context["scene_description"],
            "primary_focus_id": context["primary_focus_id"],
            "secondary_focus_ids": context["secondary_focus_ids"],
            "primary_semantic_target": context["primary_semantic_target"],
            "distance_label": context["distance_label"],
            "angle_label": context["angle_label"],
            "movement_tag": context["movement_tag"],
            "semantic_contract": context["semantic_contract"],
            "semantic_direction": candidate.get("semantic_direction") or candidate.get("direction") or "",
        }
        manifest_row = {
            **blind_row,
            "schema_version": "storyblender.candidate_validation_manifest.v1",
            "channel": validation_channel(candidate),
            "source": candidate.get("source") or "",
            "operation_depth": int(candidate.get("operation_depth") or 0),
            "algorithm_decision": "retained" if retained else "rejected",
            "retained": retained,
            "sample_reasons": entry.get("sample_reasons") or [],
            "sample_reason_categories": entry.get("sample_reason_categories") or [],
            "final_score": round(float(scores.get("final") or 0.0), 6),
            "primary_area_ratio": round(float(scores.get("primary_area_ratio") or 0.0), 6),
            "primary_visible_fraction": round(float(scores.get("primary_visible_fraction") or 0.0), 6),
            "direction_alignment": round(float(scores.get("direction_alignment") or 0.0), 6),
            "rejection_reasons": rejection_reasons,
            "rejection_categories": rejection_categories,
            "preview_qc_status": candidate.get("preview_qc_status") or preview_qc.get("status") or "",
            "preview_image_qc": preview_qc,
            "location": list(candidate.get("location") or []),
            "rotation_euler": list(candidate.get("rotation_euler") or []),
            "target": list(candidate.get("target") or []),
            "lens_mm": float(candidate.get("lens_mm") or 0.0),
        }
        manifest_rows.append(manifest_row)
        blind_rows.append(blind_row)
    return manifest_rows, blind_rows


def write_candidate_validation_outputs(output_root: Path, manifest_rows: list[dict[str, Any]], blind_rows: list[dict[str, Any]]) -> None:
    paths = validation_paths(output_root)
    append_jsonl(paths["manifest"], manifest_rows)
    append_jsonl(paths["blind_input"], blind_rows)


def movement_end_transform(camera: dict[str, Any], start: dict[str, Any], focus_center: Any) -> dict[str, Any]:
    location = mathutils.Vector(tuple(float(value) for value in start.get("location", (0.0, -3.0, 1.6))))
    movement = str(camera.get("movement_tag") or "").lower()
    direction = focus_center - location
    if direction.length <= 1e-6:
        direction = mathutils.Vector((0.0, 1.0, 0.0))
    direction.normalize()
    distance = max((focus_center - location).length, 1.0)
    if movement == "push_in":
        location += direction * max(distance * 0.035, 0.06)
    elif movement == "push_out":
        location -= direction * max(distance * 0.035, 0.06)
    elif movement in {"truck", "pan"}:
        right = direction.cross(mathutils.Vector((0.0, 0.0, 1.0)))
        if right.length <= 1e-6:
            right = mathutils.Vector((1.0, 0.0, 0.0))
        right.normalize()
        location += right * max(distance * 0.04, 0.08)
    elif movement == "orbit":
        location = focus_center + mathutils.Matrix.Rotation(math.radians(4.0), 4, "Z") @ (location - focus_center)
    end = dict(start)
    end["location"] = round_vector(location)
    end["rotation_euler"] = round_vector(look_at(location, focus_center))
    return end


def semantic_transforms(camera: dict[str, Any], bundle: dict[str, Any], center: Any, extent: Any, front: Any, right: Any) -> list[dict[str, Any]]:
    """Generate semantic seed candidates for close-ups and detail inserts only."""
    seeds: list[dict[str, Any]] = []
    seeds.extend(semantic_face_seeds(camera, bundle, center, extent, front, right))
    seeds.extend(semantic_feet_seeds(camera, bundle, center, extent, front, right))
    return seeds


def generate_candidates(scene, camera_obj, camera: dict[str, Any], bundle: dict[str, Any], details: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    basis = probe_basis(scene, camera_obj, camera, bundle, details)
    front = basis["front"]
    right = basis["right"]
    source_basis = basis["basis_source"]
    basis_front_vector = basis["basis_front_vector"]
    basis_confidence = basis["basis_confidence"]
    center = basis["center"]
    extent = basis["extent"]
    center_source = basis["center_source"]
    direction_seeds = [
        base_transform(camera, center, extent, front, right, direction)
        for direction in DIRECTIONS
    ]
    for seed in direction_seeds:
        seed["channel"] = "direction"
        seed["basis_source"] = source_basis
        seed["basis_front_vector"] = basis_front_vector
        seed["basis_confidence"] = basis_confidence
        seed["center_source"] = center_source
        seed["readability_threshold"] = 0.02
        seed.update(anchor_metadata("body_anchor", center, extent))
    preset_seeds = preset_transforms(camera, center, extent, front, right)
    for seed in preset_seeds:
        seed["channel"] = "preset"
        seed["basis_source"] = source_basis
        seed["basis_front_vector"] = basis_front_vector
        seed["basis_confidence"] = basis_confidence
        seed["center_source"] = center_source
        seed["readability_threshold"] = 0.02
        seed.update(anchor_metadata("body_anchor", center, extent))
    sem_seeds = semantic_transforms(camera, bundle, center, extent, front, right)
    for seed in sem_seeds:
        seed["channel"] = "semantic"
        seed["basis_source"] = source_basis
        seed["basis_front_vector"] = basis_front_vector
        seed["basis_confidence"] = basis_confidence
        seed["center_source"] = center_source
    candidates = []
    seen = set()
    for seed in direction_seeds + preset_seeds + sem_seeds:
        key = transform_key(seed)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(score_candidate(scene, camera_obj, seed, camera, bundle, details))
    channels = ("semantic", "direction", "preset")
    for depth in range(1, max(int(args.max_depth), 0) + 1):
        for channel_name in channels:
            if channel_name == "semantic":
                continue
            parents = [
                row for row in candidates
                if int(row.get("operation_depth") or 0) == depth - 1
                and str(row.get("channel") or "direction") == channel_name
            ]
            parents.sort(key=lambda row: float((row.get("scores") or {}).get("final") or 0.0), reverse=True)
            base_limit = max(int(args.frontier_limit), 1)
            if channel_name == "semantic":
                limit = max(4, base_limit // 2)
            else:
                limit = max(3, base_limit // 3)
            for parent in parents[:limit]:
                for operation in OPERATIONS:
                    candidate = apply_operation(parent, operation)
                    candidate["channel"] = parent.get("channel") or "direction"
                    candidate["basis_source"] = parent.get("basis_source") or source_basis
                    candidate["basis_front_vector"] = parent.get("basis_front_vector") or basis_front_vector
                    candidate["basis_confidence"] = parent.get("basis_confidence") or basis_confidence
                    candidate["center_source"] = parent.get("center_source") or center_source
                    candidate["anchor_kind"] = parent.get("anchor_kind") or "body_anchor"
                    candidate["anchor_point"] = parent.get("anchor_point") or round_vector(center)
                    if parent.get("projection_points"):
                        candidate["projection_points"] = list(parent.get("projection_points") or [])
                    if parent.get("upper_points"):
                        candidate["upper_points"] = list(parent.get("upper_points") or [])
                    candidate["readability_threshold"] = float(parent.get("readability_threshold") or 0.02)
                    key = transform_key(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(score_candidate(scene, camera_obj, candidate, camera, bundle, details))
    for index, candidate in enumerate(candidates, start=1):
        semantic_direction = normalize_direction(candidate.get("semantic_direction") or candidate.get("direction"))
        candidate["candidate_id"] = f"{camera['camera_name']}_{semantic_direction}_d{candidate.get('operation_depth', 0)}_{index:04d}"
    return candidates


def split_by_channel(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    channels: dict[str, list[dict[str, Any]]] = {"direction": [], "preset": [], "semantic": []}
    for candidate in candidates:
        channel = str(candidate.get("channel") or "direction")
        channels.setdefault(channel, []).append(candidate)
    return channels


def _same_direction_similar(a: dict[str, Any], b: dict[str, Any], pos_threshold: float = 0.15, lens_threshold: float = 8.0) -> bool:
    """True if a and b share the same direction AND are spatially/lens similar."""
    if str(a.get("direction") or "") != str(b.get("direction") or ""):
        return False
    loc_a = a.get("location") or [0.0, 0.0, 0.0]
    loc_b = b.get("location") or [0.0, 0.0, 0.0]
    dist = math.sqrt(sum((float(loc_a[i]) - float(loc_b[i])) ** 2 for i in range(3)))
    if dist > pos_threshold:
        return False
    if abs(float(a.get("lens_mm") or 35.0) - float(b.get("lens_mm") or 35.0)) > lens_threshold:
        return False
    return True


def deduplicate_channel_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove same-direction near-duplicates within a channel. Input must be pre-sorted by score desc."""
    retained: list[dict[str, Any]] = []
    for candidate in candidates:
        if not any(_same_direction_similar(candidate, existing) for existing in retained):
            retained.append(candidate)
    return retained


def select_channel_board(candidates: list[dict[str, Any]], camera: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], int]:
    eligible = [row for row in candidates if candidate_eligible(row, camera)]
    source = eligible
    if not source:
        source = fallback_candidate_pool(candidates, camera)
    source.sort(key=lambda item: candidate_rank_score(item, camera), reverse=True)
    source = deduplicate_channel_candidates(source)
    selected = source[:top_k]
    return selected, len(eligible)


def preview_image_qc(path: Path) -> dict[str, Any]:
    if Image is None:
        return {"status": "passed", "reason": "pil_unavailable"}
    try:
        image = Image.open(path).convert("L").resize((64, 36), Image.Resampling.BILINEAR)
        pixels = [float(value) for value in image.getdata()]
    except Exception as exc:
        return {"status": "failed", "reason": f"preview_qc_error:{exc}"}
    if not pixels:
        return {"status": "failed", "reason": "preview_empty_pixels"}
    mean = sum(pixels) / float(len(pixels))
    variance = sum((value - mean) ** 2 for value in pixels) / float(len(pixels))
    edge_total = 0.0
    edge_count = 0
    width, height = image.size
    for y in range(height):
        row_offset = y * width
        for x in range(width - 1):
            edge_total += abs(pixels[row_offset + x + 1] - pixels[row_offset + x])
            edge_count += 1
    for y in range(height - 1):
        row_offset = y * width
        next_offset = (y + 1) * width
        for x in range(width):
            edge_total += abs(pixels[next_offset + x] - pixels[row_offset + x])
            edge_count += 1
    edge_mean = edge_total / float(max(edge_count, 1))
    flat = variance < 18.0 and edge_mean < 2.0
    return {
        "status": "failed" if flat else "passed",
        "reason": "preview_flat_low_detail" if flat else "preview_has_detail",
        "luma_variance": round(float(variance), 4),
        "edge_mean": round(float(edge_mean), 4),
    }


def rendered_candidate_ok(candidate: dict[str, Any], camera: dict[str, Any]) -> bool:
    preview_path = Path(str(candidate.get("preview_image_path") or ""))
    if not preview_path.exists():
        candidate["preview_qc_status"] = "preview_missing"
        return False
    qc = preview_image_qc(preview_path)
    candidate["preview_image_qc"] = qc
    if qc.get("status") == "failed":
        candidate["preview_qc_status"] = str(qc.get("reason") or "preview_qc_failed")
        return False
    if not candidate_eligible(candidate, camera):
        candidate["preview_qc_status"] = "failed_post_render_candidate_gate"
        return False
    candidate["preview_qc_status"] = "passed"
    return True


def viewpoint_similarity(a: dict[str, Any], b: dict[str, Any], threshold: float = 0.15) -> bool:
    loc_a = a.get("location") or [0.0, 0.0, 0.0]
    loc_b = b.get("location") or [0.0, 0.0, 0.0]
    distance = math.sqrt(sum((float(loc_a[i]) - float(loc_b[i])) ** 2 for i in range(3)))
    if distance > threshold:
        return False
    lens_a = float(a.get("lens_mm") or 35.0)
    lens_b = float(b.get("lens_mm") or 35.0)
    if abs(lens_a - lens_b) > 8.0:
        return False
    return True


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    candidates.sort(key=lambda item: float((item.get("scores") or {}).get("final") or 0.0), reverse=True)
    retained: list[dict[str, Any]] = []
    for candidate in candidates:
        if not any(viewpoint_similarity(candidate, existing) for existing in retained):
            retained.append(candidate)
    return retained


def quality_for_camera(scene, scene_name: str, director_handoff: dict[str, Any], camera: dict[str, Any], output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    set_active_scene(scene)
    details = scene_details(director_handoff, int(camera.get("scene_id") or 0))
    bundle = focus_bundle(scene, director_handoff, camera)
    camera_data = bpy.data.cameras.new(name=f"quality_candidate_{camera['camera_name']}_data")
    camera_obj = bpy.data.objects.new(name=f"quality_candidate_{camera['camera_name']}", object_data=camera_data)
    scene.collection.objects.link(camera_obj)
    try:
        candidates = generate_candidates(scene, camera_obj, camera, bundle, details, args)
        channels = split_by_channel(candidates)
        per_channel_top_k = max(int(args.per_channel_top_k), 2)
        channel_boards: dict[str, dict[str, Any]] = {}
        merged_qualified: list[dict[str, Any]] = []
        total_eligible = 0
        candidate_dir = output_root / "quality_candidates" / f"scene_{camera['scene_id']}_shot_{camera['shot_id']}" / str(camera["camera_name"])
        preview_dir = candidate_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for channel_name, channel_candidates in channels.items():
            if not channel_candidates:
                channel_boards[channel_name] = {"selected": [], "eligible_count": 0, "board_path": None}
                continue
            selected, eligible_count = select_channel_board(channel_candidates, camera, per_channel_top_k)
            total_eligible += eligible_count
            for rank, candidate in enumerate(selected, start=1):
                preview_path = preview_dir / f"{channel_name}_{rank:02d}_{candidate['candidate_id']}.png"
                try:
                    candidate["preview_image_path"] = render_preview(scene, camera_obj, candidate, preview_path, args)
                except Exception as exc:
                    candidate["preview_error"] = str(exc)
            selected = [candidate for candidate in selected if rendered_candidate_ok(candidate, camera)]
            board_path = build_board(selected, candidate_dir / f"{camera['camera_name']}_{channel_name}_board.png")
            channel_boards[channel_name] = {"selected": selected, "eligible_count": eligible_count, "board_path": board_path}
            merged_qualified.extend(selected)
        deduplicated = deduplicate_candidates(merged_qualified)
        retained, final_eligible_count = select_candidates(deduplicated, camera, int(args.top_k))
        merged_board_path = build_board(retained, candidate_dir / f"{camera['camera_name']}_merged_board.png")
        selected = retained[0] if retained else {}
        selected_reason = selection_reason(camera, selected, final_eligible_count)
        if selected:
            selected["selection_reason"] = selected_reason
            selected["selection_source"] = "relaxed_candidate" if bool(selected.get("fallback_unqualified")) else "valid_candidate"
            selected["closeup_required"] = bool(semantic_contract(camera).get("closeup_required"))
            selected["semantic_weight_reason"] = str(semantic_contract(camera).get("semantic_weight_reason") or "")
        else:
            selected_reason = "no_valid_candidate"
        start_transform = {
            "location": list(selected.get("location") or (camera.get("start_transform") or {}).get("location") or [0.0, -3.0, 1.6]),
            "rotation_euler": list(selected.get("rotation_euler") or (camera.get("start_transform") or {}).get("rotation_euler") or [0.0, 0.0, 0.0]),
            "framing": (camera.get("start_transform") or {}).get("framing") or {},
            "lens_mm": float(selected.get("lens_mm") or camera.get("lens_mm") or 35.0),
        }
        end_transform = movement_end_transform(camera, start_transform, bundle["center"])
        scores = selected.get("scores") or {}
        occlusion_info = selected.get("occlusion_check") or {}
        report = {
            "scene_id": int(camera.get("scene_id") or 0),
            "shot_id": int(camera.get("shot_id") or 0),
            "camera_name": str(camera.get("camera_name") or ""),
            "scene_name": scene_name,
            "success": bool(selected),
            "shot_description": camera.get("shot_description") or "",
            "scene_description": camera.get("scene_description") or "",
            "camera_role": camera.get("camera_role") or "",
            "distance_label": camera.get("distance_label") or "",
            "movement_tag": camera.get("movement_tag") or "",
            "shot_contract": camera.get("shot_contract") or {},
            "candidate_count_raw": len(candidates),
            "candidate_count_eligible": total_eligible,
            "candidate_count_retained": len(retained),
            "candidate_count_deduplicated": len(deduplicated),
            "candidate_board_path": merged_board_path,
            "selection_reason": selected_reason,
            "selection_source": str(selected.get("selection_source") or ""),
            "semantic_contract": semantic_contract(camera),
            "closeup_required": bool(semantic_contract(camera).get("closeup_required")),
            "semantic_weight_reason": str(semantic_contract(camera).get("semantic_weight_reason") or ""),
            "basis_source": selected.get("basis_source") or basis_source(bundle["primary_root"]),
            "basis_front_vector": selected.get("basis_front_vector") or [],
            "basis_confidence": selected.get("basis_confidence") or "",
            "center_source": selected.get("center_source") or "primary_center",
            "anchor_kind": selected.get("anchor_kind") or "body_anchor",
            "anchor_point": selected.get("anchor_point") or round_vector(bundle["center"]),
            "channel_quota": channel_quota(camera, int(args.top_k)),
            "channel_boards": {
                name: {
                    "candidate_count": len(info["selected"]),
                    "eligible_count": info["eligible_count"],
                    "board_path": info["board_path"],
                    "selected": info["selected"],
                }
                for name, info in channel_boards.items()
            },
            "selected_candidate": selected,
            "top_candidates": retained,
            "start_transform": start_transform,
            "end_transform": end_transform,
            "quality_qc": {
                "line_of_sight_clear_ratio": float(scores.get("line_of_sight") or 0.0),
                "upper_body_visible_ratio": float(scores.get("upper_body_visibility") or 0.0),
                "primary_visible_fraction": float(scores.get("primary_visible_fraction") or 0.0),
                "primary_area_ratio": float(scores.get("primary_area_ratio") or 0.0),
                "group_subject_area_ratio": float(scores.get("group_subject_area_ratio") or 0.0),
                "direction_alignment": float(scores.get("direction_alignment") or 0.0),
                "camera_height_ok": bool(float(scores.get("camera_height_ok") or 0.0) >= 0.5),
                "occlusion_ratio": float(occlusion_info.get("occlusion_ratio") or 0.0),
                "severely_occluded": bool(occlusion_info.get("severely_occluded")),
                "top_occluder": occlusion_info.get("top_occluder"),
                "inside_scene_bounds": bool((selected.get("wall_check") or {}).get("inside_scene_bounds", False)) if selected else False,
                "required_line_of_sight_clear_ratio": required_sight(camera),
                "required_upper_body_visible_ratio": required_upper(camera),
                "required_primary_area_floor": required_primary_area(camera),
                "required_group_area_floor": float(scores.get("required_group_area_floor") or 0.0),
                "readability_threshold": float(selected.get("readability_threshold") or 0.0) if selected else 0.0,
                "readability_grade": selected.get("readability_grade") or "",
                "focus_resolution_source": bundle.get("focus_resolution_source") or "",
                "focus_conflict_resolved": bool(bundle.get("focus_conflict_resolved")),
            },
        }
        retained_ids = {str(r.get("candidate_id") or "") for r in retained}
        if bool(args.enable_candidate_validation):
            manifest_rows, blind_rows = candidate_validation_rows(
                scene=scene,
                camera_obj=camera_obj,
                camera=camera,
                candidates=candidates,
                retained_ids=retained_ids,
                quality_success=bool(selected),
                candidate_dir=candidate_dir,
                args=args,
            )
            write_candidate_validation_outputs(output_root, manifest_rows, blind_rows)
            report["candidate_validation"] = {
                "enabled": True,
                "manifest_row_count": len(manifest_rows),
                "blind_input_row_count": len(blind_rows),
                "manifest_path": str(validation_paths(output_root)["manifest"]),
                "blind_review_input_path": str(validation_paths(output_root)["blind_input"]),
            }
        rejection_rows = build_rejection_report(candidates, retained_ids, camera)
        report["rejection_report_path"] = str(candidate_dir / "rejection_report_v1.json")
        save_json(
            {
                "camera_name": str(camera.get("camera_name") or ""),
                "candidate_count_total": len(candidates),
                "candidate_count_retained": len(retained),
                "candidate_count_rejected": len(candidates) - len(retained),
                "candidates": rejection_rows,
            },
            candidate_dir / "rejection_report_v1.json",
        )
        save_json(report, candidate_dir / "quality_candidate_report_v1.json")
        return report
    finally:
        try:
            bpy.data.objects.remove(camera_obj, do_unlink=True)
        except Exception:
            pass
        try:
            bpy.data.cameras.remove(camera_data)
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    print(
        f"[ablation] quality_worker disable_semantic_height_adjust="
        f"{bool(_DISABLE_SEMANTIC_HEIGHT_ADJUST)}",
        flush=True,
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if bool(args.enable_candidate_validation):
        prepare_candidate_validation_outputs(output_root)
    camera_handoff = load_json(args.camera_handoff_path)
    director_handoff = load_json(camera_handoff["director_handoff_path"])
    rows = []
    for shot in camera_handoff.get("shots") or []:
        scene_id = int(shot.get("scene_id") or 0)
        shot_id = int(shot.get("shot_id") or 0)
        scene, scene_name = resolve_scene(scene_id, shot_id)
        if scene is None:
            for camera in shot.get("cameras") or []:
                rows.append({"scene_id": scene_id, "shot_id": shot_id, "camera_name": camera.get("camera_name"), "success": False, "error": "scene_not_found"})
            continue
        for camera in shot.get("cameras") or []:
            print(f"[quality] scene={scene_id} shot={shot_id} camera={camera.get('camera_name')}", flush=True)
            try:
                rows.append(quality_for_camera(scene, scene_name, director_handoff, camera, output_root, args))
            except Exception as exc:
                rows.append(
                    {
                        "scene_id": scene_id,
                        "shot_id": shot_id,
                        "camera_name": camera.get("camera_name"),
                        "success": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    report_path = save_json(
        {
            "schema_version": "storyblender.camera_quality_report.v1",
            "camera_handoff_path": str(Path(args.camera_handoff_path).resolve()),
            "success": bool(rows) and all(bool(row.get("success")) for row in rows),
            "rows": rows,
        },
        output_root / "outputs" / "camera_quality_report_v1.json",
    )
    print(json.dumps({"success": bool(rows), "quality_report_path": str(report_path), "row_count": len(rows)}, ensure_ascii=False, indent=2), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
