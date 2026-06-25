"""Standalone turnaround-context helpers for Plan-A."""

from __future__ import annotations

from typing import Any


TURNAROUND_DIRECTION_ORDER: tuple[str, ...] = (
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
)

SEMANTIC_DIRECTION_ORDER: tuple[str, ...] = TURNAROUND_DIRECTION_ORDER + ("top",)

SEMANTIC_TO_VIEWPOINT_ANGLE: dict[str, str] = {
    "front": "front",
    "front_right": "front_left",
    "right": "left",
    "back_right": "back_left",
    "back": "back",
    "back_left": "back_right",
    "left": "right",
    "front_left": "front_right",
}

VIEWPOINT_TO_SEMANTIC_ANGLE: dict[str, str] = {
    viewpoint_angle: semantic_angle
    for semantic_angle, viewpoint_angle in SEMANTIC_TO_VIEWPOINT_ANGLE.items()
}


def normalize_turnaround_angle(angle_name: str | None) -> str:
    """Return one normalized turnaround angle token."""

    return str(angle_name or "").strip().lower()


def to_viewpoint_turnaround_angle(semantic_angle: str | None) -> str:
    """Map one Plan-A semantic label to the Blender viewpoint label."""

    normalized = normalize_turnaround_angle(semantic_angle)
    return SEMANTIC_TO_VIEWPOINT_ANGLE.get(normalized, normalized)


def to_semantic_turnaround_angle(viewpoint_angle: str | None) -> str:
    """Map one Blender viewpoint label to the Plan-A semantic label."""

    normalized = normalize_turnaround_angle(viewpoint_angle)
    return VIEWPOINT_TO_SEMANTIC_ANGLE.get(normalized, normalized)


def ordered_turnaround_paths(turnaround_paths: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Return turnaround images in one stable display order."""

    payload = dict(turnaround_paths or {})
    ordered: list[tuple[str, str]] = []
    for direction_name in TURNAROUND_DIRECTION_ORDER:
        image_path = str(payload.get(direction_name) or "").strip()
        if image_path:
            ordered.append((direction_name, image_path))
    top_path = str(payload.get("top") or "").strip()
    if top_path:
        ordered.append(("top", top_path))
    return ordered


def semantic_angle_bias(requested_angle: str | None, candidate_direction: str) -> float:
    """Return one lightweight semantic-angle preference score."""

    angle_text = normalize_turnaround_angle(requested_angle)
    direction = normalize_turnaround_angle(candidate_direction)
    if not angle_text:
        return 0.6
    if direction in angle_text:
        return 1.0
    if "profile" in angle_text and direction in {"left", "right"}:
        return 0.92
    if "front" in angle_text and direction.startswith("front"):
        return 0.95
    if "back" in angle_text and direction.startswith("back"):
        return 0.95
    if "over shoulder" in angle_text and direction in {"back_left", "back_right"}:
        return 0.9
    return 0.55


__all__ = [
    "TURNAROUND_DIRECTION_ORDER",
    "SEMANTIC_DIRECTION_ORDER",
    "SEMANTIC_TO_VIEWPOINT_ANGLE",
    "VIEWPOINT_TO_SEMANTIC_ANGLE",
    "normalize_turnaround_angle",
    "to_viewpoint_turnaround_angle",
    "to_semantic_turnaround_angle",
    "ordered_turnaround_paths",
    "semantic_angle_bias",
]

