"""Standalone path and JSON helpers for Plan-A local engines."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    """Return one stable UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


def ensure_directory(path: str | Path) -> Path:
    """Create one directory when needed and return it as ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_jsonable(payload: Any) -> Any:
    """Convert dataclasses and ``Path`` objects into plain JSON data."""

    if is_dataclass(payload):
        return to_jsonable(asdict(payload))
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): to_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [to_jsonable(value) for value in payload]
    return payload


def save_json(payload: Any, destination: str | Path) -> Path:
    """Save one JSON payload with UTF-8 encoding."""

    target = Path(destination)
    ensure_directory(target.parent)
    target.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_json(path: str | Path) -> Any:
    """Load one JSON payload from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_scene_ids(scene_ids_text: str | Iterable[int]) -> list[int]:
    """Parse scene ids from a comma-separated string or iterable."""

    if isinstance(scene_ids_text, str):
        values = []
        for token in scene_ids_text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except ValueError:
                continue
        return values
    return [int(value) for value in scene_ids_text]


def iter_camera_instructions(shot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all camera instructions in display order."""

    instructions: list[dict[str, Any]] = []
    main_camera = shot.get("camera_instruction")
    if isinstance(main_camera, dict) and main_camera:
        instructions.append(main_camera)
    additional = shot.get("additional_camera_instructions", [])
    if isinstance(additional, list):
        for camera_instruction in additional:
            if isinstance(camera_instruction, dict) and camera_instruction:
                instructions.append(camera_instruction)
    return instructions


def filter_shots_by_scene_ids(shot_details: list[dict[str, Any]], scene_ids: list[int]) -> list[dict[str, Any]]:
    """Return deep-copiable shot dicts only for requested scenes."""

    wanted = set(scene_ids)
    selected: list[dict[str, Any]] = []
    for shot in shot_details:
        try:
            scene_id = int(shot.get("scene_id"))
        except (TypeError, ValueError):
            continue
        if scene_id in wanted:
            selected.append(json.loads(json.dumps(shot)))
    return selected


