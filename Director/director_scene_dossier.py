"""Standalone Plan-A scene dossier models and JSON helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Mapping


TURNAROUND_DIRECTIONS: tuple[str, ...] = (
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
    "top",
)

SCENE_DOSSIER_SCHEMA = "storyblender.director_scene_dossier.v1"


def _as_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _as_list_of_str(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [_as_str(item) for item in value if _as_str(item)]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {str(k): v for k, v in value.items()}
    return {}


@dataclass(slots=True)
class TurnaroundPathSet:
    """Per-character preview images for 8 directions plus top."""

    front: str | None = None
    front_right: str | None = None
    right: str | None = None
    back_right: str | None = None
    back: str | None = None
    back_left: str | None = None
    left: str | None = None
    front_left: str | None = None
    top: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            direction: _as_str(getattr(self, direction), default="")
            for direction in TURNAROUND_DIRECTIONS
            if _as_str(getattr(self, direction), default="")
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TurnaroundPathSet":
        payload = _as_dict(data)
        return cls(
            front=_as_str(payload.get("front")) or None,
            front_right=_as_str(payload.get("front_right")) or None,
            right=_as_str(payload.get("right")) or None,
            back_right=_as_str(payload.get("back_right")) or None,
            back=_as_str(payload.get("back")) or None,
            back_left=_as_str(payload.get("back_left")) or None,
            left=_as_str(payload.get("left")) or None,
            front_left=_as_str(payload.get("front_left")) or None,
            top=_as_str(payload.get("top")) or None,
        )

    def with_defaults(self) -> "TurnaroundPathSet":
        """Return a copy that always includes every known direction key."""

        values = self.to_dict()
        for direction in TURNAROUND_DIRECTIONS:
            values.setdefault(direction, "")
        return TurnaroundPathSet.from_dict(values)


@dataclass(slots=True)
class CharacterSummary:
    """Scene-level character summary with close shot previews and wide context previews."""

    character_id: str
    display_name: str = ""
    summary: str = ""
    portrait_path: str = ""
    turnaround_paths: TurnaroundPathSet = field(default_factory=TurnaroundPathSet)
    context_turnaround_paths: TurnaroundPathSet = field(default_factory=TurnaroundPathSet)
    layout_data: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "display_name": self.display_name,
            "summary": self.summary,
            "portrait_path": self.portrait_path,
            "turnaround_paths": self.turnaround_paths.with_defaults().to_dict(),
            "context_turnaround_paths": self.context_turnaround_paths.with_defaults().to_dict(),
            "layout_data": dict(self.layout_data),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CharacterSummary":
        payload = _as_dict(data)
        return cls(
            character_id=_as_str(payload.get("character_id")),
            display_name=_as_str(payload.get("display_name")),
            summary=_as_str(payload.get("summary")),
            portrait_path=_as_str(payload.get("portrait_path")),
            turnaround_paths=TurnaroundPathSet.from_dict(payload.get("turnaround_paths")),
            context_turnaround_paths=TurnaroundPathSet.from_dict(payload.get("context_turnaround_paths")),
            layout_data=_as_dict(payload.get("layout_data")),
            tags=_as_list_of_str(payload.get("tags")),
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class AssetSummary:
    """Compact scene asset record for non-character references."""

    asset_id: str
    asset_type: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "summary": self.summary,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AssetSummary":
        payload = _as_dict(data)
        return cls(
            asset_id=_as_str(payload.get("asset_id")),
            asset_type=_as_str(payload.get("asset_type")),
            summary=_as_str(payload.get("summary")),
            tags=_as_list_of_str(payload.get("tags")),
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class SceneDossier:
    """Standalone scene dossier used by Plan-A-only understanding contracts."""

    scene_id: str
    scene_top_view_path: str = ""
    layout_data: dict[str, Any] = field(default_factory=dict)
    character_summaries: list[CharacterSummary] = field(default_factory=list)
    asset_summaries: list[AssetSummary] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = SCENE_DOSSIER_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema or SCENE_DOSSIER_SCHEMA,
            "scene_id": self.scene_id,
            "scene_top_view_path": self.scene_top_view_path,
            "layout_data": dict(self.layout_data),
            "character_summaries": [item.to_dict() for item in self.character_summaries],
            "asset_summaries": [item.to_dict() for item in self.asset_summaries],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SceneDossier":
        payload = _as_dict(data)
        characters = [
            CharacterSummary.from_dict(item)
            for item in (payload.get("character_summaries") or [])
            if isinstance(item, dict)
        ]
        assets = [
            AssetSummary.from_dict(item)
            for item in (payload.get("asset_summaries") or [])
            if isinstance(item, dict)
        ]
        return cls(
            scene_id=_as_str(payload.get("scene_id")),
            scene_top_view_path=_as_str(payload.get("scene_top_view_path")),
            layout_data=_as_dict(payload.get("layout_data")),
            character_summaries=characters,
            asset_summaries=assets,
            metadata=_as_dict(payload.get("metadata")),
            schema=_as_str(payload.get("schema"), default=SCENE_DOSSIER_SCHEMA),
        )


def save_scene_dossier_json(scene_dossier: SceneDossier, output_path: str | Path) -> Path:
    """Save a scene dossier as JSON and return the resolved output path."""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(scene_dossier.to_dict(), handle, indent=2, ensure_ascii=False)
    return target


def load_scene_dossier_json(input_path: str | Path) -> SceneDossier:
    """Load one scene dossier JSON file."""

    source = Path(input_path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scene dossier JSON must be an object: {source}")
    return SceneDossier.from_dict(payload)


def save_scene_dossier_manifest(
    *,
    scene_entries: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save one multi-scene dossier manifest."""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "storyblender.director_scene_dossier_manifest.v1",
        "scene_count": len(scene_entries),
        "scenes": scene_entries,
    }
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return target


def load_scene_dossier_manifest(input_path: str | Path) -> dict[str, Any]:
    """Load one multi-scene dossier manifest JSON file."""

    source = Path(input_path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scene dossier manifest JSON must be an object: {source}")
    return payload


__all__ = [
    "TURNAROUND_DIRECTIONS",
    "SCENE_DOSSIER_SCHEMA",
    "TurnaroundPathSet",
    "CharacterSummary",
    "AssetSummary",
    "SceneDossier",
    "save_scene_dossier_json",
    "load_scene_dossier_json",
    "save_scene_dossier_manifest",
    "load_scene_dossier_manifest",
]


