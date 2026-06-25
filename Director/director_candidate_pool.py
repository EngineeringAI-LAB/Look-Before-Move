"""Standalone candidate-pool generation for Plan-A blocking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from director_shot_size_policy import ShotSizeContract, contract_for_distance
from director_turnaround_context import normalize_turnaround_angle, ordered_turnaround_paths, semantic_angle_bias


@dataclass(slots=True)
class BlockingCandidate:
    candidate_id: str
    camera_name: str
    scene_id: int
    shot_id: int
    source: str
    primary_focus_id: str
    secondary_focus_ids: list[str] = field(default_factory=list)
    direction: str = ""
    reference_image_path: str = ""
    score_components: dict[str, float] = field(default_factory=dict)
    estimated_metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _character_lookup(scene_dossier: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = scene_dossier.get("character_summaries", []) or []
    return {
        str(row.get("character_id")): row
        for row in rows
        if isinstance(row, dict) and str(row.get("character_id") or "").strip()
    }


def _estimated_primary_area(contract: ShotSizeContract, direction_name: str) -> float:
    base = contract.desired_min
    if contract.label == "close-up":
        direction_bonus = {
            "front": 0.06,
            "front_left": 0.04,
            "front_right": 0.04,
            "left": 0.02,
            "right": 0.02,
            "back_left": -0.04,
            "back_right": -0.04,
            "back": -0.08,
            "top": -0.10,
        }.get(direction_name, 0.0)
    elif contract.label == "medium":
        direction_bonus = {
            "front": 0.03,
            "front_left": 0.02,
            "front_right": 0.02,
            "left": 0.01,
            "right": 0.01,
            "back": -0.03,
            "top": -0.08,
        }.get(direction_name, 0.0)
    else:
        direction_bonus = {
            "top": 0.02,
            "front": 0.01,
            "back": 0.01,
        }.get(direction_name, 0.0)
    return max(base + direction_bonus, 0.01)


def build_unified_candidates(
    *,
    scene_dossier: dict[str, Any],
    shot_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one lightweight standalone candidate pool for local blocking."""

    start_frame_contract = shot_contract.get("start_frame_contract") or {}
    primary_focus_id = str(start_frame_contract.get("primary_focus_id") or "").strip()
    secondary_focus_ids = list(start_frame_contract.get("secondary_focus_ids") or [])
    camera_name = str(shot_contract.get("camera_name") or "")
    scene_id = int(shot_contract.get("scene_id") or 0)
    shot_id = int(shot_contract.get("shot_id") or 0)
    distance_label = str(start_frame_contract.get("distance") or "")
    angle_label = str(start_frame_contract.get("angle") or "")
    shot_size_contract = contract_for_distance(distance_label)
    characters = _character_lookup(scene_dossier)
    primary_row = characters.get(primary_focus_id) or {}

    candidates: list[BlockingCandidate] = []
    turnaround_paths = ordered_turnaround_paths(
        primary_row.get("turnaround_paths") or primary_row.get("context_turnaround_paths") or {}
    )
    for rank_index, (direction_name, image_path) in enumerate(turnaround_paths, start=1):
        semantic_direction = normalize_turnaround_angle(direction_name)
        candidate = BlockingCandidate(
            candidate_id=f"{camera_name}_turnaround_{semantic_direction}",
            camera_name=camera_name,
            scene_id=scene_id,
            shot_id=shot_id,
            source="turnaround",
            primary_focus_id=primary_focus_id,
            secondary_focus_ids=secondary_focus_ids,
            direction=semantic_direction,
            reference_image_path=image_path,
            score_components={
                "semantic_angle_score": semantic_angle_bias(angle_label, semantic_direction),
                "primary_anchor_score": 1.0 if semantic_direction != "top" else 0.35,
                "shot_size_prior_score": 1.0 if semantic_direction != "top" else 0.1,
            },
            estimated_metrics={
                "candidate_primary_area_ratio": _estimated_primary_area(shot_size_contract, semantic_direction),
                "candidate_visible_ratio": shot_size_contract.visibility_ratio_target,
                "candidate_region_distance": 0.0 if semantic_direction != "top" else 0.4,
            },
            notes=[
                "Primary-first turnaround root candidate.",
                f"Shot-size contract label={shot_size_contract.label}.",
            ],
        )
        candidates.append(candidate)

    scene_top_view_path = str(scene_dossier.get("scene_top_view_path") or "").strip()
    if scene_top_view_path and Path(scene_top_view_path).exists():
        candidates.append(
            BlockingCandidate(
                candidate_id=f"{camera_name}_scene_top_context",
                camera_name=camera_name,
                scene_id=scene_id,
                shot_id=shot_id,
                source="scene_top_context",
                primary_focus_id=primary_focus_id,
                secondary_focus_ids=secondary_focus_ids,
                direction="top",
                reference_image_path=scene_top_view_path,
                score_components={
                    "semantic_angle_score": 0.2,
                    "primary_anchor_score": 0.15,
                    "shot_size_prior_score": 0.05,
                },
                estimated_metrics={
                    "candidate_primary_area_ratio": max(shot_size_contract.hard_floor * 0.5, 0.02),
                    "candidate_visible_ratio": shot_size_contract.visibility_ratio_target,
                    "candidate_region_distance": 0.15,
                },
                notes=[
                    "Scene top context candidate retained for topology understanding.",
                    "This image informs blocking but should not dominate shot-size selection.",
                ],
            ).to_dict()
        )

    if not candidates:
        candidates.append(
            BlockingCandidate(
                candidate_id=f"{camera_name}_fallback_context",
                camera_name=camera_name,
                scene_id=scene_id,
                shot_id=shot_id,
                source="fallback",
                primary_focus_id=primary_focus_id,
                secondary_focus_ids=secondary_focus_ids,
                direction="unknown",
                reference_image_path=str(scene_dossier.get("scene_top_view_path") or ""),
                score_components={
                    "semantic_angle_score": 0.1,
                    "primary_anchor_score": 0.1,
                    "shot_size_prior_score": 0.1,
                },
                estimated_metrics={
                    "candidate_primary_area_ratio": shot_size_contract.hard_floor,
                    "candidate_visible_ratio": shot_size_contract.visibility_ratio_target,
                    "candidate_region_distance": 0.3,
                },
                notes=["Fallback candidate created because no turnaround images were available."],
            ).to_dict()
        )

    return [candidate.to_dict() if isinstance(candidate, BlockingCandidate) else dict(candidate) for candidate in candidates]


__all__ = [
    "BlockingCandidate",
    "build_unified_candidates",
]

