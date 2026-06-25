"""Standalone blocking-plan engine for Plan-A."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from director_candidate_pool import build_unified_candidates
from director_candidate_rerank import rerank_candidates
from director_engine_paths import utc_now
from director_shot_size_policy import contract_for_distance


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    return {}


@dataclass(slots=True)
class BlockingPlan:
    scene_id: int
    shot_id: int
    camera_name: str
    primary_focus_id: str
    secondary_focus_ids: list[str] = field(default_factory=list)
    shot_size_contract: dict[str, Any] = field(default_factory=dict)
    primary_first_seeding: dict[str, Any] = field(default_factory=dict)
    start_frame_contract: dict[str, Any] = field(default_factory=dict)
    keyframe_plan: list[dict[str, Any]] = field(default_factory=list)
    motion_contract: dict[str, Any] = field(default_factory=dict)
    topk_area_statistics: dict[str, Any] = field(default_factory=dict)
    shot_size_research_triggered: bool = False
    selected_candidate: dict[str, Any] = field(default_factory=dict)
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    generated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _primary_first_seeding_payload(shot_contract: dict[str, Any]) -> dict[str, Any]:
    start_frame_contract = shot_contract.get("start_frame_contract") or {}
    distance_label = start_frame_contract.get("distance")
    shot_size_contract = contract_for_distance(distance_label)
    return {
        "seed_priority": shot_size_contract.seed_priority,
        "primary_focus_id": start_frame_contract.get("primary_focus_id"),
        "start_focus_ids": list(start_frame_contract.get("start_focus_ids") or []),
        "secondary_focus_ids": list(start_frame_contract.get("secondary_focus_ids") or []),
        "start_subject_layout": _as_dict(start_frame_contract.get("start_subject_layout")),
        "notes": [
            "Start candidate generation around the primary focus anchor before allowing wider context.",
            "Do not let grouped focus extents pull close-up seeds unnecessarily wide.",
        ],
    }


def build_scene_blocking_plans(
    *,
    scene_dossier: dict[str, Any],
    shot_contracts: list[dict[str, Any]],
    top_k: int = 15,
    threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Build one standalone blocking-plan payload for each camera contract."""

    plans: list[dict[str, Any]] = []
    for shot_contract in shot_contracts:
        start_frame_contract = shot_contract.get("start_frame_contract") or {}
        distance_label = start_frame_contract.get("distance")
        shot_size_contract = contract_for_distance(distance_label)
        candidates = build_unified_candidates(
            scene_dossier=scene_dossier,
            shot_contract=shot_contract,
        )
        rerank_result = rerank_candidates(
            candidates=candidates,
            shot_contract=shot_contract,
            top_k=top_k,
            threshold=threshold,
        )
        statistics = rerank_result["topk_area_statistics"]
        selected_candidates = rerank_result["selected_candidates"]
        research_triggered = statistics.get("median", 0.0) < shot_size_contract.re_search_trigger
        selected_candidate = selected_candidates[0] if selected_candidates else {}

        plan = BlockingPlan(
            scene_id=int(shot_contract.get("scene_id") or 0),
            shot_id=int(shot_contract.get("shot_id") or 0),
            camera_name=str(shot_contract.get("camera_name") or ""),
            primary_focus_id=str(start_frame_contract.get("primary_focus_id") or ""),
            secondary_focus_ids=list(start_frame_contract.get("secondary_focus_ids") or []),
            shot_size_contract=shot_size_contract.to_dict(),
            primary_first_seeding=_primary_first_seeding_payload(shot_contract),
            start_frame_contract=dict(start_frame_contract),
            keyframe_plan=list(shot_contract.get("keyframe_plan") or []),
            motion_contract=dict(shot_contract.get("motion_contract") or {}),
            topk_area_statistics=statistics,
            shot_size_research_triggered=research_triggered,
            selected_candidate=selected_candidate,
            top_candidates=selected_candidates,
            candidate_count=len(candidates),
        ).to_dict()
        plan["blocking_engine_version"] = "director.blocking.v1"
        plan["candidate_pool_preview"] = {
            "top_k": top_k,
            "threshold": threshold,
            "candidate_ids": [candidate.get("candidate_id") for candidate in selected_candidates],
        }
        plans.append(plan)
    return plans


__all__ = [
    "BlockingPlan",
    "build_scene_blocking_plans",
]

