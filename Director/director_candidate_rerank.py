"""Standalone candidate reranking for Plan-A blocking."""

from __future__ import annotations

from typing import Any

from director_engine_metrics import clamp
from director_shot_size_policy import classify_area_fit, contract_for_distance, topk_shot_size_statistics


def _candidate_total_score(candidate: dict[str, Any], shot_contract: dict[str, Any]) -> dict[str, Any]:
    start_frame_contract = shot_contract.get("start_frame_contract") or {}
    contract = contract_for_distance(start_frame_contract.get("distance"))
    estimated_metrics = candidate.get("estimated_metrics") or {}
    score_components = dict(candidate.get("score_components") or {})
    area_ratio = float(estimated_metrics.get("candidate_primary_area_ratio") or 0.0)
    visibility_ratio = float(estimated_metrics.get("candidate_visible_ratio") or 0.0)
    region_distance = float(estimated_metrics.get("candidate_region_distance") or 1.0)
    area_fit = classify_area_fit(area_ratio, contract)

    shot_size_band_score = {
        "below_hard_floor": 0.0,
        "below_desired_band": 0.55,
        "within_desired_band": 1.0,
        "above_desired_band": 0.82,
    }[area_fit]
    visibility_score = clamp(visibility_ratio / max(contract.visibility_ratio_target, 0.01), 0.0, 1.0)
    region_score = clamp(1.0 - region_distance, 0.0, 1.0)
    primary_anchor_score = float(score_components.get("primary_anchor_score") or 0.0)
    semantic_angle_score = float(score_components.get("semantic_angle_score") or 0.0)
    prior_score = float(score_components.get("shot_size_prior_score") or 0.0)

    normalized_score = clamp(
        0.28 * shot_size_band_score
        + 0.22 * semantic_angle_score
        + 0.18 * visibility_score
        + 0.14 * region_score
        + 0.12 * primary_anchor_score
        + 0.06 * prior_score,
        0.0,
        1.0,
    )
    candidate["score_components"] = {
        **score_components,
        "shot_size_band_score": shot_size_band_score,
        "visibility_score": visibility_score,
        "region_score": region_score,
        "primary_anchor_score": primary_anchor_score,
        "semantic_angle_score": semantic_angle_score,
        "shot_size_prior_score": prior_score,
    }
    candidate["shot_size_fit"] = area_fit
    candidate["normalized_score"] = normalized_score
    return candidate


def rerank_candidates(
    *,
    candidates: list[dict[str, Any]],
    shot_contract: dict[str, Any],
    top_k: int = 15,
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Rerank one candidate pool and return top-k plus shot-size diagnostics."""

    ranked = [_candidate_total_score(dict(candidate), shot_contract) for candidate in candidates]
    ranked.sort(key=lambda item: item.get("normalized_score", 0.0), reverse=True)

    area_ratios = [
        float((candidate.get("estimated_metrics") or {}).get("candidate_primary_area_ratio") or 0.0)
        for candidate in ranked[:top_k]
    ]
    statistics = topk_shot_size_statistics(area_ratios)
    passing = [candidate for candidate in ranked if float(candidate.get("normalized_score") or 0.0) >= threshold]
    selected_pool = passing[:top_k] if passing else ranked[:top_k]
    return {
        "top_k": top_k,
        "threshold": threshold,
        "ranked_candidates": ranked,
        "selected_candidates": selected_pool,
        "topk_area_statistics": statistics,
    }


__all__ = [
    "rerank_candidates",
]

