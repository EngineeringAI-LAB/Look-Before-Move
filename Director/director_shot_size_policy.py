"""Standalone shot-size policy for Plan-A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from director_engine_metrics import safe_median, trimmed_mean


@dataclass(slots=True)
class ShotSizeContract:
    label: str
    hard_floor: float
    desired_min: float
    desired_max: float
    re_search_trigger: float
    visibility_ratio_target: float
    seed_priority: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def contract_for_distance(distance_label: str | None) -> ShotSizeContract:
    """Return the local Plan-A shot-size contract for one scripted distance."""

    normalized = str(distance_label or "").strip().lower()
    if "close" in normalized:
        return ShotSizeContract(
            label="close-up",
            hard_floor=0.18,
            desired_min=0.22,
            desired_max=0.38,
            re_search_trigger=0.18,
            visibility_ratio_target=0.90,
            seed_priority="primary_first",
        )
    if "long" in normalized or "wide" in normalized:
        return ShotSizeContract(
            label="long/wide",
            hard_floor=0.04,
            desired_min=0.05,
            desired_max=0.14,
            re_search_trigger=0.05,
            visibility_ratio_target=0.75,
            seed_priority="balanced_scene",
        )
    return ShotSizeContract(
        label="medium",
        hard_floor=0.09,
        desired_min=0.12,
        desired_max=0.24,
        re_search_trigger=0.11,
        visibility_ratio_target=0.82,
        seed_priority="primary_with_context",
    )


def classify_area_fit(area_ratio: float, contract: ShotSizeContract) -> str:
    """Classify one candidate area ratio against the shot-size contract."""

    ratio = float(area_ratio)
    if ratio < contract.hard_floor:
        return "below_hard_floor"
    if ratio < contract.desired_min:
        return "below_desired_band"
    if ratio <= contract.desired_max:
        return "within_desired_band"
    return "above_desired_band"


def topk_shot_size_statistics(area_ratios: list[float]) -> dict[str, float]:
    """Return one reusable shot-size statistics payload for candidate pools."""

    return {
        "count": len(area_ratios),
        "median": safe_median(area_ratios, default=0.0),
        "trimmed_mean": trimmed_mean(area_ratios, trim_ratio=0.2, default=0.0),
    }


__all__ = [
    "ShotSizeContract",
    "classify_area_fit",
    "contract_for_distance",
    "topk_shot_size_statistics",
]


