from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: Any, path: Path) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    ensure_directory(path.parent)
    fieldnames = [
        "segment_id",
        "status",
        "failure_reason",
        "video_path",
        "subject_perception",
        "intent_consistency",
        "trajectory_quality",
        "overall",
        "SP1_subject_coverage",
        "SP2_identity_consistency",
        "SP3_occlusion_stability",
        "IC1_shot_size_match",
        "IC2_semantic_target_match",
        "IC3_event_alignment",
        "TQ1_motion_smoothness",
        "TQ2_subject_tracking_stability",
        "TQ3_cut_continuity",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def resolve_path(value: str | None, *, base_dir: Path, workspace_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return str(base_candidate)
    return str((workspace_dir / path).resolve())


def score_mean(values: list[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def clamp_score(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return round(max(0.0, min(100.0, float(value))), 3)


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def sanitize_for_filename(value: str) -> str:
    keep = []
    for char in str(value):
        if char.isalnum() or char in ("_", "-"):
            keep.append(char)
        else:
            keep.append("_")
    text = "".join(keep).strip("_")
    return text or "segment"
