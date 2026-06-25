from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


USABLE_POSITIVE = {"good", "borderline"}
USABLE_NEGATIVE = {"bad"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def save_json(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def normalized_usable(row: dict[str, Any]) -> str:
    return str(row.get("usable") or "").strip().lower()


def human_accepts(row: dict[str, Any]) -> bool | None:
    usable = normalized_usable(row)
    if usable in USABLE_POSITIVE:
        return True
    if usable in USABLE_NEGATIVE:
        return False
    return None


def algorithm_accepts(row: dict[str, Any]) -> bool:
    if "retained" in row:
        return bool(row.get("retained"))
    return str(row.get("algorithm_decision") or "").strip().lower() == "retained"


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]
    rows: list[str] = []
    for item in raw_values:
        text = str(item or "").strip()
        if text and text not in rows:
            rows.append(text)
    return rows


def load_focus_conflict_map(output_root: str | Path | None) -> dict[str, dict[str, Any]]:
    if output_root is None:
        return {}
    camera_root = Path(output_root) / "camera_packages"
    if not camera_root.exists():
        return {}
    conflicts: dict[str, dict[str, Any]] = {}
    for path in camera_root.glob("scene_*_shot_*/*.json"):
        try:
            camera = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        camera_name = str(camera.get("camera_name") or path.stem).strip()
        if not camera_name:
            continue
        contract = camera.get("shot_contract") or {}
        if not isinstance(contract, dict):
            contract = {}
        start_contract = contract.get("start_frame_contract") or {}
        if not isinstance(start_contract, dict):
            start_contract = {}
        package_primary = str(camera.get("primary_focus_id") or "").strip()
        contract_primary = str(start_contract.get("primary_focus_id") or "").strip()
        start_ids = as_string_list(start_contract.get("start_focus_ids"))
        keyframe_primary_ids: list[str] = []
        keyframe_plan = contract.get("keyframe_plan") or []
        if isinstance(keyframe_plan, list):
            for keyframe in keyframe_plan:
                if isinstance(keyframe, dict):
                    keyframe_primary_ids.extend(as_string_list(keyframe.get("primary_focus_id")))

        corrected_primary = package_primary or contract_primary or (start_ids[0] if start_ids else "")
        conflict = False
        if start_ids:
            start_primary = start_ids[0]
            keyframe_votes = sum(1 for item in keyframe_primary_ids if item == start_primary)
            has_keyframe_consensus = bool(keyframe_primary_ids) and keyframe_votes >= max(1, len(keyframe_primary_ids) // 2 + 1)
            conflict = bool(
                has_keyframe_consensus
                and start_primary
                and ((package_primary and package_primary != start_primary) or (contract_primary and contract_primary != start_primary))
            )
            if conflict:
                corrected_primary = start_primary
        conflicts[camera_name] = {
            "focus_conflict_corrected": conflict,
            "corrected_primary_focus_id": corrected_primary,
            "original_primary_focus_id": package_primary,
            "contract_primary_focus_id": contract_primary,
            "contract_start_focus_ids": start_ids,
            "keyframe_primary_ids": sorted(set(keyframe_primary_ids)),
        }
    return conflicts


def empty_counts() -> dict[str, int]:
    return {
        "reviewed": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "algorithm_retained": 0,
        "algorithm_rejected": 0,
        "human_usable": 0,
        "human_bad": 0,
    }


def add_observation(counts: dict[str, int], algorithm_positive: bool, human_positive: bool) -> None:
    counts["reviewed"] += 1
    if algorithm_positive:
        counts["algorithm_retained"] += 1
    else:
        counts["algorithm_rejected"] += 1
    if human_positive:
        counts["human_usable"] += 1
    else:
        counts["human_bad"] += 1
    if algorithm_positive and human_positive:
        counts["tp"] += 1
    elif algorithm_positive and not human_positive:
        counts["fp"] += 1
    elif not algorithm_positive and human_positive:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


def finalize_counts(counts: dict[str, int]) -> dict[str, Any]:
    human_usable = max(int(counts.get("human_usable") or 0), 0)
    algorithm_retained = max(int(counts.get("algorithm_retained") or 0), 0)
    reviewed = max(int(counts.get("reviewed") or 0), 0)
    return {
        **counts,
        "false_negative_rate": round(float(counts["fn"]) / float(human_usable), 6) if human_usable else None,
        "false_positive_rate": round(float(counts["fp"]) / float(algorithm_retained), 6) if algorithm_retained else None,
        "accuracy": round(float(counts["tp"] + counts["tn"]) / float(reviewed), 6) if reviewed else None,
    }


def build_report(
    *,
    manifest_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    manifest_path: Path,
    review_result_path: Path,
    focus_conflicts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_by_id = {str(row.get("validation_id") or ""): row for row in manifest_rows if row.get("validation_id")}
    review_by_id = {str(row.get("validation_id") or ""): row for row in review_rows if row.get("validation_id")}
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped_review_ids: list[str] = []
    for validation_id, review in review_by_id.items():
        manifest = manifest_by_id.get(validation_id)
        if manifest is None:
            skipped_review_ids.append(validation_id)
            continue
        accepted = human_accepts(review)
        if accepted is None:
            skipped_review_ids.append(validation_id)
            continue
        joined.append((manifest, review))

    overall = empty_counts()
    by_channel: dict[str, dict[str, int]] = defaultdict(empty_counts)
    by_rejection_category: dict[str, dict[str, int]] = defaultdict(empty_counts)
    by_failure_reason: dict[str, dict[str, int]] = defaultdict(empty_counts)
    by_focus_conflict: dict[str, dict[str, int]] = defaultdict(empty_counts)
    clean_overall = empty_counts()
    clean_by_channel: dict[str, dict[str, int]] = defaultdict(empty_counts)
    focus_conflict_camera_counts: dict[str, dict[str, Any]] = {}
    focus_conflicts = focus_conflicts or {}

    for manifest, review in joined:
        algorithm_positive = algorithm_accepts(manifest)
        human_positive = bool(human_accepts(review))
        camera_name = str(manifest.get("camera_name") or "")
        focus_info = focus_conflicts.get(camera_name) or {}
        focus_conflicted = bool(focus_info.get("focus_conflict_corrected"))
        add_observation(overall, algorithm_positive, human_positive)
        channel = str(manifest.get("channel") or "unknown")
        add_observation(by_channel[channel], algorithm_positive, human_positive)
        add_observation(by_focus_conflict["conflicted" if focus_conflicted else "clean"], algorithm_positive, human_positive)
        if not focus_conflicted:
            add_observation(clean_overall, algorithm_positive, human_positive)
            add_observation(clean_by_channel[channel], algorithm_positive, human_positive)
        else:
            row = focus_conflict_camera_counts.setdefault(
                camera_name,
                {
                    "camera_name": camera_name,
                    "reviewed": 0,
                    "bad": 0,
                    "usable": 0,
                    "wrong_subject": 0,
                    "corrected_primary_focus_id": focus_info.get("corrected_primary_focus_id") or "",
                    "original_primary_focus_id": focus_info.get("original_primary_focus_id") or "",
                    "contract_primary_focus_id": focus_info.get("contract_primary_focus_id") or "",
                    "contract_start_focus_ids": focus_info.get("contract_start_focus_ids") or [],
                    "keyframe_primary_ids": focus_info.get("keyframe_primary_ids") or [],
                },
            )
            row["reviewed"] += 1
            if human_positive:
                row["usable"] += 1
            else:
                row["bad"] += 1
            if str(review.get("failure_reason") or "").strip().lower() == "wrong_subject":
                row["wrong_subject"] += 1
        for category in manifest.get("rejection_categories") or []:
            add_observation(by_rejection_category[str(category)], algorithm_positive, human_positive)
        failure_reason = str(review.get("failure_reason") or "").strip().lower()
        if failure_reason:
            add_observation(by_failure_reason[failure_reason], algorithm_positive, human_positive)

    status = "complete" if joined else "pending_review_results"
    return {
        "schema_version": "storyblender.candidate_filter_confusion_report.v1",
        "status": status,
        "manifest_path": str(manifest_path),
        "blind_review_result_path": str(review_result_path),
        "manifest_row_count": len(manifest_rows),
        "review_row_count": len(review_rows),
        "matched_review_row_count": len(joined),
        "skipped_review_ids": skipped_review_ids[:100],
        "overall": finalize_counts(overall),
        "overall_excluding_focus_conflicts": finalize_counts(clean_overall),
        "by_channel": {
            channel: finalize_counts(counts)
            for channel, counts in sorted(by_channel.items())
        },
        "by_channel_excluding_focus_conflicts": {
            channel: finalize_counts(counts)
            for channel, counts in sorted(clean_by_channel.items())
        },
        "by_focus_conflict": {
            status_name: finalize_counts(counts)
            for status_name, counts in sorted(by_focus_conflict.items())
        },
        "focus_conflict_affected_cameras": sorted(
            focus_conflict_camera_counts.values(),
            key=lambda row: (-int(row.get("wrong_subject") or 0), str(row.get("camera_name") or "")),
        ),
        "by_rejection_category": {
            category: finalize_counts(counts)
            for category, counts in sorted(by_rejection_category.items())
        },
        "by_human_failure_reason": {
            reason: finalize_counts(counts)
            for reason, counts in sorted(by_failure_reason.items())
        },
        "metric_definitions": {
            "tp": "algorithm retained and reviewer marked usable good/borderline",
            "fp": "algorithm retained but reviewer marked bad",
            "tn": "algorithm rejected and reviewer marked bad",
            "fn": "algorithm rejected but reviewer marked usable good/borderline",
            "false_negative_rate": "fn / human_usable",
            "false_positive_rate": "fp / algorithm_retained",
            "overall_excluding_focus_conflicts": "same metrics after removing cameras where start_focus_ids/keyframes conflict with package primary_focus_id",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a confusion report for direction/preset candidate filter validation.")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--review-result-path", default="")
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()
    if not args.output_root and (not args.manifest_path or not args.review_result_path):
        parser.error("Provide --output-root or both --manifest-path and --review-result-path.")
    return args


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve() if args.output_root else None
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else output_root / "outputs" / "candidate_validation_manifest_v1.jsonl"
    review_result_path = Path(args.review_result_path).resolve() if args.review_result_path else output_root / "outputs" / "candidate_blind_review_result_v1.jsonl"
    if args.output_path:
        output_path = Path(args.output_path).resolve()
    elif output_root is not None:
        output_path = output_root / "outputs" / "candidate_filter_confusion_report_v1.json"
    else:
        output_path = review_result_path.parent / "candidate_filter_confusion_report_v1.json"
    report = build_report(
        manifest_rows=read_jsonl(manifest_path),
        review_rows=read_jsonl(review_result_path),
        manifest_path=manifest_path,
        review_result_path=review_result_path,
        focus_conflicts=load_focus_conflict_map(output_root),
    )
    save_json(report, output_path)
    print(json.dumps({"success": True, "report_path": str(output_path), "status": report["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
