from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

STAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = STAGE_DIR.parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from cinestory_eval.io import (
    clamp_score,
    ensure_directory,
    load_json,
    resolve_path,
    sanitize_for_filename,
    save_json,
    score_mean,
    utc_now,
    write_jsonl,
    write_summary_csv,
)
from cinestory_eval.metrics import SegmentEvaluationError, evaluate_segment, finalize_boundary_scores
from cinestory_eval.model_zoo import VisionLanguageClient


METRIC_NAMES = [
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
REFERENCE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def default_output_root(run_id: str | None = None) -> Path:
    suffix = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return STAGE_DIR / "output" / suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CineStoryEval black-box video benchmark.")
    parser.add_argument("--benchmark-input", required=True, help="Path to benchmark_input_v1.json.")
    parser.add_argument("--output-root", default="", help="Directory for benchmark outputs.")
    parser.add_argument("--run-id", default="", help="Optional run id for default output path.")
    parser.add_argument("--runtime-config", default=str(WORKSPACE_DIR / "config" / "runtime_config.json"))
    parser.add_argument("--vlm-backend", choices=("auto", "gemini", "openai_compatible", "qwen_local", "none"), default="auto")
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--uniform-frames", type=int, default=8)
    parser.add_argument("--device", default="auto", help="Reserved for optional local model backends.")
    parser.add_argument("--reference-root", default=str(STAGE_DIR / "reference"), help="Root directory for story reference libraries.")
    parser.add_argument("--story-name", default="", help="Story reference library name under --reference-root.")
    return parser.parse_args()


def story_name_from_input(raw: dict[str, Any], explicit: str = "") -> str:
    metadata = raw.get("metadata", {})
    if explicit:
        return explicit
    if isinstance(metadata, dict):
        for key in ("story_name", "story_id", "title"):
            if metadata.get(key):
                return str(metadata[key])
    return str(raw.get("dataset_name") or "default")


def reference_dirs(reference_root: Path, story_name: str) -> list[Path]:
    candidates = [reference_root / story_name, reference_root / sanitize_for_filename(story_name)]
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def load_reference_manifest(reference_dir: Path) -> dict[str, Any]:
    manifest_path = reference_dir / "reference_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = load_json(manifest_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def reference_candidates_for_subject(subject: dict[str, Any], reference_dir_list: list[Path]) -> list[str]:
    subject_id = str(subject.get("subject_id", ""))
    identities = [subject_id]
    reference_character = str(subject.get("reference_character", "") or "")
    if reference_character and reference_character not in identities:
        identities.append(reference_character)
    found: list[str] = []
    for reference_dir in reference_dir_list:
        if not reference_dir.exists():
            continue
        manifest = load_reference_manifest(reference_dir)
        manifest_subjects = manifest.get("subjects", {}) if isinstance(manifest, dict) else {}
        aliases = manifest.get("aliases", {}) if isinstance(manifest, dict) else {}
        if isinstance(aliases, dict):
            alias = aliases.get(subject_id)
            if alias and alias not in identities:
                identities.append(str(alias))
        if isinstance(manifest_subjects, dict):
            for identity in identities:
                for value in manifest_subjects.get(identity, []) or []:
                    path = Path(str(value))
                    if not path.is_absolute():
                        path = reference_dir / path
                    found.append(str(path.resolve()))
        for identity in identities:
            for pattern in (f"{identity}__*", f"{identity}_front_view*", f"{identity}_left_view*"):
                for path in reference_dir.glob(pattern):
                    if path.is_file() and path.suffix.lower() in REFERENCE_IMAGE_SUFFIXES:
                        found.append(str(path.resolve()))
    deduped: list[str] = []
    for path in found:
        if path not in deduped:
            deduped.append(path)
    return deduped


def normalize_input(raw: dict[str, Any], *, input_path: Path, reference_root: Path, story_name: str = "") -> dict[str, Any]:
    base_dir = input_path.parent.resolve()
    resolved_story_name = story_name_from_input(raw, story_name)
    ref_dirs = reference_dirs(reference_root.resolve(), resolved_story_name)
    subjects = []
    for subject in raw.get("subjects", []):
        item = dict(subject)
        item["subject_id"] = str(item.get("subject_id", item.get("id", "")))
        explicit_references = [
            resolve_path(str(path), base_dir=base_dir, workspace_dir=WORKSPACE_DIR)
            for path in item.get("reference_images", [])
            if path
        ]
        discovered_references = reference_candidates_for_subject(item, ref_dirs)
        item["reference_images"] = []
        for path in explicit_references + discovered_references:
            if path not in item["reference_images"]:
                item["reference_images"].append(path)
        subjects.append(item)
    segments = []
    for index, segment in enumerate(raw.get("segments", [])):
        item = dict(segment)
        item["segment_id"] = str(item.get("segment_id") or item.get("id") or f"segment_{index + 1:03d}")
        item["order_index"] = int(item.get("order_index", index))
        item["video_path"] = resolve_path(str(item.get("video_path", "")), base_dir=base_dir, workspace_dir=WORKSPACE_DIR)
        if "expected_subject_ids" not in item:
            subject_id = item.get("primary_subject_id") or item.get("subject_id")
            item["expected_subject_ids"] = [str(subject_id)] if subject_id else []
        item["evaluation_scope"] = str(item.get("evaluation_scope") or ("subject" if item.get("expected_subject_ids") else "scene"))
        segments.append(item)
    segments.sort(key=lambda item: int(item.get("order_index", 0)))
    return {
        "dataset_name": raw.get("dataset_name", ""),
        "method_name": raw.get("method_name", ""),
        "fps": raw.get("fps", 24),
        "subjects": subjects,
        "segments": segments,
        "global_story_prompt": raw.get("global_story_prompt", ""),
        "metadata": raw.get("metadata", {}),
        "reference_story_name": resolved_story_name,
        "reference_dirs": [str(path) for path in ref_dirs],
    }


def validate_input(benchmark: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not benchmark.get("segments"):
        errors.append("No segments provided.")
    subject_ids = {str(subject.get("subject_id", "")) for subject in benchmark.get("subjects", [])}
    for segment in benchmark.get("segments", []):
        for subject_id in segment.get("expected_subject_ids", []):
            if subject_id and subject_id not in subject_ids:
                errors.append(f"Unknown subject id for segment {segment.get('segment_id')}: {subject_id}")
    return errors


def failure_category_for_code(code: str) -> str:
    if code in {"video_missing", "video_empty"}:
        return "generation_failed"
    if code in {"expected_subject_missing", "expected_subject_unknown", "reference_images_missing", "reference_images_unreadable"}:
        return "benchmark_input_failed"
    if code in {"evaluation_exception", "yolo_weights_missing", "yolo_load_failed", "yolo_predict_failed"}:
        return "evaluation_error"
    return "evaluation_error"


def flatten_row(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "segment_id": result.get("segment_id", ""),
        "status": result.get("status", ""),
        "generation_status": result.get("generation_status", ""),
        "evaluation_status": result.get("evaluation_status", ""),
        "failure_category": result.get("failure_category"),
        "failure_reason": result.get("failure_reason"),
        "video_path": result.get("video_path", ""),
        "subject_perception": result.get("dimensions", {}).get("subject_perception"),
        "intent_consistency": result.get("dimensions", {}).get("intent_consistency"),
        "trajectory_quality": result.get("dimensions", {}).get("trajectory_quality"),
        "overall": result.get("overall"),
    }
    row.update(result.get("metrics", {}))
    return row


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]
    generation_failed = [result for result in results if result.get("failure_category") == "generation_failed"]
    rows = [flatten_row(result) for result in successful]
    failure_reasons: dict[str, int] = {}
    failure_categories: dict[str, int] = {}
    for result in results:
        if result.get("status") == "success":
            continue
        reason = str(result.get("failure_reason") or "unknown_failure")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        category = str(result.get("failure_category") or failure_category_for_code(reason))
        failure_categories[category] = failure_categories.get(category, 0) + 1
    segment_count = len(results)
    success_only_overall = clamp_score(score_mean([row.get("overall") for row in rows]))
    success_only_dimensions = {
        "subject_perception": clamp_score(score_mean([row.get("subject_perception") for row in rows])),
        "intent_consistency": clamp_score(score_mean([row.get("intent_consistency") for row in rows])),
        "trajectory_quality": clamp_score(score_mean([row.get("trajectory_quality") for row in rows])),
    }
    success_only_metrics = {name: clamp_score(score_mean([row.get(name) for row in rows])) for name in METRIC_NAMES}

    generation_failed_count = len(generation_failed)
    scoring_denominator = len(successful) + generation_failed_count

    def missing_video_zero_mean(values: list[Any]) -> float:
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        denominator = len(numeric) + generation_failed_count
        if denominator <= 0:
            return 0.0
        return clamp_score(sum(numeric) / denominator)

    def applicable_count(values: list[Any]) -> int:
        return sum(1 for value in values if isinstance(value, (int, float)))

    missing_video_zero_dimensions = {
        "subject_perception": missing_video_zero_mean([row.get("subject_perception") for row in rows]),
        "intent_consistency": missing_video_zero_mean([row.get("intent_consistency") for row in rows]),
        "trajectory_quality": missing_video_zero_mean([row.get("trajectory_quality") for row in rows]),
    }
    missing_video_zero_metrics = {name: missing_video_zero_mean([row.get(name) for row in rows]) for name in METRIC_NAMES}
    return {
        "segment_count": segment_count,
        "generation_successful_segment_count": sum(1 for result in results if result.get("generation_status") == "success"),
        "generation_failed_segment_count": sum(1 for result in results if result.get("generation_status") == "failed"),
        "successful_segment_count": len(successful),
        "failed_segment_count": segment_count - len(successful),
        "evaluation_successful_segment_count": len(successful),
        "evaluation_failed_segment_count": segment_count - len(successful),
        "scored_segment_count": len(successful),
        "unscored_segment_count": segment_count - len(successful) - len(generation_failed),
        "score_denominator": scoring_denominator,
        "score_aggregation": "scored_segments_with_missing_video_zero",
        "overall": missing_video_zero_mean([row.get("overall") for row in rows]),
        "dimensions": missing_video_zero_dimensions,
        "metrics": missing_video_zero_metrics,
        "successful_segment_overall": success_only_overall,
        "successful_segment_dimensions": success_only_dimensions,
        "successful_segment_metrics": success_only_metrics,
        "applicable_dimension_counts": {
            "subject_perception": applicable_count([row.get("subject_perception") for row in rows]),
            "intent_consistency": applicable_count([row.get("intent_consistency") for row in rows]),
            "trajectory_quality": applicable_count([row.get("trajectory_quality") for row in rows]),
        },
        "applicable_metric_counts": {name: applicable_count([row.get(name) for row in rows]) for name in METRIC_NAMES},
        "failure_reasons": failure_reasons,
        "failure_categories": failure_categories,
        "warning_count": sum(len(result.get("warnings", [])) for result in results),
        "vlm_backends": sorted(
            {
                str(result.get("vlm", {}).get("backend", ""))
                for result in results
                if result.get("vlm", {}).get("backend")
            }
        ),
    }


def failed_segment_result(segment: dict[str, Any], *, code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    category = failure_category_for_code(code)
    return {
        "segment_id": segment.get("segment_id", ""),
        "status": "failed",
        "generation_status": "failed" if category == "generation_failed" else "success",
        "evaluation_status": "failed",
        "failure_category": category,
        "failure_reason": code,
        "failure_message": message,
        "failure_details": details or {},
        "video_path": str(segment.get("video_path", "")),
        "video_info": {},
        "metrics": {name: None for name in METRIC_NAMES},
        "dimensions": {"subject_perception": None, "intent_consistency": None, "trajectory_quality": None},
        "overall": None,
        "vlm": {},
        "observations": [],
        "head_frame_path": "",
        "tail_frame_path": "",
        "head_box": None,
        "tail_box": None,
        "warnings": ["segment_failed"],
        "boundary": {"applicable": False, "skipped_reason": "segment_failed"},
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.benchmark_input).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(args.run_id).resolve()
    ensure_directory(output_root)
    raw = load_json(input_path)
    benchmark = normalize_input(raw, input_path=input_path, reference_root=Path(args.reference_root), story_name=args.story_name)
    errors = validate_input(benchmark)
    if errors:
        save_json({"status": "failed", "errors": errors, "created_at": utc_now()}, output_root / "cinestory_report_v1.json")
        raise SystemExit("Invalid benchmark input:\n" + "\n".join(errors))
    runtime_config_path = Path(args.runtime_config).resolve() if args.runtime_config else None
    vlm_client = VisionLanguageClient(runtime_config_path=runtime_config_path, backend=args.vlm_backend, model=args.vlm_model)
    subjects_by_id = {str(subject.get("subject_id", "")): subject for subject in benchmark.get("subjects", [])}
    results: list[dict[str, Any]] = []
    for segment in benchmark["segments"]:
        segment_id = sanitize_for_filename(str(segment.get("segment_id", "")))
        evidence_dir = output_root / "evidence" / segment_id
        video_path = Path(str(segment.get("video_path", "")))
        try:
            if not video_path.exists():
                result = failed_segment_result(
                    segment,
                    code="video_missing",
                    message=f"Video file is missing for segment {segment.get('segment_id', '')}: {video_path}",
                )
            elif video_path.stat().st_size <= 0:
                result = failed_segment_result(
                    segment,
                    code="video_empty",
                    message=f"Video file is empty for segment {segment.get('segment_id', '')}: {video_path}",
                )
            else:
                result = evaluate_segment(
                    segment=segment,
                    subjects_by_id=subjects_by_id,
                    output_dir=evidence_dir,
                    vlm_client=vlm_client,
                    uniform_frames=max(3, int(args.uniform_frames)),
                )
        except SegmentEvaluationError as exc:
            result = failed_segment_result(segment, code=exc.code, message=exc.message, details=exc.details)
        except Exception as exc:
            result = failed_segment_result(
                segment,
                code="evaluation_exception",
                message=f"{exc.__class__.__name__}: {exc}",
                details={"exception_type": exc.__class__.__name__},
            )
        if result.get("status") == "success":
            result["generation_status"] = "success"
            result["evaluation_status"] = "success"
            result["failure_category"] = None
        result["order_index"] = int(segment.get("order_index", len(results)))
        result["intent"] = {
            "prompt_text": segment.get("prompt_text", ""),
            "event_description": segment.get("event_description", ""),
            "expected_subject_ids": segment.get("expected_subject_ids", []),
            "expected_shot_size": segment.get("expected_shot_size", ""),
            "expected_semantic_target": segment.get("expected_semantic_target", ""),
            "expected_camera_angle": segment.get("expected_camera_angle", ""),
            "expected_motion": segment.get("expected_motion", ""),
            "evaluation_scope": segment.get("evaluation_scope", ""),
        }
        results.append(result)
    finalize_boundary_scores(results)
    rows = [flatten_row(result) for result in sorted(results, key=lambda item: int(item.get("order_index", 0)))]
    summary = summarize(results)
    report_status = "success"
    if summary["failed_segment_count"] and summary["successful_segment_count"]:
        report_status = "partial_success"
    elif summary["failed_segment_count"]:
        report_status = "failed"
    report = {
        "schema_version": "cinestory_eval_v1",
        "created_at": utc_now(),
        "status": report_status,
        "benchmark": {
            "dataset_name": benchmark.get("dataset_name", ""),
            "method_name": benchmark.get("method_name", ""),
            "global_story_prompt": benchmark.get("global_story_prompt", ""),
            "metadata": benchmark.get("metadata", {}),
        },
        "config": {
            "uniform_frames": int(args.uniform_frames),
            "vlm_backend_requested": args.vlm_backend,
            "vlm_model": args.vlm_model or vlm_client.model,
            "device": args.device,
            "black_box_inputs_only": True,
            "reference_root": str(Path(args.reference_root).resolve()),
            "reference_story_name": benchmark.get("reference_story_name", ""),
            "reference_dirs": benchmark.get("reference_dirs", []),
        },
        "summary": summary,
        "segments": results,
    }
    save_json(report, output_root / "cinestory_report_v1.json")
    write_jsonl(results, output_root / "segment_scores.jsonl")
    write_summary_csv(rows, output_root / "summary.csv")
    save_json(benchmark, output_root / "normalized_benchmark_input_v1.json")
    return report


def main() -> int:
    args = parse_args()
    report = run_benchmark(args)
    summary = report["summary"]
    print(json.dumps({"status": report["status"], "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
