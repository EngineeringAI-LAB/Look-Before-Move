"""Analyze final preview LLM review results for one or more Cinematographer runs.

Usage:
    python tools/analyze_final_preview_results.py \
        --output-roots Cinematographer/output/the_notebook_afterfix_llm_20260502 \
                       Cinematographer/output/the_imitation_game_afterfix_llm_20260502
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Validation reasons that the OLD code treated as hard block but the NEW code treats as diagnostic-only.
OLD_CODE_DIAGNOSTIC_REASONS = {
    "llm_needs_reshoot",
    "secondary_subject_visible_false",
    "interaction_readable_false",
    "hands_visible_false",
}

# Validation reasons that are always real (camera-controllable) blocking reasons.
CAMERA_BLOCK_REASONS = {
    "primary_subject_visible_false",
    "framing_matches_intent_false",
}

# Validation reasons that indicate non-camera issues that should not block.
NON_CAMERA_REASONS = {
    "framing_non_camera_reason_ignored",
}

NON_CAMERA_KEYWORDS = (
    "environment", "set dressing", "background", "garden", "prop", "lighting",
    "action", "acting", "animation", "pose", "gesture", "facial expression",
    "expression", "emotion", "interaction", "hands", "semantic", "story",
    "narrative", "character model", "asset", "rig", "texture", "material",
    "hair", "clothing", "costume", "weather", "time of day", "skybox",
    "ground", "floor", "furniture", "vehicle", "building",
    "not performing", "does not", "missing object", "missing prop",
    "scene construction", "blender", "render quality", "low resolution",
    "workbench",
)

CAMERA_KEYWORDS = (
    "framing", "composition", "angle", "crop", "cropped", "cut off",
    "occlusion", "occluded", "visible", "visibility", "wrong subject",
    "primary subject", "subject not", "too far", "too close",
    "shot size", "camera", "view", "distance",
)


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_block_reason(reason_text: str, validation_reasons: list[str]) -> dict:
    """Classify a block and determine if it would pass under the new code."""
    text = reason_text.lower()
    vr_set = set(validation_reasons)

    # deterministic quality gate failures are real failures
    if any("deterministic" in r or "quality_no_eligible" in r or "quality_no_retained" in r for r in vr_set):
        return {"category": "deterministic_quality_gate", "would_pass_new_code": False}

    # LLM unavailable → fallback issue
    if any("llm_unavailable" in r for r in vr_set):
        return {"category": "llm_unavailable_fallback", "would_pass_new_code": False}

    # Check if ALL blocking reasons are diagnostic-only (would pass under new code)
    real_block_reasons = vr_set - OLD_CODE_DIAGNOSTIC_REASONS - NON_CAMERA_REASONS
    only_old_diagnostic = bool(vr_set & OLD_CODE_DIAGNOSTIC_REASONS) and not real_block_reasons

    # Classify by reason text
    has_camera = any(kw in text for kw in CAMERA_KEYWORDS)
    has_non_camera = any(kw in text for kw in NON_CAMERA_KEYWORDS)

    if only_old_diagnostic:
        category = "old_code_overly_strict"
    elif real_block_reasons & CAMERA_BLOCK_REASONS and not has_non_camera:
        category = "camera_controllable"
    elif has_non_camera and not has_camera:
        category = "non_camera"
    elif has_non_camera and has_camera:
        category = "mixed_overly_strict"
    elif real_block_reasons:
        category = "camera_controllable"
    else:
        category = "unknown"

    return {
        "category": category,
        "would_pass_new_code": only_old_diagnostic,
        "real_block_reasons": sorted(real_block_reasons) if real_block_reasons else [],
        "diagnostic_only_reasons": sorted(vr_set & OLD_CODE_DIAGNOSTIC_REASONS),
    }


def analyze_one_run(output_root: Path) -> dict:
    outputs_dir = output_root / "outputs"
    run_id = output_root.name

    # --- Load files ---
    quality_report_path = outputs_dir / "camera_quality_report_v1.json"
    llm_report_path = outputs_dir / "llm_selection_report_v1.json"
    handoff_path = outputs_dir / "camera_handoff_v1.json"

    quality_report = load_json(quality_report_path) if quality_report_path.exists() else {}
    llm_report = load_json(llm_report_path) if llm_report_path.exists() else {}
    handoff = load_json(handoff_path) if handoff_path.exists() else {}

    # --- Quality stage ---
    quality_rows = quality_report.get("rows") or []
    total_cameras = len(quality_rows)
    quality_success = sum(1 for r in quality_rows if r.get("success"))
    quality_fail = total_cameras - quality_success
    quality_eligible = sum(1 for r in quality_rows if int(r.get("candidate_count_eligible") or 0) > 0)
    quality_retained = sum(1 for r in quality_rows if int(r.get("candidate_count_retained") or 0) > 0)

    # --- LLM selection ---
    selection_rows = llm_report.get("selection_rows") or []
    llm_selected = sum(1 for r in selection_rows if r.get("selected_candidate_id"))
    llm_failed = total_cameras - llm_selected

    # --- Final preview review ---
    story_rows = llm_report.get("final_preview_story_rows") or llm_report.get("story_consistency_rows") or []
    blocked_names = set(llm_report.get("final_preview_blocked_cameras") or [])
    repaired_names = set(llm_report.get("final_preview_repaired_cameras") or [])

    preview_pass = 0
    preview_block = 0
    preview_repair = 0
    block_details = []

    # Also read selection_rows for detailed LLM review fields
    selection_rows = llm_report.get("selection_rows") or []
    selection_by_cam = {str(r.get("camera_name") or ""): r for r in selection_rows}

    for row in story_rows:
        cam_name = row.get("camera_name") or ""
        eligible = row.get("downstream_eligible", True)
        # story_rows have fields at top level, not nested under "review"
        validation = row.get("validation") or {}
        reason_text = str(row.get("reason") or "")
        validation_reasons = validation.get("reasons") or []
        is_valid = validation.get("valid", True)
        consistency_score = row.get("consistency_score")
        needs_reshoot = row.get("needs_reshoot")

        # Get detailed LLM review from selection_rows
        sel_row = selection_by_cam.get(cam_name) or {}
        llm_review = sel_row.get("final_preview_review") or {}
        primary_subject_visible = llm_review.get("primary_subject_visible")
        framing_matches_intent = llm_review.get("framing_matches_intent")
        secondary_subject_visible = llm_review.get("secondary_subject_visible")
        # Fallback: validation may carry these fields too
        if primary_subject_visible is None:
            primary_subject_visible = validation.get("primary_subject_visible")
        if framing_matches_intent is None:
            framing_matches_intent = validation.get("framing_matches_intent")
        llm_reason = str(llm_review.get("reason") or reason_text or "")

        if cam_name in blocked_names:
            preview_block += 1
            classification = classify_block_reason(llm_reason, validation_reasons)
            block_details.append({
                "camera_name": cam_name,
                "downstream_eligible": eligible,
                "review_success": row.get("success"),
                "primary_subject_visible": primary_subject_visible,
                "framing_matches_intent": framing_matches_intent,
                "secondary_subject_visible": secondary_subject_visible,
                "needs_reshoot": needs_reshoot,
                "consistency_score": consistency_score,
                "reason": llm_reason,
                "validation_reasons": validation_reasons,
                "validation_valid": is_valid,
                "error_class": validation.get("error_class") or "",
                **classification,
            })
        elif cam_name in repaired_names:
            preview_repair += 1
        else:
            preview_pass += 1

    # --- Handoff ---
    handoff_cameras = handoff.get("cameras") or []
    handoff_shots = handoff.get("shots") or []
    handoff_blocked = handoff.get("downstream_blocked_camera_names") or []

    result = {
        "run_id": run_id,
        "output_root": str(output_root),
        "total_cameras": total_cameras,
        "quality_success": quality_success,
        "quality_fail": quality_fail,
        "quality_eligible": quality_eligible,
        "quality_retained": quality_retained,
        "llm_selected": llm_selected,
        "llm_failed": llm_failed,
        "preview_pass": preview_pass,
        "preview_block": preview_block,
        "preview_repair": preview_repair,
        "handoff_camera_count": len(handoff_cameras),
        "handoff_shot_count": len(handoff_shots),
        "handoff_blocked_count": len(handoff_blocked),
        "handoff_blocked_names": sorted(handoff_blocked),
        "block_details": block_details,
        "success_rate_handoff": f"{len(handoff_cameras)}/{total_cameras}" if total_cameras else "N/A",
        "success_rate_pct": round(len(handoff_cameras) / total_cameras * 100, 1) if total_cameras else 0,
    }
    return result


def print_summary(results: list[dict]) -> None:
    total_all = 0
    handoff_all = 0
    block_all = 0
    would_pass_all = 0
    cat_counts = {}

    print("=" * 90)
    print("FINAL PREVIEW LLM REVIEW ANALYSIS REPORT")
    print("=" * 90)

    for r in results:
        total_all += r["total_cameras"]
        handoff_all += r["handoff_camera_count"]

        print(f"\n{'─' * 70}")
        print(f"Dataset: {r['run_id']}")
        print(f"{'─' * 70}")
        print(f"  Total cameras:             {r['total_cameras']}")
        print(f"  Quality stage success:     {r['quality_success']}")
        print(f"  Quality stage fail:        {r['quality_fail']}")
        print(f"  Quality eligible (>0):     {r['quality_eligible']}")
        print(f"  Quality retained (>0):     {r['quality_retained']}")
        print(f"  LLM selected:              {r['llm_selected']}")
        print(f"  LLM failed:                {r['llm_failed']}")
        print(f"  Final preview pass:        {r['preview_pass']}")
        print(f"  Final preview blocked:     {r['preview_block']}")
        print(f"  Final preview repaired:    {r['preview_repair']}")
        print(f"  Handoff cameras:           {r['handoff_camera_count']}")
        print(f"  Handoff shots:             {r['handoff_shot_count']}")
        print(f"  Success rate:              {r['success_rate_handoff']} ({r['success_rate_pct']}%)")

        would_pass_count = 0
        if r["block_details"]:
            print(f"\n  Final preview BLOCKED cameras ({len(r['block_details'])}):")
            for b in r["block_details"]:
                block_all += 1
                cat = b["category"]
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                wp = b.get("would_pass_new_code", False)
                if wp:
                    would_pass_count += 1
                    would_pass_all += 1
                print(f"    {b['camera_name']}:")
                print(f"      category:                {cat}")
                print(f"      would_pass_new_code:     {wp}")
                print(f"      primary_subject_visible: {b['primary_subject_visible']}")
                print(f"      framing_matches_intent:  {b['framing_matches_intent']}")
                print(f"      needs_reshoot:           {b['needs_reshoot']}")
                print(f"      consistency_score:        {b['consistency_score']}")
                print(f"      validation_reasons:       {b['validation_reasons']}")
                print(f"      diagnostic_only_reasons:  {b.get('diagnostic_only_reasons', [])}")
                print(f"      real_block_reasons:       {b.get('real_block_reasons', [])}")
                print(f"      error_class:              {b['error_class']}")
                print(f"      reason:                   {b['reason'][:150]}")
            print(f"\n  Would pass with new code: {would_pass_count}/{len(r['block_details'])}")
            print(f"  Projected handoff with new code: {r['handoff_camera_count'] + would_pass_count}/{r['total_cameras']}")

    projected_all = handoff_all + would_pass_all
    print(f"\n{'=' * 90}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Total cameras (all datasets):  {total_all}")
    print(f"  Total handoff success:         {handoff_all}")
    print(f"  Total failed (not in handoff): {total_all - handoff_all}")
    print(f"  Overall success rate:          {handoff_all}/{total_all} ({round(handoff_all / total_all * 100, 1) if total_all else 0}%)")
    print(f"\n  Would pass with new review code: {would_pass_all}")
    print(f"  Projected success with new code: {projected_all}/{total_all} ({round(projected_all / total_all * 100, 1) if total_all else 0}%)")
    print(f"\n  Block category breakdown ({block_all} total blocks):")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    print(f"\n  Category definitions:")
    print(f"    old_code_overly_strict:     Blocked ONLY by old-code diagnostic reasons (would pass now)")
    print(f"    camera_controllable:        Camera framing/angle/composition issue (true failure)")
    print(f"    non_camera:                 Scene/animation/asset issue, not camera's fault")
    print(f"    mixed_overly_strict:        Reason mentions both camera and non-camera issues")
    print(f"    deterministic_quality_gate: Quality worker found no eligible/retained candidates")
    print(f"    llm_unavailable_fallback:   LLM was unavailable, fell back to deterministic check")
    print(f"    unknown:                    Could not classify from reason text")


def save_json_report(results: list[dict], path: Path) -> None:
    aggregate = {
        "total_cameras": sum(r["total_cameras"] for r in results),
        "total_handoff": sum(r["handoff_camera_count"] for r in results),
        "total_blocked": sum(r["preview_block"] for r in results),
        "overall_success_pct": round(
            sum(r["handoff_camera_count"] for r in results)
            / max(sum(r["total_cameras"] for r in results), 1)
            * 100, 1
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "datasets": results}, f, ensure_ascii=False, indent=2)
    print(f"\nJSON report saved to: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze final preview LLM review results.")
    parser.add_argument(
        "--output-roots", nargs="+", required=True,
        help="One or more Cinematographer output root directories",
    )
    parser.add_argument("--json-output", default="", help="Optional path to save JSON report")
    args = parser.parse_args()

    results = []
    for root_str in args.output_roots:
        root = Path(root_str).resolve()
        if not root.exists():
            print(f"WARNING: {root} does not exist, skipping.", file=sys.stderr)
            continue
        results.append(analyze_one_run(root))

    if not results:
        print("No valid output roots found.", file=sys.stderr)
        return 1

    print_summary(results)

    if args.json_output:
        save_json_report(results, Path(args.json_output))
    else:
        default_json = Path(args.output_roots[0]).resolve().parent.parent.parent / "run_logs" / "afterfix_llm_20260502" / "analysis_results.json"
        default_json.parent.mkdir(parents=True, exist_ok=True)
        save_json_report(results, default_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
