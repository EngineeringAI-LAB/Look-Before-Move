"""Re-run final preview LLM review and regenerate handoff using existing outputs.

Two modes:
  --revalidate-only  : Re-apply validation logic to existing LLM review responses (no API calls).
  --rerun-llm        : Re-run VLM review for each camera using existing preview images (needs API key).

Usage:
    python tools/rerun_final_preview_review.py \
        --output-root Cinematographer/output/the_notebook_afterfix_llm_20260502 \
        --revalidate-only

    python tools/rerun_final_preview_review.py \
        --output-root Cinematographer/output/the_imitation_game_afterfix_llm_20260502 \
        --rerun-llm
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add Cinematographer to sys.path so we can import stage functions
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
CINEMATOGRAPHER_DIR = WORKSPACE_DIR / "Cinematographer"
if str(CINEMATOGRAPHER_DIR) not in sys.path:
    sys.path.insert(0, str(CINEMATOGRAPHER_DIR))

from cinematographer_stage import (
    CinematographerConfig,
    HANDOFF_POLICY_DEFAULT,
    HANDOFF_POLICY_SEMANTIC_ONLY,
    VALID_HANDOFF_POLICIES,
    build_bridge,
    build_downstream_shots,
    classify_review_reason,
    final_preview_review_validity,
    load_json,
    load_runtime_defaults,
    save_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
    shutil.copy2(path, backup)
    print(f"  Backed up: {path.name} -> {backup.name}")
    return backup


def load_camera_packages(output_root: Path) -> list[dict[str, Any]]:
    """Load all camera package JSON files from output_root/camera_packages/."""
    packages_dir = output_root / "camera_packages"
    packages: list[dict[str, Any]] = []
    if not packages_dir.exists():
        print(f"WARNING: camera_packages directory not found at {packages_dir}", file=sys.stderr)
        return packages
    for json_file in sorted(packages_dir.rglob("*.json")):
        try:
            pkg = load_json(json_file)
            if isinstance(pkg, dict) and pkg.get("camera_name"):
                pkg["camera_package_path"] = str(json_file)
                packages.append(pkg)
        except Exception as e:
            print(f"WARNING: Failed to load {json_file}: {e}", file=sys.stderr)
    return packages


def rebuild_shot_outputs(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group camera packages back into shot_outputs structure."""
    shots_map: dict[tuple[int, int], dict[str, Any]] = {}
    for pkg in packages:
        key = (int(pkg.get("scene_id") or 0), int(pkg.get("shot_id") or 0))
        if key not in shots_map:
            shots_map[key] = {
                "scene_id": key[0],
                "shot_id": key[1],
                "scene_description": pkg.get("scene_description"),
                "shot_description": pkg.get("shot_description"),
                "cameras": [],
            }
        shots_map[key]["cameras"].append(pkg)
    # Sort shots by (scene_id, shot_id)
    shot_outputs = []
    for key in sorted(shots_map.keys()):
        shot = shots_map[key]
        cameras = shot["cameras"]
        cameras.sort(key=lambda c: int(c.get("camera_index") or 0))
        bridges = [build_bridge(cameras[i], cameras[i + 1]) for i in range(len(cameras) - 1)]
        for i, cam in enumerate(cameras):
            cam["bridge_to_next"] = bridges[i] if i < len(bridges) else None
        shot["bridge_trajectories"] = bridges
        shot_outputs.append(shot)
    return shot_outputs


def revalidate_camera(camera: dict[str, Any], *, handoff_policy: str = HANDOFF_POLICY_DEFAULT) -> dict[str, Any]:
    """Re-apply final_preview_review_validity to existing LLM review data."""
    review = camera.get("final_preview_llm_review") or {}
    if not review:
        return {
            "camera_name": camera.get("camera_name"),
            "old_eligible": camera.get("downstream_eligible"),
            "new_eligible": camera.get("downstream_eligible"),
            "status": "no_review_data",
            "verdict": "unknown",
            "validation": {},
        }
    validation = final_preview_review_validity(camera, review, handoff_policy=handoff_policy)
    old_eligible = camera.get("downstream_eligible")
    new_eligible = bool(validation.get("valid"))
    camera["downstream_eligible"] = new_eligible
    review["validation"] = validation
    camera["final_preview_llm_review"] = review
    package_path = Path(str(camera.get("camera_package_path") or ""))
    if package_path.exists():
        save_json(camera, package_path)
    return {
        "camera_name": camera.get("camera_name"),
        "old_eligible": old_eligible,
        "new_eligible": new_eligible,
        "status": "changed" if old_eligible != new_eligible else "unchanged",
        "verdict": validation.get("verdict") or "unknown",
        "warnings": validation.get("warnings") or [],
        "reasons": validation.get("reasons") or [],
        "validation": validation,
    }


def rerun_llm_camera(camera: dict[str, Any], config: CinematographerConfig, *, handoff_policy: str = HANDOFF_POLICY_DEFAULT) -> dict[str, Any]:
    """Re-run VLM review for a camera using existing preview image."""
    from cinematographer_stage import llm_story_consistency_judge

    camera_name = str(camera.get("camera_name") or "")
    preview_path = str(camera.get("preview_frame_path") or camera.get("final_preview_path") or "")
    if not preview_path or not Path(preview_path).exists():
        return {
            "camera_name": camera_name,
            "old_eligible": camera.get("downstream_eligible"),
            "new_eligible": camera.get("downstream_eligible"),
            "status": "no_preview_image",
            "verdict": "unknown",
            "validation": {},
        }
    review = llm_story_consistency_judge(
        camera_package=camera,
        config=config,
        preview_path=preview_path,
        allow_candidate_preview_fallback=True,
        review_source="rerun_final_preview",
    )
    validation = final_preview_review_validity(camera, review, handoff_policy=handoff_policy)
    old_eligible = camera.get("downstream_eligible")
    new_eligible = bool(validation.get("valid"))
    camera["downstream_eligible"] = new_eligible
    review["validation"] = validation
    camera["final_preview_llm_review"] = review
    package_path = Path(str(camera.get("camera_package_path") or ""))
    if package_path.exists():
        save_json(camera, package_path)
    return {
        "camera_name": camera_name,
        "old_eligible": old_eligible,
        "new_eligible": new_eligible,
        "status": "changed" if old_eligible != new_eligible else "unchanged",
        "verdict": validation.get("verdict") or "unknown",
        "warnings": validation.get("warnings") or [],
        "reasons": validation.get("reasons") or [],
        "validation": validation,
        "review": review,
    }


def regenerate_handoff(
    shot_outputs: list[dict[str, Any]],
    output_root: Path,
    review_results: list[dict[str, Any]],
    original_handoff: dict[str, Any],
) -> Path:
    """Regenerate camera_handoff_v1.json from updated shot_outputs."""
    outputs_dir = output_root / "outputs"
    downstream_shots = build_downstream_shots(shot_outputs)
    downstream_cameras = [camera for shot in downstream_shots for camera in shot["cameras"]]
    blocked_names = sorted(
        str(camera.get("camera_name") or "")
        for shot in shot_outputs
        for camera in shot.get("cameras") or []
        if not camera.get("downstream_eligible", True)
    )
    handoff = {
        "schema_version": "storyblender.camera_handoff.v1",
        "generated_at": utc_now(),
        "run_id": original_handoff.get("run_id") or output_root.name,
        "director_handoff_path": original_handoff.get("director_handoff_path") or "",
        "fps": original_handoff.get("fps") or 24,
        "frame_size": original_handoff.get("frame_size") or {},
        "plate_size": original_handoff.get("plate_size") or {},
        "camera_quality": original_handoff.get("camera_quality") or "quality",
        "quality_report_path": original_handoff.get("quality_report_path") or "",
        "quality_summary": original_handoff.get("quality_summary") or {},
        "downstream_blocked_camera_names": blocked_names,
        "shots": downstream_shots,
        "cameras": downstream_cameras,
        "rerun_review_meta": {
            "rerun_at": utc_now(),
            "total_cameras": sum(len(shot.get("cameras") or []) for shot in shot_outputs),
            "handoff_cameras": len(downstream_cameras),
            "blocked_cameras": len(blocked_names),
            "changed_cameras": [r["camera_name"] for r in review_results if r["status"] == "changed"],
        },
    }
    handoff_path = save_json(handoff, outputs_dir / "camera_handoff_v1.json")
    return handoff_path


def update_llm_report(
    output_root: Path,
    shot_outputs: list[dict[str, Any]],
    review_results: list[dict[str, Any]],
) -> Path | None:
    """Update llm_selection_report_v1.json with new review results."""
    outputs_dir = output_root / "outputs"
    llm_report_path = outputs_dir / "llm_selection_report_v1.json"
    if not llm_report_path.exists():
        return None
    report = load_json(llm_report_path)
    all_cameras = [camera for shot in shot_outputs for camera in shot.get("cameras") or []]
    blocked_names = [
        str(camera.get("camera_name") or "")
        for camera in all_cameras
        if not camera.get("downstream_eligible", True)
    ]
    # Update selection_rows
    sel_rows = report.get("selection_rows") or []
    sel_by_name = {str(r.get("camera_name") or ""): r for r in sel_rows}
    for camera in all_cameras:
        cam_name = str(camera.get("camera_name") or "")
        row = sel_by_name.get(cam_name)
        if row is None:
            continue
        row["final_preview_review"] = camera.get("final_preview_llm_review") or {}
        row["downstream_eligible"] = bool(camera.get("downstream_eligible", True))
    # Update story rows
    story_rows = []
    for camera in all_cameras:
        review = dict(camera.get("final_preview_llm_review") or {})
        validation = dict((review.get("validation") or {}))
        story_rows.append({
            "camera_name": camera.get("camera_name"),
            "success": bool(validation.get("valid")),
            "verdict": validation.get("verdict") or ("passed_clean" if validation.get("valid") else "hard_blocked_camera_issue"),
            "preview_path": camera.get("preview_frame_path") or "",
            "review_source": review.get("review_source") or "",
            "consistency_score": review.get("consistency_score"),
            "needs_reshoot": review.get("needs_reshoot"),
            "reason": review.get("reason"),
            "camera_issue": review.get("camera_issue"),
            "non_camera_issue": review.get("non_camera_issue"),
            "hard_block_camera_reason": review.get("hard_block_camera_reason"),
            "validation": validation,
            "warnings": validation.get("warnings") or [],
            "downstream_eligible": bool(camera.get("downstream_eligible", True)),
            "final_preview_repair_trace": camera.get("final_preview_repair_trace") or [],
        })
    report["final_preview_story_rows"] = story_rows
    report["story_consistency_rows"] = story_rows
    report["final_preview_blocked_cameras"] = sorted(set(blocked_names))
    report["rerun_review_meta"] = {
        "rerun_at": utc_now(),
        "changed_cameras": [r["camera_name"] for r in review_results if r["status"] == "changed"],
    }
    save_json(report, llm_report_path)
    return llm_report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run final preview LLM review and regenerate handoff.")
    parser.add_argument("--output-root", required=True, help="Cinematographer output directory")
    parser.add_argument("--run-log-dir", default="", help="Directory to save per-run result JSON for reporting")
    parser.add_argument(
        "--handoff-policy",
        choices=list(VALID_HANDOFF_POLICIES),
        default=HANDOFF_POLICY_DEFAULT,
        help=(
            f"Handoff eligibility policy. "
            f"'{HANDOFF_POLICY_DEFAULT}': block on severe camera issues. "
            f"'{HANDOFF_POLICY_SEMANTIC_ONLY}': block only when primary subject missing."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--revalidate-only", action="store_true", help="Re-apply validation logic only (no API calls)")
    mode.add_argument("--rerun-llm", action="store_true", help="Re-run VLM review (needs API key)")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    outputs_dir = output_root / "outputs"
    if not outputs_dir.exists():
        print(f"ERROR: outputs directory not found at {outputs_dir}", file=sys.stderr)
        return 1

    # --- Load existing data ---
    print(f"Loading camera packages from {output_root / 'camera_packages'}...")
    packages = load_camera_packages(output_root)
    if not packages:
        print("ERROR: No camera packages found.", file=sys.stderr)
        return 1
    print(f"  Loaded {len(packages)} camera packages.")

    print("Rebuilding shot_outputs structure...")
    shot_outputs = rebuild_shot_outputs(packages)
    print(f"  Rebuilt {len(shot_outputs)} shots.")

    # --- Load original handoff for metadata ---
    handoff_path = outputs_dir / "camera_handoff_v1.json"
    original_handoff = load_json(handoff_path) if handoff_path.exists() else {}
    original_handoff_count = len(original_handoff.get("cameras") or [])

    # --- Backup existing files ---
    print("\nBacking up existing files...")
    backup_file(handoff_path)
    backup_file(outputs_dir / "llm_selection_report_v1.json")

    # --- Run review ---
    print(f"\nRunning review ({'revalidate-only' if args.revalidate_only else 'rerun-llm'})...")
    review_results = []
    all_cameras = [camera for shot in shot_outputs for camera in shot.get("cameras") or []]

    handoff_policy = args.handoff_policy
    print(f"  Handoff policy: {handoff_policy}")

    if args.revalidate_only:
        for camera in all_cameras:
            result = revalidate_camera(camera, handoff_policy=handoff_policy)
            review_results.append(result)
            status_icon = "+" if result["new_eligible"] and not result["old_eligible"] else (
                "-" if not result["new_eligible"] and result["old_eligible"] else "="
            )
            print(f"  [{status_icon}] {result['camera_name']}: {result['old_eligible']} -> {result['new_eligible']} ({result['status']})")
    else:
        runtime_defaults = load_runtime_defaults("cinematographer")
        config = CinematographerConfig(
            director_handoff_path=str(original_handoff.get("director_handoff_path") or ""),
            output_root=str(output_root),
            vision_model=str(runtime_defaults.get("vision_model") or "gemini-3-flash-preview"),
            anyllm_api_key=str(runtime_defaults.get("anyllm_api_key") or ""),
            anyllm_api_base=str(runtime_defaults.get("anyllm_api_base") or "https://yunwu.ai"),
            anyllm_provider=str(runtime_defaults.get("anyllm_provider") or "gemini"),
        )
        if not config.anyllm_api_key:
            print("ERROR: No API key found. Set ANYLLM_API_KEY or configure config/runtime_config.json.", file=sys.stderr)
            return 1
        for camera in all_cameras:
            cam_name = str(camera.get("camera_name") or "")
            print(f"  Reviewing {cam_name}...", end=" ", flush=True)
            result = rerun_llm_camera(camera, config, handoff_policy=handoff_policy)
            review_results.append(result)
            status_icon = "+" if result["new_eligible"] and not result["old_eligible"] else (
                "-" if not result["new_eligible"] and result["old_eligible"] else "="
            )
            print(f"[{status_icon}] {result['old_eligible']} -> {result['new_eligible']}")

    # --- Regenerate handoff ---
    print("\nRegenerating handoff...")
    new_handoff_path = regenerate_handoff(shot_outputs, output_root, review_results, original_handoff)
    new_handoff = load_json(new_handoff_path)
    new_handoff_count = len(new_handoff.get("cameras") or [])
    print(f"  Handoff saved to: {new_handoff_path}")

    # --- Update LLM report ---
    print("Updating LLM selection report...")
    llm_report_path = update_llm_report(output_root, shot_outputs, review_results)
    if llm_report_path:
        print(f"  Report saved to: {llm_report_path}")

    # --- Summary ---
    changed = [r for r in review_results if r["status"] == "changed"]
    newly_passed = [r for r in changed if r["new_eligible"]]
    newly_blocked = [r for r in changed if not r["new_eligible"]]

    # Verdict breakdown
    verdict_counts: dict[str, int] = {}
    for r in review_results:
        v = r.get("verdict") or "unknown"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    print(f"\n{'=' * 60}")
    print("RERUN SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total cameras:            {len(all_cameras)}")
    print(f"  Original handoff:         {original_handoff_count}")
    print(f"  New handoff:              {new_handoff_count}")
    print(f"  Delta:                    {new_handoff_count - original_handoff_count:+d}")
    print(f"  Changed cameras:          {len(changed)}")
    print(f"  Newly passed:             {len(newly_passed)}")
    if newly_passed:
        for r in newly_passed:
            print(f"    + {r['camera_name']} ({r.get('verdict', '?')})")
    print(f"  Newly blocked:            {len(newly_blocked)}")
    if newly_blocked:
        for r in newly_blocked:
            print(f"    - {r['camera_name']} ({r.get('verdict', '?')})")
    print(f"  Still blocked:            {len([r for r in review_results if not r['new_eligible'] and not r['old_eligible']])}")
    print(f"  Still passing:            {len([r for r in review_results if r['new_eligible'] and r['old_eligible']])}")
    print(f"\n  Verdict breakdown:")
    for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"    {v}: {c}")
    print(f"\n  Success rate: {new_handoff_count}/{len(all_cameras)} ({round(new_handoff_count / len(all_cameras) * 100, 1)}%)")

    # --- Save per-result JSON for report generation ---
    run_log_dir = Path(args.run_log_dir) if args.run_log_dir else None
    if run_log_dir:
        run_log_dir.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "output_root": str(output_root),
            "run_id": output_root.name,
            "mode": "revalidate-only" if args.revalidate_only else "rerun-llm",
            "total_cameras": len(all_cameras),
            "original_handoff": original_handoff_count,
            "new_handoff": new_handoff_count,
            "delta": new_handoff_count - original_handoff_count,
            "verdict_counts": verdict_counts,
            "newly_passed": [{"camera_name": r["camera_name"], "verdict": r.get("verdict")} for r in newly_passed],
            "newly_blocked": [{"camera_name": r["camera_name"], "verdict": r.get("verdict")} for r in newly_blocked],
            "per_camera": [
                {
                    "camera_name": r["camera_name"],
                    "old_eligible": r["old_eligible"],
                    "new_eligible": r["new_eligible"],
                    "status": r["status"],
                    "verdict": r.get("verdict"),
                    "warnings": r.get("warnings", []),
                    "reasons": r.get("reasons", []),
                }
                for r in review_results
            ],
        }
        results_path = run_log_dir / f"rerun_results_{output_root.name}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
