from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


PROTECTED_REASON_PREFIX = "deterministic_fallback_closeup_semantic_weighted"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def norm_path(value: Any) -> str:
    return str(value or "").replace("/", "\\").lower()


def candidate_id(candidate: dict[str, Any] | None) -> str:
    return str((candidate or {}).get("candidate_id") or "").strip()


def vector_close(a: Any, b: Any, tol: float = 1e-3) -> bool | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return None
    try:
        return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))
    except (TypeError, ValueError):
        return None


def scalar_close(a: Any, b: Any, tol: float = 1e-3) -> bool | None:
    if a is None or b is None:
        return None
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return None


def iter_handoff_cameras(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [camera for shot in payload.get("shots") or [] for camera in shot.get("cameras") or []]


def package_map(output_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in (output_root / "camera_packages").rglob("*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        name = str(data.get("camera_name") or path.stem)
        if name:
            result[name] = data
    return result


def quality_maps(output_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    quality = load_json(output_root / "outputs" / "camera_quality_report_v1.json")
    by_camera = {str(row.get("camera_name") or ""): row for row in quality.get("rows") or [] if row.get("camera_name")}
    rejection_by_camera: dict[str, dict[str, Any]] = {}
    for row in by_camera.values():
        path = Path(str(row.get("rejection_report_path") or ""))
        if path.exists():
            try:
                rejection_by_camera[str(row.get("camera_name"))] = load_json(path)
            except Exception:
                pass
    return by_camera, rejection_by_camera


def selection_map(output_root: Path) -> dict[str, dict[str, Any]]:
    report = load_json(output_root / "outputs" / "llm_selection_report_v1.json")
    return {str(row.get("camera_name") or ""): row for row in report.get("selection_rows") or [] if row.get("camera_name")}


def preview_maps(output_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for name in ("camera_preview_report_v1.json", "camera_preview_repair_report_v1.json"):
        report = load_json(output_root / "outputs" / name)
        for row in report.get("results") or []:
            row = dict(row)
            row["report_name"] = name
            path_key = norm_path(row.get("preview_path"))
            camera_name = str(row.get("camera_name") or "")
            if path_key:
                by_path[path_key] = row
            if camera_name:
                by_camera.setdefault(camera_name, []).append(row)
    return by_path, by_camera


def strip_json_fence(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_llm_raw_choice(raw: str) -> tuple[str, str]:
    if not raw:
        return "", ""
    try:
        parsed = json.loads(strip_json_fence(raw))
    except Exception:
        return "", "unparseable"
    if not isinstance(parsed, dict):
        return "", "non_object"
    for key in ("selected_candidate_id", "candidate_id"):
        value = str(parsed.get(key) or "").strip()
        if value:
            return value, key
    camera_name = str(parsed.get("camera_name") or "").strip()
    camera_index = str(parsed.get("camera_index") or "").strip()
    if camera_name:
        return camera_name, "camera_name"
    if camera_index:
        return camera_index, "camera_index"
    return "", "missing"


def trace_candidate_ids(trace: list[dict[str, Any]]) -> tuple[str, str, str]:
    before = ""
    after = ""
    reason = ""
    if trace:
        first = trace[0] or {}
        before = str(first.get("candidate_id") or first.get("replacement_candidate_id") or "")
        for item in trace:
            cid = str(item.get("candidate_id") or item.get("replacement_candidate_id") or "")
            if cid:
                after = cid
            failure = item.get("failure_reasons")
            if failure:
                reason = "|".join(str(x) for x in failure)
            if item.get("reason"):
                reason = str(item.get("reason"))
    return before, after, reason


def candidate_membership(final_id: str, quality_row: dict[str, Any], rejection_row: dict[str, Any]) -> dict[str, bool]:
    raw_candidates = rejection_row.get("candidates") or []
    top_candidates = quality_row.get("top_candidates") or []
    raw = False
    retained = False
    for candidate in raw_candidates:
        if str(candidate.get("candidate_id") or "") == final_id:
            raw = True
            retained = bool(candidate.get("retained"))
            break
    dedup = any(candidate_id(candidate) == final_id for candidate in top_candidates)
    selected = candidate_id(quality_row.get("selected_candidate") or {}) == final_id
    return {
        "raw_contains_final": raw,
        "retained_contains_final": retained,
        "dedup_contains_final": dedup,
        "quality_selected_is_final": selected,
    }


def transform_consistency(camera: dict[str, Any], final_candidate: dict[str, Any]) -> dict[str, Any]:
    start = camera.get("start_transform") or {}
    lens = camera.get("lens_mm", start.get("lens_mm"))
    location_match = vector_close(final_candidate.get("location"), start.get("location"))
    rotation_match = vector_close(final_candidate.get("rotation_euler"), start.get("rotation_euler"))
    lens_match = scalar_close(final_candidate.get("lens_mm"), lens)
    target_match = vector_close(final_candidate.get("target"), start.get("target"))
    checks = [value for value in (location_match, rotation_match, lens_match) if value is not None]
    return {
        "transform_candidate_location_match": location_match,
        "transform_candidate_rotation_match": rotation_match,
        "transform_candidate_lens_match": lens_match,
        "transform_candidate_target_match": target_match,
        "transform_candidate_match": bool(checks and all(checks)),
    }


def audit_output_root(output_root: Path) -> list[dict[str, Any]]:
    dataset = output_root.name
    quality_by_camera, rejection_by_camera = quality_maps(output_root)
    selections = selection_map(output_root)
    packages = package_map(output_root)
    preview_by_path, preview_by_camera = preview_maps(output_root)
    final_handoff = load_json(output_root / "outputs" / "camera_handoff_v1.json")
    final_by_camera = {str(c.get("camera_name") or ""): c for c in iter_handoff_cameras(final_handoff)}
    names = sorted(set(quality_by_camera) | set(selections) | set(packages) | set(final_by_camera))
    rows: list[dict[str, Any]] = []

    for name in names:
        quality = quality_by_camera.get(name) or {}
        selection = selections.get(name) or {}
        camera = packages.get(name) or final_by_camera.get(name) or {}
        final_candidate = camera.get("selected_candidate") or {}
        final_id = candidate_id(final_candidate)
        quality_selected = quality.get("selected_candidate") or {}
        quality_id = candidate_id(quality_selected)
        decision = selection.get("board_selection_decision") or selection.get("selection_decision") or camera.get("board_selection_decision") or {}
        llm_raw_choice, llm_raw_choice_field = parse_llm_raw_choice(str(decision.get("raw_response") or ""))
        parsed_choice = str(decision.get("selected_candidate_id") or "").strip()
        fallback_choice = str(decision.get("fallback_candidate_id") or "").strip()
        board_history = selection.get("llm_board_history") or camera.get("llm_board_history") or []
        board_ids: list[str] = []
        for item in board_history:
            for cid in item.get("candidate_ids") or []:
                if cid and cid not in board_ids:
                    board_ids.append(str(cid))
        for cid in decision.get("board_candidate_ids") or decision.get("valid_candidate_ids") or []:
            if cid and cid not in board_ids:
                board_ids.append(str(cid))
        board_candidate = parsed_choice if parsed_choice in board_ids else fallback_choice
        if not board_candidate and selection.get("selected_candidate_id") in board_ids:
            board_candidate = str(selection.get("selected_candidate_id") or "")

        repair_trace = camera.get("final_preview_repair_trace") or selection.get("final_preview_repair_trace") or []
        repair_before, repair_after, repair_reason = trace_candidate_ids(repair_trace)
        selected_before_repair = repair_before or str(selection.get("selected_candidate_id") or "")
        micro_trace = camera.get("llm_micro_adjustment_trace") or selection.get("micro_adjustment_trace") or {}
        micro_rounds = micro_trace.get("rounds") or []
        micro_before = ""
        micro_after = ""
        if micro_rounds:
            micro_before = str((micro_rounds[0] or {}).get("candidate_id_before") or "")
            for item in micro_rounds:
                cid = str(item.get("candidate_id_after") or "")
                if cid:
                    micro_after = cid
        if not micro_before:
            micro_before = fallback_choice or parsed_choice or board_candidate
        if not micro_after:
            micro_after = selected_before_repair or micro_before

        preview_path = camera.get("preview_frame_path") or camera.get("final_preview_path") or selection.get("final_preview_path")
        preview_row = preview_by_path.get(norm_path(preview_path))
        if preview_row is None:
            previews = preview_by_camera.get(name) or []
            preview_row = previews[-1] if previews else {}
        preview_candidate_id = str((preview_row or {}).get("selected_candidate_id") or "")
        final_render_source_id = str(camera.get("final_render_source_candidate_id") or "")
        if not final_render_source_id:
            final_render_source_id = final_id
        preview_candidate_match = bool(
            final_id
            and preview_candidate_id == final_id
            and final_render_source_id == final_id
        )

        replacement_happened = bool(final_id and selected_before_repair and final_id != selected_before_repair)
        diversity_trace = camera.get("diversity_reselection_trace") or selection.get("diversity_reselection_trace") or []
        continuity = camera.get("continuity_review") or {}
        replacement_reason_parts = []
        if repair_reason:
            replacement_reason_parts.append(f"final_preview_repair:{repair_reason}")
        if camera.get("selection_source"):
            replacement_reason_parts.append(f"selection_source:{camera.get('selection_source')}")
        if camera.get("selection_degrade_reason"):
            replacement_reason_parts.append(f"selection_degrade_reason:{camera.get('selection_degrade_reason')}")
        if diversity_trace:
            replacement_reason_parts.append(f"diversity_trace:{len(diversity_trace)}")
        if continuity.get("reselected_for_continuity"):
            replacement_reason_parts.append("continuity_reselected")
        replacement_reason = "; ".join(replacement_reason_parts)

        membership = candidate_membership(final_id, quality, rejection_by_camera.get(name) or {})
        transform = transform_consistency(camera, final_candidate) if final_candidate else {
            "transform_candidate_location_match": None,
            "transform_candidate_rotation_match": None,
            "transform_candidate_lens_match": None,
            "transform_candidate_target_match": None,
            "transform_candidate_match": False,
        }

        protected_expected = str(quality.get("selection_reason") or "").startswith(PROTECTED_REASON_PREFIX)
        protected_id = str(camera.get("protected_semantic_seed_candidate_id") or quality_id if protected_expected else "")
        protected_active = bool(camera.get("protected_semantic_seed_active") or decision.get("protected_semantic_seed_active"))
        final_in_board = bool(final_id and final_id in board_ids)
        unknown_fallback_wrong = bool(
            decision.get("warning") == "llm_selected_unknown_candidate"
            and fallback_choice
            and board_ids
            and fallback_choice not in board_ids
        )

        if not quality.get("success") and not final_id:
            status = "quality_failed_no_final_candidate"
        elif protected_expected and final_id == protected_id and preview_candidate_match:
            status = "protected_seed_ok"
        elif protected_expected and final_id != protected_id:
            status = "protected_seed_lost"
        elif unknown_fallback_wrong:
            status = "bug_unknown_fallback_not_on_board"
        elif not preview_candidate_match:
            status = "identity_mismatch"
        elif replacement_happened and not replacement_reason:
            status = "bug_silent_replacement"
        elif replacement_happened:
            status = "ok_replaced_with_trace"
        elif final_id and board_ids and not final_in_board and not replacement_reason:
            status = "bug_final_not_in_board_without_trace"
        else:
            status = "ok"

        rows.append(
            {
                "dataset": dataset,
                "scene_id": camera.get("scene_id") or quality.get("scene_id"),
                "shot_id": camera.get("shot_id") or quality.get("shot_id"),
                "camera_name": name,
                "quality_selected_candidate_id": quality_id,
                "quality_selected_channel": (quality_selected or {}).get("channel") or "",
                "quality_selected_source": (quality_selected or {}).get("source") or "",
                "quality_selected_lens_mm": (quality_selected or {}).get("lens_mm"),
                "quality_selected_anchor_kind": (quality_selected or {}).get("anchor_kind") or "",
                "candidate_count_raw": quality.get("candidate_count_raw"),
                "candidate_count_retained": quality.get("candidate_count_retained"),
                "candidate_count_deduplicated": quality.get("candidate_count_deduplicated"),
                "raw_contains_final": membership["raw_contains_final"],
                "retained_contains_final": membership["retained_contains_final"],
                "dedup_contains_final": membership["dedup_contains_final"],
                "llm_raw_choice": llm_raw_choice,
                "llm_raw_choice_field": llm_raw_choice_field,
                "parsed_choice": parsed_choice,
                "unknown_fallback_candidate": fallback_choice,
                "board_candidate": board_candidate,
                "board_candidate_count": len(board_ids),
                "final_in_llm_board": final_in_board,
                "micro_adjust_before": micro_before,
                "micro_adjust_after": micro_after,
                "selected_before_repair": selected_before_repair,
                "final_preview_repair_after": repair_after,
                "final_candidate": final_id,
                "final_channel": final_candidate.get("channel") or "",
                "final_source": final_candidate.get("source") or "",
                "final_lens_mm": final_candidate.get("lens_mm"),
                "final_anchor_kind": final_candidate.get("anchor_kind") or "",
                "final_closeup_strategy": final_candidate.get("closeup_strategy") or "",
                "final_render_source_candidate_id": final_render_source_id,
                "preview_report_candidate_id": preview_candidate_id,
                "preview_path": preview_path or "",
                "preview_candidate_match": preview_candidate_match,
                **transform,
                "replacement_happened": replacement_happened,
                "replacement_reason": replacement_reason,
                "protected_expected": protected_expected,
                "protected_active": protected_active,
                "protected_candidate_id": protected_id,
                "downstream_eligible": bool(camera.get("downstream_eligible", True)),
                "status": status,
            }
        )
    return rows


TABLE_COLUMNS = [
    "dataset",
    "scene_id",
    "camera_name",
    "llm_raw_choice",
    "parsed_choice",
    "board_candidate",
    "selected_before_repair",
    "final_candidate",
    "preview_candidate_match",
    "replacement_happened",
    "replacement_reason",
    "status",
]


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_identity_audit_v1.json").write_text(
        json.dumps({"rows": rows, "total": len(rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        columns = list(rows[0].keys())
        with (output_dir / "candidate_identity_audit_v1.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Candidate Identity Audit",
        "",
        f"Total cameras: {len(rows)}",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ") for col in TABLE_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "candidate_identity_audit_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for root in args.output_roots:
        rows.extend(audit_output_root(Path(root)))
    write_outputs(rows, Path(args.output_dir))
    summary: dict[str, int] = {}
    for row in rows:
        summary[str(row.get("status") or "")] = summary.get(str(row.get("status") or ""), 0) + 1
    print(json.dumps({"total": len(rows), "status_counts": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
