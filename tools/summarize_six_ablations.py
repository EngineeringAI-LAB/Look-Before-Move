"""Aggregate 6 (Fast + Quality + 4 ablations) cinestory_eval reports into a
unified set of summary artefacts.

Outputs (under run_logs/quality_and_ablations_20260502):
  - final_story_summary.{md,csv,json}
  - ablation_metric_summary.{md,csv,json}
  - per_setting_stage_summary.json
  - failed_cases.csv
  - ablation_table_latex.tex
  - ablation_analysis_draft.md
  - patch_summary.md (only stub generated here; manual edit later)
"""
from __future__ import annotations

import csv
import json
import math
import argparse
import os
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_OUTPUT_ROOT = Path(os.environ.get("LBM_EVAL_OUTPUT_ROOT", str(REPO_ROOT / "Evaluation" / "output"))).resolve()
RUN_LOG_ROOT = Path(os.environ.get("LBM_RUN_LOG_ROOT", str(REPO_ROOT / "run_logs" / "quality_and_ablations_20260502"))).resolve()
FIXED_STORIES_JSON = RUN_LOG_ROOT / "fixed_six_datasets.json"

# Display order for the final ablation table.
SETTINGS = OrderedDict(
    [
        ("quality", {"label": "Ours / Quality", "run_id": "full_scripts_except_demo_quality_20260502"}),
        ("fast", {"label": "w/o Multi-level Search (Fast Mode)", "run_id": "full_scripts_except_demo_fast_20260502"}),
        ("wo_vlm_reflection", {"label": "w/o VLM Reflection", "run_id": "full_scripts_except_demo_wo_vlm_reflection_20260502"}),
        ("wo_trajectory_grounding", {"label": "w/o Trajectory Grounding", "run_id": "full_scripts_except_demo_wo_trajectory_grounding_20260502"}),
        ("wo_semantic_height_adjust", {"label": "w/o Semantic Height Adjust", "run_id": "full_scripts_except_demo_wo_semantic_height_adjust_20260502"}),
        ("wo_pre_continuity_story_judge", {"label": "w/o Pre-Continuity Story Judge", "run_id": "full_scripts_except_demo_wo_pre_continuity_story_judge_20260502"}),
    ]
)


def run_id_for_ablation(setting_key: str, run_tag: str | None = None) -> str:
    if not run_tag:
        return SETTINGS[setting_key]["run_id"]
    base = {
        "fast": "full_scripts_except_demo_fast",
        "quality": "full_scripts_except_demo_quality",
        "wo_vlm_reflection": "full_scripts_except_demo_wo_vlm_reflection",
        "wo_trajectory_grounding": "full_scripts_except_demo_wo_trajectory_grounding",
        "wo_semantic_height_adjust": "full_scripts_except_demo_wo_semantic_height_adjust",
        "wo_pre_continuity_story_judge": "full_scripts_except_demo_wo_pre_continuity_story_judge",
    }[setting_key]
    return f"{base}_{run_tag}"


def apply_run_tag(run_tag: str) -> None:
    if not run_tag:
        return
    for setting_key, info in SETTINGS.items():
        info["run_id"] = run_id_for_ablation(setting_key, run_tag)

DIM_KEYS = (
    "Subject Perception",
    "Intent Consistency",
    "Trajectory Quality",
)
METRIC_KEYS = (
    "SP1 coverage",
    "SP2 identity",
    "SP3 occlusion",
    "IC1 shot size",
    "IC2 semantic target",
    "IC3 event alignment",
    "TQ1 smoothness",
    "TQ2 tracking",
    "TQ3 continuity",
)
ALL_NUM_KEYS = DIM_KEYS + METRIC_KEYS


def _safe_float(value):
    try:
        f = float(value)
        if math.isfinite(f):
            return f
        return None
    except (TypeError, ValueError):
        return None


def _fmt(value, digits=3):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def load_fixed_stories() -> list[str]:
    if not FIXED_STORIES_JSON.exists():
        return []
    data = json.loads(FIXED_STORIES_JSON.read_text(encoding="utf-8"))
    return [
        item["story"]
        for item in data
        if isinstance(item, dict) and item.get("selected_for_run") and item.get("story")
    ]


def parse_cinestory_report(path: Path) -> dict:
    out = {
        "exists": False,
        "successful_segment_count": 0,
        "failed_segment_count": 0,
        "metrics": {key: None for key in ALL_NUM_KEYS},
        "failure_reasons": [],
    }
    if not path.exists():
        return out
    out["exists"] = True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["failure_reasons"].append(f"report_parse_failed:{exc}")
        return out
    summary = payload.get("summary") or {}
    out["successful_segment_count"] = int(summary.get("successful_segment_count") or 0)
    out["failed_segment_count"] = int(summary.get("failed_segment_count") or 0)

    dims = summary.get("dimensions") or {}
    out["metrics"]["Subject Perception"] = _safe_float(dims.get("subject_perception"))
    out["metrics"]["Intent Consistency"] = _safe_float(dims.get("intent_consistency"))
    out["metrics"]["Trajectory Quality"] = _safe_float(dims.get("trajectory_quality"))

    m = summary.get("metrics") or {}
    out["metrics"]["SP1 coverage"] = _safe_float(m.get("SP1_subject_coverage"))
    out["metrics"]["SP2 identity"] = _safe_float(m.get("SP2_identity_consistency"))
    out["metrics"]["SP3 occlusion"] = _safe_float(m.get("SP3_occlusion_stability"))
    out["metrics"]["IC1 shot size"] = _safe_float(m.get("IC1_shot_size_match"))
    out["metrics"]["IC2 semantic target"] = _safe_float(m.get("IC2_semantic_target_match"))
    out["metrics"]["IC3 event alignment"] = _safe_float(m.get("IC3_event_alignment"))
    out["metrics"]["TQ1 smoothness"] = _safe_float(m.get("TQ1_motion_smoothness"))
    out["metrics"]["TQ2 tracking"] = _safe_float(m.get("TQ2_subject_tracking_stability"))
    out["metrics"]["TQ3 continuity"] = _safe_float(m.get("TQ3_cut_continuity"))

    # Pull any per-segment failure reasons.
    segments = payload.get("segments") or []
    for seg in segments:
        if isinstance(seg, dict) and seg.get("status") in {"failed", "error"}:
            reason = seg.get("failure_reason") or seg.get("error") or seg.get("status")
            if reason:
                out["failure_reasons"].append(str(reason))
    return out


def classify_failure(report: dict, expected_shots: int | None) -> str:
    if not report["exists"]:
        return "evaluation_failed"
    success = report["successful_segment_count"]
    failed = report["failed_segment_count"]
    if expected_shots is not None and (success + failed) == 0:
        return "evaluation_failed"
    if expected_shots is not None and success == expected_shots and failed == 0:
        return "all_success"
    if failed == 0:
        return "all_success"
    # Heuristic: aggregate the first reason
    reasons = report.get("failure_reasons") or []
    if reasons:
        return "; ".join(sorted(set(reasons))[:5])
    return "unknown_error"


def run_id_for(setting_key: str) -> str:
    return SETTINGS[setting_key]["run_id"]


def load_per_story_records(stories: list[str]) -> dict:
    """Returns nested dict: setting -> story -> record."""
    records: dict = {}
    for setting_key in SETTINGS:
        run_id = run_id_for(setting_key)
        records[setting_key] = {}
        for story in stories:
            report_path = EVAL_OUTPUT_ROOT / run_id / story / "cinestory_report_v1.json"
            parsed = parse_cinestory_report(report_path)
            records[setting_key][story] = {
                "story": story,
                "setting": setting_key,
                "setting_label": SETTINGS[setting_key]["label"],
                "run_id": run_id,
                "report_path": str(report_path),
                **parsed,
            }
    return records


def _expected_shots_for(story: str, fixed_data: list[dict]) -> int | None:
    for row in fixed_data:
        if row.get("story") == story:
            v = row.get("expected_shots")
            if isinstance(v, int):
                return v
    return None


def write_final_story_summary(records: dict, stories: list[str], fixed_data: list[dict]) -> None:
    rows: list[dict] = []
    for setting_key in SETTINGS:
        for story in stories:
            rec = records[setting_key][story]
            expected = _expected_shots_for(story, fixed_data)
            success = rec["successful_segment_count"]
            failed = rec["failed_segment_count"]
            total = success + failed
            if expected is None and total == 0:
                expected = 0
            if expected is None:
                expected = total
            overall = (
                "全部成功"
                if (failed == 0 and success > 0 and (expected == 0 or success >= expected))
                else ("失败" if not rec["exists"] else "部分成功")
            )
            failure_reason = classify_failure(rec, expected)
            row = {
                "Story": story,
                "Setting": SETTINGS[setting_key]["label"],
                "Setting key": setting_key,
                "Overall": overall,
                "成功/总段数": f"{success}/{expected if expected else total}",
                "失败原因": failure_reason,
            }
            for key in ALL_NUM_KEYS:
                row[key] = _fmt(rec["metrics"].get(key))
            rows.append(row)

    json_path = RUN_LOG_ROOT / "final_story_summary.json"
    csv_path = RUN_LOG_ROOT / "final_story_summary.csv"
    md_path = RUN_LOG_ROOT / "final_story_summary.md"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(keys) + " |\n")
            f.write("|" + "|".join(["---"] * len(keys)) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n")
    print(f"[ok] wrote {json_path}, {csv_path}, {md_path}")


def write_ablation_metric_summary(records: dict, stories: list[str], fixed_data: list[dict]) -> None:
    table_rows: list[dict] = []
    for setting_key, info in SETTINGS.items():
        per_story = records[setting_key]
        n_stories = sum(1 for story in stories if per_story[story]["exists"])
        # Aggregate per-story average where exists.
        avg_metrics: dict = {key: [] for key in ALL_NUM_KEYS}
        success_total = 0
        total_total = 0
        for story in stories:
            rec = per_story[story]
            if not rec["exists"]:
                continue
            success_total += rec["successful_segment_count"]
            expected = _expected_shots_for(story, fixed_data) or (
                rec["successful_segment_count"] + rec["failed_segment_count"]
            )
            total_total += expected
            for key in ALL_NUM_KEYS:
                v = rec["metrics"].get(key)
                if v is not None:
                    avg_metrics[key].append(v)
        avg_value = {key: (sum(vs) / len(vs) if vs else None) for key, vs in avg_metrics.items()}
        row = {
            "Setting": info["label"],
            "Setting key": setting_key,
            "#Stories": n_stories,
            "Avg. success/total": f"{success_total}/{total_total}" if total_total else "0/0",
        }
        for key in ALL_NUM_KEYS:
            row[key] = _fmt(avg_value[key])
        table_rows.append(row)

    json_path = RUN_LOG_ROOT / "ablation_metric_summary.json"
    csv_path = RUN_LOG_ROOT / "ablation_metric_summary.csv"
    md_path = RUN_LOG_ROOT / "ablation_metric_summary.md"
    json_path.write_text(json.dumps(table_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if table_rows:
        keys = list(table_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table_rows)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(keys) + " |\n")
            f.write("|" + "|".join(["---"] * len(keys)) + "|\n")
            for r in table_rows:
                f.write("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n")
    print(f"[ok] wrote {json_path}, {csv_path}, {md_path}")
    return table_rows


def write_failed_cases(records: dict, stories: list[str], fixed_data: list[dict]) -> None:
    rows: list[dict] = []
    for setting_key in SETTINGS:
        for story in stories:
            rec = records[setting_key][story]
            expected = _expected_shots_for(story, fixed_data) or (
                rec["successful_segment_count"] + rec["failed_segment_count"]
            )
            if not rec["exists"] or rec["failed_segment_count"] > 0 or rec["successful_segment_count"] < (expected or 0):
                rows.append(
                    {
                        "Setting": SETTINGS[setting_key]["label"],
                        "Setting key": setting_key,
                        "Story": story,
                        "expected_shots": expected,
                        "successful_segment_count": rec["successful_segment_count"],
                        "failed_segment_count": rec["failed_segment_count"],
                        "report_exists": rec["exists"],
                        "failure_reason": classify_failure(rec, expected),
                    }
                )
    csv_path = RUN_LOG_ROOT / "failed_cases.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Setting",
                "Setting key",
                "Story",
                "expected_shots",
                "successful_segment_count",
                "failed_segment_count",
                "report_exists",
                "failure_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {csv_path} ({len(rows)} rows)")


def write_per_setting_stage_summary(records: dict, stories: list[str]) -> None:
    out: dict = {}
    for setting_key, info in SETTINGS.items():
        story_states = []
        for story in stories:
            rec = records[setting_key][story]
            story_states.append(
                {
                    "story": story,
                    "report_exists": rec["exists"],
                    "successful_segment_count": rec["successful_segment_count"],
                    "failed_segment_count": rec["failed_segment_count"],
                }
            )
        out[setting_key] = {
            "label": info["label"],
            "run_id": info["run_id"],
            "stories": story_states,
        }
    path = RUN_LOG_ROOT / "per_setting_stage_summary.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {path}")


def write_latex_table(table_rows: list[dict]) -> None:
    if not table_rows:
        return
    metric_cols = ("Subject Perception", "Intent Consistency", "Trajectory Quality") + METRIC_KEYS
    # Column header
    latex_short = {
        "Subject Perception": "SP",
        "Intent Consistency": "IC",
        "Trajectory Quality": "TQ",
        "SP1 coverage": "SP1",
        "SP2 identity": "SP2",
        "SP3 occlusion": "SP3",
        "IC1 shot size": "IC1",
        "IC2 semantic target": "IC2",
        "IC3 event alignment": "IC3",
        "TQ1 smoothness": "TQ1",
        "TQ2 tracking": "TQ2",
        "TQ3 continuity": "TQ3",
    }
    # Determine best (max) value for each metric column across rows that have a value.
    best_values: dict[str, float | None] = {}
    for col in metric_cols:
        vals = [_safe_float(r.get(col)) for r in table_rows]
        vals = [v for v in vals if v is not None]
        best_values[col] = max(vals) if vals else None

    lines: list[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation results on the fixed test set.}")
    lines.append("\\label{tab:ablation_results}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    col_spec = "l" + "c" * len(metric_cols)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = ["Method"] + [latex_short[c] for c in metric_cols]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for r in table_rows:
        cells = [r["Setting"].replace("&", r"\\&")]
        for col in metric_cols:
            v = _safe_float(r.get(col))
            if v is None:
                cells.append("--")
                continue
            text = f"{v:.2f}"
            if best_values[col] is not None and abs(v - best_values[col]) < 1e-6:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    path = RUN_LOG_ROOT / "ablation_table_latex.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {path}")


def _delta_text(quality_value, ablation_value, key):
    if quality_value is None or ablation_value is None:
        return f"{key}: data missing"
    delta = ablation_value - quality_value
    direction = "↓" if delta < 0 else ("↑" if delta > 0 else "≈")
    return f"{key}: {ablation_value:.3f} ({direction}{abs(delta):.3f} vs Quality {quality_value:.3f})"


def write_ablation_analysis_draft(table_rows: list[dict]) -> None:
    by_key = {row["Setting key"]: row for row in table_rows}
    quality = by_key.get("quality") or {}
    quality_metrics = {
        key: _safe_float(quality.get(key)) for key in ALL_NUM_KEYS
    }
    sections: list[str] = ["# Ablation analysis draft (auto-generated)\n\nNumbers come from `ablation_metric_summary.json`. Use this as a starting point; rewrite prose freely."]
    targeted: dict[str, tuple[str, ...]] = {
        "fast": ("Subject Perception", "Intent Consistency", "Trajectory Quality"),
        "wo_vlm_reflection": ("IC2 semantic target", "IC3 event alignment", "Intent Consistency"),
        "wo_trajectory_grounding": ("TQ1 smoothness", "TQ2 tracking", "TQ3 continuity", "Trajectory Quality"),
        "wo_semantic_height_adjust": ("IC2 semantic target", "Intent Consistency"),
        "wo_pre_continuity_story_judge": ("TQ3 continuity", "IC3 event alignment"),
    }
    for setting_key in ("fast", "wo_vlm_reflection", "wo_trajectory_grounding", "wo_semantic_height_adjust", "wo_pre_continuity_story_judge"):
        row = by_key.get(setting_key)
        if not row:
            continue
        sections.append(f"\n## {SETTINGS[setting_key]['label']}\n")
        bullets = []
        for key in targeted.get(setting_key, ()):
            ablation_v = _safe_float(row.get(key))
            quality_v = quality_metrics.get(key)
            bullets.append("- " + _delta_text(quality_v, ablation_v, key))
        if not bullets:
            bullets.append("- (no targeted metrics defined)")
        sections.append("\n".join(bullets))
        sections.append(
            "\n_Note: if the targeted metric did not drop, state that explicitly and offer a hypothesis (sample size, ablation strength, evaluation noise)._"
        )
    path = RUN_LOG_ROOT / "ablation_analysis_draft.md"
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[ok] wrote {path}")


def write_patch_summary() -> None:
    path = RUN_LOG_ROOT / "patch_summary.md"
    if path.exists():
        return
    path.write_text(
        (
            "# Patch summary (2026-05-03 quality_and_ablations)\n\n"
            "## Code changes\n"
            "- `Cinematographer/cinematographer_stage.py`\n"
            "  - Added 3 ablation fields on `CinematographerConfig`: `disable_vlm_reflection`, `disable_trajectory_grounding`, `disable_semantic_height_adjust`.\n"
            "  - Registered CLI flags `--disable-vlm-reflection`, `--disable-trajectory-grounding`, `--disable-semantic-height-adjust`.\n"
            "  - Added module-level `_ABLATION_FLAGS` populated by `_set_ablation_flags(config)` at the start of `run_cinematographer`.\n"
            "  - Gated `llm_micro_adjust` and `run_final_preview_review_pipeline` when `disable_vlm_reflection`.\n"
            "  - Gated `build_camera_trajectory_plan` and `refresh_camera_trajectory` when `disable_trajectory_grounding` (naive 2-keyframe linear).\n"
            "  - Forwarded `--disable-semantic-height-adjust` to the quality worker via `script_args`.\n"
            "  - Wrote `ablation_flags` into `camera_handoff_v1.json` and `manifest.json`.\n"
            "- `Cinematographer/cinematographer_quality_worker.py`\n"
            "  - Added `--disable-semantic-height-adjust` CLI flag and module-level `_DISABLE_SEMANTIC_HEIGHT_ADJUST`.\n"
            "  - Gated 4 sites: `_collect_focus_geometry` closeup-face center.z bump, `semantic_target_point`, closeup `framing_target.z += target_z_offset`, `enforce_camera_height`.\n"
            "- `tools/run_all_scripts_except_demo.py`\n"
            "  - Added `ABLATION_PRESETS` table (fast / quality / 4 ablations).\n"
            "  - Added CLI flags `--ablation` and `--fixed-stories-from`.\n"
            "  - Per-ablation log dir under `run_logs/quality_and_ablations_20260502/<ablation>/`.\n"
            "- New helper `tools/_build_fixed_stories.py` produces `fixed_six_datasets.{json,csv}`.\n"
            "- New aggregator `tools/summarize_six_ablations.py` (this script).\n\n"
            "## Switches confirmed in `--help`\n"
            "See `_stage_help.txt` and `_orchestrator_help.txt` in this directory.\n"
        ),
        encoding="utf-8",
    )
    print(f"[ok] wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--log-root", type=str, default="")
    parser.add_argument("--fixed-stories-from", type=str, default="")
    args = parser.parse_args()

    global RUN_LOG_ROOT, FIXED_STORIES_JSON
    if args.log_root:
        RUN_LOG_ROOT = Path(args.log_root)
    if args.fixed_stories_from:
        FIXED_STORIES_JSON = Path(args.fixed_stories_from)
    apply_run_tag(args.run_tag.strip())

    RUN_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stories = load_fixed_stories()
    if not stories:
        print("[fatal] no selected stories from fixed_six_datasets.json")
        return 1
    print(f"[info] aggregating {len(stories)} stories x {len(SETTINGS)} settings")
    fixed_data = []
    if FIXED_STORIES_JSON.exists():
        fixed_data = json.loads(FIXED_STORIES_JSON.read_text(encoding="utf-8"))

    records = load_per_story_records(stories)
    write_final_story_summary(records, stories, fixed_data)
    table_rows = write_ablation_metric_summary(records, stories, fixed_data)
    write_failed_cases(records, stories, fixed_data)
    write_per_setting_stage_summary(records, stories)
    write_latex_table(table_rows)
    write_ablation_analysis_draft(table_rows)
    write_patch_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
