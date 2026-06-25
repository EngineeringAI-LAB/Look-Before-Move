from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("LBM_WORKSPACE", str(REPO_ROOT))).resolve()
TEST_ROOT = REPO_ROOT
SCRIPTS_ROOT = Path(os.environ.get("LBM_SCRIPTS_ROOT", str(WORKSPACE / "scripts"))).resolve()
PYTHON_EXE = Path(os.environ.get("LBM_PYTHON_EXE", sys.executable)).resolve()

ABLATION_PRESETS = {
    "fast": {
        "camera_quality": "fast",
        "extra_camera_args": [],
    },
    "quality": {
        "camera_quality": "quality",
        "extra_camera_args": ["--run-pre-continuity-story-judge"],
    },
    "wo_vlm_reflection": {
        "camera_quality": "quality",
        "extra_camera_args": ["--run-pre-continuity-story-judge", "--disable-vlm-reflection"],
    },
    "wo_trajectory_grounding": {
        "camera_quality": "quality",
        "extra_camera_args": ["--run-pre-continuity-story-judge", "--disable-trajectory-grounding"],
    },
    "wo_semantic_height_adjust": {
        "camera_quality": "quality",
        "extra_camera_args": ["--run-pre-continuity-story-judge", "--disable-semantic-height-adjust"],
    },
    "wo_pre_continuity_story_judge": {
        "camera_quality": "quality",
        "extra_camera_args": [],
    },
}

RUN_ID_BASE = {
    "fast": "full_scripts_except_demo_fast",
    "quality": "full_scripts_except_demo_quality",
    "wo_vlm_reflection": "full_scripts_except_demo_wo_vlm_reflection",
    "wo_trajectory_grounding": "full_scripts_except_demo_wo_trajectory_grounding",
    "wo_semantic_height_adjust": "full_scripts_except_demo_wo_semantic_height_adjust",
    "wo_pre_continuity_story_judge": "full_scripts_except_demo_wo_pre_continuity_story_judge",
}

METRIC_KEYS = (
    "Subject Perception",
    "Intent Consistency",
    "Trajectory Quality",
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


@dataclass
class GenerationResult:
    story_name: str
    status: str
    expected_shots: int | str
    director_out: str = ""
    camera_out: str = ""
    video_out: str = ""
    editor_out: str = ""
    eval_out: str = ""
    benchmark_json: str = ""
    error: str = ""


@dataclass
class EvaluationResult:
    story_name: str
    status: str
    expected_shots: int | str
    eval_out: str
    benchmark_json: str
    error: str = ""


def utc_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_root_for_run_tag(run_tag: str) -> Path:
    parts = run_tag.split("_")
    date_part = parts[0] if parts and parts[0].isdigit() and len(parts[0]) == 8 else datetime.now().strftime("%Y%m%d")
    return TEST_ROOT / "run_logs" / f"quality_and_ablations_{date_part}"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_id_for(ablation: str, run_tag: str) -> str:
    suffix = run_tag.strip() or utc_stamp()
    return f"{RUN_ID_BASE[ablation]}_{suffix}"


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HTTP_PROXY", "http://127.0.0.1:33210")
    env.setdefault("HTTPS_PROXY", "http://127.0.0.1:33210")
    env.setdefault("http_proxy", "http://127.0.0.1:33210")
    env.setdefault("https_proxy", "http://127.0.0.1:33210")
    return env


def run_cmd(cmd: list[str], cwd: Path, log_file: Path) -> tuple[bool, int]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n--- RUNNING COMMAND: {' '.join(cmd)} ---\n")
        handle.flush()
        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=command_env(),
                check=False,
            )
            rc = int(result.returncode)
        except Exception as exc:
            handle.write(f"\nEXCEPTION DURING EXECUTION: {exc}\n")
            rc = 1
        elapsed = time.monotonic() - start
        handle.write(f"\n--- EXIT CODE: {rc} (Elapsed: {elapsed:.1f}s) ---\n")
        handle.flush()
    return rc == 0, rc


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_path(*parts: str) -> str:
    return str(TEST_ROOT.joinpath(*parts))


def story_name_filters(story_filter: str) -> set[str]:
    return {item.strip() for item in story_filter.split(",") if item.strip()}


def discover_script_story_rows(scripts_root: Path, story_filter: str = "") -> list[dict[str, Any]]:
    if not scripts_root.exists():
        raise FileNotFoundError(f"Scripts root not found: {scripts_root}")
    filters = story_name_filters(story_filter)
    selected: list[dict[str, Any]] = []
    for path in sorted(scripts_root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        if path.name.lower() == "demo" or path.name.startswith("."):
            continue
        if filters and path.name not in filters:
            continue
        selected.append({"story_name": path.name, "script_path": str(path)})
    return selected


def story_rows(fixed_path: Path, story_filter: str = "") -> list[dict[str, Any]]:
    if not fixed_path.exists():
        log(f"[stories] Fixed list not found: {fixed_path}; discovering from {SCRIPTS_ROOT}")
        return discover_script_story_rows(SCRIPTS_ROOT, story_filter)

    rows = load_json(fixed_path)
    selected: list[dict[str, Any]] = []
    filters = story_name_filters(story_filter)
    for row in rows:
        if not isinstance(row, dict) or not row.get("selected_for_run"):
            continue
        story_name = str(row.get("story") or "").strip()
        script_path = str(row.get("script_path") or "").strip()
        if not story_name or not script_path:
            continue
        if filters and story_name not in filters:
            continue
        selected.append({"story_name": story_name, "script_path": script_path})
    return selected


def expected_shot_count(director_handoff: Path) -> int | str:
    try:
        payload = load_json(director_handoff)
        return len(payload.get("shot_sequence") or [])
    except Exception:
        return "N/A"


def summarize_eval(eval_out: Path) -> tuple[dict[str, Any], int, int]:
    report_path = eval_out / "cinestory_report_v1.json"
    metrics = {
        "Subject Perception": "N/A",
        "Intent Consistency": "N/A",
        "Trajectory Quality": "N/A",
        "SP1 coverage": "N/A",
        "SP2 identity": "N/A",
        "SP3 occlusion": "N/A",
        "IC1 shot size": "N/A",
        "IC2 semantic target": "N/A",
        "IC3 event alignment": "N/A",
        "TQ1 smoothness": "N/A",
        "TQ2 tracking": "N/A",
        "TQ3 continuity": "N/A",
    }
    if not report_path.exists():
        return metrics, 0, 0
    report = load_json(report_path)
    summary = report.get("summary") or {}
    success_count = int(summary.get("successful_segment_count") or 0)
    failed_count = int(summary.get("failed_segment_count") or 0)
    dims = summary.get("dimensions") or {}
    raw_metrics = summary.get("metrics") or {}
    mapping = {
        "Subject Perception": dims.get("subject_perception"),
        "Intent Consistency": dims.get("intent_consistency"),
        "Trajectory Quality": dims.get("trajectory_quality"),
        "SP1 coverage": raw_metrics.get("SP1_subject_coverage"),
        "SP2 identity": raw_metrics.get("SP2_identity_consistency"),
        "SP3 occlusion": raw_metrics.get("SP3_occlusion_stability"),
        "IC1 shot size": raw_metrics.get("IC1_shot_size_match"),
        "IC2 semantic target": raw_metrics.get("IC2_semantic_target_match"),
        "IC3 event alignment": raw_metrics.get("IC3_event_alignment"),
        "TQ1 smoothness": raw_metrics.get("TQ1_motion_smoothness"),
        "TQ2 tracking": raw_metrics.get("TQ2_subject_tracking_stability"),
        "TQ3 continuity": raw_metrics.get("TQ3_cut_continuity"),
    }
    for key, value in mapping.items():
        if isinstance(value, (int, float)):
            metrics[key] = f"{float(value):.3f}"
    return metrics, success_count, success_count + failed_count


def free_vram_gb() -> float | None:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    values: list[float] = []
    for line in text.splitlines():
        try:
            values.append(float(line.strip()) / 1024.0)
        except ValueError:
            continue
    return max(values) if values else None


def wait_for_vram(min_free_gb: float, poll_seconds: float, log_file: Path) -> None:
    if min_free_gb <= 0:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    while True:
        free = free_vram_gb()
        if free is None or free >= min_free_gb:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[eval_gate] free_vram_gb={free} threshold={min_free_gb}; starting evaluation\n")
            return
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[eval_gate] waiting: free_vram_gb={free:.2f} threshold={min_free_gb}\n")
        time.sleep(max(1.0, poll_seconds))


def run_generation(
    *,
    story: dict[str, Any],
    run_id: str,
    log_dir: Path,
    preset: dict[str, Any],
    llm_max_workers_per_story: int,
) -> GenerationResult:
    story_name = story["story_name"]
    demo_root = Path(story["script_path"])
    story_log_dir = log_dir / story_name
    stdout_log = story_log_dir / "generation_stdout.log"
    story_log_dir.mkdir(parents=True, exist_ok=True)
    director_out = TEST_ROOT / "Director" / "output" / run_id / story_name
    camera_out = TEST_ROOT / "Cinematographer" / "output" / run_id / story_name
    video_out = TEST_ROOT / "VideoEngineer" / "output" / run_id / story_name
    editor_out = TEST_ROOT / "Editor" / "output" / run_id / story_name
    eval_out = TEST_ROOT / "Evaluation" / "output" / run_id / story_name
    director_handoff = director_out / "outputs" / "director_handoff_v1.json"
    video_handoff = video_out / "outputs" / "video_handoff_v1.json"
    benchmark_json = eval_out / "benchmark_input_v1.json"

    try:
        if benchmark_json.exists() and video_handoff.exists():
            expected = expected_shot_count(director_handoff)
            with stdout_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n[resume] Existing benchmark/video handoff found; "
                    "skipping generation stages for this story.\n"
                )
            return GenerationResult(
                story_name=story_name,
                status="generation_ready",
                expected_shots=expected,
                director_out=str(director_out),
                camera_out=str(camera_out),
                video_out=str(video_out),
                editor_out=str(editor_out),
                eval_out=str(eval_out),
                benchmark_json=str(benchmark_json),
            )

        director_cmd = [
            str(PYTHON_EXE),
            stage_path("Director", "director_stage.py"),
            "--demo-root",
            str(demo_root),
            "--output-root",
            str(director_out),
            "--run-id",
            story_name,
        ]
        log(f"[gen:{story_name}] Director")
        ok, _ = run_cmd(director_cmd, WORKSPACE, stdout_log)
        if not ok:
            return GenerationResult(story_name, "director_failed", "N/A", director_out=str(director_out))

        expected = expected_shot_count(director_handoff)

        camera_cmd = [
            str(PYTHON_EXE),
            stage_path("Cinematographer", "cinematographer_stage.py"),
            "--director-handoff-path",
            str(director_handoff),
            "--output-root",
            str(camera_out),
            "--run-id",
            story_name,
            "--enable-candidate-validation",
            "--camera-quality",
            str(preset["camera_quality"]),
            "--llm-max-workers",
            str(max(1, llm_max_workers_per_story)),
        ]
        camera_cmd.extend(list(preset.get("extra_camera_args") or []))
        log(f"[gen:{story_name}] Cinematographer")
        ok, _ = run_cmd(camera_cmd, WORKSPACE, stdout_log)
        if not ok:
            return GenerationResult(story_name, "cinematographer_failed", expected, director_out=str(director_out), camera_out=str(camera_out))

        reval_cmd = [
            str(PYTHON_EXE),
            stage_path("tools", "rerun_final_preview_review.py"),
            "--output-root",
            str(camera_out),
            "--handoff-policy",
            "semantic_only",
            "--revalidate-only",
        ]
        log(f"[gen:{story_name}] Handoff override")
        ok, _ = run_cmd(reval_cmd, WORKSPACE, stdout_log)
        if not ok:
            return GenerationResult(story_name, "handoff_override_failed", expected, director_out=str(director_out), camera_out=str(camera_out))

        camera_handoff = camera_out / "outputs" / "camera_handoff_v1.json"
        video_cmd = [
            str(PYTHON_EXE),
            stage_path("VideoEngineer", "video_stage.py"),
            "--camera-handoff-path",
            str(camera_handoff),
            "--output-root",
            str(video_out),
            "--run-id",
            story_name,
        ]
        log(f"[gen:{story_name}] VideoEngineer")
        ok, _ = run_cmd(video_cmd, WORKSPACE, stdout_log)
        if not ok:
            return GenerationResult(story_name, "video_failed", expected, director_out=str(director_out), camera_out=str(camera_out), video_out=str(video_out))

        editor_cmd = [
            str(PYTHON_EXE),
            stage_path("Editor", "editor_stage.py"),
            "--video-handoff-path",
            str(video_handoff),
            "--output-root",
            str(editor_out),
            "--run-id",
            story_name,
        ]
        log(f"[gen:{story_name}] Editor")
        ok, _ = run_cmd(editor_cmd, WORKSPACE, stdout_log)
        if not ok:
            return GenerationResult(story_name, "editor_failed", expected, director_out=str(director_out), camera_out=str(camera_out), video_out=str(video_out), editor_out=str(editor_out))

        build_cmd = [
            str(PYTHON_EXE),
            stage_path("Evaluation", "scripts", "build_benchmark_input.py"),
            "--video-handoff",
            str(video_handoff),
            "--story-name",
            story_name,
            "--output",
            str(benchmark_json),
        ]
        log(f"[gen:{story_name}] Build benchmark")
        ok, _ = run_cmd(build_cmd, WORKSPACE, stdout_log)
        if not ok:
            return GenerationResult(story_name, "benchmark_failed", expected, director_out=str(director_out), camera_out=str(camera_out), video_out=str(video_out), editor_out=str(editor_out), eval_out=str(eval_out))

        return GenerationResult(
            story_name=story_name,
            status="generation_ready",
            expected_shots=expected,
            director_out=str(director_out),
            camera_out=str(camera_out),
            video_out=str(video_out),
            editor_out=str(editor_out),
            eval_out=str(eval_out),
            benchmark_json=str(benchmark_json),
        )
    except Exception as exc:
        traceback_text = traceback.format_exc()
        with stdout_log.open("a", encoding="utf-8") as handle:
            handle.write("\nGENERATION EXCEPTION\n")
            handle.write(traceback_text)
        return GenerationResult(story_name, "generation_exception", "N/A", error=f"{exc.__class__.__name__}: {exc}")


def run_evaluation(
    *,
    gen: GenerationResult,
    log_dir: Path,
    min_free_vram_gb: float,
    poll_seconds: float,
    force_eval: bool,
) -> EvaluationResult:
    story_name = gen.story_name
    eval_out = Path(gen.eval_out)
    benchmark_json = Path(gen.benchmark_json)
    story_log_dir = log_dir / story_name
    stdout_log = story_log_dir / "evaluation_stdout.log"
    story_log_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_json.exists():
        return EvaluationResult(story_name, "benchmark_missing", gen.expected_shots, str(eval_out), str(benchmark_json))

    report_path = eval_out / "cinestory_report_v1.json"
    if report_path.exists() and not force_eval:
        try:
            report = load_json(report_path)
            summary = report.get("summary") or {}
            ok = int(summary.get("successful_segment_count") or 0)
            if gen.expected_shots == "N/A" or ok == int(gen.expected_shots):
                return EvaluationResult(story_name, "evaluation_skipped_existing", gen.expected_shots, str(eval_out), str(benchmark_json))
        except Exception:
            pass

    wait_for_vram(min_free_vram_gb, poll_seconds, stdout_log)
    eval_cmd = [
        str(PYTHON_EXE),
        stage_path("Evaluation", "run_cinestory_eval.py"),
        "--benchmark-input",
        str(benchmark_json),
        "--output-root",
        str(eval_out),
        "--story-name",
        story_name,
        "--vlm-backend",
        "qwen_local",
    ]
    log(f"[eval:{story_name}] Evaluation")
    ok, _ = run_cmd(eval_cmd, WORKSPACE, stdout_log)
    if not ok:
        return EvaluationResult(story_name, "evaluation_failed", gen.expected_shots, str(eval_out), str(benchmark_json))
    return EvaluationResult(story_name, "success", gen.expected_shots, str(eval_out), str(benchmark_json))


def report_row(story_name: str, status: str, eval_out: str, expected_shots: int | str, setting: str) -> dict[str, Any]:
    metrics, success_count, total_count = summarize_eval(Path(eval_out)) if eval_out else ({}, 0, 0)
    if total_count == 0 and expected_shots != "N/A":
        try:
            total_count = int(expected_shots)
        except Exception:
            total_count = 0
    if status in {"success", "evaluation_skipped_existing"}:
        overall = "success" if expected_shots == "N/A" or success_count == int(expected_shots) else "partial_success"
        reason = "all_success" if overall == "success" else f"{total_count - success_count} failed"
    else:
        overall = "failed"
        reason = status
    return {
        "Story": story_name,
        "Setting": setting,
        "Overall": overall,
        "success/total": f"{success_count}/{total_count}",
        "failure_reason": reason,
        **metrics,
    }


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    save_json(rows, output_dir / "final_summary.json")
    if not rows:
        return
    keys = list(rows[0].keys())
    with (output_dir / "final_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "final_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(keys) + " |\n")
        handle.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run story generation in parallel, then evaluate completed stories serially.")
    parser.add_argument("--ablation", choices=tuple(ABLATION_PRESETS.keys()), default="quality")
    parser.add_argument("--fixed-stories-from", default=str(TEST_ROOT / "run_logs" / "quality_and_ablations_20260502" / "fixed_six_datasets.json"))
    parser.add_argument("--story", default="", help="Optional exact story name filter, or comma-separated names.")
    parser.add_argument("--run-tag", default="", help="Output suffix. Defaults to timestamp if omitted.")
    parser.add_argument("--log-root", default="", help="Log root. Defaults to run_logs/quality_and_ablations_<YYYYMMDD> from --run-tag.")
    parser.add_argument("--generation-workers", type=int, default=2)
    parser.add_argument("--evaluation-workers", type=int, default=1, help="Only 1 is supported for qwen_local.")
    parser.add_argument("--llm-max-workers-per-story", type=int, default=2)
    parser.add_argument("--eval-min-free-vram-gb", type=float, default=18.0)
    parser.add_argument("--eval-vram-poll-seconds", type=float, default=30.0)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved stories/run ids without running stages.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.evaluation_workers != 1:
        raise SystemExit("--evaluation-workers must stay 1 for qwen_local.")

    run_tag = args.run_tag.strip() or utc_stamp()
    run_id = run_id_for(args.ablation, run_tag)
    log_root = Path(args.log_root) if args.log_root else log_root_for_run_tag(run_tag)
    setting_log_dir = log_root / args.ablation
    setting_log_dir.mkdir(parents=True, exist_ok=True)

    preset = ABLATION_PRESETS[args.ablation]
    stories = story_rows(Path(args.fixed_stories_from), args.story)
    if not stories:
        raise SystemExit("No selected stories to run.")

    manifest = {
        "run_id": run_id,
        "run_tag": run_tag,
        "ablation": args.ablation,
        "camera_quality": preset["camera_quality"],
        "extra_camera_args": preset.get("extra_camera_args") or [],
        "fixed_stories_from": args.fixed_stories_from,
        "story_filter": args.story,
        "generation_workers": args.generation_workers,
        "evaluation_workers": args.evaluation_workers,
        "llm_max_workers_per_story": args.llm_max_workers_per_story,
        "eval_min_free_vram_gb": args.eval_min_free_vram_gb,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(manifest, setting_log_dir / "manifest.json")
    log(f"[start] run_id={run_id} ablation={args.ablation} stories={len(stories)} gen_workers={args.generation_workers}")
    if args.dry_run:
        payload = {
            "manifest": manifest,
            "stories": stories,
            "log_dir": str(setting_log_dir),
        }
        save_json(payload, setting_log_dir / "dry_run.json")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    generation_results: list[GenerationResult] = []
    evaluation_results: list[EvaluationResult] = []
    summary_rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.generation_workers)) as executor:
        futures = {
            executor.submit(
                run_generation,
                story=story,
                run_id=run_id,
                log_dir=setting_log_dir,
                preset=preset,
                llm_max_workers_per_story=args.llm_max_workers_per_story,
            ): story
            for story in stories
        }
        for future in as_completed(futures):
            story = futures[future]
            try:
                gen = future.result()
            except Exception as exc:
                gen = GenerationResult(story["story_name"], "generation_future_exception", "N/A", error=f"{exc.__class__.__name__}: {exc}")
            generation_results.append(gen)
            save_json([asdict(row) for row in generation_results], setting_log_dir / "generation_results.json")
            if gen.status != "generation_ready":
                log(f"[gen:{gen.story_name}] failed status={gen.status}")
                summary_rows.append(report_row(gen.story_name, gen.status, gen.eval_out, gen.expected_shots, args.ablation))
                write_summary(summary_rows, setting_log_dir)
                continue

            log(f"[gen:{gen.story_name}] ready; queueing serial evaluation")
            ev = run_evaluation(
                gen=gen,
                log_dir=setting_log_dir,
                min_free_vram_gb=args.eval_min_free_vram_gb,
                poll_seconds=args.eval_vram_poll_seconds,
                force_eval=bool(args.force_eval),
            )
            evaluation_results.append(ev)
            save_json([asdict(row) for row in evaluation_results], setting_log_dir / "evaluation_results.json")
            summary_rows.append(report_row(ev.story_name, ev.status, ev.eval_out, ev.expected_shots, args.ablation))
            write_summary(summary_rows, setting_log_dir)
            log(f"[eval:{ev.story_name}] status={ev.status}")

    save_json([asdict(row) for row in generation_results], setting_log_dir / "generation_results.json")
    save_json([asdict(row) for row in evaluation_results], setting_log_dir / "evaluation_results.json")
    write_summary(summary_rows, setting_log_dir)
    log(f"[done] wrote {setting_log_dir / 'final_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
