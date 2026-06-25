from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = STAGE_DIR.parent
DIRECTOR_DIR = WORKSPACE_DIR / "Director"
CINEMATOGRAPHER_DIR = WORKSPACE_DIR / "Cinematographer"
VIDEO_DIR = WORKSPACE_DIR / "VideoEngineer"
EDITOR_DIR = WORKSPACE_DIR / "Editor"
DEFAULT_DEMO_ROOT = Path(os.environ.get("LBM_DEMO_ROOT", str(WORKSPACE_DIR.parent / "scripts" / "demo"))).resolve()
DEFAULT_STAGE_PYTHON = Path(os.environ.get("LBM_PYTHON_EXE", sys.executable))
FORBIDDEN_MAIN_PREFIX = "plan" + "_a_"
FORBIDDEN_SYS_PATH_INSERT = "sys.path." + "insert"
FORBIDDEN_SYS_PATH_APPEND = "sys.path." + "append"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Any, path: Path) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def default_stage_python() -> Path:
    env_value = os.environ.get("LBM_PYTHON_EXE", "")
    if env_value:
        return Path(env_value).resolve()
    if DEFAULT_STAGE_PYTHON.exists():
        return DEFAULT_STAGE_PYTHON.resolve()
    return Path(sys.executable).resolve()


def latest_stage_file(stage_output_dir: Path, relative_path: str) -> Path:
    candidates = sorted((item for item in stage_output_dir.iterdir() if item.is_dir()), key=lambda item: item.name)
    if not candidates:
        raise FileNotFoundError(f"No output folders found in {stage_output_dir}")
    return candidates[-1] / relative_path


def audit_stage_source(stage_name: str, stage_dir: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(stage_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_MAIN_PREFIX in text:
            violations.append(f"{stage_name}: forbidden main Plan-A import/reference in {path.name}")
        if FORBIDDEN_SYS_PATH_INSERT in text or FORBIDDEN_SYS_PATH_APPEND in text:
            violations.append(f"{stage_name}: forbidden sys.path mutation in {path.name}")
    return violations


def validate_director_handoff(path: Path) -> list[str]:
    payload = load_json(path)
    violations: list[str] = []
    for key in ("schema_version", "files", "asset_index", "shot_sequence", "demo_root", "approved_story_source_path"):
        if key not in payload:
            violations.append(f"Director handoff missing '{key}'")
    files = payload.get("files") or {}
    for key in (
        "blocking_plans_path",
        "shot_contracts_path",
        "scene_understanding_path",
        "source_inventory_path",
        "director_script_path",
    ):
        candidate = Path(str(files.get(key) or ""))
        if not candidate.exists():
            violations.append(f"Director handoff file missing: {key}")
    demo_root = Path(str(payload.get("demo_root") or ""))
    if not demo_root.exists():
        violations.append("Director handoff demo_root does not exist.")
    approved_story_source_path = Path(str(payload.get("approved_story_source_path") or ""))
    if not approved_story_source_path.exists():
        violations.append("Director handoff approved_story_source_path does not exist.")
    return violations


def validate_camera_handoff(path: Path) -> list[str]:
    payload = load_json(path)
    violations: list[str] = []
    for key in ("schema_version", "director_handoff_path", "shots", "cameras"):
        if key not in payload:
            violations.append(f"Camera handoff missing '{key}'")
    for camera in payload.get("cameras") or []:
        for field in ("camera_package_path",):
            candidate = Path(str(camera.get(field) or ""))
            if not candidate.exists():
                violations.append(f"Camera handoff missing path for {camera.get('camera_name')}: {field}")
        for field in ("source_plate_path", "preview_frame_path"):
            raw_path = str(camera.get(field) or "")
            if raw_path and not Path(raw_path).exists():
                violations.append(f"Camera handoff optional path is invalid for {camera.get('camera_name')}: {field}")
    return violations


def validate_video_handoff(path: Path) -> list[str]:
    payload = load_json(path)
    violations: list[str] = []
    for key in ("schema_version", "camera_handoff_path", "clips", "clip_manifest_path"):
        if key not in payload:
            violations.append(f"Video handoff missing '{key}'")
    for clip in payload.get("clips") or []:
        for field in ("frame_dir", "clip_path", "trajectory_plan_path"):
            candidate = Path(str(clip.get(field) or ""))
            if not candidate.exists():
                violations.append(f"Video handoff missing path for {clip.get('camera_name')}: {field}")
    return violations


def validate_edit_output(path: Path) -> list[str]:
    payload = load_json(path)
    violations: list[str] = []
    for key in ("schema_version", "video_handoff_path", "export_path", "timeline"):
        if key not in payload:
            violations.append(f"Edit output missing '{key}'")
    export_path = Path(str(payload.get("export_path") or ""))
    if not export_path.exists():
        violations.append("Final export video does not exist.")
    return violations


def run_stage(stage_dir: Path, script_name: str, args: list[str], stage_label: str) -> subprocess.CompletedProcess[str]:
    command = [str(default_stage_python()), str(stage_dir / script_name), *args]
    log(f"{stage_label} started: {' '.join(command)}")
    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(stage_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def pump(stream: Any, label: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                output_queue.put((label, line))
        finally:
            output_queue.put((label, None))

    threads = [
        threading.Thread(target=pump, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=pump, args=(process.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    closed = set()
    last_heartbeat = time.monotonic()
    while process.poll() is None or len(closed) < 2:
        try:
            label, line = output_queue.get(timeout=0.5)
        except queue.Empty:
            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                log(f"{stage_label} still running ({now - started_at:.1f}s elapsed).")
                last_heartbeat = now
            continue
        if line is None:
            closed.add(label)
            continue
        if label == "stdout":
            stdout_lines.append(line)
            print(f"[{stage_label}] {line}", end="", flush=True)
        else:
            stderr_lines.append(line)
            print(f"[{stage_label}:stderr] {line}", end="", flush=True)
    returncode = process.wait()
    elapsed = time.monotonic() - started_at
    status = "completed" if returncode == 0 else f"failed rc={returncode}"
    log(f"{stage_label} {status} after {elapsed:.1f}s.")
    return subprocess.CompletedProcess(command, returncode, "".join(stdout_lines), "".join(stderr_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated four-stage StoryBlender pipeline.")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--demo-root", default=str(DEFAULT_DEMO_ROOT))
    parser.add_argument("--instruction-source-path", default="")
    parser.add_argument("--scene-ids", default="")
    parser.add_argument("--resume-from", choices=("start", "director", "cinematographer", "videoengineer"), default="start")
    parser.add_argument("--camera-quality", choices=("fast", "quality"), default="fast")
    parser.add_argument("--director-handoff-path", default="")
    parser.add_argument("--camera-handoff-path", default="")
    parser.add_argument("--video-handoff-path", default="")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo_root = Path(args.instruction_source_path).resolve() if args.instruction_source_path else Path(args.demo_root).resolve()
    output_root = ensure_directory(STAGE_DIR / "output" / args.run_id)

    audit_violations: list[str] = []
    for stage_name, stage_dir in (
        ("Director", DIRECTOR_DIR),
        ("Cinematographer", CINEMATOGRAPHER_DIR),
        ("VideoEngineer", VIDEO_DIR),
        ("Editor", EDITOR_DIR),
    ):
        audit_violations.extend(audit_stage_source(stage_name, stage_dir))
    audit_path = save_json(
        {
            "schema_version": "storyblender.contract_audit.v1",
            "generated_at": utc_now(),
            "violations": audit_violations,
        },
        output_root / "contract_audit_v1.json",
    )
    if audit_violations:
        print("Contract audit failed.")
        print(json.dumps(audit_violations, ensure_ascii=False, indent=2))
        return 1

    director_output_root = DIRECTOR_DIR / "output" / args.run_id
    camera_output_root = CINEMATOGRAPHER_DIR / "output" / args.run_id
    video_output_root = VIDEO_DIR / "output" / args.run_id
    editor_output_root = EDITOR_DIR / "output" / args.run_id

    director_handoff_path = Path(args.director_handoff_path).resolve() if args.director_handoff_path else None
    camera_handoff_path = Path(args.camera_handoff_path).resolve() if args.camera_handoff_path else None
    video_handoff_path = Path(args.video_handoff_path).resolve() if args.video_handoff_path else None
    stage_logs: dict[str, dict[str, Any]] = {}

    if args.resume_from == "start":
        result = run_stage(
            DIRECTOR_DIR,
            "run_director.py",
            [
                "--demo-root",
                str(demo_root),
                "--output-root",
                str(director_output_root),
                "--run-id",
                args.run_id,
                "--scene-ids",
                args.scene_ids,
            ],
            "Director",
        )
        stage_logs["director"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            return result.returncode
        director_handoff_path = director_output_root / "outputs" / "director_handoff_v1.json"
    if args.resume_from == "director" and director_handoff_path is None:
        director_handoff_path = latest_stage_file(DIRECTOR_DIR / "output", "outputs/director_handoff_v1.json")
    if args.resume_from in {"start", "director"}:
        if director_handoff_path is None:
            raise FileNotFoundError("Director handoff path could not be resolved.")
        violations = validate_director_handoff(director_handoff_path)
        if violations:
            print(json.dumps(violations, ensure_ascii=False, indent=2))
            return 1

    if args.resume_from in {"start", "director"}:
        result = run_stage(
            CINEMATOGRAPHER_DIR,
            "run_cinematographer.py",
            [
                "--director-handoff-path",
                str(director_handoff_path),
                "--output-root",
                str(camera_output_root),
                "--run-id",
                args.run_id,
                "--fps",
                str(args.fps),
                "--camera-quality",
                args.camera_quality,
            ],
            "Cinematographer",
        )
        stage_logs["cinematographer"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            return result.returncode
        camera_handoff_path = camera_output_root / "outputs" / "camera_handoff_v1.json"
    if args.resume_from == "cinematographer" and camera_handoff_path is None:
        camera_handoff_path = latest_stage_file(CINEMATOGRAPHER_DIR / "output", "outputs/camera_handoff_v1.json")
    if args.resume_from in {"start", "director", "cinematographer"}:
        if camera_handoff_path is None:
            raise FileNotFoundError("Camera handoff path could not be resolved.")
        violations = validate_camera_handoff(camera_handoff_path)
        if violations:
            print(json.dumps(violations, ensure_ascii=False, indent=2))
            return 1

    if args.resume_from in {"start", "director", "cinematographer"}:
        result = run_stage(
            VIDEO_DIR,
            "run_video_engineer.py",
            [
                "--camera-handoff-path",
                str(camera_handoff_path),
                "--output-root",
                str(video_output_root),
                "--run-id",
                args.run_id,
                "--fps",
                str(args.fps),
                "--resolution-x",
                str(args.resolution_x),
                "--resolution-y",
                str(args.resolution_y),
                "--camera-quality",
                args.camera_quality,
            ],
            "VideoEngineer",
        )
        stage_logs["videoengineer"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            return result.returncode
        video_handoff_path = video_output_root / "outputs" / "video_handoff_v1.json"
    if args.resume_from == "videoengineer" and video_handoff_path is None:
        video_handoff_path = latest_stage_file(VIDEO_DIR / "output", "outputs/video_handoff_v1.json")
    if args.resume_from in {"start", "director", "cinematographer", "videoengineer"}:
        if video_handoff_path is None:
            raise FileNotFoundError("Video handoff path could not be resolved.")
        violations = validate_video_handoff(video_handoff_path)
        if violations:
            print(json.dumps(violations, ensure_ascii=False, indent=2))
            return 1

    result = run_stage(
        EDITOR_DIR,
        "run_editor.py",
        [
            "--video-handoff-path",
            str(video_handoff_path),
            "--output-root",
            str(editor_output_root),
            "--run-id",
            args.run_id,
            "--fps",
            str(args.fps),
        ],
        "Editor",
    )
    stage_logs["editor"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return result.returncode

    edit_output_path = editor_output_root / "outputs" / "edit_output_v1.json"
    violations = validate_edit_output(edit_output_path)
    if violations:
        print(json.dumps(violations, ensure_ascii=False, indent=2))
        return 1

    run_manifest_path = save_json(
        {
            "schema_version": "storyblender.engine_run_manifest.v1",
            "generated_at": utc_now(),
            "run_id": args.run_id,
            "demo_root": str(demo_root),
            "scene_ids": args.scene_ids,
            "camera_quality": args.camera_quality,
            "audit_path": str(audit_path),
            "director_handoff_path": str(director_handoff_path),
            "camera_handoff_path": str(camera_handoff_path),
            "video_handoff_path": str(video_handoff_path),
            "edit_output_path": str(edit_output_path),
            "stage_logs": stage_logs,
        },
        output_root / "run_manifest_v1.json",
    )
    print("Full pipeline completed.")
    print(json.dumps({"run_manifest_path": str(run_manifest_path), "edit_output_path": str(edit_output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
