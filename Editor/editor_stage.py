from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat


STAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = STAGE_DIR.parent
VIDEO_OUTPUT_DIR = WORKSPACE_DIR / "VideoEngineer" / "output"
MATCH_CUT_DUPLICATE_THRESHOLD = 0.035
MATCH_CUT_TARGET_DELTA = 0.09
MATCH_CUT_MAX_DELTA = 0.20
MIN_TIMELINE_CLIP_SECONDS = 2.5


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


def latest_video_handoff() -> Path:
    candidates = sorted((item for item in VIDEO_OUTPUT_DIR.iterdir() if item.is_dir()), key=lambda item: item.name)
    if not candidates:
        raise FileNotFoundError("No VideoEngineer output folders found. Run VideoEngineer/run_video_engineer.py first.")
    return candidates[-1] / "outputs" / "video_handoff_v1.json"


def default_output_root(run_id: str | None = None) -> Path:
    suffix = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return STAGE_DIR / "output" / suffix


def ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"


@dataclass(slots=True)
class EditorConfig:
    video_handoff_path: str
    output_root: str
    run_id: str = ""
    fps: int = 24


def parse_args() -> EditorConfig:
    parser = argparse.ArgumentParser(description="Run the isolated Editor stage.")
    parser.add_argument("--video-handoff-path", default=str(latest_video_handoff()))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(args.run_id).resolve()
    return EditorConfig(
        video_handoff_path=args.video_handoff_path,
        output_root=str(output_root),
        run_id=args.run_id,
        fps=args.fps,
    )


def sorted_frame_paths(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("frame_*.png"))


def encode_frames(frame_dir: Path, fps: int, output_path: Path) -> subprocess.CompletedProcess[str]:
    command = [
        ffmpeg_path(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "final_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def mux_audio(video_path: Path, audio_paths: list[Path], output_path: Path) -> subprocess.CompletedProcess[str] | None:
    existing_audio = [path for path in audio_paths if path.exists()]
    if not existing_audio:
        return None
    command = [ffmpeg_path(), "-y", "-i", str(video_path)]
    for audio_path in existing_audio:
        command.extend(["-i", str(audio_path)])
    if len(existing_audio) == 1:
        command.extend(["-c:v", "copy", "-shortest", str(output_path)])
    else:
        labels = "".join(f"[{index}:a]" for index in range(1, len(existing_audio) + 1))
        filter_complex = f"{labels}amix=inputs={len(existing_audio)}:duration=longest[aout]"
        command.extend(["-filter_complex", filter_complex, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-shortest", str(output_path)])
    return subprocess.run(command, check=False, capture_output=True, text=True)


def transition_frames(previous_clip: dict[str, Any], current_clip: dict[str, Any]) -> int:
    transition = previous_clip.get("transition_to_next") or {}
    same_shot = (
        previous_clip.get("scene_id") == current_clip.get("scene_id")
        and previous_clip.get("shot_id") == current_clip.get("shot_id")
    )
    if not same_shot:
        return 0
    return int(transition.get("overlap_frames") or 0)


def frame_number_from_path(path: Path) -> int:
    stem = path.stem.split("_")[-1]
    try:
        return int(stem)
    except ValueError:
        return 0


def minimum_timeline_clip_frames(fps: int) -> int:
    return max(1, int(math.ceil(MIN_TIMELINE_CLIP_SECONDS * max(int(fps or 0), 1))))


def trim_budget(frame_paths: list[Path], min_clip_frames: int) -> int:
    return max(len(frame_paths) - min_clip_frames, 0)


def clip_duration_stats(frame_counts: list[int], fps: int) -> dict[str, Any]:
    if not frame_counts or fps <= 0:
        return {
            "min_clip_duration_seconds": 0.0,
            "mean_clip_duration_seconds": 0.0,
            "median_clip_duration_seconds": 0.0,
            "max_clip_duration_seconds": 0.0,
        }
    ordered = sorted(frame_counts)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median_frames = float(ordered[midpoint])
    else:
        median_frames = (float(ordered[midpoint - 1]) + float(ordered[midpoint])) * 0.5
    return {
        "min_clip_duration_seconds": round(float(ordered[0]) / float(fps), 3),
        "mean_clip_duration_seconds": round(sum(frame_counts) / float(len(frame_counts)) / float(fps), 3),
        "median_clip_duration_seconds": round(median_frames / float(fps), 3),
        "max_clip_duration_seconds": round(float(ordered[-1]) / float(fps), 3),
    }


def frame_difference(path_a: Path, path_b: Path) -> float:
    image_a = Image.open(path_a).convert("L").resize((96, 54))
    image_b = Image.open(path_b).convert("L").resize((96, 54))
    diff = ImageChops.difference(image_a, image_b)
    return float(ImageStat.Stat(diff).mean[0]) / 255.0


def find_same_scene_match_cut(previous_paths: list[Path], current_paths: list[Path], window: int = 16) -> dict[str, Any] | None:
    if len(previous_paths) < 10 or len(current_paths) < 10:
        return None
    previous_start = max(0, len(previous_paths) - min(window, len(previous_paths) - 4))
    current_stop = min(window, len(current_paths) - 4)
    if current_stop <= 0:
        return None
    best: dict[str, Any] | None = None
    for previous_index in range(previous_start, len(previous_paths)):
        for current_index in range(0, current_stop):
            score = frame_difference(previous_paths[previous_index], current_paths[current_index])
            if score > MATCH_CUT_MAX_DELTA:
                continue
            duplicate_risk = score < MATCH_CUT_DUPLICATE_THRESHOLD
            trim_total = (len(previous_paths) - previous_index - 1) + current_index
            selection_cost = abs(score - MATCH_CUT_TARGET_DELTA) - min(trim_total, 16) * 0.002
            if duplicate_risk:
                selection_cost += 0.12
            candidate = {
                "score": round(score, 6),
                "previous_trim": len(previous_paths) - previous_index - 1,
                "current_trim": current_index,
                "selection_cost": round(selection_cost, 6),
                "duplicate_frame_risk": duplicate_risk,
                "jump_cut_risk": score > 0.16,
            }
            if best is None or selection_cost < float(best["selection_cost"]) or (
                abs(selection_cost - float(best["selection_cost"])) < 1e-6
                and candidate["previous_trim"] + candidate["current_trim"] > int(best["previous_trim"]) + int(best["current_trim"])
            ):
                best = candidate
    if best is None:
        return None
    return best


def representative_clip_difference(previous_paths: list[Path], current_paths: list[Path]) -> float:
    if not previous_paths or not current_paths:
        return 1.0
    probes = (0.2, 0.5, 0.8)
    scores = []
    for probe in probes:
        previous_index = min(max(int(round((len(previous_paths) - 1) * probe)), 0), len(previous_paths) - 1)
        current_index = min(max(int(round((len(current_paths) - 1) * probe)), 0), len(current_paths) - 1)
        scores.append(frame_difference(previous_paths[previous_index], current_paths[current_index]))
    return sum(scores) / float(len(scores))


def detect_head_settle_trim(frame_paths: list[Path], max_window: int = 12) -> int:
    if len(frame_paths) < 12:
        return 0
    consecutive_low_motion = 0
    max_checks = min(max_window, len(frame_paths) - 1)
    for index in range(1, max_checks + 1):
        if frame_difference(frame_paths[index - 1], frame_paths[index]) < 0.016:
            consecutive_low_motion += 1
        else:
            break
    if consecutive_low_motion < 4:
        return 0
    return max(0, consecutive_low_motion - 2)


def detect_tail_settle_trim(frame_paths: list[Path], max_window: int = 18) -> int:
    if len(frame_paths) < 12:
        return 0
    consecutive_low_motion = 0
    max_checks = min(max_window, len(frame_paths) - 1)
    for offset in range(1, max_checks + 1):
        index = len(frame_paths) - offset
        if frame_difference(frame_paths[index - 1], frame_paths[index]) < 0.02:
            consecutive_low_motion += 1
        else:
            break
    if consecutive_low_motion < 4:
        return 0
    return max(0, consecutive_low_motion - 2)


def run_editor(config: EditorConfig) -> dict[str, Any]:
    output_root = ensure_directory(Path(config.output_root).resolve())
    outputs_dir = ensure_directory(output_root / "outputs")
    export_dir = ensure_directory(output_root / "exports")
    timeline_dir = ensure_directory(output_root / "timeline_frames")
    video_handoff_path = Path(config.video_handoff_path).resolve()
    video_handoff = load_json(video_handoff_path)
    fps = int(video_handoff.get("fps") or config.fps)
    min_clip_frames = minimum_timeline_clip_frames(fps)
    resolution = video_handoff.get("resolution") or {"width": 960, "height": 540}
    clips = video_handoff.get("clips") or []
    timeline_entries: list[dict[str, Any]] = []
    timeline_paths: list[Path] = []
    output_index = 0
    prepared_clips: list[dict[str, Any]] = []
    omitted_clips: list[dict[str, Any]] = []

    for clip_index, clip in enumerate(clips):
        frame_dir = Path(str(clip.get("frame_dir") or ""))
        frame_paths = sorted_frame_paths(frame_dir)
        if not frame_paths:
            continue
        if clip.get("trim_start_frames") is not None or clip.get("trim_end_frames") is not None:
            trim_start = max(0, int(clip.get("trim_start_frames") or 0))
            trim_end = max(0, int(clip.get("trim_end_frames") or 0))
        else:
            trim_start = 1 if clip_index > 0 else 0
            trim_end = 1 if clip_index < len(clips) - 1 else 0
        if len(frame_paths) - trim_start - trim_end < (min_clip_frames if len(frame_paths) >= min_clip_frames else 1):
            trim_start = 0
            trim_end = 0
        selected_paths = frame_paths[trim_start : len(frame_paths) - trim_end if trim_end else None]
        if not selected_paths:
            continue
        cut_reasons = [str(clip.get("cut_reason") or "handle_trim")]
        if len(frame_paths) < min_clip_frames:
            cut_reasons.append("source_clip_shorter_than_min_duration")
        head_settle_trim = min(detect_head_settle_trim(selected_paths), trim_budget(selected_paths, min_clip_frames))
        if clip_index > 0 and head_settle_trim > 0:
            selected_paths = selected_paths[head_settle_trim:]
            trim_start += head_settle_trim
            cut_reasons.append("head_motion_settle_trim")
        prepared_clips.append(
            {
                "clip": clip,
                "trim_start": trim_start,
                "trim_end": trim_end,
                "selected_paths": selected_paths,
                "cut_reasons": cut_reasons,
                "same_scene_match_cut": bool(clip.get("same_scene_match_cut")),
                "continuity_match_score": clip.get("continuity_match_score"),
                "selected_cut_delta": None,
                "duplicate_frame_risk": False,
                "jump_cut_risk": False,
            }
        )

    pruned_clips: list[dict[str, Any]] = []
    for prepared in prepared_clips:
        if pruned_clips:
            previous = pruned_clips[-1]
            previous_clip = previous["clip"]
            current_clip = prepared["clip"]
            same_scene = previous_clip.get("scene_id") == current_clip.get("scene_id")
            same_shot = previous_clip.get("shot_id") == current_clip.get("shot_id")
            if same_scene and same_shot:
                similarity_delta = representative_clip_difference(previous["selected_paths"], prepared["selected_paths"])
                recommended_omit = bool(current_clip.get("editor_recommended_omit"))
                omit_threshold = 0.06 if recommended_omit else 0.045
                if similarity_delta < omit_threshold:
                    previous["cut_reasons"].append("redundant_same_shot_neighbor_removed")
                    omitted_clips.append(
                        {
                            "scene_id": current_clip.get("scene_id"),
                            "shot_id": current_clip.get("shot_id"),
                            "camera_name": current_clip.get("camera_name"),
                            "omit_reason": "editor_recommended_omit_same_shot_neighbor" if recommended_omit else "redundant_same_shot_neighbor",
                            "representative_delta": round(similarity_delta, 6),
                            "kept_camera_name": previous_clip.get("camera_name"),
                            "same_shot_diversity_status": current_clip.get("same_shot_diversity_status") or "",
                            "similar_to_camera_name": current_clip.get("similar_to_camera_name") or "",
                        }
                    )
                    continue
        pruned_clips.append(prepared)
    prepared_clips = pruned_clips

    for clip_index in range(1, len(prepared_clips)):
        previous = prepared_clips[clip_index - 1]
        current = prepared_clips[clip_index]
        previous_clip = previous["clip"]
        current_clip = current["clip"]
        same_scene = previous_clip.get("scene_id") == current_clip.get("scene_id")
        different_shot = previous_clip.get("shot_id") != current_clip.get("shot_id")
        if not same_scene or not different_shot:
            continue
        match = find_same_scene_match_cut(previous["selected_paths"], current["selected_paths"])
        if not match:
            continue
        previous_extra_trim = min(int(match["previous_trim"]), trim_budget(previous["selected_paths"], min_clip_frames))
        current_extra_trim = min(int(match["current_trim"]), trim_budget(current["selected_paths"], min_clip_frames))
        if previous_extra_trim > 0:
            previous["selected_paths"] = previous["selected_paths"][: len(previous["selected_paths"]) - previous_extra_trim]
            previous["trim_end"] += previous_extra_trim
        if current_extra_trim > 0:
            current["selected_paths"] = current["selected_paths"][current_extra_trim:]
            current["trim_start"] += current_extra_trim
        previous["same_scene_match_cut"] = True
        current["same_scene_match_cut"] = True
        previous["continuity_match_score"] = float(match["score"])
        current["continuity_match_score"] = float(match["score"])
        previous["selected_cut_delta"] = float(match["score"])
        current["selected_cut_delta"] = float(match["score"])
        previous["duplicate_frame_risk"] = bool(match.get("duplicate_frame_risk"))
        current["duplicate_frame_risk"] = bool(match.get("duplicate_frame_risk"))
        previous["jump_cut_risk"] = bool(match.get("jump_cut_risk"))
        current["jump_cut_risk"] = bool(match.get("jump_cut_risk"))
        previous["cut_reasons"].append("same_scene_match_cut")
        current["cut_reasons"].append("same_scene_match_cut")
        if match.get("duplicate_frame_risk"):
            previous["cut_reasons"].append("duplicate_match_cut_risk")
            current["cut_reasons"].append("duplicate_match_cut_risk")
        if match.get("jump_cut_risk"):
            previous["cut_reasons"].append("jump_match_cut_risk")
            current["cut_reasons"].append("jump_match_cut_risk")

    for prepared in prepared_clips:
        settle_trim = min(detect_tail_settle_trim(prepared["selected_paths"]), trim_budget(prepared["selected_paths"], min_clip_frames))
        if settle_trim > 0:
            prepared["selected_paths"] = prepared["selected_paths"][: len(prepared["selected_paths"]) - settle_trim]
            prepared["trim_end"] += settle_trim
            prepared["cut_reasons"].append("tail_motion_settle_trim")

    final_pruned_clips: list[dict[str, Any]] = []
    for prepared in prepared_clips:
        if final_pruned_clips:
            previous = final_pruned_clips[-1]
            previous_clip = previous["clip"]
            current_clip = prepared["clip"]
            same_scene = previous_clip.get("scene_id") == current_clip.get("scene_id")
            same_shot = previous_clip.get("shot_id") == current_clip.get("shot_id")
            if same_scene and same_shot and previous["selected_paths"] and prepared["selected_paths"]:
                overlap_frames = transition_frames(previous_clip, current_clip)
                current_start_index = min(max(overlap_frames, 0), len(prepared["selected_paths"]) - 1)
                boundary_delta = frame_difference(previous["selected_paths"][-1], prepared["selected_paths"][current_start_index])
                recommended_omit = bool(current_clip.get("editor_recommended_omit"))
                boundary_threshold = 0.06 if recommended_omit else MATCH_CUT_DUPLICATE_THRESHOLD
                if boundary_delta < boundary_threshold:
                    previous["cut_reasons"].append("redundant_same_shot_boundary_removed")
                    omitted_clips.append(
                        {
                            "scene_id": current_clip.get("scene_id"),
                            "shot_id": current_clip.get("shot_id"),
                            "camera_name": current_clip.get("camera_name"),
                            "omit_reason": "editor_recommended_omit_same_shot_boundary" if recommended_omit else "redundant_same_shot_boundary",
                            "boundary_delta": round(boundary_delta, 6),
                            "kept_camera_name": previous_clip.get("camera_name"),
                            "same_shot_diversity_status": current_clip.get("same_shot_diversity_status") or "",
                            "similar_to_camera_name": current_clip.get("similar_to_camera_name") or "",
                        }
                    )
                    continue
        final_pruned_clips.append(prepared)
    prepared_clips = final_pruned_clips

    for clip_index, prepared in enumerate(prepared_clips):
        clip = prepared["clip"]
        frame_paths = list(prepared["selected_paths"])
        if not frame_paths:
            continue
        current_transition = transition_frames(prepared_clips[clip_index - 1]["clip"], clip) if clip_index > 0 else 0
        if current_transition > 0 and clip_index > 0:
            previous_clip = prepared_clips[clip_index - 1]["clip"]
            if previous_clip.get("scene_id") == clip.get("scene_id") and previous_clip.get("shot_id") == clip.get("shot_id"):
                current_transition = 0
                prepared["cut_reasons"].append("same_shot_overlap_disabled")
        current_transition = min(current_transition, trim_budget(frame_paths, min_clip_frames), len(timeline_paths))
        if current_transition > 0:
            for offset in range(current_transition):
                previous_path = timeline_paths[-current_transition + offset]
                previous_image = Image.open(previous_path).convert("RGBA")
                current_image = Image.open(frame_paths[offset]).convert("RGBA")
                alpha = (offset + 1) / float(current_transition + 1)
                blended = Image.blend(previous_image, current_image, alpha)
                blended.convert("RGB").save(previous_path)
            frame_paths = frame_paths[current_transition:]
        for frame_path in frame_paths:
            output_path = timeline_dir / f"final_{output_index:06d}.png"
            shutil.copy2(frame_path, output_path)
            timeline_paths.append(output_path)
            output_index += 1
        timeline_entries.append(
            {
                "timeline_index": clip_index,
                "scene_id": clip.get("scene_id"),
                "shot_id": clip.get("shot_id"),
                "camera_name": clip.get("camera_name"),
                "clip_path": clip.get("clip_path"),
                "frame_dir": clip.get("frame_dir"),
                "trajectory_plan_path": clip.get("trajectory_plan_path"),
                "trajectory_preset": clip.get("trajectory_preset"),
                "motion_qc": clip.get("motion_qc") or {},
                "quality_qc": clip.get("quality_qc") or {},
                "same_shot_diversity_review": clip.get("same_shot_diversity_review") or {},
                "same_shot_diversity_status": clip.get("same_shot_diversity_status") or "",
                "similar_to_camera_name": clip.get("similar_to_camera_name") or "",
                "editor_recommended_omit": bool(clip.get("editor_recommended_omit")),
                "editor_handles_seconds": clip.get("editor_handles_seconds") or {},
                "transition_to_next": clip.get("transition_to_next"),
                "trim_start_frames": prepared["trim_start"],
                "trim_end_frames": prepared["trim_end"],
                "transition_from_previous_frames": current_transition,
                "source_frame_count": len(prepared["selected_paths"]),
                "output_frame_count": len(frame_paths),
                "output_duration_seconds": round(len(frame_paths) / float(fps), 3) if fps else 0.0,
                "cut_in_frame": frame_number_from_path(frame_paths[0]) if frame_paths else 0,
                "cut_out_frame": frame_number_from_path(frame_paths[-1]) if frame_paths else 0,
                "cut_reason": "; ".join(dict.fromkeys(prepared["cut_reasons"])),
                "continuity_match_score": prepared["continuity_match_score"],
                "same_scene_match_cut": bool(prepared["same_scene_match_cut"]),
                "selected_cut_delta": prepared["selected_cut_delta"],
                "duplicate_frame_risk": bool(prepared["duplicate_frame_risk"]),
                "jump_cut_risk": bool(prepared["jump_cut_risk"]),
            }
        )

    export_path = export_dir / "final_edit_v1.mp4"
    encode_result = encode_frames(timeline_dir, fps, export_path)
    audio_export_path = export_dir / "final_edit_with_audio_v1.mp4"
    audio_refs = [Path(str(item)) for item in video_handoff.get("audio_refs") or [] if str(item)]
    audio_result = mux_audio(export_path, audio_refs, audio_export_path)
    final_export_path = audio_export_path if audio_result and audio_result.returncode == 0 and audio_export_path.exists() else export_path
    clip_frame_counts = [int(entry.get("output_frame_count") or 0) for entry in timeline_entries]
    clip_stats = clip_duration_stats(clip_frame_counts, fps)
    edit_output_path = save_json(
        {
            "schema_version": "storyblender.edit_output.v1",
            "generated_at": utc_now(),
            "run_id": config.run_id or output_root.name,
            "video_handoff_path": str(video_handoff_path),
            "export_path": str(final_export_path),
            "export_without_audio_path": str(export_path),
            "fps": fps,
            "resolution": resolution,
            "timeline": timeline_entries,
            "omitted_clips": omitted_clips,
            "used_audio_refs": [str(path) for path in audio_refs if path.exists()],
            "qc_summary": {
                "render_success": encode_result.returncode == 0 and export_path.exists(),
                "timeline_frame_count": len(timeline_paths),
                "timeline_duration_seconds": round(len(timeline_paths) / float(fps), 3) if fps else 0.0,
                "clip_count": len(timeline_entries),
                "omitted_clip_count": len(omitted_clips),
                "minimum_clip_duration_seconds": MIN_TIMELINE_CLIP_SECONDS,
                "minimum_clip_frame_count": min_clip_frames,
                "short_clip_count": sum(1 for count in clip_frame_counts if count < min_clip_frames),
                **clip_stats,
                "transition_count": sum(1 for entry in timeline_entries if entry["transition_from_previous_frames"] > 0),
                "same_scene_match_cut_count": sum(1 for entry in timeline_entries if entry.get("same_scene_match_cut")),
                "duplicate_frame_risk_count": sum(1 for entry in timeline_entries if entry.get("duplicate_frame_risk")),
                "jump_cut_risk_count": sum(1 for entry in timeline_entries if entry.get("jump_cut_risk")),
                "trajectory_count": sum(1 for entry in timeline_entries if entry.get("trajectory_plan_path")),
                "location_motion_count": sum(1 for entry in timeline_entries if (entry.get("motion_qc") or {}).get("has_location_change")),
                "quality_warning_count": sum(
                    1
                    for entry in timeline_entries
                    if entry.get("quality_qc")
                    and (
                        float((entry.get("quality_qc") or {}).get("line_of_sight_clear_ratio") or 0.0)
                        < float((entry.get("quality_qc") or {}).get("required_line_of_sight_clear_ratio") or 0.0)
                    )
                ),
            },
        },
        outputs_dir / "edit_output_v1.json",
    )
    manifest_path = save_json(
        {
            "schema_version": "storyblender.editor_manifest.v1",
            "generated_at": utc_now(),
            "success": True,
            "run_id": config.run_id or output_root.name,
            "video_handoff_path": str(video_handoff_path),
            "output_root": str(output_root),
            "edit_output_path": str(edit_output_path),
            "export_path": str(final_export_path),
        },
        output_root / "manifest.json",
    )
    return {
        "success": True,
        "manifest_path": str(manifest_path),
        "edit_output_path": str(edit_output_path),
        "export_path": str(final_export_path),
    }


def main() -> int:
    config = parse_args()
    result = run_editor(config)
    print("Editor completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
