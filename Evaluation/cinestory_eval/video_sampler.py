from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import ensure_directory


try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime.
    cv2 = None


@dataclass(slots=True)
class FrameSample:
    frame_index: int
    timestamp_sec: float
    frame_bgr: np.ndarray
    path: str = ""
    role: str = "uniform"


@dataclass(slots=True)
class VideoInfo:
    video_path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for CineStoryEval video sampling.")
    return cv2


def open_video(video_path: Path) -> tuple[Any, VideoInfo]:
    cv = require_cv2()
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv.CAP_PROP_FPS) or 24.0)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return cap, VideoInfo(str(video_path), fps, frame_count, width, height, duration)


def read_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    cv = require_cv2()
    cap, _ = open_video(video_path)
    try:
        cap.set(cv.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def save_frame(frame_bgr: np.ndarray, path: Path) -> str:
    cv = require_cv2()
    ensure_directory(path.parent)
    cv.imwrite(str(path), frame_bgr)
    return str(path)


def uniform_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if count <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, frame_count - 1, count)})


def estimate_motion_peak_indices(video_path: Path, frame_count: int, max_peaks: int = 2) -> list[int]:
    cv = require_cv2()
    if frame_count <= 3 or max_peaks <= 0:
        return []
    cap, _ = open_video(video_path)
    stride = max(1, frame_count // 48)
    last_gray = None
    diffs: list[tuple[float, int]] = []
    try:
        for index in range(0, frame_count, stride):
            cap.set(cv.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.resize(gray, (160, 90), interpolation=cv.INTER_AREA)
            if last_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(np.float32) - last_gray.astype(np.float32))) / 255.0)
                diffs.append((diff, index))
            last_gray = gray
    finally:
        cap.release()
    if not diffs:
        return []
    diffs.sort(reverse=True, key=lambda item: item[0])
    selected: list[int] = []
    min_gap = max(1, frame_count // 12)
    for _, index in diffs:
        if all(abs(index - existing) >= min_gap for existing in selected):
            selected.append(index)
        if len(selected) >= max_peaks:
            break
    return sorted(selected)


def sample_video(
    video_path: Path,
    evidence_dir: Path,
    *,
    uniform_count: int = 8,
    motion_peak_count: int = 2,
    write_frames: bool = True,
) -> tuple[VideoInfo, list[FrameSample]]:
    cv = require_cv2()
    cap, info = open_video(video_path)
    if info.frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    indices = set(uniform_indices(info.frame_count, uniform_count))
    indices.update(estimate_motion_peak_indices(video_path, info.frame_count, motion_peak_count))
    samples: list[FrameSample] = []
    try:
        for index in sorted(indices):
            cap.set(cv.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            timestamp = index / info.fps if info.fps > 0 else 0.0
            role = "boundary" if index in (0, info.frame_count - 1) else "keyframe"
            sample = FrameSample(index, timestamp, frame, role=role)
            if write_frames:
                frame_name = f"frame_{index:06d}.png"
                sample.path = save_frame(frame, evidence_dir / "keyframes" / frame_name)
            samples.append(sample)
    finally:
        cap.release()
    return info, samples


def dense_frames(video_path: Path, *, max_frames: int = 96) -> list[FrameSample]:
    cap, info = open_video(video_path)
    if info.frame_count <= 0:
        cap.release()
        return []
    indices = uniform_indices(info.frame_count, min(max_frames, info.frame_count))
    samples: list[FrameSample] = []
    try:
        for index in indices:
            cap.set(require_cv2().CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                samples.append(FrameSample(index, index / info.fps if info.fps > 0 else 0.0, frame, role="dense"))
    finally:
        cap.release()
    return samples


def frame_blankness(frame_bgr: np.ndarray) -> dict[str, float | bool]:
    cv = require_cv2()
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    edges = cv.Canny(gray, 60, 160)
    edge_fraction = float(np.mean(edges > 0))
    blank = variance < 8.0 and edge_fraction < 0.002
    return {
        "gray_variance": variance,
        "edge_fraction": edge_fraction,
        "is_blank": bool(blank),
    }
