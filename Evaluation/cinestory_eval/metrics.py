from __future__ import annotations

import math
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import clamp_score, ensure_directory, normalize_label, score_mean
from .model_zoo import VisionLanguageClient
from .video_sampler import FrameSample, dense_frames, frame_blankness, require_cv2, sample_video


Box = tuple[float, float, float, float]
STAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_WEIGHTS = STAGE_DIR / "models" / "yolo" / "yolo11x.pt"
REFERENCE_MATCH_THRESHOLD = 0.12
YOLO_SCORE_THRESHOLD = 0.12
_YOLO_MODEL: Any | None = None
_YOLO_ATTEMPTED = False
_YOLO_WEIGHT_PATH = ""


class SegmentEvaluationError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(slots=True)
class FrameObservation:
    frame_index: int
    timestamp_sec: float
    frame_width: int
    frame_height: int
    blank: bool
    gray_variance: float
    edge_fraction: float
    box: Box | None
    box_source: str
    box_confidence: float
    area_ratio: float
    center_score: float
    truncation: float
    identity_score: float | None
    evidence_path: str


def box_area(box: Box | None) -> float:
    if not box:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width - 1), x2)),
        max(0.0, min(float(height - 1), y2)),
    )


def center_score(box: Box | None, width: int, height: int) -> float:
    if not box or width <= 0 or height <= 0:
        return 0.0
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    return max(0.0, 1.0 - dist / 0.7071)


def truncation_ratio(box: Box | None, width: int, height: int) -> float:
    if not box or width <= 0 or height <= 0:
        return 1.0
    x1, y1, x2, y2 = box
    margin = 3.0
    hits = 0
    hits += 1 if x1 <= margin else 0
    hits += 1 if y1 <= margin else 0
    hits += 1 if x2 >= width - margin else 0
    hits += 1 if y2 >= height - margin else 0
    return hits / 4.0


def foreground_bbox(frame_bgr: np.ndarray) -> tuple[Box | None, float]:
    cv = require_cv2()
    height, width = frame_bgr.shape[:2]
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    median_gray = float(np.median(gray))
    gray_delta = cv.absdiff(gray, np.full_like(gray, int(median_gray)))
    saturation = hsv[:, :, 1]
    edges = cv.Canny(gray, 60, 160)
    mask = ((gray_delta > 18) | (saturation > 35) | (edges > 0)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.dilate(mask, kernel, iterations=1)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, Box]] = []
    frame_area = float(width * height)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = float(w * h)
        area_ratio = area / frame_area if frame_area > 0 else 0.0
        if area_ratio < 0.002 or area_ratio > 0.92:
            continue
        aspect = h / max(1.0, float(w))
        shape_weight = 1.0 if 0.35 <= aspect <= 6.0 else 0.65
        candidates.append((area_ratio * shape_weight, (float(x), float(y), float(x + w), float(y + h))))
    if not candidates:
        return None, 0.0
    candidates.sort(reverse=True, key=lambda item: item[0])
    return clamp_box(candidates[0][1], width, height), min(1.0, candidates[0][0] / 0.20)


def yolo_weight_path() -> Path:
    configured = os.environ.get("CINESTORY_YOLO_WEIGHTS", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_YOLO_WEIGHTS


def yolo_subject_bbox(frame_bgr: np.ndarray, subject: dict[str, Any]) -> tuple[Box | None, float]:
    global _YOLO_ATTEMPTED, _YOLO_MODEL, _YOLO_WEIGHT_PATH
    weight_path = yolo_weight_path()
    if not weight_path.exists():
        raise SegmentEvaluationError(
            "yolo_weights_missing",
            f"YOLO weights not found: {weight_path}",
            details={"weight_path": str(weight_path)},
        )
    resolved_weight_path = str(weight_path.resolve())
    if not _YOLO_ATTEMPTED or _YOLO_WEIGHT_PATH != resolved_weight_path:
        _YOLO_ATTEMPTED = True
        _YOLO_WEIGHT_PATH = resolved_weight_path
        try:
            from ultralytics import YOLO

            _YOLO_MODEL = YOLO(resolved_weight_path)
        except Exception as exc:
            _YOLO_MODEL = None
            raise SegmentEvaluationError(
                "yolo_load_failed",
                f"YOLO failed to load: {exc.__class__.__name__}",
                details={"weight_path": resolved_weight_path},
            ) from exc
    if _YOLO_MODEL is None:
        raise SegmentEvaluationError(
            "yolo_load_failed",
            "YOLO model is unavailable after load attempt.",
            details={"weight_path": resolved_weight_path},
        )
    height, width = frame_bgr.shape[:2]
    try:
        results = _YOLO_MODEL.predict(frame_bgr, verbose=False)
    except Exception as exc:
        raise SegmentEvaluationError(
            "yolo_predict_failed",
            f"YOLO prediction failed: {exc.__class__.__name__}",
            details={"weight_path": resolved_weight_path},
        ) from exc
    subject_type = normalize_label(subject.get("type", ""))
    best_box: Box | None = None
    best_score = 0.0
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for item in boxes:
            cls_id = int(item.cls[0]) if getattr(item, "cls", None) is not None else -1
            conf = float(item.conf[0]) if getattr(item, "conf", None) is not None else 0.0
            if subject_type in ("person", "human", "character") and cls_id != 0:
                continue
            xyxy = item.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            box = clamp_box((xyxy[0], xyxy[1], xyxy[2], xyxy[3]), width, height)
            area = box_area(box) / float(width * height) if width > 0 and height > 0 else 0.0
            score = conf * min(1.0, area / 0.18)
            if score > best_score:
                best_score = score
                best_box = box
    if best_box is None or best_score < YOLO_SCORE_THRESHOLD:
        return None, best_score
    return best_box, best_score


def orb_reference_match(frame_bgr: np.ndarray, reference_paths: list[str]) -> tuple[Box | None, float]:
    cv = require_cv2()
    if not reference_paths:
        return None, 0.0
    frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    height, width = frame_gray.shape[:2]
    orb = cv.ORB_create(nfeatures=800)
    kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)
    if des_frame is None or not kp_frame:
        return None, 0.0
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    best_box: Box | None = None
    best_score = 0.0
    for ref in reference_paths:
        ref_path = Path(ref)
        if not ref_path.exists():
            continue
        ref_img = cv.imread(str(ref_path), cv.IMREAD_GRAYSCALE)
        if ref_img is None:
            continue
        kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
        if des_ref is None or not kp_ref:
            continue
        matches = matcher.match(des_ref, des_frame)
        if len(matches) < 8:
            continue
        matches = sorted(matches, key=lambda item: item.distance)[:50]
        good = [m for m in matches if m.distance <= 72]
        if len(good) < 6:
            continue
        points = np.array([kp_frame[m.trainIdx].pt for m in good], dtype=np.float32)
        x1, y1 = np.percentile(points, 5, axis=0)
        x2, y2 = np.percentile(points, 95, axis=0)
        if x2 <= x1 or y2 <= y1:
            continue
        distance_score = max(0.0, 1.0 - float(np.mean([m.distance for m in good])) / 96.0)
        match_score = min(1.0, len(good) / 30.0) * distance_score
        if match_score > best_score:
            best_score = match_score
            best_box = clamp_box((float(x1), float(y1), float(x2), float(y2)), width, height)
    return best_box, best_score


def readable_reference_paths(reference_paths: list[str], *, subject_id: str) -> list[str]:
    cv = require_cv2()
    if not reference_paths:
        raise SegmentEvaluationError(
            "reference_images_missing",
            f"No reference images configured for subject: {subject_id}",
            details={"subject_id": subject_id},
        )
    readable: list[str] = []
    unreadable: list[str] = []
    missing: list[str] = []
    for value in reference_paths:
        path = Path(value)
        if not path.exists():
            missing.append(str(path))
            continue
        if cv.imread(str(path), cv.IMREAD_GRAYSCALE) is None:
            unreadable.append(str(path))
            continue
        readable.append(str(path))
    if missing or unreadable:
        raise SegmentEvaluationError(
            "reference_images_unreadable",
            f"Reference images are missing or unreadable for subject: {subject_id}",
            details={"subject_id": subject_id, "missing": missing, "unreadable": unreadable},
        )
    return readable


def choose_subject_box(frame_bgr: np.ndarray, subject: dict[str, Any]) -> tuple[Box | None, str, float, float | None]:
    subject_id = str(subject.get("subject_id", "primary"))
    try:
        ref_paths = readable_reference_paths([str(path) for path in subject.get("reference_images", []) if path], subject_id=subject_id)
    except SegmentEvaluationError:
        ref_paths = []
    yolo_box, yolo_score = yolo_subject_bbox(frame_bgr, subject)
    ref_box, ref_score = orb_reference_match(frame_bgr, ref_paths)
    source = "yolo_reference_orb"
    if yolo_box is None:
        source = "yolo_subject_not_detected"
    elif ref_box is None or ref_score < REFERENCE_MATCH_THRESHOLD:
        source = "identity_match_below_threshold"
    identity_score = clamp_score(ref_score * 100.0) if ref_paths else None
    return yolo_box, source, yolo_score, identity_score


def overlay_box(frame_bgr: np.ndarray, box: Box | None, output_path: Path, label: str) -> str:
    cv = require_cv2()
    ensure_directory(output_path.parent)
    canvas = frame_bgr.copy()
    if box:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(canvas, label[:32], (x1, max(18, y1 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    cv.imwrite(str(output_path), canvas)
    return str(output_path)


def observe_frames(
    samples: list[FrameSample],
    *,
    primary_subject: dict[str, Any] | None,
    evidence_dir: Path,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    subject_label = str(primary_subject.get("subject_id", "primary")) if primary_subject else "scene"
    for sample in samples:
        stats = frame_blankness(sample.frame_bgr)
        height, width = sample.frame_bgr.shape[:2]
        if stats["is_blank"]:
            box, source, confidence, identity = None, "blank_keyframe", 0.0, None
        elif primary_subject is None:
            box, source, confidence, identity = None, "no_expected_subject", 0.0, None
        else:
            box, source, confidence, identity = choose_subject_box(sample.frame_bgr, primary_subject)
        area = box_area(box) / float(width * height) if box and width > 0 and height > 0 else 0.0
        evidence_path = overlay_box(
            sample.frame_bgr,
            box,
            evidence_dir / "boxes" / f"box_{sample.frame_index:06d}.png",
            subject_label,
        )
        observations.append(
            FrameObservation(
                frame_index=sample.frame_index,
                timestamp_sec=sample.timestamp_sec,
                frame_width=width,
                frame_height=height,
                blank=bool(stats["is_blank"]),
                gray_variance=float(stats["gray_variance"]),
                edge_fraction=float(stats["edge_fraction"]),
                box=box,
                box_source=source,
                box_confidence=float(confidence),
                area_ratio=float(area),
                center_score=center_score(box, width, height),
                truncation=truncation_ratio(box, width, height),
                identity_score=identity,
                evidence_path=evidence_path,
            )
        )
    return observations


def score_subject_perception(observations: list[FrameObservation], *, applicable: bool = True) -> dict[str, Any]:
    if not applicable:
        return {"SP1_subject_coverage": None, "SP2_identity_consistency": None, "SP3_occlusion_stability": None}
    if not observations:
        return {"SP1_subject_coverage": 0.0, "SP2_identity_consistency": 0.0, "SP3_occlusion_stability": 0.0}
    valid = [obs for obs in observations if obs.box is not None and not obs.blank]
    detection_rate = len(valid) / len(observations)
    mean_area = score_mean([obs.area_ratio for obs in valid]) or 0.0
    mean_center = score_mean([obs.center_score for obs in valid]) or 0.0
    blank_rate = sum(1 for obs in observations if obs.blank) / len(observations)
    area_score = min(1.0, mean_area / 0.08)
    sp1 = (0.48 * detection_rate + 0.30 * area_score + 0.22 * mean_center) * 100.0
    sp1 *= max(0.0, 1.0 - blank_rate)
    identity_values = [obs.identity_score for obs in observations if obs.identity_score is not None]
    if identity_values:
        sp2 = score_mean(identity_values) or 0.0
    else:
        confidence = score_mean([obs.box_confidence * 100.0 for obs in valid]) or 0.0
        sp2 = confidence * 0.70 + sp1 * 0.30
    lost_rate = 1.0 - detection_rate
    mean_trunc = score_mean([obs.truncation for obs in valid]) or 1.0
    centers = []
    areas = []
    for obs in observations:
        if obs.box is None:
            continue
        x1, y1, x2, y2 = obs.box
        centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        areas.append(max(1e-6, obs.area_ratio))
    continuity = 0.0
    if len(centers) >= 2:
        jumps = [math.dist(centers[i], centers[i - 1]) for i in range(1, len(centers))]
        norm_jump = min(1.0, float(np.mean(jumps)) / 220.0)
        area_jumps = [abs(math.log(areas[i] / areas[i - 1])) for i in range(1, len(areas))]
        norm_area_jump = min(1.0, float(np.mean(area_jumps)) / 1.2) if area_jumps else 0.0
        continuity = max(0.0, 1.0 - 0.6 * norm_jump - 0.4 * norm_area_jump)
    elif len(centers) == 1:
        continuity = 0.35
    sp3 = (0.42 * (1.0 - lost_rate) + 0.28 * (1.0 - mean_trunc) + 0.30 * continuity) * 100.0
    return {
        "SP1_subject_coverage": clamp_score(sp1),
        "SP2_identity_consistency": clamp_score(sp2),
        "SP3_occlusion_stability": clamp_score(sp3),
    }


def score_shot_size(expected: str, observations: list[FrameObservation]) -> float:
    expected = normalize_label(expected)
    valid_areas = [obs.area_ratio for obs in observations if obs.box is not None and not obs.blank]
    if not valid_areas:
        return 0.0
    area = float(np.median(valid_areas))
    ranges = {
        "extreme_close_up": (0.28, 0.78),
        "close_up": (0.16, 0.65),
        "close": (0.16, 0.65),
        "medium_close_up": (0.09, 0.36),
        "medium": (0.035, 0.22),
        "medium_shot": (0.035, 0.22),
        "full_body": (0.018, 0.14),
        "wide": (0.004, 0.075),
        "wide_shot": (0.004, 0.075),
    }
    low, high = ranges.get(expected, (0.02, 0.35))
    if low <= area <= high:
        return 100.0
    if area < low:
        return max(0.0, 100.0 * area / max(low, 1e-6))
    return max(0.0, 100.0 * max(0.0, 1.0 - (area - high) / max(high, 1e-6)))


def score_semantic_target(expected: str, observations: list[FrameObservation]) -> float:
    target = normalize_label(expected)
    valid = [obs for obs in observations if obs.box is not None and not obs.blank]
    if not valid:
        return 0.0
    if target in ("", "none", "body"):
        return score_mean([obs.center_score * 100.0 for obs in valid]) or 50.0
    if target in ("face", "head", "eyes", "back_of_head"):
        close_area = score_mean([min(1.0, obs.area_ratio / 0.18) for obs in valid]) or 0.0
        upper_focus = []
        for obs in valid:
            assert obs.box is not None
            _, y1, _, y2 = obs.box
            cy = (y1 + y2) * 0.5
            height = max(1, obs.frame_height)
            upper_focus.append(1.0 if cy < 0.62 * height else max(0.0, 1.0 - (cy / height - 0.62) / 0.38))
        return clamp_score((0.60 * close_area + 0.40 * (score_mean(upper_focus) or 0.0)) * 100.0)
    if target in ("feet", "foot", "shoes"):
        lower_focus = []
        for obs in valid:
            assert obs.box is not None
            _, y1, _, y2 = obs.box
            cy = (y1 + y2) * 0.5
            height = max(1, obs.frame_height)
            lower_focus.append(1.0 if cy > 0.55 * height else max(0.0, cy / (0.55 * height)))
        detail_area = score_mean([min(1.0, obs.area_ratio / 0.08) for obs in valid]) or 0.0
        return clamp_score((0.55 * (score_mean(lower_focus) or 0.0) + 0.45 * detail_area) * 100.0)
    if target in ("full_body", "whole_body"):
        return clamp_score((score_mean([(1.0 - obs.truncation) * 100.0 for obs in valid]) or 0.0))
    if target in ("chest", "torso", "upper_body"):
        return 0.0
    return clamp_score((score_mean([obs.center_score * 100.0 for obs in valid]) or 0.0))


def score_intent_consistency(
    *,
    segment: dict[str, Any],
    observations: list[FrameObservation],
    vlm_result: dict[str, Any],
    subject_applicable: bool = True,
) -> dict[str, Any]:
    if not subject_applicable:
        return {
            "IC1_shot_size_match": None,
            "IC2_semantic_target_match": None,
            "IC3_event_alignment": clamp_score(vlm_result.get("score", 50.0)),
        }
    return {
        "IC1_shot_size_match": clamp_score(score_shot_size(str(segment.get("expected_shot_size", "")), observations)),
        "IC2_semantic_target_match": clamp_score(score_semantic_target(str(segment.get("expected_semantic_target", "")), observations)),
        "IC3_event_alignment": clamp_score(vlm_result.get("score", 50.0)),
    }


def score_motion_smoothness(video_path: Path) -> float:
    cv = require_cv2()
    samples = dense_frames(video_path, max_frames=96)
    if len(samples) < 3:
        return 50.0
    grays = [cv.resize(cv.cvtColor(sample.frame_bgr, cv.COLOR_BGR2GRAY), (160, 90), interpolation=cv.INTER_AREA) for sample in samples]
    diffs = []
    flow_magnitudes = []
    for index in range(1, len(grays)):
        prev = grays[index - 1]
        cur = grays[index]
        diffs.append(float(np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32))) / 255.0))
        flow = cv.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 11, 2, 5, 1.1, 0)
        mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(float(np.mean(mag)))
    if not diffs:
        return 50.0
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(np.array(diffs) - median))) + 1e-6
    spike_rate = float(np.mean(np.array(diffs) > median + 4.0 * mad))
    diff_cv = float(np.std(diffs) / (np.mean(diffs) + 1e-6))
    flow_cv = float(np.std(flow_magnitudes) / (np.mean(flow_magnitudes) + 1e-6)) if flow_magnitudes else 0.0
    score = 100.0 * (1.0 - min(1.0, 0.45 * diff_cv + 0.35 * flow_cv + 0.20 * spike_rate))
    return clamp_score(score)


def score_tracking_stability(observations: list[FrameObservation]) -> float:
    valid = [obs for obs in observations if obs.box is not None and not obs.blank]
    if not observations:
        return 0.0
    detection_rate = len(valid) / len(observations)
    if len(valid) < 2:
        return clamp_score(35.0 * detection_rate)
    centers = []
    areas = []
    for obs in valid:
        assert obs.box is not None
        x1, y1, x2, y2 = obs.box
        centers.append(np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32))
        areas.append(max(1e-6, obs.area_ratio))
    center_jumps = [float(np.linalg.norm(centers[i] - centers[i - 1])) for i in range(1, len(centers))]
    area_jumps = [abs(math.log(areas[i] / areas[i - 1])) for i in range(1, len(areas))]
    center_penalty = min(1.0, float(np.mean(center_jumps)) / 180.0)
    area_penalty = min(1.0, float(np.mean(area_jumps)) / 1.0) if area_jumps else 0.0
    score = 100.0 * (0.45 * detection_rate + 0.30 * (1.0 - center_penalty) + 0.25 * (1.0 - area_penalty))
    return clamp_score(score)


def score_tracking_stability_or_none(observations: list[FrameObservation], *, applicable: bool = True) -> float | None:
    if not applicable:
        return None
    return score_tracking_stability(observations)


def image_similarity(path_a: str, path_b: str) -> float:
    cv = require_cv2()
    a = cv.imread(path_a, cv.IMREAD_GRAYSCALE)
    b = cv.imread(path_b, cv.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return 0.0
    b = cv.resize(b, (a.shape[1], a.shape[0]), interpolation=cv.INTER_AREA)
    a_small = cv.resize(a, (160, 90), interpolation=cv.INTER_AREA).astype(np.float32)
    b_small = cv.resize(b, (160, 90), interpolation=cv.INTER_AREA).astype(np.float32)
    diff = float(np.mean(np.abs(a_small - b_small)) / 255.0)
    return max(0.0, 1.0 - diff)


def boundary_continuity(prev_result: dict[str, Any], cur_result: dict[str, Any]) -> dict[str, Any]:
    prev_tail = prev_result.get("tail_frame_path", "")
    cur_head = cur_result.get("head_frame_path", "")
    sim = image_similarity(prev_tail, cur_head) if prev_tail and cur_head else 0.0
    prev_box = prev_result.get("tail_box")
    cur_box = cur_result.get("head_box")
    center_jump_score = 50.0
    if prev_box and cur_box:
        px = (prev_box[0] + prev_box[2]) * 0.5
        py = (prev_box[1] + prev_box[3]) * 0.5
        cx = (cur_box[0] + cur_box[2]) * 0.5
        cy = (cur_box[1] + cur_box[3]) * 0.5
        center_jump = math.dist((px, py), (cx, cy))
        center_jump_score = 100.0 * max(0.0, 1.0 - center_jump / 360.0)
    score = 100.0 * (0.45 * sim + 0.55 * (center_jump_score / 100.0))
    return {
        "TQ3_cut_continuity": clamp_score(score),
        "boundary_similarity": round(sim, 4),
        "boundary_center_score": clamp_score(center_jump_score),
    }


def evaluate_segment(
    *,
    segment: dict[str, Any],
    subjects_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    vlm_client: VisionLanguageClient,
    uniform_frames: int = 8,
) -> dict[str, Any]:
    video_path = Path(str(segment.get("video_path", "")))
    expected_subjects = [str(item) for item in segment.get("expected_subject_ids", [])]
    primary_subject = subjects_by_id.get(expected_subjects[0]) if expected_subjects else None
    if expected_subjects and primary_subject is None:
        raise SegmentEvaluationError(
            "expected_subject_unknown",
            f"Unknown expected subject id: {expected_subjects[0]}",
            details={"subject_id": expected_subjects[0]},
        )
    info, samples = sample_video(video_path, output_dir, uniform_count=uniform_frames, motion_peak_count=2, write_frames=True)
    observations = observe_frames(samples, primary_subject=primary_subject, evidence_dir=output_dir)
    keyframe_paths = [sample.path for sample in samples if sample.path]
    vlm_result = vlm_client.score_event_alignment(segment=segment, video_path=video_path, keyframe_paths=keyframe_paths)
    subject_applicable = primary_subject is not None
    sp = score_subject_perception(observations, applicable=subject_applicable)
    ic = score_intent_consistency(segment=segment, observations=observations, vlm_result=vlm_result, subject_applicable=subject_applicable)
    tq = {
        "TQ1_motion_smoothness": score_motion_smoothness(video_path),
        "TQ2_subject_tracking_stability": score_tracking_stability_or_none(observations, applicable=subject_applicable),
        "TQ3_cut_continuity": None,
    }
    head = observations[0] if observations else None
    tail = observations[-1] if observations else None
    metrics = {**sp, **ic, **tq}
    dimensions = {
        "subject_perception": clamp_score(score_mean(list(sp.values()))) if subject_applicable else None,
        "intent_consistency": clamp_score(score_mean(list(ic.values()))),
        "trajectory_quality": clamp_score(score_mean([tq["TQ1_motion_smoothness"], tq["TQ2_subject_tracking_stability"]])),
    }
    result = {
        "segment_id": segment.get("segment_id", ""),
        "status": "success",
        "failure_reason": None,
        "failure_details": {},
        "video_path": str(video_path),
        "video_info": asdict(info),
        "metrics": metrics,
        "dimensions": dimensions,
        "overall": clamp_score(score_mean(list(dimensions.values()))),
        "vlm": vlm_result,
        "observations": [asdict(obs) for obs in observations],
        "head_frame_path": samples[0].path if samples else "",
        "tail_frame_path": samples[-1].path if samples else "",
        "head_box": list(head.box) if head and head.box else None,
        "tail_box": list(tail.box) if tail and tail.box else None,
        "warnings": [],
        "evaluation_scope": "subject" if subject_applicable else "scene",
    }
    if any(obs.blank for obs in observations):
        result["warnings"].append("blank_keyframe_detected")
    if subject_applicable:
        detected_count = sum(1 for obs in observations if obs.box is not None)
        if detected_count == 0:
            result["warnings"].append("primary_subject_not_detected")
        elif detected_count < len(observations):
            result["warnings"].append("primary_subject_partially_not_detected")
        if any(obs.box_source == "identity_match_below_threshold" for obs in observations):
            result["warnings"].append("identity_match_below_threshold")
        if any(obs.box_source == "yolo_subject_not_detected" for obs in observations):
            result["warnings"].append("yolo_subject_not_detected")
    return result


def finalize_boundary_scores(results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda row: int(row.get("order_index", 0)))
    for index, result in enumerate(ordered):
        if result.get("status") != "success":
            result["boundary"] = {"applicable": False, "skipped_reason": "segment_failed"}
            continue
        if index == 0:
            result["metrics"]["TQ3_cut_continuity"] = None
            result["boundary"] = {"applicable": False}
        elif ordered[index - 1].get("status") != "success":
            result["metrics"]["TQ3_cut_continuity"] = None
            result["boundary"] = {"applicable": False, "skipped_reason": "previous_segment_failed"}
        else:
            boundary = boundary_continuity(ordered[index - 1], result)
            result["metrics"]["TQ3_cut_continuity"] = boundary["TQ3_cut_continuity"]
            result["boundary"] = {"applicable": True, **boundary}
        tq_values = [
            result["metrics"].get("TQ1_motion_smoothness"),
            result["metrics"].get("TQ2_subject_tracking_stability"),
            result["metrics"].get("TQ3_cut_continuity"),
        ]
        result["dimensions"]["trajectory_quality"] = clamp_score(score_mean(tq_values))
        result["overall"] = clamp_score(score_mean(list(result["dimensions"].values())))
