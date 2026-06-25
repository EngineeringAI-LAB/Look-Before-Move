"""Standalone shot-contract schema and generation for Plan-A."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from director_engine_llm import call_json_response, image_path_to_data_url, llm_ready
from director_engine_paths import iter_camera_instructions, utc_now
from director_shot_size_policy import contract_for_distance


SHOT_CONTRACT_SYSTEM_PROMPT = """You are the detailed shot-contract planner for Plan-A.

You must produce a precise camera contract that downstream blocking and trajectory
stages can execute without guessing intent.

Return one JSON object with:
- scene_id
- shot_id
- camera_name
- shot_intent
- start_frame_contract
- keyframe_plan
- motion_contract
- environment_constraints
- editorial_contract
- fallback_contract

Rules:
- keyframe_plan must include kf0, kf1, kf2
- each keyframe must define primary_focus_id, secondary_focus_ids, primary_semantic_target, secondary_semantic_targets, shot_size, screen_region,
  primary_screen_area_target, primary_screen_area_floor, visibility_ratio_target,
  secondary_screen_area_min, secondary_screen_area_max, max_secondary_overlap_ratio
- primary_semantic_target must be a string indicating the specific target to focus on (e.g., "face", "back_of_head", "feet", "full_body"). Do not use "hands", "chest", "torso", or "upper_body" as a primary close-up target.
- secondary_semantic_targets must be a dictionary mapping each secondary_focus_id to its semantic target (e.g., {"vito_tuxedo": "back_of_head"}).
- Treat shot_intent as the ultimate guide for semantic targets. If it says "zooms out to show the back of Vito's head", Vito's semantic target must be "back_of_head".
- use the scene understanding and dossier to decide when secondary context enters
- treat character previews as one environmental preview set from the original scene
- treat the environmental preview set as identity/staging/context reference, not as the exact acting pose for the final shot
- treat front/back/left/right labels as subject-side semantics, not Blender viewpoint labels
- do not widen a close-up so much that it stops reading as the intended shot size
- fallback_contract must clearly say what can relax and what must never relax
- start_frame_contract must include start_focus_ids, secondary_focus_ids, primary_semantic_target, secondary_semantic_targets, start_subject_layout,
  shot_size_contract, seed_priority
- blocking_requirements must say how blocking should choose candidates before trajectory generation
- Return raw JSON only. Do not use markdown code fences.
- Keep the top-level keys exactly as requested and keep their types stable:
  - scene_id: integer
  - shot_id: integer
  - camera_name: string
  - shot_intent: string
  - start_frame_contract: object
  - keyframe_plan: list of exactly 3 objects, ordered kf0, kf1, kf2
  - motion_contract: object
  - blocking_requirements: object
  - environment_constraints: object
  - editorial_contract: object
  - fallback_contract: object
- In start_frame_contract, keep start_subject_layout as an object, not a string label.
- If you are uncertain, use empty objects, empty lists, or conservative scalar defaults instead of changing the field type.
- Do not wrap the response in markdown or ```json fences.
- Example minimal valid top-level skeleton:
  {"scene_id":1,"shot_id":1,"camera_name":"cam_x","shot_intent":"...","start_frame_contract":{"primary_semantic_target":"face","secondary_semantic_targets":{}},"keyframe_plan":[{"keyframe_id":"kf0","primary_semantic_target":"face","secondary_semantic_targets":{}},{"keyframe_id":"kf1"},{"keyframe_id":"kf2"}],"motion_contract":{},"blocking_requirements":{},"environment_constraints":{},"editorial_contract":{},"fallback_contract":{}}

Return JSON only.
"""


@dataclass(slots=True)
class KeyframeContract:
    keyframe_id: str
    primary_focus_id: str
    secondary_focus_ids: list[str] = field(default_factory=list)
    primary_semantic_target: str = "full_body"
    secondary_semantic_targets: dict[str, str] = field(default_factory=dict)
    shot_size: str = "medium shot"
    screen_region: str = "center"
    primary_screen_area_target: float = 0.14
    primary_screen_area_floor: float = 0.1
    visibility_ratio_target: float = 0.82
    secondary_screen_area_min: float = 0.0
    secondary_screen_area_max: float = 0.0
    max_secondary_overlap_ratio: float = 0.2
    must_show_face: bool = False
    must_show_hands: bool = False
    must_show_interaction: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ShotContract:
    scene_id: int
    shot_id: int
    camera_name: str
    shot_intent: str
    start_frame_contract: dict[str, Any]
    keyframe_plan: list[dict[str, Any]]
    motion_contract: dict[str, Any]
    blocking_requirements: dict[str, Any]
    environment_constraints: dict[str, Any]
    editorial_contract: dict[str, Any]
    fallback_contract: dict[str, Any]
    source: str = "deterministic_fallback"
    llm_error: str = ""
    generated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _inferred_emphasis_flags(camera_instruction: dict[str, Any]) -> dict[str, bool]:
    description = str(camera_instruction.get("description") or "").lower()
    focus_ids = list(camera_instruction.get("focus_on_ids", []) or [])
    return {
        "must_show_face": "face" in description or "facial" in description or "close-up" in description,
        "must_show_hands": False,
        "must_show_interaction": len(focus_ids) > 1 or "interaction" in description or "together" in description,
    }


def _sanitize_primary_semantic_target(value: Any, fallback: str = "full_body") -> str:
    text = _as_string(value, fallback).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"hand", "hands", "chest", "torso", "upper_body"}:
        return "full_body"
    if text in {"face", "eyes", "head", "back_of_head", "feet", "foot", "shoes", "full_body"}:
        return text
    return fallback


def _sanitize_semantic_target_map(value: Any) -> dict[str, str]:
    rows = _as_dict(value)
    sanitized: dict[str, str] = {}
    for focus_id, target in rows.items():
        key = _as_string(focus_id, "")
        if not key:
            continue
        sanitized[key] = _sanitize_primary_semantic_target(target, "full_body")
    return sanitized


def _as_string(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        result = []
        for item in value:
            text = _as_string(item, "")
            if text:
                result.append(text)
        return result
    if isinstance(value, tuple):
        return _as_string_list(list(value))
    text = _as_string(value, "")
    return [text] if text else []


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return fallback


def _is_number_sequence(value: Any, *, minimum_length: int = 2) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) < minimum_length:
        return False
    try:
        for item in list(value)[:minimum_length]:
            float(item)
    except Exception:
        return False
    return True


def _valid_layout_value(value: Any) -> bool:
    if _is_number_sequence(value):
        return True
    if not isinstance(value, dict):
        return False
    for key in ("center", "location", "screen_center", "screen_position", "screen_region"):
        if key in value and _is_number_sequence(value.get(key)):
            return True
    return False


def _sanitize_start_subject_layout(value: Any) -> dict[str, Any]:
    layout = _as_dict(value)
    if not layout:
        return {}
    if _valid_layout_value(layout):
        return layout
    sanitized: dict[str, Any] = {}
    for key, item in layout.items():
        if _valid_layout_value(item):
            sanitized[str(key)] = item
    return sanitized


def _normalize_motion_contract(value: Any) -> dict[str, Any]:
    motion = _as_dict(value)
    if not motion:
        return {}
    motion_type = _as_string(motion.get("motion_type"), "")
    normalized = motion_type.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"dolly_back", "dolly_backward", "pull_back", "pullback"}:
        motion["motion_type"] = "dolly_out"
    return motion


def _subject_layout_from_dossier(scene_dossier: dict[str, Any], focus_id: str) -> dict[str, Any]:
    for row in scene_dossier.get("character_summaries", []) or []:
        if str(row.get("character_id") or "") == str(focus_id or ""):
            return dict(row.get("layout_data") or {})
    return {}


WEAK_CHARACTER_ID_TOKENS = {
    "adult",
    "aged",
    "elder",
    "elderly",
    "middle",
    "older",
    "old",
    "young",
}


def _character_ids_from_dossier(scene_dossier: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for row in scene_dossier.get("character_summaries", []) or []:
        character_id = _as_string(row.get("character_id"), "")
        if character_id and character_id not in rows:
            rows.append(character_id)
    return rows


def _character_match_tokens(character_id: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in str(character_id or "").replace("-", "_").split("_"):
        token = raw_token.strip().lower()
        if len(token) <= 1 or token in WEAK_CHARACTER_ID_TOKENS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def _first_token_hit(text: str, tokens: list[str]) -> int | None:
    positions: list[int] = []
    for token in tokens:
        hit = re.search(rf"\b{re.escape(token)}\b", text)
        if hit:
            positions.append(hit.start())
    return min(positions) if positions else None


def _token_match_stats(text: str, tokens: list[str]) -> tuple[int, int, int] | None:
    positions: list[int] = []
    matched = 0
    for token in tokens:
        hit = re.search(rf"\b{re.escape(token)}\b", text)
        if hit:
            matched += 1
            positions.append(hit.start())
    if not positions:
        return None
    return min(positions), len(tokens) - matched, matched


def _narrative_primary_focus_id(scene_dossier: dict[str, Any], shot: dict[str, Any], camera_instruction: dict[str, Any]) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            shot.get("shot_description"),
            camera_instruction.get("description"),
        )
    ).lower()
    rows: list[tuple[int, int, int, int, str]] = []
    for order, character_id in enumerate(_character_ids_from_dossier(scene_dossier)):
        tokens = _character_match_tokens(character_id)
        match_stats = _token_match_stats(text, tokens)
        if match_stats is not None:
            hit, missing_count, matched_count = match_stats
            rows.append((hit, missing_count, -matched_count, order, character_id))
    if not rows:
        return ""
    rows.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return rows[0][4]


def _keyframe_primary_ids(contract: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    keyframes = contract.get("keyframe_plan") or []
    if not isinstance(keyframes, list):
        return rows
    for keyframe in keyframes:
        if isinstance(keyframe, dict):
            rows.extend(_as_string_list(keyframe.get("primary_focus_id")))
    return rows


def _keyframe_majority_primary_id(contract: dict[str, Any]) -> str:
    ids = _keyframe_primary_ids(contract)
    if not ids:
        return ""
    counts: dict[str, int] = {}
    for focus_id in ids:
        counts[focus_id] = counts.get(focus_id, 0) + 1
    focus_id, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return focus_id if count >= max(1, len(ids) // 2 + 1) else ""


def _merge_focus_ids(*groups: list[str]) -> list[str]:
    rows: list[str] = []
    for group in groups:
        for focus_id in group:
            text = _as_string(focus_id, "")
            if text and text not in rows:
                rows.append(text)
    return rows


def _repair_focus_consistency(
    *,
    scene_dossier: dict[str, Any],
    shot: dict[str, Any],
    camera_instruction: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Ensure every focus-bearing field agrees before Director hands off downstream."""
    repaired = dict(contract)
    start = dict(repaired.get("start_frame_contract") or {})
    original_primary = _as_string(start.get("primary_focus_id"), "")
    start_ids = _as_string_list(start.get("start_focus_ids"))
    keyframe_majority = _keyframe_majority_primary_id(repaired)
    narrative_primary = _narrative_primary_focus_id(scene_dossier, shot, camera_instruction)
    camera_focus_ids = _as_string_list(camera_instruction.get("focus_on_ids"))
    narrative_text = " ".join(
        str(value or "")
        for value in (
            shot.get("shot_description"),
            camera_instruction.get("description"),
        )
    ).lower()
    original_primary_mentioned = bool(_first_token_hit(narrative_text, _character_match_tokens(original_primary)))

    canonical_primary = original_primary or (start_ids[0] if start_ids else "") or (camera_focus_ids[0] if camera_focus_ids else "")
    evidence: list[str] = []
    if narrative_primary and original_primary and narrative_primary != original_primary and not original_primary_mentioned:
        canonical_primary = narrative_primary
        evidence.append("narrative_primary_over_unmentioned_primary")
    elif start_ids and keyframe_majority and start_ids[0] == keyframe_majority:
        canonical_primary = start_ids[0]
        evidence.append("start_focus_ids_and_keyframes_agree")
    elif narrative_primary and narrative_primary in start_ids:
        canonical_primary = narrative_primary
        evidence.append("narrative_primary_in_start_focus_ids")
    elif narrative_primary and keyframe_majority == narrative_primary:
        canonical_primary = narrative_primary
        evidence.append("narrative_primary_matches_keyframes")
    elif keyframe_majority:
        canonical_primary = keyframe_majority
        evidence.append("keyframe_majority")

    if not canonical_primary:
        repaired["start_frame_contract"] = start
        return repaired

    keyframe_secondaries: list[str] = []
    for keyframe in repaired.get("keyframe_plan") or []:
        if isinstance(keyframe, dict):
            keyframe_secondaries.extend(_as_string_list(keyframe.get("secondary_focus_ids")))
    secondary_ids = [
        focus_id
        for focus_id in _merge_focus_ids(
            _as_string_list(start.get("secondary_focus_ids")),
            start_ids,
            camera_focus_ids,
            keyframe_secondaries,
            [original_primary],
        )
        if focus_id != canonical_primary
    ]

    original_start = dict(start)
    start["primary_focus_id"] = canonical_primary
    start["start_focus_ids"] = [canonical_primary]
    start["secondary_focus_ids"] = secondary_ids
    semantic_targets = _sanitize_semantic_target_map(start.get("secondary_semantic_targets") or {})
    semantic_targets.pop(canonical_primary, None)
    for focus_id in secondary_ids:
        semantic_targets.setdefault(focus_id, "full_body")
    start["secondary_semantic_targets"] = semantic_targets

    layout = _sanitize_start_subject_layout(start.get("start_subject_layout"))
    layout_object = _as_string(layout.get("object_name"), "") if isinstance(layout, dict) else ""
    if not layout or (layout_object and layout_object != canonical_primary):
        dossier_layout = _subject_layout_from_dossier(scene_dossier, canonical_primary)
        if dossier_layout:
            start["start_subject_layout"] = dossier_layout
    repaired["start_frame_contract"] = start

    keyframes = []
    for row in repaired.get("keyframe_plan") or []:
        if not isinstance(row, dict):
            continue
        keyframe = dict(row)
        keyframe_original_primary = _as_string(keyframe.get("primary_focus_id"), "")
        keyframe["primary_focus_id"] = canonical_primary
        keyframe["secondary_focus_ids"] = [
            focus_id
            for focus_id in _merge_focus_ids(
                _as_string_list(keyframe.get("secondary_focus_ids")),
                [keyframe_original_primary],
                secondary_ids,
            )
            if focus_id != canonical_primary
        ]
        keyframe_targets = _sanitize_semantic_target_map(keyframe.get("secondary_semantic_targets") or {})
        keyframe_targets.pop(canonical_primary, None)
        for focus_id in keyframe["secondary_focus_ids"]:
            keyframe_targets.setdefault(focus_id, "full_body")
        keyframe["secondary_semantic_targets"] = keyframe_targets
        keyframes.append(keyframe)
    if keyframes:
        repaired["keyframe_plan"] = keyframes

    applied = bool(original_primary and original_primary != canonical_primary) or bool(original_start != start)
    if applied:
        repaired["focus_consistency_repair"] = {
            "applied": True,
            "original_primary_focus_id": original_primary,
            "canonical_primary_focus_id": canonical_primary,
            "original_start_focus_ids": start_ids,
            "keyframe_majority_primary_id": keyframe_majority,
            "narrative_primary_focus_id": narrative_primary,
            "evidence": evidence,
        }
    else:
        repaired.setdefault("focus_consistency_repair", {"applied": False})
    return repaired


def _normalize_keyframe_plan(payload_value: Any, fallback_keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload_value, dict):
        sequence = [payload_value.get("kf0"), payload_value.get("kf1"), payload_value.get("kf2")]
    elif isinstance(payload_value, list):
        sequence = payload_value[:3]
    else:
        return fallback_keyframes

    normalized: list[dict[str, Any]] = []
    for index, fallback in enumerate(fallback_keyframes):
        row = sequence[index] if index < len(sequence) else None
        row_dict = _as_dict(row)
        merged = dict(fallback)
        if row_dict:
            merged["keyframe_id"] = _as_string(row_dict.get("keyframe_id"), fallback.get("keyframe_id", f"kf{index}"))
            merged["primary_focus_id"] = _as_string(row_dict.get("primary_focus_id"), fallback.get("primary_focus_id", ""))
            merged["secondary_focus_ids"] = _as_string_list(row_dict.get("secondary_focus_ids")) or list(fallback.get("secondary_focus_ids", []))
            merged["primary_semantic_target"] = _sanitize_primary_semantic_target(
                row_dict.get("primary_semantic_target"),
                _sanitize_primary_semantic_target(fallback.get("primary_semantic_target"), "full_body"),
            )
            merged["secondary_semantic_targets"] = _sanitize_semantic_target_map(
                row_dict.get("secondary_semantic_targets")
            ) or _sanitize_semantic_target_map(fallback.get("secondary_semantic_targets", {}))
            merged["shot_size"] = _as_string(row_dict.get("shot_size"), fallback.get("shot_size", "medium shot"))
            merged["screen_region"] = _as_string(row_dict.get("screen_region"), fallback.get("screen_region", "center"))
            for key in (
                "primary_screen_area_target",
                "primary_screen_area_floor",
                "visibility_ratio_target",
                "secondary_screen_area_min",
                "secondary_screen_area_max",
                "max_secondary_overlap_ratio",
            ):
                merged[key] = _as_float(row_dict.get(key), float(fallback.get(key, 0.0)))
            for key in ("must_show_face", "must_show_hands", "must_show_interaction"):
                merged[key] = False if key == "must_show_hands" else _as_bool(row_dict.get(key), bool(fallback.get(key, False)))
            notes = _as_string_list(row_dict.get("notes"))
            if notes:
                merged["notes"] = notes
        normalized.append(merged)
    return normalized


def _deterministic_keyframes(scene_dossier: dict[str, Any], camera_instruction: dict[str, Any]) -> list[dict[str, Any]]:
    focus_ids = list(camera_instruction.get("focus_on_ids", []) or [])
    primary_focus_id = str(camera_instruction.get("primary_focus_id") or (focus_ids[0] if focus_ids else ""))
    secondary_focus_ids = [focus_id for focus_id in focus_ids if focus_id != primary_focus_id]
    distance_label = str(camera_instruction.get("distance") or "medium shot")
    shot_size_contract = contract_for_distance(distance_label)
    emphasis = _inferred_emphasis_flags(camera_instruction)
    start_region = "center" if shot_size_contract.label == "close-up" else "center_left"
    primary_semantic_target = "face" if emphasis["must_show_face"] else "full_body"
    secondary_semantic_targets = {fid: "full_body" for fid in secondary_focus_ids}

    kf0 = KeyframeContract(
        keyframe_id="kf0",
        primary_focus_id=primary_focus_id,
        secondary_focus_ids=[] if shot_size_contract.label == "close-up" else list(secondary_focus_ids),
        primary_semantic_target=primary_semantic_target,
        secondary_semantic_targets=dict(secondary_semantic_targets) if shot_size_contract.label != "close-up" else {},
        shot_size=distance_label,
        screen_region=start_region,
        primary_screen_area_target=shot_size_contract.desired_min,
        primary_screen_area_floor=shot_size_contract.hard_floor,
        visibility_ratio_target=shot_size_contract.visibility_ratio_target,
        secondary_screen_area_min=0.0,
        secondary_screen_area_max=0.0 if shot_size_contract.label == "close-up" else 0.12,
        max_secondary_overlap_ratio=0.12 if shot_size_contract.label == "close-up" else 0.2,
        must_show_face=emphasis["must_show_face"],
        must_show_hands=False,
        must_show_interaction=False,
        notes=["Start on the primary emotional subject without letting context dominate."],
    ).to_dict()
    kf1 = KeyframeContract(
        keyframe_id="kf1",
        primary_focus_id=primary_focus_id,
        secondary_focus_ids=list(secondary_focus_ids),
        primary_semantic_target=primary_semantic_target,
        secondary_semantic_targets=dict(secondary_semantic_targets),
        shot_size=distance_label,
        screen_region=start_region,
        primary_screen_area_target=max(shot_size_contract.desired_min * 0.92, shot_size_contract.hard_floor),
        primary_screen_area_floor=shot_size_contract.hard_floor,
        visibility_ratio_target=max(shot_size_contract.visibility_ratio_target - 0.04, 0.7),
        secondary_screen_area_min=0.04 if secondary_focus_ids else 0.0,
        secondary_screen_area_max=0.12 if shot_size_contract.label == "close-up" else 0.18,
        max_secondary_overlap_ratio=0.28,
        must_show_face=emphasis["must_show_face"],
        must_show_hands=False,
        must_show_interaction=emphasis["must_show_interaction"],
        notes=["Introduce context while preserving the primary subject as the narrative anchor."],
    ).to_dict()
    kf2 = KeyframeContract(
        keyframe_id="kf2",
        primary_focus_id=primary_focus_id,
        secondary_focus_ids=list(secondary_focus_ids),
        primary_semantic_target=primary_semantic_target,
        secondary_semantic_targets=dict(secondary_semantic_targets),
        shot_size=distance_label,
        screen_region="center_left" if start_region == "center" else start_region,
        primary_screen_area_target=max(shot_size_contract.desired_min * 0.86, shot_size_contract.hard_floor),
        primary_screen_area_floor=max(shot_size_contract.hard_floor * 0.94, 0.04),
        visibility_ratio_target=max(shot_size_contract.visibility_ratio_target - 0.06, 0.68),
        secondary_screen_area_min=0.05 if secondary_focus_ids else 0.0,
        secondary_screen_area_max=0.16 if shot_size_contract.label == "close-up" else 0.22,
        max_secondary_overlap_ratio=0.32,
        must_show_face=emphasis["must_show_face"],
        must_show_hands=False,
        must_show_interaction=emphasis["must_show_interaction"],
        notes=["End slightly wider only if the primary subject remains the clear story center."],
    ).to_dict()
    return [kf0, kf1, kf2]


def _deterministic_contract(
    scene_dossier: dict[str, Any],
    scene_understanding: dict[str, Any],
    shot: dict[str, Any],
    camera_instruction: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    focus_ids = list(camera_instruction.get("focus_on_ids", []) or [])
    primary_focus_id = str(camera_instruction.get("primary_focus_id") or (focus_ids[0] if focus_ids else ""))
    secondary_focus_ids = [focus_id for focus_id in focus_ids if focus_id != primary_focus_id]
    distance_label = str(camera_instruction.get("distance") or "medium shot")
    shot_size_contract = contract_for_distance(distance_label)
    keyframes = _deterministic_keyframes(scene_dossier, camera_instruction)
    start_subject_layout = _subject_layout_from_dossier(scene_dossier, primary_focus_id)
    contract = ShotContract(
        scene_id=int(shot.get("scene_id") or 0),
        shot_id=int(shot.get("shot_id") or 0),
        camera_name=str(camera_instruction.get("camera_name") or ""),
        shot_intent=str(camera_instruction.get("description") or "Preserve scripted focus and readability."),
        start_frame_contract={
            "primary_focus_id": primary_focus_id,
            "primary_semantic_target": keyframes[0]["primary_semantic_target"],
            "start_focus_ids": [primary_focus_id] if primary_focus_id else [],
            "secondary_focus_ids": secondary_focus_ids,
            "secondary_semantic_targets": keyframes[0].get("secondary_semantic_targets") or {},
            "distance": camera_instruction.get("distance"),
            "angle": camera_instruction.get("angle"),
            "screen_region": keyframes[0]["screen_region"],
            "primary_screen_area_target": keyframes[0]["primary_screen_area_target"],
            "primary_screen_area_floor": keyframes[0]["primary_screen_area_floor"],
            "visibility_ratio_target": keyframes[0]["visibility_ratio_target"],
            "shot_size_contract": shot_size_contract.to_dict(),
            "seed_priority": shot_size_contract.seed_priority,
            "start_subject_layout": start_subject_layout,
        },
        keyframe_plan=keyframes,
        motion_contract={
            "movement": camera_instruction.get("movement"),
            "direction": camera_instruction.get("direction"),
            "allow_semantic_light_dynamic": True,
            "must_preserve_primary_anchor": True,
            "sync_to_action_duration": False,
            "story_priority": "story_first",
            "keyframe_count": len(keyframes),
        },
        blocking_requirements={
            "scene_context_required": True,
            "use_scene_top_view": True,
            "use_character_turnarounds": True,
            "prefer_primary_first_seeding": shot_size_contract.seed_priority == "primary_first",
            "top_k_before_llm_selection": 15,
            "top_k_threshold": 0.85,
            "shot_size_research_trigger": shot_size_contract.re_search_trigger,
            "planning_constraints": list(scene_understanding.get("planning_constraints") or []),
        },
        environment_constraints={
            "must_avoid_wall_leaks": True,
            "must_avoid_room_shell_exit": True,
            "must_avoid_severe_focus_occlusion": True,
        },
        editorial_contract={
            "source_clip_policy": "full_action_plus_editorial_margin",
            "allow_trim_after_review": True,
        },
        fallback_contract={
            "may_reduce_motion_before_reframing_subject_importance": True,
            "must_not_drop_primary_focus": True,
            "must_not_downgrade_close_up_to_medium_without_explicit_failure_reason": True,
            "must_preserve_shot_size_hard_floor": True,
            "must_preserve_primary_first_start_frame": shot_size_contract.seed_priority == "primary_first",
        },
        source="deterministic_fallback",
        llm_error=error,
    )
    return contract.to_dict()


def _merge_shot_contract_payload(
    base_contract: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    contract = dict(base_contract)
    contract["shot_intent"] = _as_string(payload.get("shot_intent"), _as_string(contract.get("shot_intent"), ""))

    start_frame_contract = dict(contract.get("start_frame_contract") or {})
    payload_start = _as_dict(payload.get("start_frame_contract"))
    if payload_start:
        start_frame_contract["primary_focus_id"] = _as_string(
            payload_start.get("primary_focus_id"),
            _as_string(start_frame_contract.get("primary_focus_id"), ""),
        )
        start_frame_contract["primary_semantic_target"] = _sanitize_primary_semantic_target(
            payload_start.get("primary_semantic_target"),
            _sanitize_primary_semantic_target(start_frame_contract.get("primary_semantic_target"), "full_body"),
        )
        start_focus_ids = _as_string_list(payload_start.get("start_focus_ids"))
        if start_focus_ids:
            start_frame_contract["start_focus_ids"] = start_focus_ids
        secondary_focus_ids = _as_string_list(payload_start.get("secondary_focus_ids"))
        if secondary_focus_ids or "secondary_focus_ids" in payload_start:
            start_frame_contract["secondary_focus_ids"] = secondary_focus_ids
        start_frame_contract["secondary_semantic_targets"] = _sanitize_semantic_target_map(
            payload_start.get("secondary_semantic_targets") or start_frame_contract.get("secondary_semantic_targets") or {}
        )
        for key in ("distance", "angle", "screen_region", "seed_priority"):
            start_frame_contract[key] = _as_string(payload_start.get(key), _as_string(start_frame_contract.get(key), ""))
        for key in ("primary_screen_area_target", "primary_screen_area_floor", "visibility_ratio_target"):
            start_frame_contract[key] = _as_float(payload_start.get(key), float(start_frame_contract.get(key, 0.0)))
        payload_layout = _sanitize_start_subject_layout(payload_start.get("start_subject_layout"))
        if payload_layout:
            start_frame_contract["start_subject_layout"] = payload_layout
        payload_shot_size_contract = _as_dict(payload_start.get("shot_size_contract"))
        if payload_shot_size_contract:
            start_frame_contract["shot_size_contract"] = payload_shot_size_contract
    contract["start_frame_contract"] = start_frame_contract

    contract["keyframe_plan"] = _normalize_keyframe_plan(payload.get("keyframe_plan"), list(contract.get("keyframe_plan") or []))

    payload_motion = _normalize_motion_contract(payload.get("motion_contract"))
    if payload_motion:
        merged_motion = dict(contract.get("motion_contract") or {})
        merged_motion.update(payload_motion)
        contract["motion_contract"] = merged_motion

    for section_key in (
        "blocking_requirements",
        "environment_constraints",
        "editorial_contract",
        "fallback_contract",
    ):
        payload_section = _as_dict(payload.get(section_key))
        if payload_section:
            merged_section = dict(contract.get(section_key) or {})
            merged_section.update(payload_section)
            contract[section_key] = merged_section
    return contract


def _referenced_character_rows(scene_dossier: dict[str, Any], focus_ids: list[str]) -> list[dict[str, Any]]:
    character_rows = scene_dossier.get("character_summaries", []) or []
    return [row for row in character_rows if row.get("character_id") in focus_ids]


def _build_llm_user_content(
    *,
    scene_dossier: dict[str, Any],
    scene_understanding: dict[str, Any],
    shot: dict[str, Any],
    camera_instruction: dict[str, Any],
) -> list[dict[str, Any]]:
    focus_ids = list(camera_instruction.get("focus_on_ids", []) or [])
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Scene dossier scene_id={scene_dossier.get('scene_id')}\n"
                f"Scene layout data: {scene_dossier.get('layout_data')}\n"
                f"Scene metadata: {scene_dossier.get('metadata')}\n"
                f"Asset summaries: {scene_dossier.get('asset_summaries')}\n"
                f"Scene understanding: {scene_understanding}\n"
                f"Shot: scene_id={shot.get('scene_id')} shot_id={shot.get('shot_id')}\n"
                f"Camera instruction: {camera_instruction}"
            ),
        }
    ]
    scene_top_view_path = str(scene_dossier.get("scene_top_view_path") or "").strip()
    if scene_top_view_path and Path(scene_top_view_path).exists():
        user_content.append({"type": "text", "text": "Scene top view for spatial context."})
        user_content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(scene_top_view_path)}})

    for row in _referenced_character_rows(scene_dossier, focus_ids):
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Referenced character {row.get('character_id')}.\n"
                    f"Layout data: {row.get('layout_data')}\n"
                    f"Summary: {row.get('summary')}"
                ),
            }
        )
        turnaround_paths = row.get("context_turnaround_paths") or {}
        for direction_name, image_path in turnaround_paths.items():
            image_path = str(image_path or "").strip()
            if not image_path or not Path(image_path).exists():
                continue
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"{row.get('character_id')} environmental preview {direction_name} "
                        "(subject-side semantic label)."
                    ),
                }
            )
            user_content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path)}})
    return user_content


def generate_scene_shot_contracts(
    *,
    scene_dossier: dict[str, Any],
    scene_understanding: dict[str, Any],
    scene_shots: list[dict[str, Any]],
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str,
    keyframe_count: int = 3,
) -> list[dict[str, Any]]:
    """Generate detailed shot contracts for every camera in one scene."""

    _ = keyframe_count
    contracts: list[dict[str, Any]] = []
    for shot in scene_shots:
        for camera_instruction in iter_camera_instructions(shot):
            if not llm_ready(model=vision_model, api_key=anyllm_api_key):
                contract = _deterministic_contract(
                    scene_dossier,
                    scene_understanding,
                    shot,
                    camera_instruction,
                    error="llm_unavailable",
                )
                contracts.append(
                    _repair_focus_consistency(
                        scene_dossier=scene_dossier,
                        shot=shot,
                        camera_instruction=camera_instruction,
                        contract=contract,
                    )
                )
                continue

            user_content = _build_llm_user_content(
                scene_dossier=scene_dossier,
                scene_understanding=scene_understanding,
                shot=shot,
                camera_instruction=camera_instruction,
            )
            payload, error, raw_text = call_json_response(
                model=vision_model,
                system_prompt=SHOT_CONTRACT_SYSTEM_PROMPT,
                user_content=user_content,
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                provider=anyllm_provider,
            )
            if payload is None:
                contract = _deterministic_contract(
                    scene_dossier,
                    scene_understanding,
                    shot,
                    camera_instruction,
                    error=error or "llm_failed",
                )
                contracts.append(
                    _repair_focus_consistency(
                        scene_dossier=scene_dossier,
                        shot=shot,
                        camera_instruction=camera_instruction,
                        contract=contract,
                    )
                )
                continue

            contract = _deterministic_contract(scene_dossier, scene_understanding, shot, camera_instruction)
            contract = _merge_shot_contract_payload(contract, payload)
            contract = _repair_focus_consistency(
                scene_dossier=scene_dossier,
                shot=shot,
                camera_instruction=camera_instruction,
                contract=contract,
            )
            contract["source"] = "llm_shot_contract"
            contract["raw_llm_response"] = raw_text
            contracts.append(contract)
    return contracts


__all__ = [
    "KeyframeContract",
    "ShotContract",
    "generate_scene_shot_contracts",
]


