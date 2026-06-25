"""Standalone Plan-A scene-understanding generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from director_engine_llm import call_json_response, image_path_to_data_url, llm_ready
from director_engine_paths import utc_now


SCENE_UNDERSTANDING_SYSTEM_PROMPT = """You are the scene-understanding planner for Plan-A.

You are given:
- a scene dossier with layout data
- per-character environmental preview images from the original Blender scene (front/back/left/right plus diagonals and top)
- a scene top view
- shot summaries for the scene

Produce one JSON object with these keys:
- scene_id
- summary
- character_relationships
- spatial_observations
- occlusion_risks
- recommended_shot_biases
- planning_constraints

Guidelines:
- Focus on concrete staging and camera-planning implications.
- Explain who dominates the scene, who is foreground/background power context, and which subjects should anchor close-ups.
- Use top view for spatial topology and environmental preview images for subject-side understanding.
- Treat environmental preview images as reference views for identity, side visibility, wardrobe silhouette, and scene context.
- Do not over-interpret rigid arm pose or neutral standing pose in the previews as the literal acting beat of the scripted shot.
- Interpret front/back/left/right as subject-side semantics: they describe which side of the character the camera sees, not raw Blender viewpoint labels.
- Be specific about occlusion or wall-risk zones when the dossier suggests them.
- Return raw JSON only. Do not use markdown code fences.
- Keep every top-level field in the requested type:
  - scene_id: string
  - summary: string
  - character_relationships: list of strings
  - spatial_observations: list of strings
  - occlusion_risks: list of strings
  - recommended_shot_biases: list of strings
  - planning_constraints: list of strings
- If you are uncertain, use an empty list or concise string instead of changing the field type.
- Do not wrap the response in markdown or ```json fences.
- Example minimal valid response:
  {"scene_id":"1","summary":"...", "character_relationships":[], "spatial_observations":[], "occlusion_risks":[], "recommended_shot_biases":[], "planning_constraints":[]}
Return JSON only.
"""


@dataclass(slots=True)
class SceneUnderstandingResult:
    scene_id: str
    summary: str
    character_relationships: list[str] = field(default_factory=list)
    spatial_observations: list[str] = field(default_factory=list)
    occlusion_risks: list[str] = field(default_factory=list)
    recommended_shot_biases: list[str] = field(default_factory=list)
    planning_constraints: list[str] = field(default_factory=list)
    source: str = "deterministic_fallback"
    llm_error: str = ""
    generated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_scene_summary(scene_dossier: dict[str, Any], scene_shots: list[dict[str, Any]]) -> str:
    scene_id = scene_dossier.get("scene_id")
    character_rows = scene_dossier.get("character_summaries", []) or []
    character_ids = [row.get("character_id") for row in character_rows if row.get("character_id")]
    shot_count = len(scene_shots)
    return (
        f"Scene {scene_id} contains {len(character_ids)} tracked characters: {', '.join(character_ids)}. "
        f"The scene currently has {shot_count} scripted shots in the filtered payload."
    )


def _as_string(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


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


def _deterministic_scene_understanding(scene_dossier: dict[str, Any], scene_shots: list[dict[str, Any]], error: str = "") -> dict[str, Any]:
    character_rows = scene_dossier.get("character_summaries", []) or []
    character_ids = [row.get("character_id") for row in character_rows if row.get("character_id")]
    spatial_observations = []
    for row in character_rows:
        layout = row.get("layout_data") or {}
        center = layout.get("center") or layout.get("location")
        if center:
            spatial_observations.append(f"{row.get('character_id')} center={center}")
    occlusion_risks = []
    notable_objects = ((scene_dossier.get("layout_data") or {}).get("notable_scene_objects") or [])[:8]
    if notable_objects:
        occlusion_risks.append("Potential structure or furniture blockers: " + ", ".join(str(item) for item in notable_objects))
    recommended_shot_biases = []
    if character_ids:
        recommended_shot_biases.append(
            f"Use {character_ids[0]} as the conservative primary anchor when a shot contract does not specify a clearer start focus."
        )
    recommended_shot_biases.append("Use scene top view for wall and furniture risk understanding before generating shot contracts.")
    planning_constraints = [
        "Do not let foreground context overpower the primary emotional subject in close-up contracts.",
        "Do not place the camera outside plausible room or wall bounds inferred from the scene dossier.",
    ]
    return SceneUnderstandingResult(
        scene_id=str(scene_dossier.get("scene_id") or ""),
        summary=_build_scene_summary(scene_dossier, scene_shots),
        character_relationships=[
            "Scene understanding fallback: infer relationship emphasis from shot descriptions and character co-occurrence."
        ],
        spatial_observations=spatial_observations,
        occlusion_risks=occlusion_risks,
        recommended_shot_biases=recommended_shot_biases,
        planning_constraints=planning_constraints,
        source="deterministic_fallback",
        llm_error=error,
    ).to_dict()


def _merge_scene_understanding_payload(
    scene_dossier: dict[str, Any],
    scene_shots: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    raw_text: str,
) -> dict[str, Any]:
    result = _deterministic_scene_understanding(scene_dossier, scene_shots)
    result["scene_id"] = _as_string(payload.get("scene_id"), _as_string(result.get("scene_id"), ""))
    result["summary"] = _as_string(payload.get("summary"), _as_string(result.get("summary"), ""))
    for key in (
        "character_relationships",
        "spatial_observations",
        "occlusion_risks",
        "recommended_shot_biases",
        "planning_constraints",
    ):
        llm_value = _as_string_list(payload.get(key))
        if llm_value:
            result[key] = llm_value
    result["source"] = "llm_scene_understanding"
    result["llm_error"] = ""
    result["raw_llm_response"] = raw_text
    return result


def generate_scene_understanding(
    *,
    scene_dossier: dict[str, Any],
    scene_shots: list[dict[str, Any]],
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str,
) -> dict[str, Any]:
    """Generate one scene-understanding record from dossier plus images."""

    if not llm_ready(model=vision_model, api_key=anyllm_api_key):
        return _deterministic_scene_understanding(scene_dossier, scene_shots, error="llm_unavailable")

    scene_top_view_path = str(scene_dossier.get("scene_top_view_path") or "").strip()
    character_rows = scene_dossier.get("character_summaries", []) or []
    shot_summaries = []
    for shot in scene_shots:
        shot_summaries.append(
            {
                "scene_id": shot.get("scene_id"),
                "shot_id": shot.get("shot_id"),
                "camera_instruction": shot.get("camera_instruction"),
                "additional_camera_count": len(shot.get("additional_camera_instructions", []) or []),
            }
        )

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Scene dossier for scene {scene_dossier.get('scene_id')}.\n"
                f"Layout data: {scene_dossier.get('layout_data')}\n"
                f"Scene metadata: {scene_dossier.get('metadata')}\n"
                f"Asset summaries: {scene_dossier.get('asset_summaries')}\n"
                f"Shot summaries: {shot_summaries}"
            ),
        }
    ]
    if scene_top_view_path and Path(scene_top_view_path).exists():
        user_content.append({"type": "text", "text": "Scene top view for overall spatial understanding."})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path_to_data_url(scene_top_view_path)},
            }
        )

    for row in character_rows:
        character_id = row.get("character_id")
        turnaround_paths = row.get("context_turnaround_paths") or {}
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Character {character_id}.\n"
                    f"Display name: {row.get('display_name')}\n"
                    f"Layout data: {row.get('layout_data')}\n"
                    f"Summary: {row.get('summary')}"
                ),
            }
        )
        for direction_name, image_path in turnaround_paths.items():
            image_path = str(image_path or "").strip()
            if not image_path or not Path(image_path).exists():
                continue
            user_content.append(
                {
                    "type": "text",
                    "text": f"{character_id} environmental preview {direction_name} (subject-side semantic label).",
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(image_path)},
                }
            )

    payload, error, raw_text = call_json_response(
        model=vision_model,
        system_prompt=SCENE_UNDERSTANDING_SYSTEM_PROMPT,
        user_content=user_content,
        api_key=anyllm_api_key,
        api_base=anyllm_api_base,
        provider=anyllm_provider,
    )
    if payload is None:
        return _deterministic_scene_understanding(scene_dossier, scene_shots, error=error or "llm_failed")
    return _merge_scene_understanding_payload(scene_dossier, scene_shots, payload, raw_text=raw_text)


__all__ = [
    "SceneUnderstandingResult",
    "generate_scene_understanding",
]


