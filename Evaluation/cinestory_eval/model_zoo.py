from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from .io import load_json

_QWEN_MODEL: Any | None = None
_QWEN_PROCESSOR: Any | None = None
STAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QWEN_CACHE_DIR = STAGE_DIR / "models" / "huggingface"
MODEL_LANGUAGE_POLICY = (
    "Language policy: all instructions, field descriptions, free-form reasons, "
    "and natural-language output must be written in English."
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def assert_english_model_text(text: str, *, field: str) -> None:
    if CJK_RE.search(text or ""):
        raise ValueError(f"model_input_must_be_english: cjk_text_found_at_{field}")


def load_runtime_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def shared_runtime_value(runtime: dict[str, Any], key: str, default: str = "") -> str:
    shared = runtime.get("shared") if isinstance(runtime, dict) else {}
    value = shared.get(key) if isinstance(shared, dict) else None
    return str(value or default)


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except Exception:
        return None


class VisionLanguageClient:
    def __init__(self, *, runtime_config_path: Path | None, backend: str = "auto", model: str = "") -> None:
        self.runtime = load_runtime_config(runtime_config_path)
        self.backend = backend
        if backend == "qwen_local" and not model:
            self.model = os.environ.get("CINESTORY_QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        else:
            self.model = model or shared_runtime_value(self.runtime, "vision_model", "gemini-3-flash-preview")
        self.api_key = os.environ.get("CINESTORY_VLM_API_KEY") or shared_runtime_value(self.runtime, "anyllm_api_key")
        self.api_base = os.environ.get("CINESTORY_VLM_API_BASE") or shared_runtime_value(self.runtime, "anyllm_api_base")
        self.provider = os.environ.get("CINESTORY_VLM_PROVIDER") or shared_runtime_value(self.runtime, "anyllm_provider")
        self.qwen_cache_dir = Path(os.environ.get("CINESTORY_QWEN_CACHE_DIR", str(DEFAULT_QWEN_CACHE_DIR)))

    def enabled_backend(self) -> str:
        if self.backend == "none":
            return "none"
        if self.backend in ("gemini", "openai_compatible", "qwen_local"):
            return self.backend
        provider = (self.provider or "").lower()
        model = (self.model or "").lower()
        if self.api_key and ("gemini" in provider or "gemini" in model):
            return "gemini"
        if self.api_key and self.api_base:
            return "openai_compatible"
        return "none"

    def score_event_alignment(
        self,
        *,
        segment: dict[str, Any],
        video_path: Path,
        keyframe_paths: list[str],
    ) -> dict[str, Any]:
        backend = self.enabled_backend()
        if backend == "none":
            return {
                "score": 50.0,
                "backend": "none",
                "warning": "VLM disabled or not configured; IC3 uses neutral fallback.",
                "raw": None,
            }
        prompt = self._event_prompt(segment)
        try:
            if backend == "gemini":
                return self._score_with_gemini(prompt=prompt, video_path=video_path)
            if backend == "qwen_local":
                return self._score_with_qwen_local(prompt=prompt, video_path=video_path)
            if backend == "openai_compatible":
                return self._score_with_openai_compatible(prompt=prompt, keyframe_paths=keyframe_paths)
        except Exception:
            if backend == "qwen_local" and self.backend == "qwen_local":
                raise
            exc = sys.exc_info()[1]
            return {
                "score": 50.0,
                "backend": backend,
                "warning": f"VLM call failed; neutral fallback used: {exc.__class__.__name__}",
                "raw": None,
            }
        return {"score": 50.0, "backend": backend, "warning": "Unsupported VLM backend.", "raw": None}

    def _event_prompt(self, segment: dict[str, Any]) -> str:
        payload = {
            "segment_id": segment.get("segment_id", ""),
            "prompt_text": segment.get("prompt_text", ""),
            "event_description": segment.get("event_description", ""),
            "expected_subject_ids": segment.get("expected_subject_ids", []),
            "evaluation_scope": segment.get("evaluation_scope", ""),
            "expected_shot_size": segment.get("expected_shot_size", ""),
            "expected_semantic_target": segment.get("expected_semantic_target", ""),
            "expected_camera_angle": segment.get("expected_camera_angle", ""),
            "expected_motion": segment.get("expected_motion", ""),
        }
        subject_instruction = (
            "If expected_subject_ids is empty, evaluate the scene/action/camera intent without requiring a visible person. "
            "If expected_subject_ids is non-empty, do not reward frames where the expected subject is absent. "
        )
        prompt = (
            "You are a strict film-shot evaluation model. Evaluate whether the video matches the intended event, "
            "camera view, shot size, and semantic target. Do not reward blank frames. "
            f"{subject_instruction}"
            "Return only JSON with keys: score (0-100 integer), reasons (array of short strings), "
            "subject_visible (boolean), intent_match (boolean).\n"
            f"{MODEL_LANGUAGE_POLICY}\n"
            f"Intent JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        assert_english_model_text(prompt, field="cinestory_event_prompt")
        return prompt

    def _score_with_gemini(self, *, prompt: str, video_path: Path) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("missing Gemini API key")
        try:
            from google import genai
        except Exception as exc:
            raise RuntimeError("google-genai is not installed") from exc
        client = genai.Client(api_key=self.api_key)
        uploaded = client.files.upload(file=str(video_path))
        for _ in range(120):
            state = getattr(getattr(uploaded, "state", None), "name", "")
            if state and state != "PROCESSING":
                break
            time.sleep(1.0)
            uploaded = client.files.get(name=uploaded.name)
        response = client.models.generate_content(model=self.model, contents=[uploaded, prompt])
        text = getattr(response, "text", "") or ""
        parsed = extract_json_object(text) or {}
        score = float(parsed.get("score", 50.0))
        return {"score": max(0.0, min(100.0, score)), "backend": "gemini", "raw": parsed or text}

    def _score_with_qwen_local(self, *, prompt: str, video_path: Path) -> dict[str, Any]:
        global _QWEN_MODEL, _QWEN_PROCESSOR
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError("transformers, torch, and qwen-vl-utils are required for qwen_local") from exc
        if _QWEN_MODEL is None or _QWEN_PROCESSOR is None:
            model_path = Path(self.model)
            cache_dir = None if model_path.exists() else str(self.qwen_cache_dir)
            _QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model,
                cache_dir=cache_dir,
                local_files_only=True,
                torch_dtype="auto",
                device_map="auto",
            )
            _QWEN_PROCESSOR = AutoProcessor.from_pretrained(self.model, cache_dir=cache_dir, local_files_only=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": 1.0},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = _QWEN_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _QWEN_PROCESSOR(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = getattr(_QWEN_MODEL, "device", None)
        if device is not None:
            inputs = inputs.to(device)
        with torch.no_grad():
            generated_ids = _QWEN_MODEL.generate(**inputs, max_new_tokens=512, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = _QWEN_PROCESSOR.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = extract_json_object(output_text) or {}
        score = float(parsed.get("score", 50.0))
        return {"score": max(0.0, min(100.0, score)), "backend": "qwen_local", "raw": parsed or output_text}

    def _score_with_openai_compatible(self, *, prompt: str, keyframe_paths: list[str]) -> dict[str, Any]:
        if not self.api_key or not self.api_base:
            raise RuntimeError("missing OpenAI-compatible API configuration")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in keyframe_paths[:8]:
            image_path = Path(path)
            if not image_path.exists():
                continue
            data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 512,
        }
        url = str(self.api_base).rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            import requests
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return {"score": 50.0, "backend": "openai_compatible", "warning": f"request_failed: {exc}", "raw": None}
        # Robust response parsing (matches cinematographer_llm_adapter behavior)
        text = ""
        try:
            text = str(data["choices"][0]["message"]["content"] or "")
        except Exception:
            # Fallback: yunwu.ai may return non-standard structure; try stringifying top-level
            text = str(data)
        parsed = extract_json_object(text) or {}
        score = float(parsed.get("score", 50.0))
        return {"score": max(0.0, min(100.0, score)), "backend": "openai_compatible", "raw": parsed or text}
