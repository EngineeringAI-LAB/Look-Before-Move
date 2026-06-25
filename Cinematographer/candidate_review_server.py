from __future__ import annotations

import argparse
import json
import mimetypes
import socketserver
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from candidate_filter_validation import build_report, read_jsonl, save_json


REVIEW_FIELDS = {
    "usable",
    "primary_visible",
    "secondary_visible",
    "semantic_satisfied",
    "framing_matches_intent",
    "direction_matches",
    "failure_reason",
    "notes",
}


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Candidate Filter Review</title>
  <style>
    :root { color-scheme: light; font-family: Arial, "Microsoft YaHei", sans-serif; }
    body { margin: 0; background: #f3f4f6; color: #111827; }
    .app { height: 100vh; display: grid; grid-template-columns: minmax(640px, 1fr) 360px; }
    .viewer { display: flex; align-items: center; justify-content: center; padding: 10px; background: #111827; overflow: hidden; }
    .viewer img { width: 100%; height: 100%; max-width: 100%; max-height: calc(100vh - 20px); object-fit: contain; background: #000; }
    .panel { overflow: auto; padding: 14px; background: #fff; border-left: 1px solid #d1d5db; }
    h1 { font-size: 18px; margin: 0 0 8px; }
    .muted { color: #6b7280; font-size: 12px; }
    .warn { color: #b45309; font-size: 12px; display: block; margin-top: 3px; }
    .meta { display: grid; gap: 7px; margin: 12px 0; font-size: 13px; }
    .meta div { padding-bottom: 6px; border-bottom: 1px solid #eef0f3; }
    .refs { display: grid; gap: 10px; margin: 12px 0; }
    .ref-card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; background: #fafafa; }
    .ref-title { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; margin-bottom: 6px; }
    .ref-role { color: #6b7280; }
    .ref-images { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
    .ref-images figure { margin: 0; min-width: 0; }
    .ref-images img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border: 1px solid #d1d5db; border-radius: 4px; background: #111827; }
    .ref-images figcaption { font-size: 10px; color: #6b7280; text-align: center; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .label { color: #6b7280; display: block; font-size: 11px; margin-bottom: 2px; text-transform: uppercase; }
    .buttons { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0; }
    button, select, textarea { font: inherit; }
    button { border: 1px solid #cbd5e1; background: #fff; padding: 9px 10px; border-radius: 6px; cursor: pointer; }
    button.active { border-color: #111827; background: #111827; color: #fff; }
    button.primary { background: #2563eb; border-color: #2563eb; color: #fff; width: 100%; }
    button.secondary { width: 100%; }
    select, textarea { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; }
    textarea { min-height: 64px; resize: vertical; }
    .field { margin: 10px 0; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .nav { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }
    .progress { height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 10px 0; }
    .bar { height: 100%; width: 0%; background: #16a34a; }
    .kbd { font-family: Consolas, monospace; background: #eef2ff; border: 1px solid #c7d2fe; padding: 1px 4px; border-radius: 4px; }
    pre { white-space: pre-wrap; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; font-size: 12px; }
  </style>
</head>
<body>
  <div class="app">
    <main class="viewer"><img id="image" alt="candidate" /></main>
    <aside class="panel">
      <h1>Direction/Preset 候选筛选验证</h1>
      <div class="muted"><span id="counter">0 / 0</span> · 已标注 <span id="done">0</span></div>
      <div class="progress"><div id="bar" class="bar"></div></div>

      <div class="nav">
        <button id="prev">上一张</button>
        <button id="next">下一张</button>
      </div>

      <div class="buttons" data-field="usable">
        <button data-value="good">Good <span class="kbd">1</span></button>
        <button data-value="borderline">Borderline <span class="kbd">2</span></button>
        <button data-value="bad">Bad <span class="kbd">3</span></button>
      </div>

      <div class="row">
        <div class="field">
          <label class="label">Primary Visible</label>
          <select id="primary_visible">
            <option value="">未选</option><option>true</option><option>false</option>
          </select>
        </div>
        <div class="field">
          <label class="label">Secondary Visible</label>
          <select id="secondary_visible">
            <option value="">未选</option><option>true</option><option>false</option><option>not_required</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label class="label">Semantic Satisfied</label>
          <select id="semantic_satisfied">
            <option value="">未选</option><option>true</option><option>false</option>
          </select>
        </div>
        <div class="field">
          <label class="label">Direction Matches</label>
          <select id="direction_matches">
            <option value="">未选</option><option>true</option><option>false</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label class="label">Framing Matches Intent</label>
        <select id="framing_matches_intent">
          <option value="">未选</option><option>true</option><option>false</option>
        </select>
      </div>
      <div class="field">
        <label class="label">Failure Reason</label>
        <select id="failure_reason">
          <option value="">无 / 未选</option>
          <option value="occlusion">遮挡</option>
          <option value="outside_scene_bounds">出界</option>
          <option value="wrong_subject">主体错</option>
          <option value="too_far">构图太远</option>
          <option value="too_close">构图太近</option>
          <option value="wrong_direction">方向错</option>
          <option value="semantic_mismatch">语义不符</option>
          <option value="other">其他</option>
        </select>
      </div>
      <div class="field">
        <label class="label">Notes</label>
        <textarea id="notes"></textarea>
      </div>
      <button class="primary" id="save">保存并下一张 <span class="kbd">S</span></button>
      <button class="secondary" id="report" style="margin-top:8px;">生成混淆矩阵报告</button>

      <section class="meta">
        <div><span class="label">Camera / Candidate</span><span id="id"></span></div>
        <div><span class="label">Shot</span><span id="shot"></span></div>
        <div><span class="label">Scene</span><span id="scene"></span></div>
        <div><span class="label">Focus</span><span id="focus"></span></div>
        <div><span class="label">Character References</span><section id="refs" class="refs"></section></div>
        <div><span class="label">Semantic</span><span id="semantic"></span></div>
        <div><span class="label">Motion</span><span id="motion"></span></div>
        <div><span class="label">Semantic Contract</span><pre id="contract"></pre></div>
      </section>
      <div class="muted">快捷键：1 good，2 borderline，3 bad，A/D 上一张/下一张，S 保存。</div>
    </aside>
  </div>
<script>
let items = [];
let reviews = {};
let index = 0;
let currentUsable = "";

function $(id) { return document.getElementById(id); }

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function setButtons(field, value) {
  document.querySelectorAll(`[data-field="${field}"] button`).forEach(btn => {
    btn.classList.toggle("active", btn.dataset.value === value);
  });
}

function current() { return items[index]; }

function setForm(review) {
  currentUsable = review.usable || "";
  setButtons("usable", currentUsable);
  for (const id of ["primary_visible","secondary_visible","semantic_satisfied","framing_matches_intent","direction_matches","failure_reason","notes"]) {
    $(id).value = review[id] || "";
  }
}

function getForm() {
  return {
    validation_id: current().validation_id,
    usable: currentUsable,
    primary_visible: $("primary_visible").value,
    secondary_visible: $("secondary_visible").value,
    semantic_satisfied: $("semantic_satisfied").value,
    framing_matches_intent: $("framing_matches_intent").value,
    direction_matches: $("direction_matches").value,
    failure_reason: $("failure_reason").value,
    notes: $("notes").value
  };
}

function hasFormInput(payload) {
  return Boolean(
    payload.usable ||
    payload.primary_visible ||
    payload.secondary_visible ||
    payload.semantic_satisfied ||
    payload.framing_matches_intent ||
    payload.direction_matches ||
    payload.failure_reason ||
    payload.notes
  );
}

function render() {
  const item = current();
  if (!item) return;
  $("image").src = `/image?path=${encodeURIComponent(item.candidate_image_path)}`;
  $("counter").textContent = `${index + 1} / ${items.length}`;
  $("done").textContent = Object.keys(reviews).length;
  $("bar").style.width = `${items.length ? (Object.keys(reviews).length / items.length) * 100 : 0}%`;
  $("id").textContent = `${item.camera_name} / ${item.candidate_id}`;
  $("shot").textContent = item.shot_description || "";
  $("scene").textContent = item.scene_description || "";
  const focusBits = [`primary: ${item.primary_focus_id || ""}`, `secondary: ${(item.secondary_focus_ids || []).join(", ")}`];
  if (item.focus_conflict_corrected) {
    const original = item.original_primary_focus_id || item.contract_primary_focus_id || "";
    focusBits.push(`corrected from: ${original}`);
  }
  $("focus").textContent = focusBits.join(" | ");
  if (item.focus_conflict_corrected) {
    const detail = `contract primary: ${item.contract_primary_focus_id || ""}; start focus: ${(item.contract_start_focus_ids || []).join(", ")}; keyframes: ${(item.keyframe_primary_ids || []).join(", ")}`;
    const warning = document.createElement("span");
    warning.className = "warn";
    warning.textContent = detail;
    $("focus").appendChild(warning);
  }
  renderRefs(item.character_references || []);
  $("semantic").textContent = `${item.primary_semantic_target || ""} | ${item.distance_label || ""} | ${item.semantic_direction || ""}`;
  $("motion").textContent = `${item.angle_label || ""} | ${item.movement_tag || ""}`;
  $("contract").textContent = JSON.stringify(item.semantic_contract || {}, null, 2);
  setForm(reviews[item.validation_id] || {});
}

function renderRefs(refs) {
  const host = $("refs");
  host.innerHTML = "";
  if (!refs.length) {
    host.textContent = "未找到人物参考图";
    return;
  }
  for (const ref of refs) {
    const card = document.createElement("article");
    card.className = "ref-card";
    const title = document.createElement("div");
    title.className = "ref-title";
    title.innerHTML = `<strong>${ref.character_id || ""}</strong><span class="ref-role">${ref.role || ""}</span>`;
    card.appendChild(title);
    const images = document.createElement("div");
    images.className = "ref-images";
    const preferred = ["front", "front_left", "front_right", "left", "right", "back", "top"];
    const paths = ref.context_turnaround_paths || ref.turnaround_paths || {};
    for (const key of preferred) {
      if (!paths[key]) continue;
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = `/image?path=${encodeURIComponent(paths[key])}`;
      const cap = document.createElement("figcaption");
      cap.textContent = key;
      fig.appendChild(img);
      fig.appendChild(cap);
      images.appendChild(fig);
      if (images.children.length >= 3) break;
    }
    card.appendChild(images);
    host.appendChild(card);
  }
}

async function save(next=true, quiet=false) {
  const payload = getForm();
  if (!payload.usable) {
    if (!quiet) alert("先选 usable: good / borderline / bad");
    return false;
  }
  const res = await api("/api/review", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
  reviews[payload.validation_id] = res.review;
  if (next && index < items.length - 1) index++;
  render();
  return true;
}

async function navigate(delta) {
  const target = index + delta;
  if (target < 0 || target >= items.length) return;
  const payload = getForm();
  const alreadyReviewed = Boolean(reviews[payload.validation_id]);
  if (hasFormInput(payload) || alreadyReviewed) {
    const saved = await save(false, false);
    if (!saved) return;
  }
  index = target;
  render();
}

async function load() {
  const data = await api("/api/items");
  items = data.items || [];
  reviews = data.reviews || {};
  const firstUnreviewed = items.findIndex(item => !reviews[item.validation_id]);
  index = firstUnreviewed >= 0 ? firstUnreviewed : 0;
  render();
}

document.querySelectorAll('[data-field="usable"] button').forEach(btn => {
  btn.addEventListener("click", () => { currentUsable = btn.dataset.value; setButtons("usable", currentUsable); });
});
$("prev").onclick = () => navigate(-1);
$("next").onclick = () => navigate(1);
$("save").onclick = () => save(true);
$("report").onclick = async () => {
  const res = await api("/api/report", {method:"POST"});
  alert(`报告已生成：${res.report_path}\nstatus=${res.status}`);
};
window.addEventListener("keydown", ev => {
  if (ev.target.tagName === "TEXTAREA" || ev.target.tagName === "SELECT") return;
  if (ev.key === "1") { currentUsable = "good"; setButtons("usable", currentUsable); }
  if (ev.key === "2") { currentUsable = "borderline"; setButtons("usable", currentUsable); }
  if (ev.key === "3") { currentUsable = "bad"; setButtons("usable", currentUsable); }
  if (ev.key.toLowerCase() === "a") { navigate(-1); }
  if (ev.key.toLowerCase() === "d") { navigate(1); }
  if (ev.key.toLowerCase() === "s") { ev.preventDefault(); save(true); }
});
load().catch(err => alert(err.message));
</script>
</body>
</html>"""


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row.get("validation_id") or ""): row for row in rows if row.get("validation_id")}


def write_reviews(path: Path, reviews: dict[str, dict[str, Any]], item_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_ids = [validation_id for validation_id in item_order if validation_id in reviews]
    ordered_ids.extend(sorted(validation_id for validation_id in reviews if validation_id not in set(ordered_ids)))
    with path.open("w", encoding="utf-8") as handle:
        for validation_id in ordered_ids:
            handle.write(json.dumps(reviews[validation_id], ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def director_run_root(output_root: Path) -> Path | None:
    for path in (
        output_root / "manifest.json",
        output_root / "outputs" / "camera_handoff_quality_input_v1.json",
        output_root / "outputs" / "camera_handoff_v1.json",
    ):
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        handoff_path = Path(str(payload.get("director_handoff_path") or ""))
        if handoff_path.exists():
            return handoff_path.resolve().parent.parent
    return None


def scene_dossier_path(director_root: Path | None, scene_id: Any) -> Path | None:
    if director_root is None:
        return None
    try:
        scene_number = int(scene_id)
    except (TypeError, ValueError):
        return None
    path = director_root / "scene_dossiers" / f"scene_{scene_number}" / "scene_dossier.json"
    return path if path.exists() else None


def load_scene_character_map(director_root: Path | None) -> dict[int, dict[str, dict[str, Any]]]:
    if director_root is None:
        return {}
    rows: dict[int, dict[str, dict[str, Any]]] = {}
    dossier_root = director_root / "scene_dossiers"
    for dossier_path in dossier_root.glob("scene_*/scene_dossier.json"):
        try:
            payload = load_json(dossier_path)
        except Exception:
            continue
        try:
            scene_id = int(payload.get("scene_id") or dossier_path.parent.name.split("_")[-1])
        except (TypeError, ValueError):
            continue
        scene_map: dict[str, dict[str, Any]] = {}
        for row in payload.get("character_summaries") or []:
            character_id = str(row.get("character_id") or "").strip()
            if not character_id:
                continue
            scene_map[character_id] = {
                "character_id": character_id,
                "display_name": row.get("display_name") or character_id,
                "portrait_path": row.get("portrait_path") or "",
                "turnaround_paths": row.get("turnaround_paths") or {},
                "context_turnaround_paths": row.get("context_turnaround_paths") or {},
            }
        rows[scene_id] = scene_map
    return rows


def dedupe_focus_ids(primary_id: str, secondary_ids: list[Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    primary = str(primary_id or "").strip()
    if primary:
        rows.append(("primary", primary))
        seen.add(primary)
    for item in secondary_ids or []:
        value = str(item or "").strip()
        if value and value not in seen:
            rows.append(("secondary", value))
            seen.add(value)
    return rows


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]
    rows: list[str] = []
    for item in raw_values:
        text = str(item or "").strip()
        if text and text not in rows:
            rows.append(text)
    return rows


def load_focus_correction_map(output_root: Path) -> dict[str, dict[str, Any]]:
    """Derive review-facing focus from contract/keyframes when package metadata conflicts."""
    camera_root = output_root / "camera_packages"
    if not camera_root.exists():
        return {}
    corrections: dict[str, dict[str, Any]] = {}
    for path in camera_root.glob("scene_*_shot_*/*.json"):
        try:
            camera = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        camera_name = str(camera.get("camera_name") or path.stem).strip()
        if not camera_name:
            continue
        contract = camera.get("shot_contract") or {}
        if not isinstance(contract, dict):
            contract = {}
        start_contract = contract.get("start_frame_contract") or {}
        if not isinstance(start_contract, dict):
            start_contract = {}
        package_primary = str(camera.get("primary_focus_id") or "").strip()
        contract_primary = str(start_contract.get("primary_focus_id") or "").strip()
        start_ids = as_string_list(start_contract.get("start_focus_ids"))
        contract_secondary = as_string_list(start_contract.get("secondary_focus_ids"))
        keyframe_plan = contract.get("keyframe_plan") or []
        keyframe_primary_ids = []
        if isinstance(keyframe_plan, list):
            for keyframe in keyframe_plan:
                if isinstance(keyframe, dict):
                    keyframe_primary_ids.extend(as_string_list(keyframe.get("primary_focus_id")))

        corrected_primary = package_primary or contract_primary or (start_ids[0] if start_ids else "")
        conflict_reason = ""
        if start_ids:
            start_primary = start_ids[0]
            keyframe_votes = sum(1 for item in keyframe_primary_ids if item == start_primary)
            has_keyframe_consensus = bool(keyframe_primary_ids) and keyframe_votes >= max(1, len(keyframe_primary_ids) // 2 + 1)
            conflicts_with_package = package_primary and package_primary != start_primary
            conflicts_with_contract = contract_primary and contract_primary != start_primary
            if (conflicts_with_package or conflicts_with_contract) and has_keyframe_consensus:
                corrected_primary = start_primary
                conflict_reason = (
                    "start_focus_ids and keyframe_plan agree on this subject; "
                    "package/start_frame primary_focus_id pointed elsewhere"
                )

        secondary_ids: list[str] = []
        for focus_id in contract_secondary + start_ids + as_string_list(camera.get("secondary_focus_ids")):
            if focus_id and focus_id != corrected_primary and focus_id not in secondary_ids:
                secondary_ids.append(focus_id)
        if package_primary and package_primary != corrected_primary and package_primary not in secondary_ids:
            secondary_ids.insert(0, package_primary)

        corrections[camera_name] = {
            "primary_focus_id": corrected_primary,
            "secondary_focus_ids": secondary_ids,
            "original_primary_focus_id": package_primary,
            "contract_primary_focus_id": contract_primary,
            "contract_start_focus_ids": start_ids,
            "keyframe_primary_ids": sorted(set(keyframe_primary_ids)),
            "focus_conflict_corrected": bool(conflict_reason),
            "focus_conflict_reason": conflict_reason,
        }
    return corrections


def enrich_items_with_character_references(
    items: list[dict[str, Any]],
    character_maps: dict[int, dict[str, dict[str, Any]]],
    focus_corrections: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        camera_name = str(row.get("camera_name") or "").strip()
        correction = focus_corrections.get(camera_name) or {}
        if correction.get("primary_focus_id"):
            row["original_primary_focus_id"] = correction.get("original_primary_focus_id") or row.get("primary_focus_id") or ""
            row["primary_focus_id"] = correction["primary_focus_id"]
            row["secondary_focus_ids"] = correction.get("secondary_focus_ids") or row.get("secondary_focus_ids") or []
            row["focus_conflict_corrected"] = bool(correction.get("focus_conflict_corrected"))
            row["focus_conflict_reason"] = correction.get("focus_conflict_reason") or ""
            row["contract_primary_focus_id"] = correction.get("contract_primary_focus_id") or ""
            row["contract_start_focus_ids"] = correction.get("contract_start_focus_ids") or []
            row["keyframe_primary_ids"] = correction.get("keyframe_primary_ids") or []
        try:
            scene_id = int(row.get("scene_id") or 0)
        except (TypeError, ValueError):
            scene_id = 0
        scene_map = character_maps.get(scene_id) or {}
        refs: list[dict[str, Any]] = []
        for role, focus_id in dedupe_focus_ids(str(row.get("primary_focus_id") or ""), row.get("secondary_focus_ids") or []):
            ref = scene_map.get(focus_id)
            if not ref:
                continue
            refs.append({**ref, "role": role})
        row["character_references"] = refs
        enriched.append(row)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local visual review UI for candidate filter validation.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


class ReviewState:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.resolve()
        self.director_root = director_run_root(self.output_root)
        outputs_dir = self.output_root / "outputs"
        self.input_path = outputs_dir / "candidate_blind_review_input_v1.jsonl"
        self.result_path = outputs_dir / "candidate_blind_review_result_v1.jsonl"
        self.manifest_path = outputs_dir / "candidate_validation_manifest_v1.jsonl"
        self.report_path = outputs_dir / "candidate_filter_confusion_report_v1.json"
        self.character_maps = load_scene_character_map(self.director_root)
        self.focus_corrections = load_focus_correction_map(self.output_root)
        self.items = enrich_items_with_character_references(
            read_jsonl(self.input_path),
            self.character_maps,
            self.focus_corrections,
        )
        self.item_order = [str(item.get("validation_id") or "") for item in self.items if item.get("validation_id")]
        self.reviews = load_reviews(self.result_path)

    def refresh_reviews(self) -> None:
        self.reviews = load_reviews(self.result_path)

    def image_allowed(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            if not resolved.exists() or not resolved.is_file():
                return False
            if resolved.is_relative_to(self.output_root):
                return True
            if self.director_root is not None and resolved.is_relative_to(self.director_root):
                return True
            return False
        except Exception:
            return False

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        validation_id = str(payload.get("validation_id") or "").strip()
        if not validation_id:
            raise ValueError("validation_id is required")
        if validation_id not in set(self.item_order):
            raise ValueError(f"Unknown validation_id: {validation_id}")
        row = {
            "schema_version": "storyblender.candidate_blind_review_result.v1",
            "validation_id": validation_id,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        for key in REVIEW_FIELDS:
            value = payload.get(key)
            if value is not None:
                row[key] = str(value)
        self.reviews[validation_id] = row
        write_reviews(self.result_path, self.reviews, self.item_order)
        return row

    def write_report(self) -> dict[str, Any]:
        report = build_report(
            manifest_rows=read_jsonl(self.manifest_path),
            review_rows=read_jsonl(self.result_path),
            manifest_path=self.manifest_path,
            review_result_path=self.result_path,
            focus_conflicts=self.focus_corrections,
        )
        save_json(report, self.report_path)
        return report


def make_handler(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            return payload if isinstance(payload, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/items":
                state.refresh_reviews()
                self._send_json({"items": state.items, "reviews": state.reviews})
                return
            if parsed.path == "/image":
                query = urllib.parse.parse_qs(parsed.query)
                image_path = Path(str((query.get("path") or [""])[0]))
                if not state.image_allowed(image_path):
                    self.send_error(403, "image path is not allowed")
                    return
                content = image_path.read_bytes()
                mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/api/review":
                    review = state.save_review(self._read_json())
                    self._send_json({"success": True, "review": review})
                    return
                if self.path == "/api/report":
                    report = state.write_report()
                    self._send_json({"success": True, "status": report.get("status"), "report_path": str(state.report_path)})
                    return
                self.send_error(404)
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, status=400)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> int:
    args = parse_args()
    state = ReviewState(Path(args.output_root))
    if not state.items:
        raise SystemExit(f"No review items found: {state.input_path}")
    handler = make_handler(state)
    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(
            json.dumps(
                {
                    "success": True,
                    "url": url,
                    "item_count": len(state.items),
                    "director_root": str(state.director_root or ""),
                    "scene_reference_count": len(state.character_maps),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if args.open:
            import webbrowser

            webbrowser.open(url)
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
