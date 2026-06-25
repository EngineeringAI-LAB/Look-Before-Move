from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(os.environ.get("LBM_SCRIPTS_ROOT", str(WORKSPACE.parent / "scripts"))).resolve()

FIVE_STORIES = [
    "Scent of a Woman",
    "The Bridges of Madison County",
    "The Godfather",
    "The Godfather Part II",
    "The Godfather Part III",
]


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the fixed five-story manifest used for Ours/CCD comparison.")
    parser.add_argument(
        "--output",
        default=str(WORKSPACE / "run_logs" / "quality_and_ablations_20260503" / "fixed_five_story_comparison.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for story in FIVE_STORIES:
        script_path = SCRIPTS_ROOT / story
        if not script_path.exists():
            missing.append(str(script_path))
        rows.append(
            {
                "story": story,
                "script_path": str(script_path),
                "selected_for_run": True,
                "comparison_set": "five_story_ours_ccd",
            }
        )
    save_json(rows, Path(args.output))
    print(f"Wrote {args.output}")
    if missing:
        print("Missing script directories:")
        for item in missing:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
