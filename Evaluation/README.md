# CineStoryEval

CineStoryEval is the segment-level evaluator used by Look-Before-Move. It scores generated clips with three metric groups:

- Subject perception (SP)
- Intent consistency (IC)
- Trajectory quality (TQ)

The evaluator code is included in this repository. Model weights, reference images, benchmark inputs, evidence frames, and reports are local runtime assets and are not committed.

## Local Model Assets

Default locations:

```text
Evaluation/models/yolo/yolo11x.pt
Evaluation/models/huggingface/
Evaluation/reference/
Evaluation/output/
```

Environment overrides:

```powershell
$env:CINESTORY_YOLO_WEIGHTS = "D:\path\to\yolo11x.pt"
$env:CINESTORY_QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
$env:CINESTORY_QWEN_CACHE_DIR = "D:\path\to\huggingface_cache"
```

## Qwen Setup

Install dependencies:

```powershell
.\Evaluation\scripts\install_qwen_local.ps1
```

Prefetch the model before using `--vlm-backend qwen_local`:

```powershell
.\Evaluation\scripts\prefetch_qwen_model.ps1 `
  -ModelId "Qwen/Qwen2.5-VL-7B-Instruct" `
  -CacheDir ".\Evaluation\models\huggingface"
```

The evaluator loads Qwen with `local_files_only=True`, so an online model id is not enough unless the cache already contains the full snapshot.

## Reference Library

Subject identity metrics need reference images. Build them from a story's `formatted_model` directory:

```powershell
.\Evaluation\scripts\build_reference_library.ps1 `
  -FormattedModelDir "D:\path\to\scripts\The Godfather\formatted_model" `
  -StoryName "the_godfather"
```

The generated library is stored under:

```text
Evaluation/reference/<story_name>/
```

## Benchmark Input

Build an evaluator input from a `video_handoff_v1.json`:

```powershell
.\Evaluation\scripts\build_benchmark_input.ps1 `
  -VideoHandoff "VideoEngineer\output\<run_id>\<story>\outputs\video_handoff_v1.json" `
  -StoryName "the_godfather" `
  -Output "Evaluation\config\benchmark_input_the_godfather.json"
```

`Evaluation/config/benchmark_input_v1.example.json` documents the expected schema.

## Run

Smoke test without VLM:

```powershell
.\.venv\Scripts\python.exe Evaluation\run_cinestory_eval.py `
  --benchmark-input Evaluation\config\benchmark_input_v1.example.json `
  --output-root Evaluation\output\debug_eval `
  --reference-root Evaluation\reference `
  --story-name debug_story `
  --vlm-backend none
```

Full local-Qwen evaluation:

```powershell
.\.venv\Scripts\python.exe Evaluation\run_cinestory_eval.py `
  --benchmark-input Evaluation\config\benchmark_input_the_godfather.json `
  --output-root Evaluation\output\the_godfather `
  --reference-root Evaluation\reference `
  --story-name the_godfather `
  --vlm-backend qwen_local
```

Outputs:

- `cinestory_report_v1.json`
- `segment_scores.jsonl`
- `summary.csv`
- `normalized_benchmark_input_v1.json`
- `evidence/<segment_id>/keyframes`
- `evidence/<segment_id>/boxes`

## Failure Policy

Generated videos are evaluated normally when readable. Missing/empty videos are generation failures. Metrics that are not applicable to a segment are written as `null` and excluded from that metric's mean. If local Qwen is explicitly requested and cannot load, the segment is marked as an evaluation failure instead of silently using a neutral VLM score.
