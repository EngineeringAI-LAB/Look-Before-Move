# 🎬 Look-Before-Move

Look-Before-Move is a narrative-grounded camera planning pipeline for Blender scenes. Given a story/script directory, it builds scene context, searches camera viewpoints, plans motion, renders shot clips, and evaluates the generated result with segment-level cinematic metrics.

The released repository contains code only. Scene assets, story datasets, generated videos, model checkpoints, paper figures, logs, and benchmark outputs are intentionally excluded.

## ✨ Overview

The system is organized as four executable stages:

1. `Director`: parses story inputs, builds scene context, shot contracts, and blocking plans.
2. `Cinematographer`: performs candidate viewpoint generation, validation, Monte Carlo/quality search, VLM reflection, semantic height adjustment, and trajectory grounding.
3. `VideoEngineer`: converts camera plans into executable Blender trajectories and renders clips.
4. `Editor`: assembles rendered clips and emits the final edit manifest.

`Engine/run_full_pipeline.py` runs these stages end to end. `Evaluation/` contains CineStoryEval, the metric pipeline used for SP, IC, and TQ reporting.

## 🧱 Repository Structure

```text
Look-Before-Move/
|-- Director/                 # Story parsing, scene context, shot contracts, blocking
|-- Cinematographer/          # Viewpoint search, candidate validation, trajectory planning
|-- VideoEngineer/            # Blender rendering and clip handoff generation
|-- Editor/                   # Timeline assembly
|-- Engine/                   # End-to-end pipeline runner
|-- Evaluation/               # CineStoryEval metric code and Qwen helper scripts
|-- tools/                    # Batch runners, audit tools, ablation summaries
|-- config/                   # Example runtime configuration
|-- requirements.txt          # Python dependencies for code + evaluation
`-- .env.example              # Environment variable template
```

## 🚀 Quick Start

### 1) Requirements

- Windows or Linux with Python 3.11+.
- Blender 4.5 is recommended. The code calls Blender in background mode and uses `bpy` inside Blender subprocesses.
- NVIDIA GPU is recommended for quality mode and local Qwen evaluation.
- FFmpeg should be available on `PATH` for robust video handling.

Install the Python environment:

```powershell
git clone https://github.com/EngineeringAI-LAB/Look-Before-Move.git
cd Look-Before-Move

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your CUDA/PyTorch version differs, edit the first lines of `requirements.txt` or install a PyTorch wheel that matches your driver before installing the remaining packages.

### 2) Configuration

Copy the environment template and edit local paths/secrets:

```powershell
Copy-Item .env.example .env
notepad .env
```

Important variables:

- `STORYBLENDER_BLENDER_EXE`: absolute path to `blender.exe`, or leave unset if `blender` is on `PATH`.
- `ANYLLM_API_KEY`, `ANYLLM_API_BASE`, `ANYLLM_PROVIDER`: OpenAI-compatible or AnyLLM-compatible VLM endpoint used by Director/Cinematographer.
- `STORYBLENDER_VISION_MODEL`: vision model name for VLM reflection and selection.
- `LBM_SCRIPTS_ROOT`: directory containing story asset folders.
- `CINESTORY_QWEN_MODEL`: local path or Hugging Face model id for the evaluator. Default: `Qwen/Qwen2.5-VL-7B-Instruct`.
- `CINESTORY_QWEN_CACHE_DIR`: local Hugging Face cache for Qwen weights. Default: `Evaluation/models/huggingface`.
- `CINESTORY_YOLO_WEIGHTS`: local YOLO weight path. Default: `Evaluation/models/yolo/yolo11x.pt`.

The code also supports `config/runtime_config.json`; use `config/runtime_config.example.json` as a template. Do not commit real API keys.

## 🧠 Qwen and Evaluator Weights

CineStoryEval can run event-alignment scoring with a local Qwen video-language model. The evaluator loads with `local_files_only=True`, so weights must be prefetched before evaluation.

Install Qwen dependencies:

```powershell
.\Evaluation\scripts\install_qwen_local.ps1
```

Download Qwen weights:

```powershell
.\Evaluation\scripts\prefetch_qwen_model.ps1 `
  -ModelId "Qwen/Qwen2.5-VL-7B-Instruct" `
  -CacheDir ".\Evaluation\models\huggingface"
```

Equivalent Hugging Face CLI command:

```powershell
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct `
  --cache-dir .\Evaluation\models\huggingface
```

YOLO weights are required for strict subject detection. Place `yolo11x.pt` at:

```text
Evaluation/models/yolo/yolo11x.pt
```

or set:

```powershell
$env:CINESTORY_YOLO_WEIGHTS = "D:\path\to\yolo11x.pt"
```

Model directories are ignored by git.

## 📁 Input Data Layout

Each story directory should contain the Blender scene, story text, formatted models, layout scripts, and related assets. A typical directory looks like:

```text
scripts/
`-- The Godfather/
    |-- story.txt
    |-- formatted_model/
    |-- layout_script/
    |-- animated_models/
    |-- supplementary_assets/
    `-- *.blend
```

The repository does not include these assets.

## 🎥 Run One Story

```powershell
$env:STORYBLENDER_BLENDER_EXE = "<path-to-blender>\blender.exe"
$env:ANYLLM_API_KEY = "<your-api-key>"
$env:ANYLLM_API_BASE = "https://your-openai-compatible-endpoint"
$env:ANYLLM_PROVIDER = "openai"

.\.venv\Scripts\python.exe Engine\run_full_pipeline.py `
  --demo-root "<path-to-scripts>\The Godfather" `
  --run-id "the_godfather_quality" `
  --camera-quality quality
```

Useful quality options are exposed by `Cinematographer/cinematographer_stage.py`, including:

- `--camera-quality fast|quality`
- `--run-pre-continuity-story-judge`
- `--disable-vlm-reflection`
- `--disable-trajectory-grounding`
- `--disable-semantic-height-adjust`

## 🏃 Run All Stories in Quality Mode

Set the story root and run the batch orchestrator:

```powershell
$env:LBM_SCRIPTS_ROOT = "<path-to-scripts>"
$env:LBM_PYTHON_EXE = ".\.venv\Scripts\python.exe"

.\.venv\Scripts\python.exe tools\run_all_scripts_parallel_generation.py `
  --ablation quality `
  --run-tag full_quality `
  --fixed-stories-from ".\missing_manifest_for_auto_discovery.json" `
  --generation-workers 2 `
  --evaluation-workers 1 `
  --llm-max-workers-per-story 2 `
  --eval-min-free-vram-gb 18
```

When `--fixed-stories-from` does not exist, the script discovers all directories under `LBM_SCRIPTS_ROOT` except `demo`.

## 📊 Run Evaluation

Build benchmark input from a video handoff:

```powershell
.\Evaluation\scripts\build_benchmark_input.ps1 `
  -VideoHandoff "VideoEngineer\output\<run_id>\<story>\outputs\video_handoff_v1.json" `
  -StoryName "the_godfather" `
  -Output "Evaluation\config\benchmark_input_the_godfather.json"
```

Evaluate with local Qwen:

```powershell
.\.venv\Scripts\python.exe Evaluation\run_cinestory_eval.py `
  --benchmark-input Evaluation\config\benchmark_input_the_godfather.json `
  --output-root Evaluation\output\the_godfather `
  --reference-root Evaluation\reference `
  --story-name the_godfather `
  --vlm-backend qwen_local
```

Use `--vlm-backend none` for a no-VLM smoke test.

## 📦 Outputs

Generated artifacts are written under stage-local `output/` directories and are ignored by git:

- `Director/output/.../outputs/director_handoff_v1.json`
- `Cinematographer/output/.../outputs/camera_handoff_v1.json`
- `VideoEngineer/output/.../outputs/video_handoff_v1.json`
- `Editor/output/.../exports/final_edit_v1.mp4`
- `Evaluation/output/.../cinestory_report_v1.json`

## 📚 Citation

```bibtex
@misc{bian2026lookbeforemovenarrativegroundedworldvisual,
      title={Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds}, 
      author={Jiaming Bian and Bingliang Li and Yuehao Wu and Pichao Wang and Zhi Wang and Hailan Ma and Huadong Mo and Zhenhong Sun},
      year={2026},
      eprint={2606.26964},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.26964}, 
}
```
