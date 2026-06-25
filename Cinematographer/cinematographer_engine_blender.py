from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def run_blender_python_script(
    *,
    blender_exe: str | Path,
    blend_file: str | Path,
    python_script: str | Path,
    script_args: list[str],
    workdir: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    timeout_seconds: int,
    background: bool = True,
) -> dict[str, Any]:
    command = [str(blender_exe)]
    if background:
        command.append("-b")
    command.append("--python-use-system-env")
    command.extend([str(blend_file), "--python", str(python_script), "--", *script_args])

    stdout_target = Path(stdout_path)
    stderr_target = Path(stderr_path)
    stdout_target.parent.mkdir(parents=True, exist_ok=True)
    stderr_target.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    script_parent = str(Path(python_script).resolve().parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = script_parent if not existing_pythonpath else f"{script_parent}{os.pathsep}{existing_pythonpath}"
    with stdout_target.open("w", encoding="utf-8") as stdout_handle, stderr_target.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=str(workdir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
    return {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout_path": str(stdout_target),
        "stderr_path": str(stderr_target),
    }
