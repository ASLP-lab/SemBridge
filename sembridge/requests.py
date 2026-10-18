"""JSON inputs and generation options for SemBridge inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DEFAULT_GENERATION = {
    "max_new_patches": 500,
    "min_new_patches": 2,
    "flow_steps": None,
    "cfg_value": None,
    "temperature": None,
    "seed": 42,
}


def resolve_generation_options(
    settings: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply explicit overrides over request settings, then inference defaults."""
    result = dict(DEFAULT_GENERATION)
    for values in (settings, overrides):
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError("generation must be a JSON object")
        unknown = values.keys() - DEFAULT_GENERATION.keys()
        if unknown:
            raise ValueError(f"unknown generation settings: {', '.join(sorted(unknown))}")
        result.update({key: value for key, value in values.items() if value is not None})
    return result


def validate_generation_options(options: dict[str, Any]) -> None:
    """Check resolved sampling values at the inference entry point."""
    for key in ("max_new_patches", "min_new_patches", "seed", "flow_steps"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    if options["max_new_patches"] < 1 or not 0 <= options["min_new_patches"] <= options["max_new_patches"]:
        raise ValueError("require 0 <= min_new_patches <= max_new_patches and max_new_patches > 0")
    if options["flow_steps"] < 2:
        raise ValueError("flow_steps must be at least 2")
    for key in ("cfg_value", "temperature"):
        value = options[key]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"{key} must be a finite, non-negative number")


def load_request(
    path: str | Path,
    *,
    task: str | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return keyword arguments for ``SemBridge.generate``.

    Reference paths are relative to the request JSON, not the working directory.
    Optional ``generation`` settings are flattened into the returned arguments.
    """
    request_path = Path(path).expanduser().resolve()
    record = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("request must contain one JSON object")
    allowed = {"task", "text", "notes", "segments", "prompt_audio", "prompt_text",
               "prompt_task", "generation"}
    unknown = record.keys() - allowed
    if unknown:
        raise ValueError(f"unknown request fields: {', '.join(sorted(unknown))}")
    record_task = record.get("task", task)
    if record_task == "mixed":
        record_task = "unified"
    if not isinstance(record_task, str) or record_task not in {"tts", "svs", "unified"}:
        raise ValueError("request task must be tts, svs, or unified")
    if task is not None and record_task != task:
        raise ValueError(f"request task {record_task!r} does not match task {task!r}")
    record["task"] = record_task
    prompt_audio = record.get("prompt_audio")
    if prompt_audio is not None:
        if not isinstance(prompt_audio, str) or not prompt_audio:
            raise ValueError("prompt_audio must be a non-empty path string")
        prompt_path = Path(prompt_audio).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = request_path.parent / prompt_path
        prompt_path = prompt_path.resolve()
        record["prompt_audio"] = prompt_path
    record.update(resolve_generation_options(record.pop("generation", None), generation_overrides))
    return record
