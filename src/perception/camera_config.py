"""Per-camera perception configuration: counting lines calibrated per view.

configs/cameras.json is the committed source of truth (camera ids and line
geometry only — no endpoints). CLI tools resolve lines in this order:
explicit --line args > this config > the generic DEFAULT_LINE fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.perception.crossing import CountingLine, parse_line_spec

DEFAULT_CONFIG_PATH = Path("configs/cameras.json")


def load_camera_lines(
    camera_id: str, path: Path = DEFAULT_CONFIG_PATH
) -> list[CountingLine] | None:
    """Counting lines for a camera, or None if the camera isn't configured."""
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get(camera_id.lower())
    if not entry:
        return None
    return [parse_line_spec(spec) for spec in entry["lines"]]


def resolve_lines(
    camera_id: str,
    cli_specs: list[str] | None,
    fallback_spec: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[CountingLine]:
    """Resolution order: explicit CLI specs > per-camera config > fallback."""
    if cli_specs:
        return [parse_line_spec(s) for s in cli_specs]
    configured = load_camera_lines(camera_id, config_path)
    if configured:
        return configured
    return [parse_line_spec(fallback_spec)]
