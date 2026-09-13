from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_latest_output(data_dir: Path) -> dict[str, Any]:
    file_path = data_dir / "latest_output.json"
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_runs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.iterdir() if p.is_dir()])
