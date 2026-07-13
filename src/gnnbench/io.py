"""Small IO helpers shared by the pipeline stages."""

import json
from pathlib import Path


def write_report(payload: dict[str, object], path: Path) -> None:
    """Write a stage report as pretty-printed JSON, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2)
    path.write_text(content + "\n", encoding="utf-8")
