"""Small IO helpers shared by the pipeline stages."""

import json
from pathlib import Path

_REPORT_DECIMAL_PLACES = 5


def write_report(payload: dict[str, object], path: Path) -> None:
    """Write a stage report as pretty-printed JSON, rounding floats and creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(_round_floats(payload, _REPORT_DECIMAL_PLACES), indent=2)
    path.write_text(content + "\n", encoding="utf-8")


def _round_floats(value: object, ndigits: int) -> object:
    """Round floats recursively before JSON serialization."""
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: _round_floats(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v, ndigits) for v in value]
    return value
