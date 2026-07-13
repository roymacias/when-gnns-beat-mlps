"""Typed loading of the YAML experiment configuration.

The repository root is derived from the package location. All other paths
are read from ``config/common.yaml`` and resolved relative to that root.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    """Absolute locations of the pipeline directories."""

    data: Path
    splits: Path
    reports: Path
    artifacts: Path


@dataclass(frozen=True)
class CommonConfig:
    """Protocol shared by every experiment: seeds, device, and paths."""

    seeds: list[int]
    device: str
    paths: PathsConfig


@dataclass(frozen=True)
class LinkSplitConfig:
    """Edge-holdout fractions for link prediction."""

    num_val: float
    num_test: float


@dataclass(frozen=True)
class GraphSplitConfig:
    """Stratified split fractions over graphs for graph classification."""

    train: float
    val: float
    test: float


def repo_root() -> Path:
    """Repository root, derived from the package location."""
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise TypeError(f"Expected {path} to contain a YAML mapping.")
    return content


def load_common() -> CommonConfig:
    """Load ``config/common.yaml`` with paths resolved to absolute."""
    root = repo_root()
    raw = _load_yaml(root / "config" / "common.yaml")
    device = str(raw["device"])
    if device != "cpu":
        raise ValueError(f"Unsupported device: {device!r}")
    paths = PathsConfig(
        data=root / raw["paths"]["data"],
        splits=root / raw["paths"]["splits"],
        reports=root / raw["paths"]["reports"],
        artifacts=root / raw["paths"]["artifacts"],
    )
    return CommonConfig(seeds=list(raw["seeds"]), device=device, paths=paths)


def load_link_split() -> LinkSplitConfig:
    """Load the split block of ``config/link.yaml``."""
    raw = _load_yaml(repo_root() / "config" / "link.yaml")["split"]
    return LinkSplitConfig(num_val=float(raw["num_val"]), num_test=float(raw["num_test"]))


def load_graph_split() -> GraphSplitConfig:
    """Load the split block of ``config/graph.yaml``."""
    raw = _load_yaml(repo_root() / "config" / "graph.yaml")["split"]
    return GraphSplitConfig(
        train=float(raw["train"]), val=float(raw["val"]), test=float(raw["test"])
    )
