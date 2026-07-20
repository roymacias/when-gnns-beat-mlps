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
    """Protocol shared by every experiment: seeds, device, paths, tiebreak."""

    seeds: list[int]
    device: str
    paths: PathsConfig
    median_tiebreak: str


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


@dataclass(frozen=True)
class CommonBaselineConfig:
    """MLP hyperparameters for the node and graph baselines."""

    hidden_dims: list[int]
    dropout: float


@dataclass(frozen=True)
class GCNConfig:
    """GCN hyperparameters for node classification."""

    hidden_dim: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings shared by both models of a task."""

    optimizer: str
    learning_rate: float
    weight_decay: float
    epochs: int


@dataclass(frozen=True)
class NodeConfig:
    """Full configuration of the node-classification experiment."""

    gcn: GCNConfig
    baseline: CommonBaselineConfig
    training: TrainingConfig


@dataclass(frozen=True)
class GAEConfig:
    """GAE encoder hyperparameters (two GCN layers)."""

    hidden_dim: int
    embedding_dim: int
    dropout: float


@dataclass(frozen=True)
class LinkBaselineConfig:
    """MLP encoder hyperparameters for the link-prediction baseline."""

    hidden_dims: list[int]
    embedding_dim: int
    dropout: float


@dataclass(frozen=True)
class LinkConfig:
    """Full configuration of the link-prediction experiment."""

    gae: GAEConfig
    baseline: LinkBaselineConfig
    training: TrainingConfig


@dataclass(frozen=True)
class GINConfig:
    """GIN hyperparameters (message-passing layers, epsilon fixed at 0)."""

    hidden_dim: int
    num_layers: int
    dropout: float


@dataclass(frozen=True)
class GraphTrainingConfig:
    """Optimization settings for graph classification (adds a batch size)."""

    optimizer: str
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int


@dataclass(frozen=True)
class GraphConfig:
    """Full configuration of the graph-classification experiment."""

    gin: GINConfig
    baseline: CommonBaselineConfig
    training: GraphTrainingConfig


def repo_root() -> Path:
    """Repository root, derived from the package location (src layout)."""
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
    if device not in ("cpu"):
        raise ValueError(f"Unsupported device: {device!r}")
    median_tiebreak = str(raw["median_tiebreak"])
    if median_tiebreak not in ("upper", "lower"):
        raise ValueError(f"Unsupported median tiebreak: {median_tiebreak!r}")
    paths = PathsConfig(
        data=root / raw["paths"]["data"],
        splits=root / raw["paths"]["splits"],
        reports=root / raw["paths"]["reports"],
        artifacts=root / raw["paths"]["artifacts"],
    )
    return CommonConfig(
        seeds=list(raw["seeds"]),
        device=device,
        paths=paths,
        median_tiebreak=median_tiebreak,
    )


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


def load_node() -> NodeConfig:
    """Load the node-level task configuration from ``config/node.yaml``."""
    raw = _load_yaml(repo_root() / "config" / "node.yaml")
    gcn = raw["model"]["gcn"]
    baseline = raw["model"]["baseline"]
    training = raw["training"]
    return NodeConfig(
        gcn=GCNConfig(
            hidden_dim=int(gcn["hidden_dim"]),
            dropout=float(gcn["dropout"]),
        ),
        baseline=CommonBaselineConfig(
            hidden_dims=[int(d) for d in baseline["hidden_dims"]],
            dropout=float(baseline["dropout"]),
        ),
        training=TrainingConfig(
            optimizer=str(training["optimizer"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            epochs=int(training["epochs"]),
        ),
    )


def load_link() -> LinkConfig:
    """Load the edge-level task configuration from ``config/link.yaml``."""
    raw = _load_yaml(repo_root() / "config" / "link.yaml")
    gae = raw["model"]["gae"]
    baseline = raw["model"]["baseline"]
    training = raw["training"]
    return LinkConfig(
        gae=GAEConfig(
            hidden_dim=int(gae["hidden_dim"]),
            embedding_dim=int(gae["embedding_dim"]),
            dropout=float(gae["dropout"]),
        ),
        baseline=LinkBaselineConfig(
            hidden_dims=[int(d) for d in baseline["hidden_dims"]],
            embedding_dim=int(baseline["embedding_dim"]),
            dropout=float(baseline["dropout"]),
        ),
        training=TrainingConfig(
            optimizer=str(training["optimizer"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            epochs=int(training["epochs"]),
        ),
    )


def load_graph() -> GraphConfig:
    """Load the graph-level task configuration from ``config/graph.yaml``."""
    raw = _load_yaml(repo_root() / "config" / "graph.yaml")
    gin = raw["model"]["gin"]
    baseline = raw["model"]["baseline"]
    training = raw["training"]
    return GraphConfig(
        gin=GINConfig(
            hidden_dim=int(gin["hidden_dim"]),
            num_layers=int(gin["num_layers"]),
            dropout=float(gin["dropout"]),
        ),
        baseline=CommonBaselineConfig(
            hidden_dims=[int(d) for d in baseline["hidden_dims"]],
            dropout=float(baseline["dropout"]),
        ),
        training=GraphTrainingConfig(
            optimizer=str(training["optimizer"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
        ),
    )
