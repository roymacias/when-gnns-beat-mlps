"""Evaluate the trained graph-classification models on NCI1.

Run as ``python -m gnnbench.evaluation.graph`` after training. Loads the best
weights of every seed, measures test performance, and reports each model's
distribution with its median, mean, and standard deviation. Each model's own
median run drives its figures: the loss curves and the confusion matrix.
The figures and the evaluation report go to ``reports/evaluation/graph/``.
"""

import json
from typing import NamedTuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gnnbench.config import CommonConfig, GraphConfig, load_common, load_graph
from gnnbench.evaluation.figures import HistoryReport, plot_confusion, plot_loss_curves
from gnnbench.evaluation.helpers import median_seed
from gnnbench.io import write_report
from gnnbench.models import GIN, GraphMLP


class SeedResult(NamedTuple):
    """Test accuracy of both models for one seed."""

    seed: int
    gin: float
    baseline: float


def _build_models(in_dim: int, num_classes: int, graph_cfg: GraphConfig) -> tuple[GIN, GraphMLP]:
    """Instantiate the GIN and baseline models."""
    gin = GIN(in_dim, graph_cfg.gin.hidden_dim, graph_cfg.gin.num_layers, num_classes, 0.0)
    baseline = GraphMLP(in_dim, graph_cfg.baseline.hidden_dims, num_classes, 0.0)
    return gin, baseline


def _load_weights(common: CommonConfig, model: str, seed: int) -> dict[str, Tensor]:
    """Load the saved weights of one model for one seed."""
    path = common.paths.artifacts / "graph" / model / f"weights_seed_{seed:02d}.pt"
    return torch.load(path, weights_only=True)


def _load_training_report(common: CommonConfig, model: str, seed: int) -> HistoryReport:
    """Load the training report of one model for one seed."""
    path = common.paths.reports / "training" / "graph" / model / f"report_seed_{seed:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _test_loader(
    common: CommonConfig, dataset: TUDataset, seed: int, batch_size: int
) -> DataLoader:
    """Build the test data loader for one seed."""
    split = torch.load(common.paths.splits / "graph" / f"seed_{seed:02d}.pt", weights_only=True)
    return DataLoader(dataset[split["test_idx"]], batch_size=batch_size, shuffle=False)


def _predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted and true labels over a data loader."""
    preds: list[int] = []
    trues: list[int] = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            logits = model(data.x, data.edge_index, data.batch)
            preds.extend(logits.argmax(dim=1).tolist())
            trues.extend(data.y.tolist())
    return np.array(preds), np.array(trues)


def _normalized_confusion(trues: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Row-normalized confusion matrix (each true-class row sums to one)."""
    matrix = confusion_matrix(trues, preds, labels=list(range(num_classes))).astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(row_sums, 1.0, None)


def main() -> None:
    """Measure test accuracy per seed; each model's median run drives its figures."""
    common = load_common()
    graph_cfg = load_graph()
    dataset = TUDataset(root=str(common.paths.data), name="NCI1")
    num_classes = dataset.num_classes
    batch_size = graph_cfg.training.batch_size
    gin, baseline = _build_models(dataset.num_features, num_classes, graph_cfg)

    per_seed: list[SeedResult] = []
    for seed in tqdm(common.seeds, desc="evaluation", unit="seed"):
        loader = _test_loader(common, dataset, seed, batch_size)
        gin.load_state_dict(_load_weights(common, "gin", seed))
        baseline.load_state_dict(_load_weights(common, "baseline", seed))
        gin_preds, gin_trues = _predictions(gin, loader)
        base_preds, base_trues = _predictions(baseline, loader)
        per_seed.append(
            SeedResult(
                seed=seed,
                gin=float((gin_preds == gin_trues).mean()),
                baseline=float((base_preds == base_trues).mean()),
            )
        )

    seeds = [row.seed for row in per_seed]
    gin_scores = [row.gin for row in per_seed]
    gin_distribution = sorted(
        [{"seed": r.seed, "accuracy": r.gin} for r in per_seed],
        key=lambda r: r["accuracy"],
    )
    base_scores = [row.baseline for row in per_seed]
    base_distribution = sorted(
        [{"seed": r.seed, "accuracy": r.baseline} for r in per_seed],
        key=lambda r: r["accuracy"],
    )
    gin_seed, gin_median = median_seed(seeds, gin_scores, common.median_tiebreak)
    base_seed, base_median = median_seed(seeds, base_scores, common.median_tiebreak)

    figures_dir = common.paths.reports / "evaluation" / "graph"

    # Loss figure seeded.
    plot_loss_curves(
        _load_training_report(common, "gin", gin_seed),
        _load_training_report(common, "baseline", base_seed),
        "GIN",
        "MLP",
        figures_dir / "graph-loss.pdf",
    )

    # Confusion figure seeded.
    gin.load_state_dict(_load_weights(common, "gin", gin_seed))
    gin_preds, gin_trues = _predictions(gin, _test_loader(common, dataset, gin_seed, batch_size))
    baseline.load_state_dict(_load_weights(common, "baseline", base_seed))
    base_preds, base_trues = _predictions(
        baseline, _test_loader(common, dataset, base_seed, batch_size)
    )
    plot_confusion(
        _normalized_confusion(gin_trues, gin_preds, num_classes),
        _normalized_confusion(base_trues, base_preds, num_classes),
        ["inactive", "active"],
        "GIN",
        "MLP",
        figures_dir / "graph-confusion.pdf",
    )

    gin_arr = np.array(gin_scores)
    base_arr = np.array(base_scores)
    write_report(
        {
            "metrics": {
                "accuracy": {
                    "distribution": {
                        "gin": gin_distribution,
                        "baseline": base_distribution,
                    },
                    "median_seed": {
                        "gin": gin_seed,
                        "baseline": base_seed,
                    },
                    "median": {
                        "gin": gin_median,
                        "baseline": base_median,
                    },
                    "mean": {
                        "gin": float(gin_arr.mean()),
                        "baseline": float(base_arr.mean()),
                    },
                    "std": {
                        "gin": float(gin_arr.std(ddof=1)),
                        "baseline": float(base_arr.std(ddof=1)),
                    },
                },
            },
        },
        figures_dir / "results.json",
    )
    print(f"Graph evaluation complete; outputs in {figures_dir}.")


if __name__ == "__main__":
    main()
