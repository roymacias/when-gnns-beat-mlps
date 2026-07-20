"""Evaluate the trained node-classification models on Cora.

Run as ``python -m gnnbench.evaluation.node`` after training. Loads the best
weights of every seed, measures test performance, and reports each model's
distribution with its median, mean, and standard deviation. Each model's own
median run drives its figures: the loss curve and the UMAP projection.
The figures and the evaluation report go to ``reports/evaluation/node/``.
"""

import json
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

from gnnbench.config import CommonConfig, NodeConfig, load_common, load_node
from gnnbench.evaluation.figures import HistoryReport, plot_loss_curves, plot_umap
from gnnbench.evaluation.helpers import median_seed
from gnnbench.io import write_report
from gnnbench.models import GCN, MLP


class SeedResult(NamedTuple):
    """Test accuracy of both models for one seed."""

    seed: int
    gcn: float
    baseline: float


def _build_models(data: Data, node_cfg: NodeConfig) -> tuple[GCN, MLP]:
    """Instantiate the GCN and baseline models."""
    in_dim = data.x.size(1)
    out_dim = int(data.y.max()) + 1
    gcn = GCN(in_dim, node_cfg.gcn.hidden_dim, out_dim, node_cfg.gcn.dropout)
    baseline = MLP(in_dim, node_cfg.baseline.hidden_dims, out_dim, node_cfg.baseline.dropout)
    return gcn, baseline


def _load_weights(common: CommonConfig, model: str, seed: int) -> dict[str, Tensor]:
    """Load the saved weights of one model for one seed."""
    path = common.paths.artifacts / "node" / model / f"weights_seed_{seed:02d}.pt"
    return torch.load(path, weights_only=True)


def _load_training_report(common: CommonConfig, model: str, seed: int) -> HistoryReport:
    """Load the training report of one model for one seed."""
    path = common.paths.reports / "training" / "node" / model / f"report_seed_{seed:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_hidden(common: CommonConfig, baseline: MLP, x: Tensor, seed: int) -> Tensor:
    """Hidden representation before the output layer of the baseline."""
    baseline.load_state_dict(_load_weights(common, "baseline", seed))
    baseline.eval()
    with torch.no_grad():
        for layer in baseline.hidden:
            x = torch.relu(layer(x))
        return x


def _gcn_hidden(common: CommonConfig, gcn: GCN, data: Data, seed: int) -> Tensor:
    """Hidden representation before the output layer of the GCN."""
    gcn.load_state_dict(_load_weights(common, "gcn", seed))
    gcn.eval()
    with torch.no_grad():
        return gcn.conv1(data.x, data.edge_index).relu()


def _test_accuracy(logits: Tensor, data: Data) -> float:
    """Compute the test accuracy."""
    predicted = logits[data.test_mask].argmax(dim=1)
    return float((predicted == data.y[data.test_mask]).float().mean())


def main() -> None:
    """Measure test accuracy per seed; each model's median run drives its figures."""
    common = load_common()
    node_cfg = load_node()
    data = Planetoid(root=str(common.paths.data), name="Cora", transform=NormalizeFeatures())[0]
    gcn, baseline = _build_models(data, node_cfg)

    per_seed: list[SeedResult] = []
    for seed in tqdm(common.seeds, desc="evaluation", unit="seed"):
        gcn.load_state_dict(_load_weights(common, "gcn", seed))
        baseline.load_state_dict(_load_weights(common, "baseline", seed))
        gcn.eval()
        baseline.eval()
        with torch.no_grad():
            gcn_acc = _test_accuracy(gcn(data.x, data.edge_index), data)
            baseline_acc = _test_accuracy(baseline(data.x), data)
        per_seed.append(SeedResult(seed=seed, gcn=gcn_acc, baseline=baseline_acc))

    seeds = [row.seed for row in per_seed]
    gcn_scores = [row.gcn for row in per_seed]
    gcn_distribution = sorted(
        [{"seed": r.seed, "accuracy": r.gcn} for r in per_seed],
        key=lambda r: r["accuracy"],
    )
    baseline_scores = [row.baseline for row in per_seed]
    baseline_distribution = sorted(
        [{"seed": r.seed, "accuracy": r.baseline} for r in per_seed],
        key=lambda r: r["accuracy"],
    )
    gcn_seed, gcn_median = median_seed(seeds, gcn_scores, common.median_tiebreak)
    baseline_seed, baseline_median = median_seed(seeds, baseline_scores, common.median_tiebreak)

    figures_dir = common.paths.reports / "evaluation" / "node"

    # Loss figure seeded.
    plot_loss_curves(
        _load_training_report(common, "gcn", gcn_seed),
        _load_training_report(common, "baseline", baseline_seed),
        "GCN",
        "MLP",
        figures_dir / "node-loss.pdf",
    )

    # UMAP figure seeded (Cora split is seed-independent).
    gcn_hidden = _gcn_hidden(common, gcn, data, gcn_seed)
    baseline_hidden = _baseline_hidden(common, baseline, data.x, baseline_seed)
    plot_umap(
        gcn_hidden[data.test_mask].numpy(),
        baseline_hidden[data.test_mask].numpy(),
        data.y[data.test_mask].numpy(),
        "GCN",
        "MLP",
        figures_dir / "node-embeddings.pdf",
    )

    gcn_arr = np.array(gcn_scores)
    baseline_arr = np.array(baseline_scores)
    write_report(
        {
            "metrics": {
                "accuracy": {
                    "distribution": {
                        "gcn": gcn_distribution,
                        "baseline": baseline_distribution,
                    },
                    "median_seed": {
                        "gcn": gcn_seed,
                        "baseline": baseline_seed,
                    },
                    "median": {
                        "gcn": gcn_median,
                        "baseline": baseline_median,
                    },
                    "mean": {
                        "gcn": float(gcn_arr.mean()),
                        "baseline": float(baseline_arr.mean()),
                    },
                    "std": {
                        "gcn": float(gcn_arr.std(ddof=1)),
                        "baseline": float(baseline_arr.std(ddof=1)),
                    },
                },
            },
        },
        figures_dir / "results.json",
    )
    print(f"Node evaluation complete; outputs in {figures_dir}.")


if __name__ == "__main__":
    main()
