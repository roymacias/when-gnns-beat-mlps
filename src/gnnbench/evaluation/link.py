"""Evaluate the trained link-prediction models on Amazon Computers.

Run as ``python -m gnnbench.evaluation.link`` after training. Loads the best
weights of every seed, measures test performance on the fixed
evaluation pairs, and reports each model's distribution with its median, mean,
and standard deviation. Each model's own median run (by test AUC) drives its
figures: the loss curves and the ROC curve. The figures and the evaluation
report go to ``reports/evaluation/link/``.
"""

import json
from typing import NamedTuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor
from torch_geometric.datasets import Amazon
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

from gnnbench.config import CommonConfig, LinkConfig, load_common, load_link
from gnnbench.evaluation.figures import HistoryReport, plot_loss_curves, plot_roc
from gnnbench.evaluation.helpers import median_seed
from gnnbench.io import write_report
from gnnbench.models import GCN, MLP, decode


class SeedResult(NamedTuple):
    """Test AUC and average precision of both models for one seed."""

    seed: int
    gae_auc: float
    gae_ap: float
    baseline_auc: float
    baseline_ap: float


def _build_models(in_dim: int, link_cfg: LinkConfig) -> tuple[GCN, MLP]:
    """Instantiate the GCN and baseline encoders."""
    gae = GCN(in_dim, link_cfg.gae.hidden_dim, link_cfg.gae.embedding_dim, 0.0)
    baseline = MLP(in_dim, link_cfg.baseline.hidden_dims, link_cfg.baseline.embedding_dim, 0.0)
    return gae, baseline


def _load_weights(common: CommonConfig, model: str, seed: int) -> dict[str, Tensor]:
    """Load the saved weights of one model for one seed."""
    path = common.paths.artifacts / "link" / model / f"weights_seed_{seed:02d}.pt"
    return torch.load(path, weights_only=True)


def _load_split(common: CommonConfig, seed: int) -> dict[str, Tensor]:
    """Load the dataset splits for one seed."""
    return torch.load(common.paths.splits / "link" / f"seed_{seed:02d}.pt", weights_only=True)


def _load_training_report(common: CommonConfig, model: str, seed: int) -> HistoryReport:
    """Load the training report of one model for one seed."""
    path = common.paths.reports / "training" / "link" / model / f"report_seed_{seed:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _scores(z: Tensor, index: Tensor) -> np.ndarray:
    """Sigmoid probabilities for the evaluation pairs."""
    return torch.sigmoid(decode(z, index)).numpy()


def _embed(model: GCN | MLP, x: Tensor, message: Tensor, is_gnn: bool) -> Tensor:
    """Return node embeddings from the selected encoder."""
    return model(x, message) if is_gnn else model(x)


def _roc_curve(
    common: CommonConfig,
    model: GCN | MLP,
    name: str,
    seed: int,
    x: Tensor,
    is_gnn: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ROC curve of one model for one seed."""
    split = _load_split(common, seed)
    message = split["message_edge_index"]
    test_index = split["test_edge_label_index"]
    test_label = split["test_edge_label"].numpy()
    model.load_state_dict(_load_weights(common, name, seed))
    model.eval()
    with torch.no_grad():
        scores = _scores(_embed(model, x, message, is_gnn), test_index)
    fpr, tpr, _ = roc_curve(test_label, scores)
    return fpr, tpr


def main() -> None:
    """Measure test AUC/AP per seed; each model's median AUC run drives its figures."""
    common = load_common()
    link_cfg = load_link()
    data = Amazon(root=str(common.paths.data), name="Computers", transform=NormalizeFeatures())[0]
    x = data.x
    gae, baseline = _build_models(x.size(1), link_cfg)

    per_seed: list[SeedResult] = []
    for seed in tqdm(common.seeds, desc="evaluation", unit="seed"):
        split = _load_split(common, seed)
        message = split["message_edge_index"]
        test_index = split["test_edge_label_index"]
        test_label = split["test_edge_label"].numpy()

        gae.load_state_dict(_load_weights(common, "gae", seed))
        baseline.load_state_dict(_load_weights(common, "baseline", seed))
        gae.eval()
        baseline.eval()
        with torch.no_grad():
            gae_scores = _scores(_embed(gae, x, message, True), test_index)
            base_scores = _scores(_embed(baseline, x, message, False), test_index)

        per_seed.append(
            SeedResult(
                seed=seed,
                gae_auc=float(roc_auc_score(test_label, gae_scores)),
                gae_ap=float(average_precision_score(test_label, gae_scores)),
                baseline_auc=float(roc_auc_score(test_label, base_scores)),
                baseline_ap=float(average_precision_score(test_label, base_scores)),
            )
        )

    seeds = [row.seed for row in per_seed]

    gae_auc = [row.gae_auc for row in per_seed]
    gae_auc_distribution = sorted(
        [{"seed": r.seed, "auc": r.gae_auc} for r in per_seed],
        key=lambda r: r["auc"],
    )
    gae_ap = [row.gae_ap for row in per_seed]
    gae_ap_distribution = sorted(
        [{"seed": r.seed, "ap": r.gae_ap} for r in per_seed],
        key=lambda r: r["ap"],
    )

    base_auc = [row.baseline_auc for row in per_seed]
    base_auc_distribution = sorted(
        [{"seed": r.seed, "auc": r.baseline_auc} for r in per_seed],
        key=lambda r: r["auc"],
    )
    base_ap = [row.baseline_ap for row in per_seed]
    base_ap_distribution = sorted(
        [{"seed": r.seed, "ap": r.baseline_ap} for r in per_seed],
        key=lambda r: r["ap"],
    )

    gae_auc_seed, gae_auc_median = median_seed(seeds, gae_auc, common.median_tiebreak)
    base_auc_seed, base_auc_median = median_seed(seeds, base_auc, common.median_tiebreak)
    gae_ap_seed, gae_ap_median = median_seed(seeds, gae_ap, common.median_tiebreak)
    base_ap_seed, base_ap_median = median_seed(seeds, base_ap, common.median_tiebreak)

    figures_dir = common.paths.reports / "evaluation" / "link"

    # Loss figure seeded.
    plot_loss_curves(
        _load_training_report(common, "gae", gae_auc_seed),
        _load_training_report(common, "baseline", base_auc_seed),
        "GAE",
        "MLP",
        figures_dir / "link-loss.pdf",
    )

    # ROC figure seeded.
    gae_fpr, gae_tpr = _roc_curve(
        common,
        gae,
        "gae",
        gae_auc_seed,
        x,
        True,
    )
    base_fpr, base_tpr = _roc_curve(
        common,
        baseline,
        "baseline",
        base_auc_seed,
        x,
        False,
    )
    plot_roc(
        (gae_fpr, gae_tpr),
        (base_fpr, base_tpr),
        gae_auc_median,
        base_auc_median,
        "GAE",
        "MLP",
        figures_dir / "link-roc.pdf",
    )

    gae_auc_arr = np.array(gae_auc)
    gae_ap_arr = np.array(gae_ap)
    base_auc_arr = np.array(base_auc)
    base_ap_arr = np.array(base_ap)

    write_report(
        {
            "metrics": {
                "auc": {
                    "distribution": {
                        "gae": gae_auc_distribution,
                        "baseline": base_auc_distribution,
                    },
                    "median_seed": {
                        "gae": gae_auc_seed,
                        "baseline": base_auc_seed,
                    },
                    "median": {
                        "gae": gae_auc_median,
                        "baseline": base_auc_median,
                    },
                    "mean": {
                        "gae": float(gae_auc_arr.mean()),
                        "baseline": float(base_auc_arr.mean()),
                    },
                    "std": {
                        "gae": float(gae_auc_arr.std(ddof=1)),
                        "baseline": float(base_auc_arr.std(ddof=1)),
                    },
                },
                "ap": {
                    "distribution": {
                        "gae": gae_ap_distribution,
                        "baseline": base_ap_distribution,
                    },
                    "median_seed": {
                        "gae": gae_ap_seed,
                        "baseline": base_ap_seed,
                    },
                    "median": {
                        "gae": gae_ap_median,
                        "baseline": base_ap_median,
                    },
                    "mean": {
                        "gae": float(gae_ap_arr.mean()),
                        "baseline": float(base_ap_arr.mean()),
                    },
                    "std": {
                        "gae": float(gae_ap_arr.std(ddof=1)),
                        "baseline": float(base_ap_arr.std(ddof=1)),
                    },
                },
            },
        },
        figures_dir / "results.json",
    )
    print(f"Link evaluation complete; outputs in {figures_dir}.")


if __name__ == "__main__":
    main()
