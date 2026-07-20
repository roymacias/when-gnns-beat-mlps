"""Result-figure generation (matplotlib, PGF backend, LaTeX-typeset text).

All figures use the PGF backend so text is set by LaTeX in the document font,
and are written as vector PDFs.
"""

from pathlib import Path
from typing import Literal, TypedDict, cast

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("pgf")

import matplotlib.pyplot as plt  # noqa: E402
import umap  # noqa: E402

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "font.family": "serif",
        "text.usetex": True,
        "pgf.preamble": r"\usepackage{mathpazo}",
    }
)

GNN_COLOR = "#0072B2"
BASELINE_COLOR = "#E69F00"
CHANCE_COLOR = "#999999"
OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


class HistoryRecord(TypedDict):
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


class HistoryReport(TypedDict):
    history: list[HistoryRecord]


HistoryKey = Literal[
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
]


def plot_loss_curves(
    gnn_report: HistoryReport,
    baseline_report: HistoryReport,
    gnn_name: str,
    baseline_name: str,
    path: Path,
) -> None:
    """Two panels: training loss (left) and validation loss (right).

    Each panel superposes both the GNN and the baseline models, and the two
    panels use independent vertical scales.
    """
    gnn_epochs = _history(gnn_report, "epoch")
    base_epochs = _history(baseline_report, "epoch")

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6))
    for ax, phase, title in (
        (axes[0], "train_loss", "training"),
        (axes[1], "val_loss", "validation"),
    ):
        phase = cast(HistoryKey, phase)
        ax.plot(
            gnn_epochs,
            _history(gnn_report, phase),
            color=GNN_COLOR,
            label=gnn_name,
        )
        ax.plot(
            base_epochs,
            _history(baseline_report, phase),
            color=BASELINE_COLOR,
            label=baseline_name,
        )
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(frameon=False)
    axes[0].set_ylabel("loss")
    _save(fig, path)


def plot_umap(
    gnn_hidden: np.ndarray,
    baseline_hidden: np.ndarray,
    labels: np.ndarray,
    gnn_name: str,
    baseline_name: str,
    path: Path,
) -> None:
    """Two panels: 2-D UMAP of GNN and baseline representations, colored by class.

    Each panel is an independent projection; coordinates are not comparable
    across panels. Class identity uses the Okabe-Ito palette (one hue per class).
    ``random_state`` is fixed so the projection is reproducible from its
    input (already seed-determined).
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    for ax, hidden, title in (
        (axes[0], gnn_hidden, gnn_name),
        (axes[1], baseline_hidden, baseline_name),
    ):
        reducer = umap.UMAP(n_components=2, random_state=0)
        coords = reducer.fit_transform(hidden)
        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=6,
                color=OKABE_ITO[int(cls) % len(OKABE_ITO)],
                label=str(int(cls)),
                linewidths=0,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, path)


def plot_roc(
    gnn_curve: tuple[np.ndarray, np.ndarray],
    baseline_curve: tuple[np.ndarray, np.ndarray],
    gnn_auc: float,
    baseline_auc: float,
    gnn_name: str,
    baseline_name: str,
    path: Path,
) -> None:
    """Single axes: superposed ROC curves for GNN and baseline plus the chance line.

    The superposed curves show the shape of each model's performance, and the
    AUC of each curve is annotated in the legend.
    """
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot(*gnn_curve, color=GNN_COLOR, label=f"{gnn_name} (AUC {gnn_auc:.3f})")
    ax.plot(
        *baseline_curve,
        color=BASELINE_COLOR,
        label=f"{baseline_name} (AUC {baseline_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color=CHANCE_COLOR, linestyle="--", label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def plot_confusion(
    gnn_cm: np.ndarray,
    baseline_cm: np.ndarray,
    labels: list[str],
    gnn_name: str,
    baseline_name: str,
    path: Path,
) -> None:
    """Two panels: GNN and baseline confusion matrices under a shared Viridis scale.

    The class labels are drawn once, on the left panel, since both axes are the same.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    for ax, matrix, title in (
        (axes[0], gnn_cm, gnn_name),
        (axes[1], baseline_cm, baseline_name),
    ):
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    axes[0].set_ylabel("true")
    axes[1].tick_params(labelleft=False)
    fig.colorbar(image, ax=axes, fraction=0.046)
    _save(fig, path)


def _history(report: HistoryReport, key: HistoryKey) -> list[int | float]:
    """Extract a per-epoch series from a training report history."""
    return [record[key] for record in report["history"]]


def _save(fig: Figure, path: Path) -> None:
    """Write the figure as a vector PDF and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
