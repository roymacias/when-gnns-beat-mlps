"""Train GAE and the baseline-encoder on Amazon Computers over the fixed set of seeds.

Run as ``python -m gnnbench.training.link`` after the data stage. For each seed
the pre-materialized split (training message graph, training positives, and the
fixed evaluation pairs) is loaded; both encoders are trained to reconstruct the
training edges through the shared inner-product decoder, with fresh negative
pairs sampled every epoch. Training negatives are excluded against the complete
graph so a real validation or test edge is never presented as a negative. The
weights achieving the highest validation performance are kept and saved
under ``artifacts/link/{gae,baseline}/weights_seed_XX.pt``. A per-seed,
per-model report is written to ``reports/training/link/{gae,baseline}/``.
"""

import copy
from collections.abc import Callable

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch_geometric.datasets import Amazon
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from gnnbench.config import CommonConfig, LinkConfig, load_common, load_link
from gnnbench.io import write_report
from gnnbench.models import GCN, MLP, decode
from gnnbench.seeds import set_seed


def _train_encoder(
    encode: Callable[[], Tensor],
    model: nn.Module,
    split: dict[str, Tensor],
    full_edge_index: Tensor,
    num_nodes: int,
    link_cfg: LinkConfig,
) -> dict[str, object]:
    """Train one encoder over full-batch; return training results."""
    cfg = link_cfg.training
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    train_pos = split["train_pos_edge_label_index"]
    val_index = split["val_edge_label_index"]
    val_label = split["val_edge_label"]

    history: list[dict[str, float | int]] = []
    best_val_auc = -1.0
    best_epoch = -1
    best_state: dict[str, Tensor] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = encode()
        # Fresh negatives each epoch, excluded against the complete graph.
        train_neg = negative_sampling(
            edge_index=full_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_pos.size(1),
        )
        pos_scores = decode(z, train_pos)
        neg_scores = decode(z, train_neg)
        scores = torch.cat([pos_scores, neg_scores])
        targets = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        loss = loss_fn(scores, targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = encode()
            val_scores = decode(z, val_index)
            val_loss = loss_fn(val_scores, val_label.float())
            auc = float(roc_auc_score(val_label.numpy(), val_scores.numpy()))

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss),
                "val_loss": float(val_loss),
                "val_auc": auc,
            }
        )
        if auc > best_val_auc:
            best_val_auc = auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    return {
        "best_state": best_state,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "history": history,
    }


def _run_seed(
    x: Tensor,
    full_edge_index: Tensor,
    split: dict[str, Tensor],
    link_cfg: LinkConfig,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Train both models for one seed; return their results keyed by model name."""
    in_dim = x.size(1)
    num_nodes = x.size(0)
    message = split["message_edge_index"]
    results: dict[str, dict[str, object]] = {}

    set_seed(seed)
    gae = GCN(in_dim, link_cfg.gae.hidden_dim, link_cfg.gae.embedding_dim, link_cfg.gae.dropout)
    results["gae"] = _train_encoder(
        lambda: gae(x, message),
        gae,
        split,
        full_edge_index,
        num_nodes,
        link_cfg,
    )

    set_seed(seed)
    baseline = MLP(
        in_dim,
        link_cfg.baseline.hidden_dims,
        link_cfg.baseline.embedding_dim,
        link_cfg.baseline.dropout,
    )
    results["baseline"] = _train_encoder(
        lambda: baseline(x),
        baseline,
        split,
        full_edge_index,
        num_nodes,
        link_cfg,
    )

    return results


def main() -> None:
    """Train both encoders over every seed and persist weights and reports."""
    common: CommonConfig = load_common()
    link_cfg = load_link()
    data = Amazon(root=str(common.paths.data), name="Computers", transform=NormalizeFeatures())[0]
    x = data.x
    full_edge_index = data.edge_index  # all real edges, both directions

    for seed in tqdm(common.seeds, desc="training", unit="seed"):
        split = torch.load(common.paths.splits / "link" / f"seed_{seed:02d}.pt", weights_only=True)
        results = _run_seed(x, full_edge_index, split, link_cfg, seed)
        for name, result in results.items():
            weights_dir = common.paths.artifacts / "link" / name
            weights_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result["best_state"], weights_dir / f"weights_seed_{seed:02d}.pt")

            write_report(
                {
                    "seed": seed,
                    "model": name,
                    "parameters": result["parameters"],
                    "best_epoch": result["best_epoch"],
                    "best_val_auc": result["best_val_auc"],
                    "history": result["history"],
                },
                common.paths.reports / "training" / "link" / name / f"report_seed_{seed:02d}.json",
            )

    print(
        f"Link training completed for {len(common.seeds)} seeds; "
        f"weights in {common.paths.artifacts / 'link'}, "
        f"reports in {common.paths.reports / 'training' / 'link'}."
    )


if __name__ == "__main__":
    main()
