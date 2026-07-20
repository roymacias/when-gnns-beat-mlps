"""Train GCN and the baseline on Cora over the fixed set of seeds.

Run as ``python -m gnnbench.training.node`` after the data stage. For each seed
and each model the network is trained full-batch for the configured number of
epochs; the weights achieving the highest validation performance are kept and
saved under ``artifacts/node/{gcn,baseline}/weights_seed_XX.pt``. A per-seed,
per-model report is written to ``reports/training/node/{gcn,baseline}/``.
"""

import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

from gnnbench.config import CommonConfig, NodeConfig, load_common, load_node
from gnnbench.io import write_report
from gnnbench.models import GCN, MLP
from gnnbench.seeds import set_seed


def _train_model(
    forward: Callable[[], Tensor],
    model: nn.Module,
    data: Data,
    node_cfg: NodeConfig,
) -> dict[str, object]:
    """Train one model over full-batch; return training results.

    ``forward`` closes over the model and returns the logits. This callable
    lets the same training loop handle both the GCN and the baseline.
    """
    cfg = node_cfg.training
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_epoch = -1
    best_state: dict[str, Tensor] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward()
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = forward()
            val_loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            predicted = logits[data.val_mask].argmax(dim=1)
            val_acc = float((predicted == data.y[data.val_mask]).float().mean())

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss),
                "val_loss": float(val_loss),
                "val_accuracy": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    return {
        "best_state": best_state,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "history": history,
    }


def _run_seed(data: Data, node_cfg: NodeConfig, seed: int) -> dict[str, dict[str, object]]:
    """Train both models for one seed; return their results keyed by model name."""
    in_dim = data.x.size(1)
    out_dim = int(data.y.max()) + 1
    results: dict[str, dict[str, object]] = {}

    set_seed(seed)
    gcn = GCN(in_dim, node_cfg.gcn.hidden_dim, out_dim, node_cfg.gcn.dropout)
    results["gcn"] = _train_model(lambda: gcn(data.x, data.edge_index), gcn, data, node_cfg)

    set_seed(seed)
    baseline = MLP(in_dim, node_cfg.baseline.hidden_dims, out_dim, node_cfg.baseline.dropout)
    results["baseline"] = _train_model(lambda: baseline(data.x), baseline, data, node_cfg)

    return results


def main() -> None:
    """Train both models over every seed and persist weights and reports."""
    common: CommonConfig = load_common()
    node_cfg = load_node()
    data = Planetoid(root=str(common.paths.data), name="Cora", transform=NormalizeFeatures())[0]

    for seed in tqdm(common.seeds, desc="training", unit="seed"):
        results = _run_seed(data, node_cfg, seed)
        for name, result in results.items():
            weights_dir = common.paths.artifacts / "node" / name
            weights_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result["best_state"], weights_dir / f"weights_seed_{seed:02d}.pt")

            write_report(
                {
                    "seed": seed,
                    "model": name,
                    "parameters": result["parameters"],
                    "best_epoch": result["best_epoch"],
                    "best_val_accuracy": result["best_val_accuracy"],
                    "history": result["history"],
                },
                common.paths.reports / "training" / "node" / name / f"report_seed_{seed:02d}.json",
            )

    print(
        f"Node training completed for {len(common.seeds)} seeds; "
        f"weights in {common.paths.artifacts / 'node'}, "
        f"reports in {common.paths.reports / 'training' / 'node'}."
    )


if __name__ == "__main__":
    main()
