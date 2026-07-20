"""Train GIN and the baseline on NCI1 over the fixed set of seeds.

Run as ``python -m gnnbench.training.graph`` after the data stage. For each seed
the stratified split indices are loaded; both models are trained over
mini-batches of graphs (a seeded generator makes the shuffle deterministic),
each graph pooled to a single vector by the shared sum readout. The weights
achieving the highest validation performance are kept and saved under
``artifacts/graph/{gin,baseline}/weights_seed_XX.pt``. A per-seed,
per-model report is written to ``reports/training/graph/{gin,baseline}/``.
"""

import copy

import torch
from torch import Tensor, nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gnnbench.config import CommonConfig, GraphConfig, load_common, load_graph
from gnnbench.io import write_report
from gnnbench.models import GIN, GraphMLP
from gnnbench.seeds import set_seed


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    graph_cfg: GraphConfig,
) -> dict[str, object]:
    """Train one model over mini-batches; return training results."""
    cfg = graph_cfg.training
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
        epoch_loss = 0.0
        n_graphs = 0
        for data in train_loader:
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(logits, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * int(data.num_graphs)
            n_graphs += int(data.num_graphs)
        train_loss = epoch_loss / n_graphs

        model.eval()
        epoch_loss = 0.0
        n_graphs = 0
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                logits = model(data.x, data.edge_index, data.batch)
                epoch_loss += float(loss_fn(logits, data.y)) * int(data.num_graphs)
                correct += int((logits.argmax(dim=1) == data.y).sum())
                n_graphs += int(data.num_graphs)
        val_loss = epoch_loss / n_graphs
        val_acc = float(correct / n_graphs)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
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


def _loaders(
    dataset: TUDataset, split: dict[str, Tensor], batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    """Training loader (shuffle seeded per model) and validation loader."""
    train_set = dataset[split["train_idx"]]
    val_set = dataset[split["val_idx"]]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main() -> None:
    """Train both models over every seed and persist weights and reports."""
    common: CommonConfig = load_common()
    graph_cfg = load_graph()
    dataset = TUDataset(root=str(common.paths.data), name="NCI1")
    in_dim = dataset.num_features
    num_classes = dataset.num_classes

    for seed in tqdm(common.seeds, desc="training", unit="seed"):
        split = torch.load(common.paths.splits / "graph" / f"seed_{seed:02d}.pt", weights_only=True)
        gin_train_loader, gin_val_loader = _loaders(
            dataset, split, graph_cfg.training.batch_size, seed
        )
        base_train_loader, base_val_loader = _loaders(
            dataset, split, graph_cfg.training.batch_size, seed
        )

        set_seed(seed)
        gin = GIN(
            in_dim,
            graph_cfg.gin.hidden_dim,
            graph_cfg.gin.num_layers,
            num_classes,
            graph_cfg.gin.dropout,
        )
        set_seed(seed)
        baseline = GraphMLP(
            in_dim, graph_cfg.baseline.hidden_dims, num_classes, graph_cfg.baseline.dropout
        )

        results = {
            "gin": _train_model(gin, gin_train_loader, gin_val_loader, graph_cfg),
            "baseline": _train_model(baseline, base_train_loader, base_val_loader, graph_cfg),
        }
        for name, result in results.items():
            weights_dir = common.paths.artifacts / "graph" / name
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
                common.paths.reports / "training" / "graph" / name / f"report_seed_{seed:02d}.json",
            )

    print(
        f"Graph training completed for {len(common.seeds)} seeds; "
        f"weights in {common.paths.artifacts / 'graph'}, "
        f"reports in {common.paths.reports / 'training' / 'graph'}."
    )


if __name__ == "__main__":
    main()
