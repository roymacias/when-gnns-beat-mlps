"""Materialize seeded splits and certify their integrity.

Run as ``python -m gnnbench.data.splits`` after the download stage.

Per task:

- **node** (Cora): nothing is materialized; the canonical Planetoid masks are
  validated (disjointness) and their sizes reported.
- **link** (Amazon Computers): one file per seed with the training message
  graph and the evaluation pairs. The message graph is pinned to *training
  edges only* for both validation and test (PyTorch Geometric's default would
  add validation edges to the test message graph); the evaluation therefore
  never sees a held-out edge during message passing.
- **graph** (NCI1): one file per seed with stratified train/val/test graph
  indices.

Integrity violations on the real data raise ``ValueError`` at pipeline time
and abort the stage.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid, TUDataset
from torch_geometric.transforms import RandomLinkSplit

from gnnbench.config import load_common, load_graph_split, load_link_split
from gnnbench.io import write_report
from gnnbench.seeds import set_seed

# ---------------------------------------------------------------------------
# Graph classification: stratified index split
# ---------------------------------------------------------------------------


def graph_split_sizes(total: int, train: float, val: float, test: float) -> tuple[int, int, int]:
    """Integer subset sizes: floor for train and val, remainder to test."""
    total_fraction = train + val + test
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1, got {total_fraction}.")
    n_train = int(total * train)
    n_val = int(total * val)
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


def stratified_graph_split(
    labels: np.ndarray, sizes: tuple[int, int, int], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test indices with exact integer sizes."""
    _, n_val, n_test = sizes
    indices = np.arange(labels.shape[0])
    rest, test_idx = train_test_split(
        indices,
        test_size=n_test,
        stratify=labels,
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        rest,
        test_size=n_val,
        stratify=labels[rest],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Link prediction: edge split with pinned message graph
# ---------------------------------------------------------------------------


def canonical_edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Return undirected edges as canonical ``(min, max)`` pairs."""
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    return {(min(a, b), max(a, b)) for a, b in zip(src, dst, strict=True)}


def _sample_unique_negatives(
    num_nodes: int,
    forbidden: set[tuple[int, int]],
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``count`` unique undirected negative edges.

    Returns a ``[2, count]`` tensor of canonical ``(min, max)`` node pairs.
    Self-loops, forbidden pairs, and duplicate undirected edges are rejected until
    ``count`` unique negatives have been collected.
    """
    chosen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    while len(result) < count:
        size = 2 * (count - len(result)) + 16
        batch = torch.randint(0, num_nodes, (2, size), generator=generator)
        for a, b in zip(batch[0].tolist(), batch[1].tolist(), strict=True):
            if a == b:
                continue
            pair = (min(a, b), max(a, b))
            if pair in forbidden or pair in chosen:
                continue
            chosen.add(pair)
            result.append(pair)
            if len(result) == count:
                break
    return torch.tensor(result, dtype=torch.long).t().contiguous()


def _labeled_pairs(
    positives: torch.Tensor, negatives: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate positive and negative pairs into an index/label tensor pair."""
    index = torch.cat([positives, negatives], dim=1)
    label = torch.cat([torch.ones(positives.size(1)), torch.zeros(negatives.size(1))])
    return index, label


def link_split(data: Data, num_val: float, num_test: float, seed: int) -> tuple[Data, Data, Data]:
    """Seeded edge split with custom unique undirected 1:1 evaluation negatives.

    RandomLinkSplit provides the positive evaluation edges. Its sampled negatives
    are discarded and replaced with negatives generated separately, which are
    guaranteed distinct as undirected pairs. The message graph is pinned to
    training edges for both validation and test.
    """
    set_seed(seed)
    transform = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.0,  # positives only; negatives generated separately
    )
    train_data, val_data, test_data = transform(data)

    num_nodes = int(data.num_nodes)
    # A negative must be a non-edge of the original graph. RandomLinkSplit removes
    # the val/test edges from the training graph, so building the forbidden set
    # from train_data.edge_index alone would allow those real edges to be
    # sampled as negatives.
    forbidden = canonical_edge_set(data.edge_index)
    generator = torch.Generator().manual_seed(seed)

    for split in (val_data, test_data):
        positives = split.edge_label_index  # ratio=0.0, positives only
        negatives = _sample_unique_negatives(num_nodes, forbidden, positives.size(1), generator)
        forbidden |= canonical_edge_set(negatives)  # keep val and test negatives disjoint
        # Pin the message graph to training edges.
        split.edge_index = train_data.edge_index
        split.edge_label_index, split.edge_label = _labeled_pairs(positives, negatives)

    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# Integrity validators: raise ValueError on violation (real-data checks)
# ---------------------------------------------------------------------------


def validate_partition(parts: tuple[np.ndarray, ...], total: int) -> None:
    """Subsets must be pairwise disjoint and jointly cover ``total`` items."""
    sets = [set(part.tolist()) for part in parts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise ValueError(f"Subsets {i} and {j} overlap on {len(overlap)} indices.")
    union = set().union(*sets)
    if len(union) != total or sum(len(s) for s in sets) != total:
        raise ValueError(f"Subsets do not partition {total} items (union={len(union)}).")


def validate_stratification(
    labels: np.ndarray, parts: tuple[np.ndarray, ...], tolerance: float = 0.05
) -> None:
    """Per-subset class proportions must match the global ones within tolerance."""
    classes = np.unique(labels)
    global_props = {int(c): float(np.mean(labels == c)) for c in classes}
    for index, part in enumerate(parts):
        part_labels = labels[part]
        for c, expected in global_props.items():
            got = float(np.mean(part_labels == c))
            if abs(got - expected) > tolerance:
                raise ValueError(
                    f"Subset {index}: class {c} proportion {got:.3f} deviates from "
                    f"{expected:.3f} beyond tolerance {tolerance}."
                )


def _eval_pairs(split: Data) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Positive and negative evaluation pairs of a link split as canonical sets."""
    pairs = split.edge_label_index
    labels = split.edge_label
    positives = canonical_edge_set(pairs[:, labels == 1])
    negatives = canonical_edge_set(pairs[:, labels == 0])
    return positives, negatives


def validate_link_split(
    full_edge_index: torch.Tensor, train_data: Data, val_data: Data, test_data: Data
) -> None:
    """No evaluation pair may leak into the message graph; negatives must be non-edges."""
    message_edges = canonical_edge_set(train_data.edge_index)
    true_edges = canonical_edge_set(full_edge_index)
    for name, split in (("val", val_data), ("test", test_data)):
        positives, negatives = _eval_pairs(split)
        leaked = positives & message_edges
        if leaked:
            raise ValueError(f"{name}: {len(leaked)} positive pairs leak into the message graph.")
        fake_negatives = negatives & true_edges
        if fake_negatives:
            raise ValueError(f"{name}: {len(fake_negatives)} negatives are actual edges.")
        if len(negatives) != len(positives):
            raise ValueError(
                f"{name}: {len(negatives)} negatives for {len(positives)} positives (expected 1:1)."
            )
        if positives & true_edges != positives:
            raise ValueError(f"{name}: some positive pairs are not edges of the original graph.")


def validate_canonical_masks(data: Data) -> dict[str, object]:
    """Validate Cora mask disjointness; report per-split sizes and class counts."""
    masks = {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }
    names = list(masks)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = int((masks[names[i]] & masks[names[j]]).sum())
            if overlap:
                raise ValueError(f"Masks {names[i]} and {names[j]} overlap on {overlap} nodes.")

    labels = data.y
    report: dict[str, object] = {}
    for name, mask in masks.items():
        split_labels = labels[mask]
        classes = torch.unique(split_labels)
        counts = {str(int(c)): int((split_labels == c).sum()) for c in classes}
        report[name] = {"size": int(mask.sum()), "class_counts": counts}
    return report


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Validate/materialize the splits of the three tasks and write reports."""
    common = load_common()
    root = str(common.paths.data)
    reports = common.paths.reports / "data"

    # Node: canonical masks, nothing materialized.
    cora = Planetoid(root=root, name="Cora")[0]
    splits = validate_canonical_masks(cora)
    write_report(
        {"strategy": "canonical Planetoid masks", "splits": splits},
        reports / "node" / "splits.json",
    )

    # Link: seeded edge splits with custom evaluation negatives.
    link_cfg = load_link_split()
    amazon = Amazon(root=root, name="Computers")[0]
    link_dir = common.paths.splits / "link"
    link_dir.mkdir(parents=True, exist_ok=True)
    link_seeds: dict[str, object] = {}
    for seed in common.seeds:
        train_data, val_data, test_data = link_split(
            amazon, link_cfg.num_val, link_cfg.num_test, seed
        )
        validate_link_split(amazon.edge_index, train_data, val_data, test_data)
        torch.save(
            {
                "message_edge_index": train_data.edge_index,
                "train_pos_edge_label_index": train_data.edge_label_index,
                "val_edge_label_index": val_data.edge_label_index,
                "val_edge_label": val_data.edge_label,
                "test_edge_label_index": test_data.edge_label_index,
                "test_edge_label": test_data.edge_label,
            },
            link_dir / f"seed_{seed:02d}.pt",
        )
        link_seeds[f"seed_{seed:02d}"] = {
            "message_undirected_edges": int(train_data.edge_index.size(1)) // 2,
            "val_pairs": int(val_data.edge_label.numel()),
            "test_pairs": int(test_data.edge_label.numel()),
        }
    write_report(
        {
            "strategy": "RandomLinkSplit, message graph pinned to training edges",
            "num_val": link_cfg.num_val,
            "num_test": link_cfg.num_test,
            "negative_ratio_eval": 1.0,
            "seeds": link_seeds,
        },
        reports / "link" / "splits.json",
    )

    # Graph: seeded stratified index splits.
    graph_cfg = load_graph_split()
    nci1 = TUDataset(root=root, name="NCI1")
    labels = np.array([int(graph.y.item()) for graph in nci1])
    sizes3 = graph_split_sizes(len(labels), graph_cfg.train, graph_cfg.val, graph_cfg.test)
    graph_dir = common.paths.splits / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_seeds: dict[str, object] = {}
    for seed in common.seeds:
        parts = stratified_graph_split(labels, sizes3, seed)
        validate_partition(parts, len(labels))
        validate_stratification(labels, parts)
        train_idx, val_idx, test_idx = parts
        torch.save(
            {
                "train_idx": torch.from_numpy(train_idx),
                "val_idx": torch.from_numpy(val_idx),
                "test_idx": torch.from_numpy(test_idx),
            },
            graph_dir / f"seed_{seed:02d}.pt",
        )
        graph_seeds[f"seed_{seed:02d}"] = {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        }
    write_report(
        {
            "strategy": "stratified over graph labels",
            "fractions": {"train": graph_cfg.train, "val": graph_cfg.val, "test": graph_cfg.test},
            "seeds": graph_seeds,
        },
        reports / "graph" / "splits.json",
    )

    print(f"Splits materialized under {common.paths.splits}; reports written to {reports}.")


if __name__ == "__main__":
    main()
