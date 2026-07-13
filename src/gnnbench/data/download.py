"""Download the benchmark datasets and report their statistics.

Run as ``python -m gnnbench.data.download``. Datasets are stored under the
configured data root (PyTorch Geometric manages its own ``raw/`` and
``processed/`` layout inside it), and one statistics report is written for
each task under ``reports/data/{node,link,graph}/dataset.json``.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid, TUDataset

from gnnbench.config import load_common
from gnnbench.io import write_report


def edge_count(data: Data, is_undirected: bool) -> int:
    """Count graph edges, accounting for PyG's bidirectional storage."""
    assert data.edge_index is not None  # dataset contract
    num_edges = int(data.edge_index.size(1))
    return num_edges // 2 if is_undirected else num_edges


def average_degree(num_edges: int, num_nodes: int, is_undirected: bool) -> float:
    """Average node degree of a graph."""
    return 2 * num_edges / num_nodes if is_undirected else num_edges / num_nodes


def node_dataset_stats(data: Data, num_classes: int) -> dict[str, object]:
    """Statistics of the node-classification dataset (Cora)."""
    assert data.x is not None and data.y is not None  # dataset contract
    num_nodes = int(data.num_nodes)
    is_undirected = bool(data.is_undirected())
    num_edges = edge_count(data, is_undirected)
    avg_degree = average_degree(num_edges, num_nodes, is_undirected)
    classes = torch.unique(data.y)
    class_distribution = {str(int(c)): int((data.y == c).sum()) for c in classes}
    return {
        "dataset": "Cora",
        "nodes": num_nodes,
        "is_undirected": is_undirected,
        "edges": num_edges,
        "avg_degree": round(avg_degree, 2),
        "feature_dim": int(data.x.size(1)),
        "classes": num_classes,
        "class_distribution": class_distribution,
    }


def link_dataset_stats(data: Data) -> dict[str, object]:
    """Statistics of the link-prediction dataset (Amazon Computers)."""
    assert data.x is not None  # dataset contract
    num_nodes = int(data.num_nodes)
    is_undirected = bool(data.is_undirected())
    num_edges = edge_count(data, is_undirected)
    avg_degree = average_degree(num_edges, num_nodes, is_undirected)
    return {
        "dataset": "Amazon Computers",
        "nodes": num_nodes,
        "is_undirected": is_undirected,
        "edges": num_edges,
        "avg_degree": round(avg_degree, 2),
        "feature_dim": int(data.x.size(1)),
    }


def graph_dataset_stats(dataset: TUDataset) -> dict[str, object]:
    """Statistics of the graph-classification dataset (NCI1)."""
    class_counts: dict[int, int] = {}
    node_counts: list[int] = []
    edge_counts: list[int] = []
    is_undirected = True
    for graph in dataset:
        assert graph.x is not None and graph.y is not None  # dataset contract
        node_counts.append(int(graph.num_nodes or 0))
        graph_is_undirected = bool(graph.is_undirected())
        edge_counts.append(edge_count(graph, graph_is_undirected))
        label = int(graph.y.item())
        class_counts[label] = class_counts.get(label, 0) + 1
        is_undirected &= graph_is_undirected
    total = len(dataset)
    return {
        "dataset": "NCI1",
        "graphs": total,
        "is_undirected": is_undirected,
        "avg_nodes_per_graph": round(sum(node_counts) / total, 2),
        "avg_edges_per_graph": round(sum(edge_counts) / total, 2),
        "feature_dim": dataset.num_features,
        "classes": dataset.num_classes,
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
    }


def main() -> None:
    """Download the three benchmarks and write their statistics reports."""
    common = load_common()
    root = str(common.paths.data)
    reports = common.paths.reports / "data"

    cora = Planetoid(root=root, name="Cora")
    write_report(node_dataset_stats(cora[0], cora.num_classes), reports / "node" / "dataset.json")

    amazon = Amazon(root=root, name="Computers")
    write_report(link_dataset_stats(amazon[0]), reports / "link" / "dataset.json")

    nci1 = TUDataset(root=root, name="NCI1")
    write_report(graph_dataset_stats(nci1), reports / "graph" / "dataset.json")

    print(f"Datasets ready under {root}; reports written to {reports}.")


if __name__ == "__main__":
    main()
