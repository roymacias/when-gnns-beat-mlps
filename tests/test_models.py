"""Forward-shape and parameter-count invariants for all benchmark models."""

import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected

from gnnbench.models import GCN, MLP


def _toy_graph(features: int, nodes: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """A small connected graph with random node features."""
    x = torch.randn(nodes, features)
    edge_index = to_undirected(
        torch.stack([torch.arange(nodes), (torch.arange(nodes) + 1) % nodes])
    )
    return x, edge_index


def _count_parameters(model: nn.Module) -> int:
    """Number of learnable parameters (bias terms included)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------- Node classification --------------------

NODE_IN = 1433  # Cora feature dimension
GCN_HIDDEN = 16
MLP_HIDDEN = [32]
NODE_CLASSES = 7
NODE_DROPOUT = 0.5


def test_gcn_forward_shape() -> None:
    x, edge_index = _toy_graph(features=NODE_IN)
    gcn = GCN(NODE_IN, GCN_HIDDEN, NODE_CLASSES, NODE_DROPOUT)
    gcn.eval()
    out = gcn(x, edge_index)
    assert out.shape == (x.size(0), NODE_CLASSES)


def test_mlp_forward_shape() -> None:
    x, _ = _toy_graph(features=NODE_IN)
    mlp = MLP(NODE_IN, MLP_HIDDEN, NODE_CLASSES, NODE_DROPOUT)
    mlp.eval()
    out = mlp(x)
    assert out.shape == (x.size(0), NODE_CLASSES)


def test_mlp_baseline_larger_than_gcn() -> None:
    """The baseline must carry more parameters than the GCN."""
    gcn = GCN(NODE_IN, GCN_HIDDEN, NODE_CLASSES, NODE_DROPOUT)
    mlp = MLP(NODE_IN, MLP_HIDDEN, NODE_CLASSES, NODE_DROPOUT)
    assert _count_parameters(mlp) > _count_parameters(gcn)


def test_parameter_count_includes_bias() -> None:
    """Parameter counts include bias terms."""
    mlp = MLP(NODE_IN, [8], NODE_CLASSES, NODE_DROPOUT)
    expected = NODE_IN * 8 + 8 + 8 * NODE_CLASSES + NODE_CLASSES  # weights + biases
    assert _count_parameters(mlp) == expected
