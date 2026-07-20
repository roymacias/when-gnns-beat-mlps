"""Two-layer GCN for node classification.

Two message-passing layers give every node a two-hop receptive field.
"""

from torch import Tensor, nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """GCN with two graph-convolutional layers."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Map node features ``[N, in_dim]`` and structure to ``[N, out_dim]``."""
        x = self.conv1(x, edge_index).relu()
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
