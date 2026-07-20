"""Graph-classification models: GIN and the MLP baseline.

GIN uses message passing, the baseline processes each node independently.
Keeping readout and head identical isolates the effect of message passing.
"""

from collections.abc import Sequence

from torch import Tensor, nn
from torch_geometric.nn import GINConv, global_add_pool


def _gin_update_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """MLP used as a GIN layer's update function."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GIN(nn.Module):
    """Graph Isomorphism Network with sum readout and a linear classifier.

    Each GIN layer performs message passing, a two-layer update MLP with batch
    normalization and ReLU, followed by batch normalization, ReLU, and dropout,
    with epsilon fixed at 0.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.gins = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        widths = [in_dim] + [hidden_dim] * num_layers
        for layer in range(num_layers):
            self.gins.append(
                GINConv(_gin_update_mlp(widths[layer], widths[layer + 1]), train_eps=False)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Map a batch of graphs to class logits ``[B, num_classes]``."""
        for gin, batch_norm in zip(self.gins, self.batch_norms, strict=True):
            x = gin(x, edge_index)
            x = batch_norm(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        pooled = global_add_pool(x, batch)
        return self.head(pooled)


class GraphMLP(nn.Module):
    """MLP baseline with the same readout and head as GIN.

    Each block mirrors GIN's update path --- linear map, batch normalization,
    ReLU --- applied to the node's own features instead of to an aggregated
    neighborhood. Normalization, activation, dropout, readout, and head match GIN's,
    so the comparison isolates the absence of message passing.
    """

    def __init__(
        self, in_dim: int, hidden_dims: Sequence[int], num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        widths = [in_dim, *hidden_dims]
        for i, width in enumerate(hidden_dims):
            self.layers.append(nn.Linear(widths[i], widths[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(width))
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Map a batch of graphs to class logits ``[B, num_classes]``.

        ``edge_index`` is accepted to match GIN's signature but unused:
        the baseline does not exchange messages.
        """
        del edge_index
        for layer, batch_norm in zip(self.layers, self.batch_norms, strict=True):
            x = layer(x)
            x = batch_norm(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        pooled = global_add_pool(x, batch)
        return self.head(pooled)
