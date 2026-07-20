"""MLP baseline, shared by every task."""

from collections.abc import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    """Multilayer perceptron over node features (no graph structure)."""

    def __init__(
        self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.dropout = dropout
        widths = [in_dim, *hidden_dims]
        self.hidden = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(len(hidden_dims))
        )
        self.out = nn.Linear(widths[-1], out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map node features ``[N, in_dim]`` to ``[N, out_dim]``."""
        for layer in self.hidden:
            x = nn.functional.relu(layer(x))
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)
