"""Model definitions: the GNN architectures and the MLP baseline."""

from gnnbench.models.decoder import decode
from gnnbench.models.gcn import GCN
from gnnbench.models.gin import GIN, GraphMLP
from gnnbench.models.mlp import MLP

__all__ = ["GCN", "GIN", "MLP", "GraphMLP", "decode"]
