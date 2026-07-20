"""Inner-product decoder for link prediction.

The decoder scores each node pair by the inner product of its embeddings.
"""

from torch import Tensor


def decode(z: Tensor, edge_index: Tensor) -> Tensor:
    """Score each pair in ``edge_index`` as the inner product of its embeddings.

    ``z`` are node embeddings ``[N, d]`` and ``edge_index`` is ``[2, E]``.
    The returned ``[E]`` tensor holds logits.
    """
    source, target = edge_index
    return (z[source] * z[target]).sum(dim=-1)
