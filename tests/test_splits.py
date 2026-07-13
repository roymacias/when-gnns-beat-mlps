"""Split logic invariants, exercised offline on synthetic inputs."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from gnnbench.data.splits import (
    canonical_edge_set,
    graph_split_sizes,
    link_split,
    stratified_graph_split,
    validate_link_split,
    validate_partition,
    validate_stratification,
)


def synthetic_labels(zeros: int = 60, ones: int = 40) -> np.ndarray:
    """Binary labels with a configurable class imbalance."""
    return np.concatenate([np.zeros(zeros, dtype=np.int64), np.ones(ones, dtype=np.int64)])


def synthetic_graph(nodes: int = 40, chords: int = 40, seed: int = 66) -> Data:
    """Undirected ring plus random chords, with node features."""
    generator = torch.Generator().manual_seed(seed)
    ring = torch.stack([torch.arange(nodes), torch.arange(1, nodes + 1) % nodes])
    extra = torch.randint(0, nodes, (2, chords), generator=generator)
    edge_index = to_undirected(torch.cat([ring, extra], dim=1))
    features = torch.randn(nodes, 8, generator=generator)
    return Data(x=features, edge_index=edge_index, num_nodes=nodes)


class TestGraphSplit:
    def test_deterministic_given_seed(self) -> None:
        labels = synthetic_labels()
        sizes = graph_split_sizes(len(labels), 0.8, 0.1, 0.1)
        first = stratified_graph_split(labels, sizes, seed=3)
        second = stratified_graph_split(labels, sizes, seed=3)
        other = stratified_graph_split(labels, sizes, seed=4)
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a, b)
        assert any(not np.array_equal(a, c) for a, c in zip(first, other, strict=True))

    def test_sizes_floor_train_val_remainder_test(self) -> None:
        assert graph_split_sizes(97, 0.8, 0.1, 0.1) == (77, 9, 11)

    def test_sizes_reject_bad_fractions(self) -> None:
        with pytest.raises(ValueError):
            graph_split_sizes(100, 0.8, 0.1, 0.2)

    def test_partition_and_stratification(self) -> None:
        labels = synthetic_labels()
        sizes = graph_split_sizes(len(labels), 0.8, 0.1, 0.1)
        parts = stratified_graph_split(labels, sizes, seed=0)
        validate_partition(parts, len(labels))
        validate_stratification(labels, parts, tolerance=0.06)

    def test_validate_partition_rejects_overlap(self) -> None:
        with pytest.raises(ValueError):
            validate_partition((np.array([0, 1, 2]), np.array([2, 3])), total=4)

    def test_validate_partition_rejects_missing_items(self) -> None:
        with pytest.raises(ValueError):
            validate_partition((np.array([0, 1]), np.array([2])), total=5)


class TestLinkSplit:
    def test_deterministic_given_seed(self) -> None:
        data = synthetic_graph()
        first = link_split(data, num_val=0.1, num_test=0.2, seed=5)
        second = link_split(data, num_val=0.1, num_test=0.2, seed=5)
        assert torch.equal(first[2].edge_label_index, second[2].edge_label_index)
        assert torch.equal(first[0].edge_index, second[0].edge_index)

    @pytest.mark.parametrize("seed", range(15))
    def test_negative_invariants_hold_across_seeds(self, seed: int) -> None:
        """Directly assert the negative-generation contract over many seeds.

        Unlike the single validate_link_split call in the pipeline, this checks
        the invariants that motivated the custom negative sampling --- exactly one
        negative per positive, negatives distinct as undirected pairs, and no
        negative equal to a real edge --- on every split and every seed.
        """
        data = synthetic_graph()
        assert data.edge_index is not None
        real_edges = canonical_edge_set(data.edge_index)
        _, val_data, test_data = link_split(data, num_val=0.15, num_test=0.25, seed=seed)
        for split in (val_data, test_data):
            labels = split.edge_label
            positives = split.edge_label_index[:, labels == 1]
            negatives = split.edge_label_index[:, labels == 0]
            canonical_negatives = canonical_edge_set(negatives)
            # 1:1 in raw count, and no undirected duplicates collapsed the negatives.
            assert negatives.size(1) == positives.size(1)
            assert len(canonical_negatives) == negatives.size(1)
            # No negative is a real edge (in either orientation).
            assert not (canonical_negatives & real_edges)
            # No self-loops among negatives.
            assert all(a != b for a, b in canonical_negatives)

    def test_negatives_unique_in_dense_regime(self) -> None:
        """A small, dense graph makes non-edges scarce and collisions likely.

        The sampled negatives must still be unique and preserve the 1:1 ratio.
        """
        data = synthetic_graph(nodes=16, chords=40, seed=7)
        assert data.edge_index is not None
        real_edges = canonical_edge_set(data.edge_index)
        _, val_data, test_data = link_split(data, num_val=0.2, num_test=0.2, seed=0)
        for split in (val_data, test_data):
            labels = split.edge_label
            positives = split.edge_label_index[:, labels == 1]
            negatives = split.edge_label_index[:, labels == 0]
            canonical_negatives = canonical_edge_set(negatives)
            assert negatives.size(1) == positives.size(1)
            assert len(canonical_negatives) == negatives.size(1)
            assert not (canonical_negatives & real_edges)

    def test_val_and_test_negatives_disjoint(self) -> None:
        data = synthetic_graph()
        _, val_data, test_data = link_split(data, num_val=0.15, num_test=0.25, seed=3)
        val_neg = canonical_edge_set(val_data.edge_label_index[:, val_data.edge_label == 0])
        test_neg = canonical_edge_set(test_data.edge_label_index[:, test_data.edge_label == 0])
        assert not (val_neg & test_neg)

    def test_message_graph_pinned_to_train_edges(self) -> None:
        data = synthetic_graph()
        train_data, val_data, test_data = link_split(data, num_val=0.1, num_test=0.2, seed=1)
        assert torch.equal(test_data.edge_index, train_data.edge_index)
        assert torch.equal(val_data.edge_index, train_data.edge_index)

    def test_split_passes_integrity_validation(self) -> None:
        data = synthetic_graph()
        assert data.edge_index is not None
        train_data, val_data, test_data = link_split(data, num_val=0.1, num_test=0.2, seed=0)
        validate_link_split(data.edge_index, train_data, val_data, test_data)

    def test_validator_catches_injected_leakage(self) -> None:
        data = synthetic_graph()
        assert data.edge_index is not None
        train_data, val_data, test_data = link_split(data, num_val=0.1, num_test=0.2, seed=2)
        labels = val_data.edge_label
        leaked_pair = val_data.edge_label_index[:, labels == 1][:, :1]
        tampered = torch.cat([train_data.edge_index, leaked_pair, leaked_pair.flip(0)], dim=1)
        train_data.edge_index = tampered
        with pytest.raises(ValueError):
            validate_link_split(data.edge_index, train_data, val_data, test_data)
