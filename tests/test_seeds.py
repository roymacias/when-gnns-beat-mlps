"""Tests for deterministic RNG seeding across supported libraries."""

import random

import numpy as np
import torch

from gnnbench.seeds import set_seed


def test_python_random_seeded() -> None:
    set_seed(7)
    first = [random.random() for _ in range(16)]
    set_seed(7)
    second = [random.random() for _ in range(16)]
    assert first == second


def test_torch_stream_seeded() -> None:
    set_seed(7)
    first = torch.rand(16)
    set_seed(7)
    second = torch.rand(16)
    assert torch.equal(first, second)


def test_numpy_stream_seeded() -> None:
    set_seed(7)
    first = np.random.rand(16)
    set_seed(7)
    second = np.random.rand(16)
    assert np.array_equal(first, second)


def test_different_seeds_differ() -> None:
    set_seed(7)
    first = torch.rand(16)
    set_seed(8)
    second = torch.rand(16)
    assert not torch.equal(first, second)


def test_deterministic_algorithms_enforced() -> None:
    set_seed(0)
    assert torch.are_deterministic_algorithms_enabled()
