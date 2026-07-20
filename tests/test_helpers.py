"""Median-index selection for the representative run."""

from gnnbench.evaluation.helpers import _median_index


def test_even_upper() -> None:
    assert _median_index(10, "upper") == 5  # rank 6 of 10


def test_even_lower() -> None:
    assert _median_index(10, "lower") == 4  # rank 5 of 10


def test_odd_is_central() -> None:
    assert _median_index(11) == 5
    assert _median_index(11) == 5


def test_adjusts_to_count() -> None:
    assert _median_index(20, "upper") == 10
    assert _median_index(13, "lower") == 6
