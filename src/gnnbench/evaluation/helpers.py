"""Helpers shared by the per-task evaluation pipelines."""


def median_seed(seeds: list[int], scores: list[float], tiebreak: str) -> tuple[int, float]:
    """Return the seed and score of the median run sorting by score."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    central = order[_median_index(len(scores), tiebreak)]
    return seeds[central], scores[central]


def _median_index(n: int, tiebreak: str = "upper") -> int:
    """Return the zero-based median index for ``n`` elements.

    For an even number of elements, ``tiebreak`` selects the lower or upper median
    (default: ``"upper"``).
    """
    if n % 2 == 1:
        return n // 2
    return n // 2 if tiebreak == "upper" else n // 2 - 1
