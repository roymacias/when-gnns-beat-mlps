"""Seeding: one seed determines every stochastic component of a run."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG the pipelines consume and enforce determinism.

    Covers Python's ``random``, NumPy's legacy global RNG, and torch. Also
    enables torch's deterministic algorithms so operations without a
    deterministic implementation raise an error instead of producing
    non-deterministic results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
