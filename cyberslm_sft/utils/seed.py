"""
CyberSLM SFT — Reproducibility
================================
Centralised seeding so every run is deterministic given the same seed.
"""

from __future__ import annotations

import logging
import os
import random

import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set the RNG seed for Python, NumPy (if installed), and PyTorch.

    Parameters
    ----------
    seed: Integer seed value.  Use the same value across runs for
          deterministic results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic CUDA ops — may reduce throughput slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    logger.debug("Global seed set to %d", seed)
