from __future__ import annotations

import numpy as np


Array = np.ndarray


def quantile_higher(values: Array, q: float) -> float:
    """Return the q-quantile using the 'higher' interpolation (numpy-version agnostic)."""
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))
