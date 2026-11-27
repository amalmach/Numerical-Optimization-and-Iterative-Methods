"""
Acquisition functions for Bayesian global optimization.

Currently implements Expected Improvement (EI) for minimization.
"""

import numpy as np
from scipy.stats import norm


def expected_improvement(mean, std, best_value, xi=0.0):
    """
    Compute Expected Improvement for minimization.

    Parameters
    ----------
    mean : array_like
        Gaussian process predictive mean.
    std : array_like
        Gaussian process predictive standard deviation.
    best_value : float
        Best objective value observed so far.
    xi : float, optional
        Nonnegative exploration parameter.

    Returns
    -------
    ndarray or float
        Expected Improvement values.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    if not np.isfinite(best_value):
        raise ValueError("best_value must be finite.")

    if not np.isfinite(xi) or xi < 0:
        raise ValueError("xi must be nonnegative and finite.")

    mean, std = np.broadcast_arrays(mean, std)

    if not np.all(np.isfinite(mean)):
        raise ValueError("Predictive means must be finite.")

    if not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError(
            "Predictive standard deviations must be finite and nonnegative."
        )

    improvement = best_value - mean - xi

    # Handle zero uncertainty without division by zero.
    safe_std = np.where(std > 0, std, 1.0)

    z = improvement / safe_std

    ei_positive = (
        improvement * norm.cdf(z)
        + std * norm.pdf(z)
    )

    ei = np.where(
        std > 0,
        ei_positive,
        np.maximum(improvement, 0.0),
    )

    ei = np.maximum(ei, 0.0)

    return float(ei) if ei.ndim == 0 else ei
