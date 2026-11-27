"""
Rosenbrock-type optimization benchmark.

Reuses the objective and analytical gradient from the
Spectral Gradient Solver.

The formulation uses a coefficient of 1 rather than the
conventional Rosenbrock coefficient of 100.
"""

import numpy as np

from gradient_methods.spectral_gradient_solver.objectives import JR


def rosenbrock(x):
    """
    Evaluate the Rosenbrock-type objective.

    Parameters
    ----------
    x : array_like, shape (n,)
        Input vector with n >= 2.

    Returns
    -------
    float
        Objective value.
    """
    value, _ = JR(x)
    return value


def rosenbrock_gradient(x):
    """
    Evaluate the analytical Rosenbrock-type gradient.

    Parameters
    ----------
    x : array_like, shape (n,)
        Input vector with n >= 2.

    Returns
    -------
    ndarray, shape (n,)
        Analytical gradient.
    """
    _, gradient = JR(x)
    return gradient


def rosenbrock_hessian(x):
    """
    Evaluate the analytical Rosenbrock-type Hessian.

    Parameters
    ----------
    x : array_like, shape (n,)
        Input vector with n >= 2.

    Returns
    -------
    ndarray, shape (n, n)
        Symmetric Hessian matrix.
    """
    # Reuse JR's existing input validation.
    JR(x)
    x = np.asarray(x, dtype=float)

    n = x.size
    H = np.zeros((n, n), dtype=float)

    w = x[1:] - x[:-1] ** 2

    for i in range(n - 1):
        H[i, i] += 12.0 * x[i] ** 2 - 4.0 * x[i + 1] + 2.0
        H[i + 1, i + 1] += 2.0

        H[i, i + 1] = -4.0 * x[i]
        H[i + 1, i] = -4.0 * x[i]

    return H
