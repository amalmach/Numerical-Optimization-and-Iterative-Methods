"""Objective functions for spectral gradient experiments."""

import numpy as np


def _vector(x):
    """Validate an optimization vector."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a nonempty one-dimensional vector.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")
    return x


def J1(x):
    """J1(x) = sum((x_i - 1)^2)."""
    x = _vector(x)
    d = x - 1.0
    return float(d @ d), 2.0 * d


def J2(x):
    """J2(x) = sum((x_i - i)^2), i = 1,...,n."""
    x = _vector(x)
    d = x - np.arange(1, x.size + 1, dtype=float)
    return float(d @ d), 2.0 * d


def tridiag(n):
    """Construct the tridiagonal matrix A = tridiag(-1, 2, -1)."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    A = 2.0 * np.eye(n)
    A += -np.eye(n, k=1) - np.eye(n, k=-1)
    return A


def J5(x):
    """J5(x) = 0.5*x.T*A*x - 1.T*x + ||x||^2."""
    x = _vector(x)
    Ax = tridiag(x.size) @ x
    value = 0.5 * (x @ Ax) - np.sum(x) + (x @ x)
    gradient = Ax - np.ones_like(x) + 2.0 * x
    return float(value), gradient


def J6(x):
    """J6(x) = 0.5*x.T*A*x - 1.T*x + sum(x_i^4)."""
    x = _vector(x)
    Ax = tridiag(x.size) @ x
    value = 0.5 * (x @ Ax) - np.sum(x) + np.sum(x**4)
    gradient = Ax - np.ones_like(x) + 4.0 * x**3
    return float(value), gradient


def JR(x):
    """Rosenbrock-type objective from the original notebook."""
    x = _vector(x)
    if x.size < 2:
        raise ValueError("JR requires dimension >= 2.")

    w = x[1:] - x[:-1]**2
    d = x[:-1] - 1.0

    value = float(w @ w + d @ d)

    gradient = np.zeros_like(x)
    gradient[:-1] += -4.0 * x[:-1] * w + 2.0 * d
    gradient[1:] += 2.0 * w

    return value, gradient


OBJECTIVES = {1: J1, 2: J2, 5: J5, 6: J6, "R": JR}


def cost(objective, x):
    """Return objective value and gradient."""
    try:
        function = OBJECTIVES[objective]
    except (KeyError, TypeError):
        raise ValueError(
            "objective must be 1, 2, 5, 6, or 'R'."
        ) from None

    return function(x)
