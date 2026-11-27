"""
Benchmark functions for global optimization.

Includes the modified Branin function used in the
Efficient Global Optimization (EGO) experiments.
"""

import numpy as np


def modified_branin(x1, x2):
    """
    Evaluate the modified Branin function.

    Parameters
    ----------
    x1 : float or ndarray
        First coordinate, defined on [-5, 10].
    x2 : float or ndarray
        Second coordinate, defined on [0, 15].

    Returns
    -------
    float or ndarray
        Objective function value.

    Notes
    -----
    The modified Branin function includes an additional
    linear term, 5*x1, compared with the classical Branin function.
    """
    a = 5.1 / (4.0 * np.pi**2)
    b = 5.0 / np.pi
    c = 1.0 - 1.0 / (8.0 * np.pi)

    return (
        (x2 - a * x1**2 + b * x1 - 6.0)**2
        + 10.0 * (c * np.cos(x1) + 1.0)
        + 5.0 * x1
    )


def modified_branin_normalized(y1, y2):
    """
    Evaluate the modified Branin function on [0, 1]^2.

    The coordinate transformation is

        x1 = 15*y1 - 5
        x2 = 15*y2

    Parameters
    ----------
    y1, y2 : float or ndarray
        Normalized coordinates.

    Returns
    -------
    float or ndarray
        Objective function value.
    """
    x1 = 15.0 * np.asarray(y1) - 5.0
    x2 = 15.0 * np.asarray(y2)

    return modified_branin(x1, x2)
