#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Steepest Descent Method
=======================

Iterative method for solving the linear system

    Ax = b

where A is symmetric positive definite.
"""

import numpy as np


def steepest_descent(
    A,
    b,
    x0=None,
    tol=1e-8,
    max_iter=1000,
):
    """
    Solve Ax = b using the Steepest Descent algorithm.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Symmetric positive-definite matrix.

    b : array_like, shape (n,)
        Right-hand-side vector.

    x0 : array_like, optional
        Initial approximation. Defaults to zero.

    tol : float
        Absolute residual tolerance.

    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    dict
        Dictionary containing:
        - solution
        - iterations
        - converged
        - residual_norms
        - history
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n = b.size

    if A.shape != (n, n):
        raise ValueError(
            "Matrix A must be square and compatible with b."
        )

    if not np.allclose(A, A.T):
        raise ValueError(
            "Steepest Descent requires a symmetric matrix."
        )

    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Steepest Descent requires a positive-definite matrix."
        ) from exc

    if tol <= 0:
        raise ValueError("Tolerance must be positive.")

    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative.")

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()

        if x.size != n:
            raise ValueError("Initial guess has incorrect dimensions.")

    r = b - A @ x

    residual_norm = np.linalg.norm(r)

    residual_norms = [residual_norm]
    history = [x.copy()]

    converged = residual_norm <= tol
    iterations = 0

    for k in range(max_iter):

        if converged:
            break

        Ar = A @ r

        denominator = np.dot(r, Ar)

        if denominator <= 0:
            raise ArithmeticError(
                "Steepest Descent encountered a nonpositive denominator."
            )

        alpha = np.dot(r, r) / denominator

        x = x + alpha * r

        r = b - A @ x

        residual_norm = np.linalg.norm(r)

        residual_norms.append(residual_norm)
        history.append(x.copy())

        iterations = k + 1

        converged = residual_norm <= tol

    return {
        "solution": x,
        "iterations": iterations,
        "converged": converged,
        "residual_norms": np.asarray(residual_norms),
        "history": np.asarray(history),
    }
