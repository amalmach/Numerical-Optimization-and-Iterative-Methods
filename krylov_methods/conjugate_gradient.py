#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conjugate Gradient Method
=========================

Iterative method for solving

    Ax = b

where A is symmetric positive definite.
"""

import numpy as np


def conjugate_gradient(
    A,
    b,
    x0=None,
    tol=1e-8,
    max_iter=1000,
):
    """
    Solve Ax = b using the Conjugate Gradient algorithm.

    Parameters
    ----------
    A : array_like
        Symmetric positive-definite matrix.

    b : array_like
        Right-hand-side vector.

    x0 : array_like, optional
        Initial approximation.

    tol : float
        Absolute residual tolerance.

    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    dict
        Numerical solution and convergence information.
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
            "Conjugate Gradient requires a symmetric matrix."
        )

    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Conjugate Gradient requires a positive-definite matrix."
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
            raise ValueError(
                "Initial guess has incorrect dimensions."
            )

    r = b - A @ x
    p = r.copy()

    rr = np.dot(r, r)

    residual_norm = np.sqrt(rr)

    residual_norms = [residual_norm]
    history = [x.copy()]

    converged = residual_norm <= tol
    iterations = 0

    for k in range(max_iter):

        if converged:
            break

        Ap = A @ p

        denominator = np.dot(p, Ap)

        if denominator <= 0:
            raise ArithmeticError(
                "CG encountered a nonpositive denominator."
            )

        alpha = rr / denominator

        x = x + alpha * p

        r = r - alpha * Ap

        rr_new = np.dot(r, r)

        residual_norm = np.sqrt(rr_new)

        residual_norms.append(residual_norm)
        history.append(x.copy())

        iterations = k + 1

        converged = residual_norm <= tol

        if converged:
            break

        beta = rr_new / rr

        p = r + beta * p

        rr = rr_new

    return {
        "solution": x,
        "iterations": iterations,
        "converged": converged,
        "residual_norms": np.asarray(residual_norms),
        "history": np.asarray(history),
    }
