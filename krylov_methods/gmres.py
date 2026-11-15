#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generalized Minimal Residual Method (GMRES)
===========================================

Solve Ax = b using the Arnoldi process and
residual minimization.
"""

import numpy as np


def gmres(
    A,
    b,
    x0=None,
    tol=1e-8,
    max_iter=100,
):
    """
    Solve Ax = b using unrestarted GMRES.

    Parameters
    ----------
    A : array_like
        Square system matrix.

    b : array_like
        Right-hand-side vector.

    x0 : array_like, optional
        Initial approximation.

    tol : float
        Absolute residual tolerance.

    max_iter : int
        Maximum number of Arnoldi iterations.

    Returns
    -------
    dict
        Solution, convergence status, residual history,
        and approximate-solution history.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n = b.size

    if A.shape != (n, n):
        raise ValueError(
            "Matrix A must be square and compatible with b."
        )

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

    x_initial = x.copy()

    r0 = b - A @ x_initial
    beta = np.linalg.norm(r0)

    residual_norms = [beta]
    history = [x_initial.copy()]

    converged = beta <= tol
    iterations = 0

    if converged or max_iter == 0:
        return {
            "solution": x,
            "iterations": iterations,
            "converged": converged,
            "residual_norms": np.asarray(residual_norms),
            "history": np.asarray(history),
        }

    m = min(max_iter, n)

    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))

    V[:, 0] = r0 / beta

    g = np.zeros(m + 1)
    g[0] = beta

    for k in range(m):

        # Arnoldi iteration
        w = A @ V[:, k]

        for j in range(k + 1):
            H[j, k] = np.dot(V[:, j], w)
            w = w - H[j, k] * V[:, j]

        H[k + 1, k] = np.linalg.norm(w)

        if H[k + 1, k] > 0:
            V[:, k + 1] = w / H[k + 1, k]

        # Solve the GMRES least-squares problem
        Hk = H[:k + 2, :k + 1]
        gk = g[:k + 2]

        y, *_ = np.linalg.lstsq(
            Hk,
            gk,
            rcond=None,
        )

        x = x_initial + V[:, :k + 1] @ y

        residual = b - A @ x
        residual_norm = np.linalg.norm(residual)

        residual_norms.append(residual_norm)
        history.append(x.copy())

        iterations = k + 1
        converged = residual_norm <= tol

        if converged:
            break

        # Arnoldi breakdown
        if H[k + 1, k] == 0:
            break

    return {
        "solution": x,
        "iterations": iterations,
        "converged": converged,
        "residual_norms": np.asarray(residual_norms),
        "history": np.asarray(history),
    }
