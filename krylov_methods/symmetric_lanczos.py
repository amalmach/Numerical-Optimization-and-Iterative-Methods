#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Symmetric Lanczos Algorithm
===========================

Lanczos tridiagonalization and Galerkin approximation
for solving symmetric linear systems

    Ax = b.
"""

import numpy as np


def symmetric_lanczos(
    A,
    b,
    x0=None,
    m=None,
    tol=1e-10,
):
    """
    Apply the symmetric Lanczos algorithm.

    Parameters
    ----------
    A : array_like
        Symmetric system matrix.

    b : array_like
        Right-hand-side vector.

    x0 : array_like, optional
        Initial approximation.

    m : int, optional
        Maximum number of Lanczos iterations.

    tol : float
        Absolute residual tolerance.

    Returns
    -------
    dict
        Lanczos basis, tridiagonal matrix,
        approximate solution, and convergence information.

    Notes
    -----
    The approximate solution is obtained using

        x_m = x0 + V_m y_m,

    where

        T_m y_m = beta_0 e_1.

    This Galerkin approximation requires a nonsingular
    projected matrix T_m.
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
            "Symmetric Lanczos requires a symmetric matrix."
        )

    if tol <= 0:
        raise ValueError("Tolerance must be positive.")

    if m is None:
        m = n

    if m < 1:
        raise ValueError("m must be positive.")

    m = min(m, n)

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
    beta0 = np.linalg.norm(r0)

    residual_norms = [beta0]
    history = [x_initial.copy()]

    if beta0 <= tol:
        return {
            "solution": x_initial,
            "iterations": 0,
            "converged": True,
            "residual_norms": np.asarray(residual_norms),
            "history": np.asarray(history),
            "basis": np.empty((n, 0)),
            "tridiagonal": np.empty((0, 0)),
            "ritz_values": np.array([]),
        }

    V = np.zeros((n, m))
    alpha = np.zeros(m)
    beta = np.zeros(m - 1)

    V[:, 0] = r0 / beta0

    v_previous = np.zeros(n)
    beta_previous = 0.0

    iterations = 0
    converged = False

    for k in range(m):

        v = V[:, k]

        # Lanczos three-term recurrence
        w = A @ v - beta_previous * v_previous

        alpha[k] = np.dot(v, w)

        w = w - alpha[k] * v

        beta_next = np.linalg.norm(w)

        iterations = k + 1

        # Construct the projected tridiagonal matrix
        T = np.diag(alpha[:iterations])

        if iterations > 1:
            off_diagonal = beta[:iterations - 1]

            T += np.diag(off_diagonal, 1)
            T += np.diag(off_diagonal, -1)

        # Galerkin approximation
        rhs = np.zeros(iterations)
        rhs[0] = beta0

        try:
            y = np.linalg.solve(T, rhs)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError(
                "The projected Lanczos matrix is singular. "
                "The Galerkin approximation is undefined."
            ) from exc

        x = x_initial + V[:, :iterations] @ y

        residual_norm = np.linalg.norm(b - A @ x)

        residual_norms.append(residual_norm)
        history.append(x.copy())

        converged = residual_norm <= tol

        if converged:
            break

        # Stop if Krylov subspace cannot expand further
        if k == m - 1:
            break

        # Lanczos termination
        if beta_next <= 1e-14:
            break

        beta[k] = beta_next

        V[:, k + 1] = w / beta_next

        v_previous = v
        beta_previous = beta_next

    return {
        "solution": x,
        "iterations": iterations,
        "converged": converged,
        "residual_norms": np.asarray(residual_norms),
        "history": np.asarray(history),
        "basis": V[:, :iterations],
        "tridiagonal": T,
        "ritz_values": np.linalg.eigvalsh(T),
    }
