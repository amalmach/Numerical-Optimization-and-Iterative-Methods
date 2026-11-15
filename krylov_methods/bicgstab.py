#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bi-Conjugate Gradient Stabilized Method (BiCGSTAB)
==================================================

Iterative method for solving the linear system

    Ax = b

for general nonsymmetric matrices.
"""

import numpy as np


def bicgstab(
    A,
    b,
    x0=None,
    tol=1e-8,
    max_iter=1000,
):
    """
    Solve Ax = b using the BiCGSTAB algorithm.

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
        Maximum number of iterations.

    Returns
    -------
    dict
        Solution and convergence diagnostics.
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

    r = b - A @ x
    r_hat = r.copy()

    p = np.zeros(n)
    v = np.zeros(n)

    rho_old = 1.0
    alpha = 1.0
    omega = 1.0

    residual_norm = np.linalg.norm(r)

    residual_norms = [residual_norm]
    history = [x.copy()]

    converged = residual_norm <= tol
    iterations = 0

    for k in range(max_iter):

        if converged:
            break

        rho = np.dot(r_hat, r)

        if abs(rho) < 1e-30:
            raise ArithmeticError(
                "BiCGSTAB breakdown: rho is zero."
            )

        if k == 0:
            p = r.copy()
        else:
            if abs(omega) < 1e-30:
                raise ArithmeticError(
                    "BiCGSTAB breakdown: omega is zero."
                )

            beta = (rho / rho_old) * (alpha / omega)

            p = r + beta * (p - omega * v)

        v = A @ p

        denominator = np.dot(r_hat, v)

        if abs(denominator) < 1e-30:
            raise ArithmeticError(
                "BiCGSTAB breakdown: denominator is zero."
            )

        alpha = rho / denominator

        s = r - alpha * v

        # Early convergence
        if np.linalg.norm(s) <= tol:

            x = x + alpha * p

            residual_norm = np.linalg.norm(b - A @ x)

            residual_norms.append(residual_norm)
            history.append(x.copy())

            iterations = k + 1
            converged = residual_norm <= tol

            break

        t = A @ s

        tt = np.dot(t, t)

        if abs(tt) < 1e-30:
            raise ArithmeticError(
                "BiCGSTAB breakdown: ||A s|| is zero."
            )

        omega = np.dot(t, s) / tt

        if abs(omega) < 1e-30:
            raise ArithmeticError(
                "BiCGSTAB breakdown: omega is zero."
            )

        x = x + alpha * p + omega * s

        r = s - omega * t

        residual_norm = np.linalg.norm(b - A @ x)

        residual_norms.append(residual_norm)
        history.append(x.copy())

        iterations = k + 1
        converged = residual_norm <= tol

        rho_old = rho

    return {
        "solution": x,
        "iterations": iterations,
        "converged": converged,
        "residual_norms": np.asarray(residual_norms),
        "history": np.asarray(history),
    }
