#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bi-Conjugate Gradient Stabilized Method (BiCGSTAB).

With true-residual checks and optional stationary
RIGHT preconditioning. Supports dense/sparse/matrix-free real or complex A.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse.linalg as spl

from .preconditioners import select_preconditioner


@dataclass
class BiCGSTABResult:
    """Solution and diagnostics from a BiCGSTAB run."""

    solution: np.ndarray
    iterations: int
    converged: bool
    residual_norms: np.ndarray
    history: np.ndarray | None
    message: str
    breakdown: bool

    @property
    def x(self):
        """Alias compatible with the GMRES result object."""
        return self.solution


def _matvec(A, n, dtype):
    if not callable(A) and not isinstance(A, spl.LinearOperator):
        if not hasattr(A, "shape") or A.shape != (n, n):
            raise ValueError("A must be square and compatible with b")
    elif hasattr(A, "shape") and A.shape != (n, n):
        raise ValueError("Operator shape is incompatible with b")

    def action(v):
        result = np.asarray(A(v) if callable(A) else A @ v,
                            dtype=dtype).reshape(-1)
        if result.shape != (n,) or not np.all(np.isfinite(result)):
            raise ValueError("A returned an invalid matrix-vector product")
        return result

    return action


def solve_bicgstab(A, b, x0=None, *, rtol=1e-8, atol=0.0,
                   max_iter=None, preconditioner=None, M=None,
                   preconditioner_options=None, shadow_residual=None,
                   store_history=False, residual_recompute=25,
                   callback: Callable | None = None, verbose=False):
    """Solve Ax=b using right-preconditioned BiCGSTAB.

    Parameters
    ----------
    A : matrix, sparse matrix, LinearOperator, or matvec callable
    b : array_like, shape (n,) or (n,1)
    x0 : initial approximation; defaults to zero
    rtol, atol : stopping condition ||b-Ax|| <= max(atol, rtol*||b||)
    max_iter : maximum BiCGSTAB iterations; defaults to 2*n
    preconditioner : None, 'jacobi', 'lu', 'ilu', 'ic0', 'ssor', 'custom'
    M : custom inverse-action callable/LinearOperator or preconditioner matrix
        with preconditioner='custom'.
    preconditioner_options : constructor settings, e.g. ILU drop_tol
    shadow_residual : optional fixed shadow vector; defaults to r0
    store_history : whether to store all iterate vectors (O(n*iterations))
    residual_recompute : every this many iterations refresh residual exactly;
        0 disables periodic replacement. True residual is always checked when
        an iteration appears converged.
    callback : optional callback(iteration, true_residual_norm)

    Returns
    -------
    BiCGSTABResult. Numerical breakdown returns converged=False with a
    detailed message, rather than silently returning an invalid solution.

    Notes
    -----
    The algorithm computes M^{-1}p and M^{-1}s before multiplying by A.
    That is stationary RIGHT preconditioning (not left preconditioning).
    It minimizes the residual of the original Ax=b, and uses the true
    residual norm for the stopping decision. Complex inner products use
    np.vdot (conjugating the shadow vector).
    """
    b = np.asarray(b).reshape(-1)
    if b.size == 0 or not np.issubdtype(b.dtype, np.number):
        raise ValueError("b must be a nonempty numeric vector")
    n = b.size
    dtype = np.result_type(b.dtype, getattr(A, "dtype", float),
                           getattr(M, "dtype", float), np.float64)
    b = b.astype(dtype)
    x = np.zeros(n, dtype=dtype) if x0 is None else np.asarray(x0, dtype=dtype).reshape(-1).copy()
    if x.shape != (n,):
        raise ValueError("x0 has incompatible size")
    if not (np.isfinite(rtol) and np.isfinite(atol)) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    if max_iter is None:
        max_iter = 2 * n
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if not isinstance(residual_recompute, (int, np.integer)) or residual_recompute < 0:
        raise ValueError("residual_recompute must be a nonnegative integer")

    mv = _matvec(A, n, dtype)
    psolve = select_preconditioner(
        A, preconditioner, M=M,
        **({} if preconditioner_options is None else dict(preconditioner_options)))

    def precondition(v):
        out = np.asarray(psolve(v), dtype=dtype).reshape(-1)
        if out.shape != (n,) or not np.all(np.isfinite(out)):
            raise ValueError("Preconditioner returned an invalid vector")
        return out

    tol_stop = max(atol, rtol * np.linalg.norm(b))
    r = b - mv(x)
    norm = float(np.linalg.norm(r))
    residuals = [norm]
    x_history = [x.copy()] if store_history else None

    def result(k, converged, message, breakdown=False):
        return BiCGSTABResult(x.copy(), k, converged, np.asarray(residuals),
                              np.asarray(x_history) if store_history else None,
                              message, breakdown)

    def record(k, rr):
        value = float(np.linalg.norm(rr))
        residuals.append(value)
        if store_history:
            x_history.append(x.copy())
        if callback is not None:
            callback(k, value)
        if verbose:
            print(f"iteration {k}: true residual norm={value:.6e}")
        return value

    if norm <= tol_stop:
        return result(0, True, "initial guess converged")
    if max_iter == 0:
        return result(0, False, "zero iteration budget")

    r_hat = r.copy() if shadow_residual is None else np.asarray(
        shadow_residual, dtype=dtype).reshape(-1).copy()
    if r_hat.shape != (n,) or np.linalg.norm(r_hat) == 0:
        raise ValueError("shadow_residual must be a nonzero vector of length n")
    p = np.zeros(n, dtype=dtype)
    v = np.zeros(n, dtype=dtype)
    rho_previous = dtype.type(1) if hasattr(dtype, "type") else 1.0
    alpha = 1.0
    omega = 1.0
    small = np.finfo(float).eps * 20

    for k in range(1, max_iter + 1):
        rho = np.vdot(r_hat, r)
        if abs(rho) <= small * np.linalg.norm(r_hat) * np.linalg.norm(r):
            return result(k - 1, False, "breakdown: shadow residual orthogonal to residual", True)

        if k == 1:
            p = r.copy()
        else:
            if abs(omega) <= small:
                return result(k - 1, False, "breakdown: omega approximately zero", True)
            beta = (rho / rho_previous) * (alpha / omega)
            p = r + beta * (p - omega * v)

        p_hat = precondition(p)
        v = mv(p_hat)
        denominator = np.vdot(r_hat, v)
        if abs(denominator) <= small * np.linalg.norm(r_hat) * np.linalg.norm(v):
            return result(k - 1, False, "breakdown: shadow vector orthogonal to A M^-1 p", True)
        alpha = rho / denominator
        s = r - alpha * v

        if np.linalg.norm(s) <= tol_stop:
            x += alpha * p_hat
            r = b - mv(x)  # never trust recurrence alone for convergence
            norm = record(k, r)
            if norm <= tol_stop:
                return result(k, True, "converged after alpha update")
            # A discrepancy between predicted and true residual means we
            # must restart the recurrence; report it explicitly for now.
            return result(k, False, "residual gap after early convergence", True)

        s_hat = precondition(s)
        t = mv(s_hat)
        tt = np.vdot(t, t)
        if abs(tt) <= small * np.linalg.norm(t) ** 2 or tt == 0:
            return result(k - 1, False, "breakdown: A M^-1 s approximately zero", True)
        omega = np.vdot(t, s) / tt
        if abs(omega) <= small:
            return result(k - 1, False, "breakdown: omega approximately zero", True)

        x += alpha * p_hat + omega * s_hat
        r = s - omega * t
        if residual_recompute and k % residual_recompute == 0:
            r = b - mv(x)
        # Always use the true residual for reported diagnostics and stopping.
        true_r = b - mv(x)
        norm = record(k, true_r)
        if norm <= tol_stop:
            return result(k, True, "converged")
        rho_previous = rho

    return result(max_iter, False, "maximum iteration count reached")


def bicgstab(A, b, x0=None, tol=1e-8, max_iter=1000):
    """Backward-compatible simple interface returning the original dict."""
    result = solve_bicgstab(A, b, x0=x0, rtol=0.0, atol=tol,
                            max_iter=max_iter, store_history=True)
    return dict(solution=result.solution, iterations=result.iterations,
                converged=result.converged, residual_norms=result.residual_norms,
                history=result.history, message=result.message,
                breakdown=result.breakdown)


def bicgstab_solver(A, b, x0=None, M=None, rtol=1e-8, atol=1e-8,
                    max_iter=None):
    """Legacy entry point returning only x, with right preconditioning.

    M is an inverse-action callable/LinearOperator or a matrix to be
    factorized. Raises RuntimeError on failed convergence.
    """
    result = solve_bicgstab(A, b, x0=x0, M=M,
                            preconditioner="custom" if M is not None else None,
                            rtol=rtol, atol=atol, max_iter=max_iter)
    if not result.converged:
        raise RuntimeError(result.message)
    return result.solution
