"""Steepest Descent with exact line search for real SPD/complex HPD systems."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator


@dataclass
class SteepestDescentResult:
    solution: np.ndarray
    iterations: int
    converged: bool
    residual_norms: np.ndarray
    history: np.ndarray | None
    breakdown: bool
    message: str


def solve_steepest_descent(A, b, x0=None, *, rtol=1e-8, atol=1e-12,
                          max_iter=None, store_history=False,
                          callback=None):
    """Solve Ax=b by exact-line-search Steepest Descent.

    A must be Hermitian positive definite. Explicit dense matrices are
    validated by Cholesky; sparse and matrix-free operators are assumed HPD.
    The stopping condition uses the true residual:
    ||b-Ax||_2 <= max(atol, rtol * ||b||_2).
    """
    b = np.asarray(b)
    if b.ndim not in (1, 2) or (b.ndim == 2 and 1 not in b.shape):
        raise ValueError("b must have shape (n,), (n,1), or (1,n).")
    b = b.reshape(-1)
    if b.size == 0 or not np.issubdtype(b.dtype, np.number):
        raise ValueError("b must be a nonempty numeric vector.")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain finite values.")
    n = b.size
    if not (np.isfinite(rtol) and np.isfinite(atol)) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative.")
    if max_iter is None:
        max_iter = 10 * n
    if (isinstance(max_iter, (bool, np.bool_)) or
            not isinstance(max_iter, (int, np.integer)) or max_iter < 0):
        raise ValueError("max_iter must be a nonnegative integer.")

    if callable(A) and not isinstance(A, LinearOperator):
        probe = np.asarray(A(b)).reshape(-1)
        if probe.shape != (n,) or not np.all(np.isfinite(probe)):
            raise ValueError("Callable A must return a finite vector of length n.")
        op = LinearOperator((n, n), matvec=A,
                            dtype=np.result_type(b.dtype, probe.dtype, float))
    else:
        op = aslinearoperator(A)
    if op.shape != (n, n):
        raise ValueError("A must have shape (len(b), len(b)).")

    if isinstance(A, np.ndarray):
        if not np.all(np.isfinite(A)) or not np.allclose(A, A.conj().T):
            raise ValueError("Dense A must be finite and Hermitian.")
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Dense A must be positive definite.") from exc

    dtype = np.result_type(b.dtype, op.dtype, float)
    b = b.astype(dtype, copy=False)
    x = np.zeros(n, dtype=dtype) if x0 is None else np.asarray(
        x0, dtype=dtype).reshape(-1).copy()
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite vector of length n.")

    threshold = max(float(atol), float(rtol) * np.linalg.norm(b))
    residual_norms = []
    history = [] if store_history else None

    def mv(v):
        result = np.asarray(op.matvec(v)).reshape(-1)
        if result.shape != (n,) or not np.all(np.isfinite(result)):
            raise ArithmeticError("A returned an invalid matrix-vector product.")
        if np.iscomplexobj(result) and not np.issubdtype(dtype, np.complexfloating):
            raise ValueError("Complex operator requires a complex dtype.")
        return np.asarray(result, dtype=dtype)

    def record(k):
        norm = float(np.linalg.norm(b - mv(x)))
        residual_norms.append(norm)
        if history is not None:
            history.append(x.copy())
        if callback is not None and k > 0:
            callback(k, x.copy(), norm)
        return norm

    def finish(k, converged, message, breakdown=False):
        return SteepestDescentResult(
            x.copy(), k, converged, np.asarray(residual_norms),
            np.asarray(history) if history is not None else None,
            breakdown, message,
        )

    r = b - mv(x)
    if record(0) <= threshold:
        return finish(0, True, "Initial guess satisfies tolerance.")

    for k in range(1, max_iter + 1):
        Ar = mv(r)
        denominator = np.vdot(r, Ar)
        rr = np.vdot(r, r).real
        scale = max(1.0, np.linalg.norm(r) * np.linalg.norm(Ar))
        if (not np.isfinite(denominator) or
                abs(denominator.imag) > 1e-10 * scale or
                denominator.real <= 0):
            return finish(k - 1, False, "Nonpositive or non-Hermitian curvature.", True)
        alpha = rr / denominator.real
        x += alpha * r
        # Recompute the true residual to avoid recursive-residual drift.
        r = b - mv(x)
        if record(k) <= threshold:
            return finish(k, True, "Converged.")

    return finish(max_iter, False, "Maximum iterations reached.")


def steepest_descent(A, b, x0=None, tol=1e-8, max_iter=1000):
    """Backward-compatible interface matching the original dictionary API."""
    result = solve_steepest_descent(
        A, b, x0=x0, rtol=0.0, atol=tol,
        max_iter=max_iter, store_history=True,
    )
    return {
        "solution": result.solution,
        "iterations": result.iterations,
        "converged": result.converged,
        "residual_norms": result.residual_norms,
        "history": result.history,
    }
