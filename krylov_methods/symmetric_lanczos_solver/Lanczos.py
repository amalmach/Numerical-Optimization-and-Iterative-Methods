"""Lanczos tridiagonalization for symmetric/Hermitian linear operators."""

from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator


@dataclass
class LanczosResult:
    basis: np.ndarray
    tridiagonal: np.ndarray
    ritz_values: np.ndarray
    iterations: int
    beta_next: float
    next_vector: np.ndarray | None
    stopped_early: bool
    message: str


def lanczos_tridiagonalize(A, v0, m=None, *, reorthogonalize=False, breakdown_tol=None):
    """Build a Lanczos basis for a Hermitian matrix/operator.

    Parameters
    ----------
    A : dense/sparse matrix, LinearOperator, or callable
        Hermitian matrix-vector product. Callables are assumed to be n-by-n.
    v0 : array_like
        Nonzero starting vector, normalized internally.
    m : int, optional
        Maximum Krylov dimension, capped at n.
    reorthogonalize : bool
        Apply two passes of full reorthogonalization, useful in finite precision.
    breakdown_tol : float, optional
        Threshold for a vanishing next Lanczos vector. When omitted, use a
        scale-sensitive floating-point threshold.

    Notes
    -----
    A must be Hermitian. Only explicit dense matrices are checked directly;
    the requirement is the caller's responsibility for other operators.
    """
    v0 = np.asarray(v0)
    if v0.ndim not in (1, 2) or (v0.ndim == 2 and 1 not in v0.shape):
        raise ValueError("v0 must have shape (n,), (n,1), or (1,n).")
    v0 = v0.reshape(-1)
    if v0.size == 0 or not np.issubdtype(v0.dtype, np.number):
        raise ValueError("v0 must be a nonempty numerical vector.")
    if not np.all(np.isfinite(v0)):
        raise ValueError("v0 must contain finite numerical values.")
    n = len(v0)
    if callable(A) and not isinstance(A, LinearOperator):
        # Probe once so a complex callable with a real starting vector is
        # assigned a complex dtype rather than silently losing information.
        probe = np.asarray(A(v0)).reshape(-1)
        if probe.shape != (n,) or not np.all(np.isfinite(probe)):
            raise ValueError("Callable A must return a finite vector of length n.")
        op = LinearOperator(
            (n, n), matvec=A,
            dtype=np.result_type(v0.dtype, probe.dtype, float),
        )
    else:
        op = aslinearoperator(A)
    if op.shape != (n, n):
        raise ValueError("A must have shape (len(v0), len(v0)).")
    if isinstance(A, np.ndarray) and not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian.")
    if m is None:
        m = n
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer.")
    m = min(int(m), n)
    if breakdown_tol is not None and (not np.isfinite(breakdown_tol) or breakdown_tol < 0):
        raise ValueError("breakdown_tol must be finite and nonnegative.")

    dtype = np.result_type(v0.dtype, op.dtype, float)
    q = np.asarray(v0, dtype=dtype)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("v0 must have nonzero finite norm.")
    q = q / norm
    V = np.zeros((n, m), dtype=dtype)
    alpha = np.zeros(m, dtype=float)
    beta = np.zeros(max(0, m - 1), dtype=float)
    old = np.zeros(n, dtype=dtype)
    previous_beta = 0.0
    next_vector = None
    beta_next = 0.0
    stopped = False
    message = "Requested Krylov dimension reached."

    for j in range(m):
        V[:, j] = q
        Aq = np.asarray(op.matvec(q)).reshape(-1)
        if Aq.shape != (n,) or not np.all(np.isfinite(Aq)):
            raise ArithmeticError("Invalid result from matrix-vector product.")
        if np.iscomplexobj(Aq) and not np.issubdtype(dtype, np.complexfloating):
            raise ValueError("Complex matrix-vector products require a complex operator dtype.")
        w = np.asarray(Aq, dtype=dtype).copy()
        w -= previous_beta * old
        a = np.vdot(q, w)
        if abs(a.imag) > 1e-10 * max(1.0, abs(a.real)):
            raise ValueError("The operator does not appear Hermitian.")
        alpha[j] = a.real
        w -= alpha[j] * q
        if reorthogonalize:
            Q = V[:, :j + 1]
            for _ in range(2):
                w -= Q @ (Q.conj().T @ w)
        beta_next = float(np.linalg.norm(w))
        scale = max(1.0, float(np.linalg.norm(Aq)))
        threshold = (10 * np.finfo(float).eps * scale
                     if breakdown_tol is None else breakdown_tol)
        if beta_next <= threshold:
            stopped = True
            message = "Lanczos invariant subspace reached (happy breakdown)."
            break
        if j == m - 1:
            next_vector = w / beta_next
            break
        beta[j] = beta_next
        old, q = q, w / beta_next
        previous_beta = beta_next

    k = j + 1
    T = np.diag(alpha[:k])
    if k > 1:
        T += np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
    return LanczosResult(V[:, :k], T, np.linalg.eigvalsh(T), k,
                         beta_next, next_vector, stopped, message)
