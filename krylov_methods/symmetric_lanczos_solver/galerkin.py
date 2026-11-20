"""Optional Lanczos Galerkin approximation to Ax=b."""

from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .Lanczos import lanczos_tridiagonalize, LanczosResult


@dataclass
class GalerkinResult:
    solution: np.ndarray | None
    converged: bool
    residual_norm: float | None
    lanczos: LanczosResult
    message: str


def lanczos_galerkin(A, b, x0=None, m=None, tol=1e-10, reorthogonalize=False):
    """Project Ax=b onto the Lanczos space; report singular T without raising.

    For general symmetric indefinite systems T may be singular, even if A is
    nonsingular. MINRES is preferable for a robust indefinite linear solve.
    """
    b = np.asarray(b)
    if b.ndim not in (1, 2) or (b.ndim == 2 and 1 not in b.shape):
        raise ValueError("b must have shape (n,), (n,1), or (1,n).")
    b = b.reshape(-1)
    if b.size == 0 or not np.issubdtype(b.dtype, np.number):
        raise ValueError("b must be a nonempty numerical vector.")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain finite numerical values.")
    n = b.size
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
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
    dtype = np.result_type(b.dtype, op.dtype, float)
    b = b.astype(dtype, copy=False)
    x = np.zeros(n, dtype=dtype) if x0 is None else np.asarray(x0, dtype=dtype).reshape(-1).copy()
    if x.shape != (n,):
        raise ValueError("x0 has incompatible shape.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must contain finite numerical values.")
    r = b - op.matvec(x)
    beta0 = np.linalg.norm(r)
    if beta0 <= tol:
        empty = LanczosResult(np.empty((n, 0), dtype=dtype), np.empty((0, 0)),
                              np.array([]), 0, 0.0, None, True, "Initial convergence.")
        return GalerkinResult(x, True, float(beta0), empty, "Initial guess converged.")
    lanczos = lanczos_tridiagonalize(A, r, m=m, reorthogonalize=reorthogonalize)
    rhs = np.zeros(lanczos.iterations, dtype=dtype)
    rhs[0] = beta0
    try:
        y = np.linalg.solve(lanczos.tridiagonal, rhs)
    except np.linalg.LinAlgError:
        return GalerkinResult(None, False, None, lanczos,
                              "Projected matrix singular; use MINRES for indefinite systems.")
    x = x + lanczos.basis @ y
    residual = float(np.linalg.norm(b - op.matvec(x)))
    return GalerkinResult(x, residual <= tol, residual, lanczos,
                          "Converged." if residual <= tol else "Krylov dimension reached.")
