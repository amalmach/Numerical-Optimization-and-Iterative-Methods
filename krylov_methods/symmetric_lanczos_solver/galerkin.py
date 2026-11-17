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
    b = np.asarray(b).reshape(-1)
    n = b.size
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if callable(A) and not isinstance(A, LinearOperator):
        op = LinearOperator((n, n), matvec=A, dtype=np.result_type(b.dtype, float))
    else:
        op = aslinearoperator(A)
    dtype = np.result_type(b.dtype, op.dtype, float)
    b = b.astype(dtype, copy=False)
    x = np.zeros(n, dtype=dtype) if x0 is None else np.asarray(x0, dtype=dtype).reshape(-1).copy()
    if x.shape != (n,):
        raise ValueError("x0 has incompatible shape.")
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
