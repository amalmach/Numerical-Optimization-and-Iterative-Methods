"""Preconditioned Conjugate Gradient for real SPD / complex Hermitian PD systems."""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .preconditioners import build_preconditioner


@dataclass
class ConjugateGradientResult:
    solution: np.ndarray
    iterations: int
    converged: bool
    residual_norms: np.ndarray
    history: np.ndarray | None
    breakdown: bool
    message: str


def solve_conjugate_gradient(
    A, b, x0=None, rtol=1e-8, atol=1e-12, max_iter=None,
    preconditioner=None, preconditioner_options=None, M=None,
    store_history=False,
):
    """Solve Ax=b by fixed-preconditioner CG.

    A must be Hermitian positive definite; the inverse-action
    preconditioner must also be fixed Hermitian positive definite.
    A may be dense, sparse, LinearOperator, or a callable A(x).
    Callable A is assumed to map vectors of length len(b) to the same length.
    Stopping threshold: max(atol, rtol*||b||_2).
    """
    b = np.asarray(b).reshape(-1)
    if b.ndim != 1 or not b.size or not np.issubdtype(b.dtype, np.number):
        raise ValueError("b must be a nonempty numerical vector.")
    n = b.size
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative.")
    if max_iter is None:
        max_iter = n * 10
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer.")

    if callable(A) and not isinstance(A, LinearOperator):
        operator = LinearOperator((n, n), matvec=A, dtype=np.result_type(b.dtype, float))
        explicit = operator
    else:
        explicit = A
        operator = aslinearoperator(A)
    if operator.shape != (n, n):
        raise ValueError("A has incompatible shape.")

    dtype = np.result_type(b.dtype, operator.dtype, float)
    b = b.astype(dtype, copy=False)
    x = np.zeros(n, dtype=dtype) if x0 is None else np.asarray(x0, dtype=dtype).reshape(-1).copy()
    if x.shape != (n,):
        raise ValueError("x0 has incompatible shape.")

    apply_M = build_preconditioner(explicit, preconditioner, M, preconditioner_options)
    threshold = max(float(atol), float(rtol) * np.linalg.norm(b))
    residuals = []
    history = [] if store_history else None

    def record(x_current):
        norm = float(np.linalg.norm(b - operator.matvec(x_current)))
        residuals.append(norm)
        if history is not None:
            history.append(x_current.copy())
        return norm

    def finish(k, converged, breakdown, message):
        return ConjugateGradientResult(
            x.copy(), k, converged, np.asarray(residuals),
            np.asarray(history) if history is not None else None,
            breakdown, message,
        )

    r = b - operator.matvec(x)
    if record(x) <= threshold:
        return finish(0, True, False, "Initial guess satisfies tolerance.")
    if max_iter == 0:
        return finish(0, False, False, "Iteration budget exhausted.")

    z = np.asarray(apply_M(r), dtype=dtype).reshape(-1)
    if z.shape != (n,) or not np.all(np.isfinite(z)):
        return finish(0, False, True, "Invalid preconditioner output.")
    rz = np.vdot(r, z)
    p = z.copy()

    for k in range(1, max_iter + 1):
        Ap = np.asarray(operator.matvec(p), dtype=dtype).reshape(-1)
        pAp = np.vdot(p, Ap)
        scale = max(1.0, np.linalg.norm(p) * np.linalg.norm(Ap))
        if not np.isfinite(pAp) or abs(pAp.imag) > 1e-10 * scale or pAp.real <= 0:
            return finish(k - 1, False, True, "Nonpositive or non-Hermitian curvature.")
        if not np.isfinite(rz) or abs(rz.imag) > 1e-10 * max(1.0, abs(rz.real)) or rz.real <= 0:
            return finish(k - 1, False, True, "Preconditioner is not positive definite.")

        alpha = rz.real / pAp.real
        x = x + alpha * p
        r = r - alpha * Ap
        # Periodically recompute r to limit recursive residual drift.
        if k % 50 == 0:
            r = b - operator.matvec(x)
        if record(x) <= threshold:
            return finish(k, True, False, "Converged.")

        z = np.asarray(apply_M(r), dtype=dtype).reshape(-1)
        if z.shape != (n,) or not np.all(np.isfinite(z)):
            return finish(k, False, True, "Invalid preconditioner output.")
        rz_next = np.vdot(r, z)
        if not np.isfinite(rz_next) or rz_next.real <= 0 or abs(rz_next.imag) > 1e-10 * max(1.0, abs(rz_next.real)):
            return finish(k, False, True, "Preconditioner is not positive definite.")
        beta = rz_next.real / rz.real
        p = z + beta * p
        rz = rz_next

    return finish(max_iter, False, False, "Maximum iterations reached.")
