"""Fixed SPD inverse-action preconditioners for Conjugate Gradient."""

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, aslinearoperator


def build_preconditioner(A, kind=None, M=None, options=None):
    """Return a callable implementing z=M^{-1}r.

    The supplied inverse action MUST represent a fixed Hermitian positive-
    definite operator; this cannot be verified for arbitrary callables.
    """
    options = {} if options is None else dict(options)
    n = A.shape[0]

    if kind is None:
        return lambda r: np.array(r, copy=True)

    if kind == "custom":
        if M is None:
            raise ValueError("Custom preconditioning requires M.")
        if callable(M):
            return M
        if isinstance(M, LinearOperator):
            return M.matvec
        mat = M.toarray() if sparse.issparse(M) else np.asarray(M)
        if mat.shape != (n, n):
            raise ValueError("M has incompatible dimensions.")
        factor = cho_factor(mat, check_finite=True)
        return lambda r: cho_solve(factor, r)

    if kind == "jacobi":
        if isinstance(A, LinearOperator):
            raise ValueError("Jacobi requires an explicit matrix.")
        diagonal = A.diagonal() if sparse.issparse(A) else np.diag(A)
        if np.any(~np.isfinite(diagonal)) or np.any(np.real(diagonal) <= 0):
            raise ValueError("Jacobi requires a positive finite diagonal.")
        return lambda r: r / diagonal

    if kind == "cholesky":
        if isinstance(A, LinearOperator):
            raise ValueError("Cholesky requires an explicit matrix.")
        mat = A.toarray() if sparse.issparse(A) else np.asarray(A)
        factor = cho_factor(mat, check_finite=True)
        return lambda r: cho_solve(factor, r)

    if kind == "ssor":
        if isinstance(A, LinearOperator):
            raise ValueError("SSOR requires an explicit matrix.")
        omega = float(options.get("omega", 1.0))
        if not 0 < omega < 2:
            raise ValueError("SSOR requires 0 < omega < 2.")
        mat = A.toarray() if sparse.issparse(A) else np.asarray(A)
        d = np.diag(mat)
        if np.any(np.real(d) <= 0):
            raise ValueError("SSOR requires a positive diagonal.")
        lower = np.diag(d / omega) + np.tril(mat, -1)
        upper = np.diag(d / omega) + np.triu(mat, 1)
        scale = (2 - omega) / omega
        # Inverse of (omega/(2-omega))*(D/omega+L)*D^-1*(D/omega+U)
        def apply(r):
            y = np.linalg.solve(lower, r)
            return scale * np.linalg.solve(upper, d * y)
        return apply

    raise ValueError(f"Unknown preconditioner: {kind!r}")
