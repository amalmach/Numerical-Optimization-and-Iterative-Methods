"""Left preconditioners: each returned operator *applies* M^{-1}."""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def _matrix(A):
    if isinstance(A, spl.LinearOperator) or callable(A):
        raise ValueError("This preconditioner needs an explicit matrix A")
    A = A if sp.issparse(A) else np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    return A


def build_preconditioner(M=None, *, n=None, dtype=float):
    """Return a callable applying inverse preconditioning.

    M may be an already-inverted LinearOperator/callable or an explicit
    matrix whose inverse action is formed once by factorization.
    """
    if M is None:
        return lambda x: x.copy()
    if isinstance(M, spl.LinearOperator):
        return lambda x: M @ x
    if callable(M):
        return M
    if sp.issparse(M):
        solve = spl.factorized(M.tocsc())
        return solve
    lu, piv = la.lu_factor(np.asarray(M))
    return lambda x: la.lu_solve((lu, piv), x)


def Jacobi_Diag_Precond_LO(A, diag_tol=1e-15, diag_shift=0.0):
    A = _matrix(A)
    d = np.asarray(A.diagonal() if sp.issparse(A) else np.diag(A)) + diag_shift
    if np.any(np.abs(d) <= diag_tol):
        raise ValueError("Jacobi requires a nonzero shifted diagonal")
    return spl.LinearOperator(A.shape, matvec=lambda v: v / d, dtype=d.dtype)


def Exact_LU_Precond_LO(A, diag_shift=0.0):
    A = _matrix(A)
    n = A.shape[0]
    if sp.issparse(A):
        shifted = A.tocsc() + diag_shift * sp.eye(n, format="csc")
        lu = spl.splu(shifted)
        return spl.LinearOperator(A.shape, matvec=lu.solve, dtype=shifted.dtype)
    shifted = np.asarray(A) + diag_shift * np.eye(n)
    lu, piv = la.lu_factor(shifted)
    return spl.LinearOperator(A.shape, matvec=lambda v: la.lu_solve((lu, piv), v), dtype=shifted.dtype)


def ILU_Precond_LO(A, diag_shift=0.0, drop_tol=1e-4, fill_factor=10):
    A = _matrix(A)
    n = A.shape[0]
    shifted = sp.csc_matrix(A) + diag_shift * sp.eye(n, format="csc")
    ilu = spl.spilu(shifted, drop_tol=drop_tol, fill_factor=fill_factor)
    return spl.LinearOperator(A.shape, matvec=ilu.solve, dtype=shifted.dtype)


def Incomplete_Cholesky_Precond_LO(A, tol=1e-12):
    """IC(0) on the existing lower sparsity pattern; real SPD matrices only.

    Intended as an educational implementation, not a high-performance IC.
    """
    A = _matrix(A)
    dense = A.toarray() if sp.issparse(A) else np.asarray(A)
    if np.iscomplexobj(dense) or not np.allclose(dense, dense.T, atol=tol):
        raise ValueError("IC(0) requires a real symmetric matrix")
    n = A.shape[0]
    L = np.zeros_like(dense, dtype=float)
    lower_pattern = np.tril(dense != 0, k=-1)
    for i in range(n):
        for j in range(i):
            if lower_pattern[i, j]:
                L[i, j] = (dense[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
        pivot = dense[i, i] - L[i, :i] @ L[i, :i]
        if pivot <= tol:
            raise ValueError(f"IC(0) nonpositive pivot at row {i}")
        L[i, i] = np.sqrt(pivot)
    return spl.LinearOperator(A.shape, matvec=lambda v: la.solve_triangular(
        L.T, la.solve_triangular(L, v, lower=True), lower=False), dtype=float)


def SSOR_Precond_LO(A, omega=1.0, diag_shift=0.0):
    """Apply the standard SSOR inverse, up to a harmless scalar factor."""
    if not 0 < omega < 2:
        raise ValueError("SSOR requires 0 < omega < 2")
    A = _matrix(A)
    S = sp.csc_matrix(A)
    n = S.shape[0]
    if diag_shift:
        S = S + diag_shift * sp.eye(n, format="csc")
    d = S.diagonal()
    if np.any(abs(d) < 1e-15):
        raise ValueError("SSOR requires nonzero diagonal")
    lower = sp.tril(S, -1, format="csc") + sp.diags(d / omega, format="csc")
    upper = sp.triu(S, 1, format="csc") + sp.diags(d / omega, format="csc")

    def solve(v):
        y = spl.spsolve_triangular(lower, v, lower=True)
        y = d * y
        return ((2 - omega) / omega) * spl.spsolve_triangular(upper, y, lower=False)

    return spl.LinearOperator(S.shape, matvec=solve, dtype=S.dtype)


def select_preconditioner(A, kind=None, *, M=None, **options):
    """Choose none, jacobi, lu, ilu, ic0, ssor, or custom.

    A preconditioner is an inverse action, NOT an explicit inverse matrix.
    """
    if kind is None or str(kind).lower() in ("none", "identity"):
        if M is not None:
            raise ValueError("Use preconditioner='custom' when providing M")
        return build_preconditioner(None)
    kind = str(kind).lower()
    factories = {
        "jacobi": Jacobi_Diag_Precond_LO,
        "lu": Exact_LU_Precond_LO,
        "ilu": ILU_Precond_LO,
        "ic0": Incomplete_Cholesky_Precond_LO,
        "ssor": SSOR_Precond_LO,
    }
    if kind == "custom":
        if M is None:
            raise ValueError("Provide M for custom preconditioning")
        return build_preconditioner(M)
    if kind not in factories:
        raise ValueError(f"Unknown preconditioner {kind!r}; choose {list(factories)} or 'custom'")
    return build_preconditioner(factories[kind](A, **options))
