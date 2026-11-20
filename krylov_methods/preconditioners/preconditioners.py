"""Reusable fixed preconditioner inverse actions z = M^{-1} r.

The functions return callables; they never form an explicit inverse.
Whether a solver applies the action on the left, right, or in PCG is
determined by the solver, not by this module.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as spla


def _explicit(A):
    if isinstance(A, spla.LinearOperator) or callable(A):
        raise ValueError("This preconditioner needs an explicit matrix A.")
    A = A if sparse.issparse(A) else np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    return A


def _factor_inverse(M, *, spd=False):
    """Return inverse action for an explicit *preconditioner matrix* M."""
    if spd:
        mat = M.toarray() if sparse.issparse(M) else np.asarray(M)
        factor = linalg.cho_factor(mat, check_finite=True)
        return lambda r: linalg.cho_solve(factor, r)
    if sparse.issparse(M):
        solver = spla.splu(M.tocsc())
        return solver.solve
    lu, piv = linalg.lu_factor(np.asarray(M))
    return lambda r: linalg.lu_solve((lu, piv), r)


def make_preconditioner(A, kind=None, *, M=None, options=None, require_spd=False):
    """Build a reusable inverse-action callable.

    Parameters
    ----------
    A : matrix or LinearOperator
        System matrix. Named preconditioners need explicit A.
    kind : None, 'jacobi', 'lu', 'ilu', 'cholesky', 'ic0', 'ssor', 'custom'
        Chosen inverse-action type. None selects the identity.
    M : callable, LinearOperator, or explicit matrix, optional
        Used only for 'custom'. Callable/operator MUST already apply the
        inverse; explicit M is factored once and solved on each call.
    options : dict, optional
        Jacobi: diag_shift, diag_tol; LU: diag_shift; ILU: diag_shift,
        drop_tol, fill_factor; IC(0): tol; SSOR: omega, diag_shift.
    require_spd : bool
        Restrict to configurations appropriate for fixed-preconditioner CG.
        The user remains responsible for SPD system and custom actions.

    Returns
    -------
    callable
        A fixed action apply(r) = M^{-1} r.
    """
    options = dict(options or {})
    kind = "none" if kind is None else str(kind).lower()
    if kind == "identity":
        kind = "none"
    valid = {"none", "jacobi", "lu", "ilu", "cholesky", "ic0", "ssor", "custom"}
    if kind not in valid:
        raise ValueError(f"Unknown preconditioner {kind!r}; choose {sorted(valid)}")
    if require_spd and kind in {"ilu", "lu"}:
        raise ValueError("Generic LU/ILU is not guaranteed SPD; not enabled for PCG.")
    if kind != "custom" and M is not None:
        raise ValueError("Pass M only when kind='custom'.")
    if kind in {"none", "custom"} and options:
        raise ValueError(f"Unexpected options for {kind}: {list(options)}")
    if kind == "none":
        return lambda r: np.array(r, copy=True)
    if kind == "custom":
        if M is None:
            raise ValueError("Custom preconditioning requires M.")
        if isinstance(M, spla.LinearOperator):
            return M.matvec
        if callable(M):
            return M
        return _factor_inverse(M, spd=require_spd)

    A = _explicit(A)
    n = A.shape[0]
    if kind == "jacobi":
        diag_shift = options.pop("diag_shift", 0.0)
        diag_tol = options.pop("diag_tol", 1e-15)
        d = np.asarray(A.diagonal() if sparse.issparse(A) else np.diag(A)) + diag_shift
        if np.any(~np.isfinite(d)) or np.any(np.abs(d) <= diag_tol):
            raise ValueError("Jacobi requires a finite nonzero shifted diagonal.")
        if require_spd and np.any(np.abs(np.imag(d)) > 1e-12):
            raise ValueError("PCG Jacobi requires a real positive diagonal.")
        if require_spd and np.any(np.real(d) <= 0):
            raise ValueError("PCG Jacobi requires positive diagonal entries.")
        action = lambda r: r / d
    elif kind == "lu":
        shift = options.pop("diag_shift", 0.0)
        shifted = A + shift * sparse.eye(n, format="csc") if sparse.issparse(A) else A + shift * np.eye(n)
        action = _factor_inverse(shifted)
    elif kind == "ilu":
        shift = options.pop("diag_shift", 0.0)
        drop_tol = options.pop("drop_tol", 1e-4)
        fill_factor = options.pop("fill_factor", 10)
        shifted = sparse.csc_matrix(A) + shift * sparse.eye(n, format="csc")
        action = spla.spilu(shifted, drop_tol=drop_tol, fill_factor=fill_factor).solve
    elif kind == "cholesky":
        if options:
            raise ValueError("Cholesky does not accept options.")
        action = _factor_inverse(A, spd=True)
    elif kind == "ic0":
        tol = options.pop("tol", 1e-12)
        mat = A.toarray() if sparse.issparse(A) else np.asarray(A)
        if np.iscomplexobj(mat) or not np.allclose(mat, mat.T, atol=tol):
            raise ValueError("This IC(0) implementation requires real symmetric A.")
        L = np.zeros((n, n), dtype=float)
        pattern = np.tril(mat != 0, k=-1)
        for i in range(n):
            for j in range(i):
                if pattern[i, j]:
                    L[i, j] = (mat[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
            pivot = mat[i, i] - L[i, :i] @ L[i, :i]
            if pivot <= tol:
                raise ValueError(f"IC(0) nonpositive pivot at row {i}.")
            L[i, i] = np.sqrt(pivot)

        def action(r):
            y = linalg.solve_triangular(L, r, lower=True)
            return linalg.solve_triangular(L.T, y, lower=False)
    else:  # SSOR
        omega = float(options.pop("omega", 1.0))
        shift = options.pop("diag_shift", 0.0)
        if not 0 < omega < 2:
            raise ValueError("SSOR requires 0 < omega < 2.")
        S = sparse.csc_matrix(A) + shift * sparse.eye(n, format="csc")
        d = S.diagonal()
        if np.any(abs(d) < 1e-15):
            raise ValueError("SSOR requires a nonzero diagonal.")
        if require_spd and (np.any(np.real(d) <= 0) or np.any(abs(np.imag(d)) > 1e-12)):
            raise ValueError("PCG SSOR requires a real positive diagonal.")
        lower = sparse.tril(S, -1, format="csc") + sparse.diags(d / omega, format="csc")
        upper = sparse.triu(S, 1, format="csc") + sparse.diags(d / omega, format="csc")

        def action(r):
            y = spla.spsolve_triangular(lower, r, lower=True)
            return ((2 - omega) / omega) * spla.spsolve_triangular(upper, d * y, lower=False)

    if options:
        raise ValueError(f"Unexpected options for {kind}: {sorted(options)}")
    return action


def build_preconditioner(A, kind=None, *, M=None, options=None, require_spd=False):
    """Alias for make_preconditioner with the same explicit arguments."""
    return make_preconditioner(A, kind, M=M, options=options, require_spd=require_spd)
