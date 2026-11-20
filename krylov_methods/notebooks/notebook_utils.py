"""Shared test problems, inverse actions, and plots for Krylov notebooks.

All test problems have reproducible random seeds and known exact solutions.
The plotting helpers support both per-iteration and explicitly indexed
histories (useful for restarted GMRES, which records some data per cycle).
"""

import numpy as np
import matplotlib.pyplot as plt


def spd_problem(n=30, condition=1e4, seed=42):
    """Build a real SPD system with prescribed spectral condition number."""
    _validate_spectrum(n, condition)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigenvalues = np.geomspace(1.0, condition, n)
    A = (Q * eigenvalues) @ Q.T
    A = (A + A.T) / 2
    x_true = rng.standard_normal(n)
    return A, A @ x_true, x_true


def hermitian_pd_problem(n=30, condition=1e4, seed=42):
    """Build a complex Hermitian PD system with a known exact solution."""
    _validate_spectrum(n, condition)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(Z)
    eigenvalues = np.geomspace(1.0, condition, n)
    A = (Q * eigenvalues) @ Q.conj().T
    A = (A + A.conj().T) / 2
    x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return A, A @ x_true, x_true


def _validate_spectrum(n, condition):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    if not np.isfinite(condition) or condition < 1:
        raise ValueError("condition must be finite and at least 1.")


def nonsymmetric_problem(n=30, seed=42):
    """Build a reproducible real, generally nonsymmetric dense system."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    A += 2.5 * np.eye(n)
    x_true = rng.standard_normal(n)
    return A, A @ x_true, x_true


def symmetric_indefinite_problem(n=30, condition=1e4, seed=42):
    """Create a nonsingular real symmetric indefinite system, with known x*."""
    _validate_spectrum(n, condition)
    if n < 2:
        raise ValueError("A symmetric indefinite example requires n >= 2.")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigenvalues = np.geomspace(1.0, condition, n)
    eigenvalues[::2] *= -1.0
    A = (Q * eigenvalues) @ Q.T
    A = (A + A.T) / 2
    x_true = rng.standard_normal(n)
    return A, A @ x_true, x_true


def ritz_spectrum_plot(A, ritz_values, title="Lanczos spectral approximation"):
    """Plot exact Hermitian eigenvalues against approximate Ritz values."""
    eigs = np.linalg.eigvalsh(A)
    ritz_values = np.asarray(ritz_values, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.scatter(eigs, np.zeros_like(eigs), marker="|", s=260,
               label="Exact eigenvalues")
    ax.scatter(ritz_values, np.ones_like(ritz_values), marker="x", s=75,
               label="Ritz values")
    ax.set(yticks=[0, 1], yticklabels=["Exact eigenvalues", "Ritz values"],
           xlabel="Eigenvalue", title=title)
    ax.set_ylim(-0.45, 1.45)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()
    return fig, ax


def ritz_residual_plot(A, basis, tridiagonal,
                       title="Lanczos — Ritz-pair residuals"):
    """Plot ||A u_i - theta_i u_i|| where T y_i=theta_i y_i, u_i=V y_i."""
    theta, Y = np.linalg.eigh(tridiagonal)
    U = np.asarray(basis) @ Y
    residuals = np.linalg.norm(A @ U - U * theta, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(np.arange(1, len(residuals) + 1),
                np.maximum(residuals, np.finfo(float).tiny), "o-")
    ax.set(xlabel="Ritz pair index (sorted by Ritz value)",
           ylabel=r"$\|Au_i-\theta_i u_i\|_2$", title=title)
    ax.grid(True, which="both", alpha=0.3)
    ax.margins(y=0.12)
    fig.tight_layout()
    plt.show()
    return residuals


def block_jacobi_inverse(A, block_size=4):
    """Return a fixed block-Jacobi inverse action for an SPD/HPD matrix.

    Each principal diagonal block is factored with Cholesky. This is an
    educational *dense* implementation, not a sparse direct solver.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not A.size:
        raise ValueError("A must be a nonempty square matrix.")
    if not np.all(np.isfinite(A)) or not np.allclose(A, A.conj().T):
        raise ValueError("A must be finite and Hermitian.")
    if (isinstance(block_size, (bool, np.bool_)) or
            not isinstance(block_size, (int, np.integer)) or block_size < 1):
        raise ValueError("block_size must be a positive integer.")

    n = A.shape[0]
    blocks = []
    for start in range(0, n, block_size):
        section = slice(start, min(n, start + block_size))
        L = np.linalg.cholesky(A[section, section])
        blocks.append((section, L))

    def apply(r):
        r = np.asarray(r)
        if r.shape != (n,):
            raise ValueError("Preconditioner input must have shape (n,).")
        out = np.empty(n, dtype=np.result_type(r.dtype, A.dtype))
        for section, L in blocks:
            out[section] = np.linalg.solve(
                L.conj().T, np.linalg.solve(L, r[section])
            )
        return out

    return apply


def _history_xy(data, label):
    """Normalize a history or an (iteration indices, values) tuple."""
    if isinstance(data, tuple) and len(data) == 2:
        indices = np.asarray(data[0])
        values = np.asarray(data[1], dtype=float)
    else:
        values = np.asarray(data, dtype=float)
        indices = np.arange(values.size)
    if indices.ndim != 1 or values.ndim != 1 or indices.shape != values.shape:
        raise ValueError(f"Invalid history dimensions for {label!r}.")
    if not np.all(np.isfinite(indices)) or not np.all(np.isfinite(values)):
        raise ValueError(f"History {label!r} contains non-finite values.")
    if np.any(values < 0):
        raise ValueError(f"History {label!r} contains negative norms.")
    return indices, values


def _convergence_plot(histories, title, ylabel, xlabel, threshold=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    tiny = np.finfo(float).tiny
    for label, data in histories.items():
        indices, values = _history_xy(data, label)
        if values.size:
            ax.semilogy(
                indices, np.maximum(values, tiny), marker="o", markersize=3,
                linewidth=1.7, label=label,
            )
    if threshold is not None:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative.")
        ax.axhline(
            max(threshold, tiny), color="black", linestyle="--",
            linewidth=1, label="Stopping threshold",
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, which="both", alpha=0.25)
    if ax.lines:
        ax.legend(loc="best")
    ax.margins(y=0.12)
    fig.tight_layout()
    plt.show()
    return fig, ax


def residual_plot(histories, threshold, title="Residual convergence", xlabel="Iteration"):
    """Plot residual norms; values can be arrays or (iterations, norms).

    Only pass *true* residuals here. For left-preconditioned GMRES, plot
    the Givens estimates separately because they have a different meaning.
    """
    return _convergence_plot(
        histories, title, r"True residual $\|b-Ax_k\|_2$", xlabel,
        threshold=threshold,
    )


def error_plot(histories, title="Relative solution-error convergence", xlabel="Iteration"):
    """Plot relative errors; accepts arrays or (iterations, errors)."""
    return _convergence_plot(
        histories, title, r"Relative error $\|x_k-x^*\|_2/\|x^*\|_2$",
        xlabel,
    )


def report(label, solution, x_true, b, A, iterations, converged):
    """Print solution diagnostics using the true residual and known solution."""
    solution = np.asarray(solution).reshape(-1)
    x_true = np.asarray(x_true).reshape(-1)
    residual = np.linalg.norm(np.asarray(b).reshape(-1) - A @ solution)
    exact_norm = np.linalg.norm(x_true)
    error = np.linalg.norm(solution - x_true)
    relative_error = error / exact_norm if exact_norm else error
    print(
        f"{label:20} converged={str(converged):5} "
        f"iterations={iterations:4d} residual={residual:.2e} "
        f"relative error={relative_error:.2e}"
    )
