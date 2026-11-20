"""Demonstrate Lanczos Ritz values and the Galerkin projection."""

import numpy as np
from .Lanczos import lanczos_tridiagonalize
from .galerkin import lanczos_galerkin


def main():
    n = 20
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.random.default_rng(42).normal(size=n)

    result = lanczos_tridiagonalize(A, b, m=12, reorthogonalize=True)
    Q, T = result.basis, result.tridiagonal
    print("Matrix dimension:", n)
    print("Hermitian error:", np.linalg.norm(A - A.conj().T))
    print("Condition number:", np.linalg.cond(A))
    print("Requested Krylov dimension:", 12)
    print("Krylov dimension:", result.iterations)
    print("Stopped early:", result.stopped_early, "—", result.message)
    print("Orthogonality error:", np.linalg.norm(Q.conj().T @ Q - np.eye(result.iterations)))
    print("Projection error:", np.linalg.norm(Q.conj().T @ A @ Q - T))
    print("Ritz values:", result.ritz_values)
    ritz_values, Y = np.linalg.eigh(T)
    U = Q @ Y
    print("Ritz-pair residual norms:", np.linalg.norm(A @ U - U * ritz_values, axis=0))

    projected = lanczos_galerkin(A, b, m=n, tol=1e-10)
    print("Galerkin converged:", projected.converged)
    print("Final residual:", projected.residual_norm)
    if projected.solution is not None:
        exact = np.linalg.solve(A, b)
        print("Relative solution error:",
              np.linalg.norm(projected.solution - exact) / np.linalg.norm(exact))


if __name__ == "__main__":
    main()
