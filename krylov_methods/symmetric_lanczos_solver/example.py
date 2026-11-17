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
    print("Krylov dimension:", result.iterations)
    print("Orthogonality error:", np.linalg.norm(Q.T @ Q - np.eye(result.iterations)))
    print("Projection error:", np.linalg.norm(Q.T @ A @ Q - T))
    print("Ritz values:", result.ritz_values)

    projected = lanczos_galerkin(A, b, m=n, tol=1e-10)
    print("Galerkin converged:", projected.converged)
    print("Final residual:", projected.residual_norm)


if __name__ == "__main__":
    main()
