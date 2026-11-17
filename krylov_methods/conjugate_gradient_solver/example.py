"""Run with python -m conjugate_gradient_solver.example from its parent."""

import numpy as np
from .ConjugateGradient import solve_conjugate_gradient


def main():
    n = 30
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.ones(n)
    for kind in (None, "jacobi", "ssor"):
        result = solve_conjugate_gradient(
            A, b, preconditioner=kind, rtol=1e-10,
            store_history=True,
        )
        print(f"{kind or 'none':>8}: converged={result.converged}, "
              f"iterations={result.iterations}, "
              f"residual={result.residual_norms[-1]:.3e}")


if __name__ == "__main__":
    main()
