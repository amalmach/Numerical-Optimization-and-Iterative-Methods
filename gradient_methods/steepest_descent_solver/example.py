"""Run: python3 -m gradient_methods.steepest_descent_solver.example"""

import numpy as np
from scipy.sparse.linalg import cg
from .SteepestDescent import solve_steepest_descent


def main():
    n = 20
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    x_true = np.random.default_rng(42).standard_normal(n)
    b = A @ x_true
    result = solve_steepest_descent(
        A, b, rtol=1e-10, atol=1e-12, max_iter=10000,
    )
    x_cg, info = cg(A, b, rtol=1e-10, atol=1e-12)
    print(f"Matrix dimension: {n}; condition number: {np.linalg.cond(A):.3e}")
    print(f"Steepest Descent: converged={result.converged}, "
          f"iterations={result.iterations}, "
          f"residual={result.residual_norms[-1]:.3e}, "
          f"relative error={np.linalg.norm(result.solution-x_true)/np.linalg.norm(x_true):.3e}")
    print(f"SciPy CG: converged={info == 0}, residual={np.linalg.norm(b-A@x_cg):.3e}, "
          f"relative error={np.linalg.norm(x_cg-x_true)/np.linalg.norm(x_true):.3e}")


if __name__ == "__main__":
    main()
