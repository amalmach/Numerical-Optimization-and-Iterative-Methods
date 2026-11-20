"""
Numerical demonstration of BiCGSTAB.

Run from the repository root:

    python3 -m krylov_methods.bicgstab_solver.example
"""

import numpy as np
import scipy.sparse as sp

from .BiCGSTAB import solve_bicgstab


def main():
    n = 60
    A = sp.diags([-np.ones(n - 1), 4 * np.ones(n),
                  -0.7 * np.ones(n - 1)], [-1, 0, 1], format="csr")
    x_exact = np.ones(n)
    b = A @ x_exact
    for name in (None, "jacobi", "ilu", "ssor"):
        result = solve_bicgstab(A, b, preconditioner=name, rtol=1e-10)
        relative_error = (
            np.linalg.norm(result.solution - x_exact)
            / np.linalg.norm(x_exact)
        )
        print(
                f"{str(name):8} "
                f"converged={result.converged} "
                f"iterations={result.iterations:3d} "
                f"residual={result.residual_norms[-1]:.3e} "
                f"rel_error={relative_error:.3e}"
            )


if __name__ == "__main__":
    main()
