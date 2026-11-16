"""Run: python -m bicgstab_solver.example"""

import numpy as np
import scipy.sparse as sp

from .BiCGSTAB import solve_bicgstab


def main():
    n = 60
    A = sp.diags([-np.ones(n - 1), 4 * np.ones(n),
                  -0.7 * np.ones(n - 1)], [-1, 0, 1], format="csr")
    b = np.ones(n)
    for name in (None, "jacobi", "ilu", "ssor"):
        result = solve_bicgstab(A, b, preconditioner=name, rtol=1e-10)
        print(f"{str(name):8} converged={result.converged} "
              f"iterations={result.iterations:3} "
              f"true residual={result.residual_norms[-1]:.3e}")


if __name__ == "__main__":
    main()
