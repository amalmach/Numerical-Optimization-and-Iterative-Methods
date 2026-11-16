"""Run from repository root with `python -m gmres_solver.example`."""

import numpy as np
import scipy.sparse as sp

from .GMRES import gmres


def main():
    A = sp.diags([-np.ones(49), 4 * np.ones(50), -np.ones(49)],
                 offsets=[-1, 0, 1], format="csr")
    b = np.ones(50)
    for label, options in (
        ("unrestarted", dict(restart=None, maxiter=50)),
        ("restarted", dict(restart=10, maxiter=100)),
        ("restarted + Jacobi", dict(restart=10, maxiter=100,
                                     preconditioner="jacobi")),
        ("restarted + ILU", dict(restart=10, maxiter=100,
                                  preconditioner="ilu")),
    ):
        result = gmres(A, b, rtol=1e-10, **options)
        print(f"{label:22} converged={result.converged} "
              f"iterations={result.iterations:3} "
              f"true residual={result.residual_norm:.3e}")


if __name__ == "__main__":
    main()
