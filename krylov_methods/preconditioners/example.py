"""Run from repo root: python -m krylov_methods.preconditioners.example."""

import numpy as np
from . import make_preconditioner


def main():
    A = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
    r = np.array([1., 2., 3.])
    for kind in (None, "jacobi", "ssor", "cholesky", "ic0"):
        inverse_action = make_preconditioner(A, kind, require_spd=True)
        z = inverse_action(r)
        print(f"{str(kind):10} ||r-Az||_2={np.linalg.norm(r - A @ z):.3e}")
    print("Note: the application above illustrates inverse actions, not CG iterations.")


if __name__ == "__main__":
    main()
