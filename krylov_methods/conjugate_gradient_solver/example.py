"""
Numerical demonstration of Conjugate Gradient and PCG.

Run from the repository root:

    python3 -m krylov_methods.conjugate_gradient_solver.example
"""

import numpy as np

from .ConjugateGradient import solve_conjugate_gradient


def main():

    # -------------------------------------------------
    # Symmetric positive-definite test system
    # -------------------------------------------------

    n = 30

    A = (
        2 * np.eye(n)
        - np.eye(n, k=1)
        - np.eye(n, k=-1)
    )

    # Known exact solution
    x_exact = np.ones(n)

    b = A @ x_exact

    print("=" * 65)
    print("CONJUGATE GRADIENT — NUMERICAL DEMONSTRATION")
    print("=" * 65)

    print(f"\nMatrix dimension: {n}")
    print(f"Condition number: {np.linalg.cond(A):.3e}")
    print(f"Symmetry error: {np.linalg.norm(A - A.T):.3e}")

    print("\nPreconditioner comparison")
    print("-" * 65)

    # -------------------------------------------------
    # Available preconditioners
    # -------------------------------------------------

    preconditioners = (
        None,
        "jacobi",
        "ssor",
        "cholesky",
        "ic0",
        "custom",
    )

    # Custom inverse action: diagonal scaling
    diagonal = np.diag(A)

    def custom_inverse(r):
        return r / diagonal

    for kind in preconditioners:

        options = {}

        if kind == "custom":
            options["M"] = custom_inverse

        result = solve_conjugate_gradient(
            A,
            b,
            preconditioner=kind,
            rtol=1e-10,
            atol=1e-12,
            **options,
        )

        relative_error = (
            np.linalg.norm(result.solution - x_exact)
            / np.linalg.norm(x_exact)
        )

        label = kind if kind is not None else "none"

        print(
            f"{label:12} "
            f"converged={str(result.converged):5} "
            f"iterations={result.iterations:4d} "
            f"residual={result.residual_norms[-1]:.3e} "
            f"rel_error={relative_error:.3e}"
        )

        if result.breakdown:
            print(f"  Breakdown: {result.message}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
