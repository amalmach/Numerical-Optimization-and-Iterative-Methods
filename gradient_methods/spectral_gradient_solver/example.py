"""Reproducible experiments comparing BB1 and BB2."""

import numpy as np

from gradient_methods.spectral_gradient_solver import (
    solve_spectral_gradient,
)


def main():
    dimension = 10
    initial_guess = np.zeros(dimension)

    objectives = {
        1: "J1 — Quadratic, target vector of ones",
        2: "J2 — Quadratic, target vector (1, 2, ..., n)",
        5: "J5 — Tridiagonal quadratic",
        6: "J6 — Quartic regularization",
        "R": "JR — Rosenbrock",
    }

    print("=" * 85)
    print("BARZILAI–BORWEIN SPECTRAL GRADIENT COMPARISON")
    print("=" * 85)
    print(f"Dimension: {dimension}")
    print("Initial step: 0.25")
    print("Gradient tolerance: 1e-8")
    print("Maximum iterations: 300")

    for objective, description in objectives.items():
        print(f"\n{description}")
        print("-" * 85)

        print(
            f"{'Method':<10}"
            f"{'Iterations':>12}"
            f"{'Converged':>13}"
            f"{'Final cost':>18}"
            f"{'Gradient norm':>20}"
        )

        for variant in ("BB1", "BB2"):
            result = solve_spectral_gradient(
                initial_guess,
                variant=variant,
                objective=objective,
                initial_step=0.25,
                max_iter=300,
                tol=1e-8,
            )

            print(
                f"{variant:<10}"
                f"{result.iterations:>12}"
                f"{str(result.converged):>13}"
                f"{result.cost_history[-1]:>18.6e}"
                f"{result.gradient_norms[-1]:>20.6e}"
            )


if __name__ == "__main__":
    main()
