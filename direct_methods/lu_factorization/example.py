"""
Example: LU factorization with partial pivoting.

Demonstrates:
- PA = LU factorization
- Solving Ax = b using LU
- Comparison with NumPy and SciPy

Run from the repository root:

    python -m direct_methods.lu_factorization.example
"""

import numpy as np
from scipy.linalg import solve

from .lu import lu_factorization, lu_solve


def main():
    """Demonstrate LU factorization and linear system solving."""

    A = np.array([
        [0.0, 2.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0],
    ])

    b = np.array([3.0, 2.0, 9.0])

    print("=" * 60)
    print("LU FACTORIZATION WITH PARTIAL PIVOTING")
    print("=" * 60)

    P, L, U = lu_factorization(A)

    print("\nOriginal matrix A:")
    print(A)

    print("\nPermutation matrix P:")
    print(P)

    print("\nLower triangular matrix L:")
    print(L)

    print("\nUpper triangular matrix U:")
    print(U)

    factorization_residual = np.linalg.norm(P @ A - L @ U)

    print("\n" + "-" * 60)
    print("FACTORIZATION VALIDATION")
    print("-" * 60)

    print(f"||PA - LU||₂ = {factorization_residual:.6e}")

    x_lu = lu_solve(A, b)
    x_numpy = np.linalg.solve(A, b)
    x_scipy = solve(A, b)

    print("\n" + "-" * 60)
    print("LINEAR SYSTEM SOLUTION")
    print("-" * 60)

    print("\nRight-hand side b:")
    print(b)

    print("\nLU solution:")
    print(x_lu)

    print("\nNumPy solution:")
    print(x_numpy)

    print("\nSciPy solution:")
    print(x_scipy)

    residual = np.linalg.norm(A @ x_lu - b)

    relative_residual = residual / (
        np.linalg.norm(A) * np.linalg.norm(x_lu)
        + np.linalg.norm(b)
    )

    print("\n" + "-" * 60)
    print("NUMERICAL ACCURACY")
    print("-" * 60)

    print(f"Absolute residual: {residual:.6e}")
    print(f"Relative residual: {relative_residual:.6e}")

    print("\nValidation:")
    print("PA = LU:", np.allclose(P @ A, L @ U))
    print("LU = NumPy:", np.allclose(x_lu, x_numpy))
    print("LU = SciPy:", np.allclose(x_lu, x_scipy))


if __name__ == "__main__":
    main()
