"""
Reproducible examples of elementary numerical operations.

Run from the repository root:

    python -m scientific_utilities.numerical_operations.example
"""

import numpy as np

from .operations import (
    closest,
    norm_two,
    AxB,
    matvec,
    scalar,
    vecvec,
    matT,
    vec_asarray,
)


def main():
    """Demonstrate eight numerical operations."""

    print("=" * 60)
    print("SCIENTIFIC UTILITIES — NUMERICAL OPERATIONS")
    print("=" * 60)

    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    B = np.array([
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    u = np.array([1.0, 2.0])
    v = np.array([3.0, 4.0])

    print("\n1. Nearest value")
    nearest = closest([0.1, 0.4, 0.8], 0.7)
    print("Closest value to 0.7:", nearest)

    print("\n2. Euclidean norm")
    print("Implementation:", norm_two(u))
    print("NumPy:         ", np.linalg.norm(u))

    print("\n3. Matrix multiplication")
    print("Implementation:\n", AxB(A, B))
    print("NumPy:\n", A @ B)

    print("\n4. Matrix-vector multiplication")
    print("Implementation:", matvec(A, u))
    print("NumPy:         ", A @ u)

    print("\n5. Scalar product")
    print("Implementation:", scalar(u, v))
    print("NumPy:         ", np.dot(u, v))

    print("\n6. Outer product")
    print("Implementation:\n", vecvec(u, v))
    print("NumPy:\n", np.outer(u, v))

    print("\n7. Matrix transpose")
    print("Implementation:\n", matT(A))
    print("NumPy:\n", A.T)

    print("\n8. Nested-vector conversion")
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    print(vec_asarray(vectors))

    # Numerical verification
    assert nearest == 0.8
    assert np.isclose(norm_two(u), np.linalg.norm(u))
    assert np.allclose(AxB(A, B), A @ B)
    assert np.allclose(matvec(A, u), A @ u)
    assert np.isclose(scalar(u, v), np.dot(u, v))
    assert np.allclose(vecvec(u, v), np.outer(u, v))
    assert np.allclose(matT(A), A.T)
    assert np.allclose(vec_asarray(vectors), np.asarray(vectors))

    print("\n" + "=" * 60)
    print("All eight numerical operations validated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
