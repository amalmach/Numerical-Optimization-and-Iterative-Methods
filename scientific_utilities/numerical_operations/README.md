# Numerical Operations

Educational implementations of fundamental numerical operations for scientific computing.

This module modernizes elementary vector and matrix operations from the repository's original scripts while preserving their educational purpose.

The implementations use NumPy for array representation and validation. Several operations use explicit loops to illustrate the underlying mathematics.

## 1. Available Operations

| Function | Description |
|---|---|
| `closest(lst, val)` | Find the nearest value in a sequence |
| `norm_two(v)` | Compute the Euclidean norm |
| `AxB(X, Y)` | Compute a matrix–matrix product |
| `matvec(A, v)` | Compute a matrix–vector product |
| `scalar(u, v)` | Compute a dot product |
| `vecvec(u, v)` | Compute an outer product |
| `matT(A)` | Compute a matrix transpose |
| `vec_asarray(v)` | Convert nested vectors into a 2D array |

All operations accept finite real-valued inputs.

## 2. Mathematical Formulation

### 2.1 Nearest Value

Given a finite sequence

\[
S=\{s_1,s_2,\ldots,s_n\}
\]

and a target value \(t\), find

\[
s^* \in \operatorname*{arg\,min}_{s_i\in S}|s_i-t|.
\]

When several elements are equally close to the target, `closest()` returns the last such element in the input sequence.

### 2.2 Euclidean Norm

For a vector \(v\in\mathbb{R}^n\), the Euclidean norm is

\[
\|v\|_2=\sqrt{\sum_{i=1}^{n}v_i^2}.
\]

The implementation uses `numpy.linalg.norm` for reliable numerical evaluation.

### 2.3 Matrix Multiplication

Let

\[
A\in\mathbb{R}^{m\times n},
\qquad
B\in\mathbb{R}^{n\times p}.
\]

Their product \(C=AB\) belongs to \(\mathbb{R}^{m\times p}\), with entries

\[
C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}.
\]

The educational implementation uses three nested loops.

### 2.4 Matrix–Vector Multiplication

For

\[
A\in\mathbb{R}^{m\times n},
\qquad
v\in\mathbb{R}^{n},
\]

the matrix–vector product \(w=Av\) is defined by

\[
w_i=\sum_{j=1}^{n}A_{ij}v_j.
\]

The implementation supports rectangular matrices.

### 2.5 Dot Product

For vectors \(u,v\in\mathbb{R}^{n}\),

\[
u^\top v=\sum_{i=1}^{n}u_iv_i.
\]

Both vectors must have the same number of components.

### 2.6 Outer Product

For

\[
u\in\mathbb{R}^{m},
\qquad
v\in\mathbb{R}^{n},
\]

their outer product is

\[
C=uv^\top\in\mathbb{R}^{m\times n},
\]

where

\[
C_{ij}=u_iv_j.
\]

Unlike the dot product, the vectors do not need to have the same length.

### 2.7 Matrix Transpose

For \(A\in\mathbb{R}^{m\times n}\), the transpose satisfies

\[
(A^\top)_{ij}=A_{ji}.
\]

The resulting matrix has shape \(n\times m\).

### 2.8 Nested-Vector Conversion

The function `vec_asarray()` converts a rectangular collection of numerical vectors into a two-dimensional NumPy array.

For example,

\[
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
\]

is represented as a NumPy array of shape `(2, 2)`.

Each input vector becomes one row.

## 3. Usage

Run these commands from the repository root.

```python
import numpy as np

from scientific_utilities.numerical_operations import (
    closest,
    norm_two,
    AxB,
    matvec,
    scalar,
    vecvec,
    matT,
    vec_asarray,
)

A = np.array([[1., 2.], [3., 4.]])
B = np.array([[5., 6.], [7., 8.]])

u = np.array([1., 2.])
v = np.array([3., 4.])

print(closest([0.1, 0.4, 0.8], 0.7))
print(norm_two(u))
print(AxB(A, B))
print(matvec(A, u))
print(scalar(u, v))
print(vecvec(u, v))
print(matT(A))
print(vec_asarray([[1, 2], [3, 4]]))
```

## 4. Reproducible Example

From the repository root:

```bash
python -m scientific_utilities.numerical_operations.example
```

The example demonstrates all eight operations and verifies their results against NumPy.

## 5. Automated Tests

Run:

```bash
python -m pytest -v scientific_utilities/numerical_operations/test_operations.py
```

The test suite includes 16 tests covering:

- Numerical correctness.
- Comparisons against NumPy.
- Rectangular matrix operations.
- Dimension compatibility.
- Invalid and nonfinite inputs.
- Nearest-value tie-breaking behavior.

## 6. Input Validation

The operations validate the shape and numerical finiteness of their inputs.

In particular:

- Vectors must be nonempty and one-dimensional.
- Matrices must be nonempty and two-dimensional.
- Matrix and vector dimensions must be compatible.
- NaN and infinite entries are rejected.
- Irregular nested vectors are rejected.

Invalid inputs raise `ValueError` in the supported validation cases.

## 7. Computational Complexity

| Operation | Time complexity |
|---|---|
| `closest()` | \(O(n)\) |
| `norm_two()` | \(O(n)\) |
| `AxB()` | \(O(mnp)\) |
| `matvec()` | \(O(mn)\) |
| `scalar()` | \(O(n)\) |
| `vecvec()` | \(O(mn)\) |
| `matT()` | \(O(mn)\) |
| `vec_asarray()` | \(O(mn)\) |

These expressions describe the principal computational work for valid inputs.

## 8. Scope and Limitations

This module is intended primarily for education, reproducibility, and numerical experimentation.

Several functions use explicit Python loops to make their mathematical definitions transparent. They are not intended to outperform NumPy's optimized implementations.

The module currently supports finite real-valued arrays. Complex arithmetic, sparse matrices, and batched matrix operations are outside its scope.

LU factorization is intentionally excluded because it has a dedicated implementation in:

[`direct_methods/lu_factorization/`](../../direct_methods/lu_factorization/)

## 9. Legacy Sources

This module is derived from concepts demonstrated in the original repository files:

- `closest.py`
- `numpy_scipy_func.py`

The original scripts are preserved for historical reference.

The modernized functions provide a reusable API, input validation, automated tests, and reproducible examples.
