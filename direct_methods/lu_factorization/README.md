<div align="center">

# LU Factorization with Partial Pivoting

A Python implementation of LU factorization with partial pivoting for solving dense linear systems.

</div>

---

## 1. Mathematical Formulation

Consider a linear system

<p align="center">$\displaystyle
Ax=b,
$</p>

where $A\in\mathbb{R}^{n\times n}$, $x\in\mathbb{R}^n$, and $b\in\mathbb{R}^n$.

LU factorization decomposes the coefficient matrix into triangular factors.

With partial pivoting, the decomposition takes the form

<p align="center">$\displaystyle
\boxed{PA=LU},
$</p>

where:

- $P$ is a permutation matrix.
- $L$ is a unit lower triangular matrix.
- $U$ is an upper triangular matrix.

Partial pivoting improves the numerical stability of Gaussian elimination by selecting the largest available pivot in absolute value within the current column.

## 2. LU Factorization Algorithm

At elimination step $k$, select the pivot row

<p align="center">$\displaystyle
p=\underset{k\leq i<n}{\operatorname{argmax}}\,|U_{ik}|.
$</p>

Exchange rows $k$ and $p$ in $U$ and $P$.

Previously computed multipliers in $L$ must also be exchanged.

For each row $i>k$, compute the multiplier

<p align="center">$\displaystyle
L_{ik}=\frac{U_{ik}}{U_{kk}}.
$</p>

Update the remaining entries of $U$:

<p align="center">$\displaystyle
U_{ij}\leftarrow U_{ij}-L_{ik}U_{kj},
\qquad j>k.
$</p>

After all elimination steps, the resulting matrices satisfy

<p align="center">$\displaystyle
PA=LU
$</p>

up to floating-point rounding errors.

## 3. Solving Linear Systems

Once the LU factors have been computed, the original system

<p align="center">$\displaystyle
Ax=b
$</p>

can be rewritten as

<p align="center">$\displaystyle
LUx=Pb.
$</p>

The solution is obtained through two triangular solves.

### 3.1 Forward Substitution

First solve

<p align="center">$\displaystyle
Ly=Pb.
$</p>

For $i=0,\ldots,n-1$,

<p align="center">$\displaystyle
y_i=
\frac{(Pb)_i-\displaystyle\sum_{j=0}^{i-1}L_{ij}y_j}
{L_{ii}}.
$</p>

For the unit lower triangular factor produced by LU factorization, $L_{ii}=1$.

### 3.2 Backward Substitution

Next solve

<p align="center">$\displaystyle
Ux=y.
$</p>

For $i=n-1,\ldots,0$,

<p align="center">$\displaystyle
x_i=
\frac{y_i-\displaystyle\sum_{j=i+1}^{n-1}U_{ij}x_j}
{U_{ii}}.
$</p>

This procedure avoids explicitly constructing $A^{-1}$.

## 4. Computational Complexity

For a dense $n\times n$ matrix:

| Operation | Computational complexity |
|---|---|
| LU factorization | $O(n^3)$ |
| Forward substitution | $O(n^2)$ |
| Backward substitution | $O(n^2)$ |
| Solving after factorization | $O(n^2)$ |
| Storing the LU factors | $O(n^2)$ |

The leading arithmetic cost of conventional dense LU factorization is approximately

<p align="center">$\displaystyle
\frac{2}{3}n^3
$</p>

floating-point operations.

If several systems share the same coefficient matrix but have different right-hand sides, reusing the LU factors can significantly reduce the computational cost.

The current `lu_solve()` interface performs a new factorization for every call.

## 5. Python Usage

Run the following example from the repository root:

```python
import numpy as np

from direct_methods.lu_factorization import (
    lu_factorization,
    lu_solve,
)

A = np.array([
    [0.0, 2.0, 1.0],
    [1.0, 1.0, 0.0],
    [2.0, 3.0, 4.0],
])

b = np.array([3.0, 2.0, 9.0])

P, L, U = lu_factorization(A)

print("Factorization residual:")
print(np.linalg.norm(P @ A - L @ U))

x = lu_solve(A, b)

print("Solution:", x)
print("Linear system residual:", np.linalg.norm(A @ x - b))
```

Expected solution:

<p align="center">$\displaystyle
\boxed{x=(1,1,1)^T}.
$</p>

## 6. Numerical Validation

The factorization can be checked using

<p align="center">$\displaystyle
r_{\mathrm{LU}}=\|PA-LU\|_F.
$</p>

The linear system solution can be checked using

<p align="center">$\displaystyle
r=\|Ax-b\|_2.
$</p>

A useful normalized residual is

<p align="center">$\displaystyle
r_{\mathrm{rel}}=
\frac{\|Ax-b\|_2}
{\|A\|_2\|x\|_2+\|b\|_2}.
$</p>

Small residuals indicate that the computed factors or solution satisfy the corresponding equations accurately.

However, a small residual does not necessarily imply a small forward error when the coefficient matrix is ill-conditioned.

## 7. Testing

Run the automated tests from the repository root:

```bash
python -m pytest -v direct_methods/lu_factorization/test_lu.py
```

The current test suite contains 14 tests covering factorization, pivoting, triangular substitution, linear system solving, singular matrices, and comparisons with NumPy and SciPy.

## 8. Reproducible Example

Run:

```bash
python -m direct_methods.lu_factorization.example
```

The example displays the permutation and triangular matrices, solves a linear system, and compares the result with NumPy and SciPy.

## 9. Limitations

The current implementation:

- Supports dense, real-valued matrices.
- Solves systems with a single right-hand-side vector.
- Uses partial rather than complete pivoting.
- Does not estimate the condition number.
- Uses a heuristic threshold to detect numerically negligible pivots.
- Is intended for numerical education and experimentation rather than replacing optimized LAPACK implementations.

For large-scale sparse systems, sparse direct solvers or iterative Krylov methods may be more appropriate.

## 10. Relationship to Other Methods

LU factorization is a direct linear algebra method.

It complements the iterative algorithms available in this repository, particularly GMRES and other Krylov subspace methods.

LU factorization can also be used as a building block for preconditioning and Newton-type numerical algorithms.
