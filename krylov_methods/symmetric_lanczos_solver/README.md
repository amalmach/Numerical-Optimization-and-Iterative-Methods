<div align="center">

# Symmetric Lanczos Algorithm

Numerical Lanczos tridiagonalization for symmetric and complex Hermitian
matrices, with an optional Galerkin approximation to linear systems.

</div>

### Main idea

Lanczos is a Krylov subspace algorithm, but primarily a basis construction
and tridiagonalization procedure, *not* itself a general-purpose linear solver.
Its symmetric tridiagonal projection is useful for eigenvalue estimation and
underlies linear solvers such as CG and MINRES.

The Symmetric Lanczos algorithm constructs an
orthonormal basis of a Krylov subspace associated
with a symmetric matrix.

Its principal objective is to transform a large
symmetric matrix into a smaller symmetric
tridiagonal matrix.

This transformation is useful for eigenvalue
approximation and Krylov projection methods.

### Files

| File | Purpose |
|:---|:---|
| [Lanczos.py](Lanczos.py) | Symmetric/Hermitian Lanczos tridiagonalization |
| [galerkin.py](galerkin.py) | Optional projected linear-system approximation |
| [test_lanczos.py](test_lanczos.py) | Regression tests for orthogonality and Ritz values |
| [example.py](example.py) | Reproducible experiment |

### Mathematical formulation

For a symmetric matrix $A$ and nonzero starting vector $r_0=b-Ax_0$, let

<p align="center">
\beta_0=\|r_0\|_2,\qquad v_1=r_0/\beta_0,\qquad v_0=0.$
</p>

The Lanczos recurrence is (For $k=1,2,\ldots,m$):

<p align="center">
$w_k=Av_k-\beta_{k-1}v_{k-1},$
</p>

<p align="center">
$\alpha_k=v_k^Tw_k,$
</p>

<p align="center">
$w_k\leftarrow w_k-\alpha_kv_k,$
</p>

<p align="center">
$\beta_k=\|w_k\|_2,$
</p>

<p align="center">
$v_{k+1}=\frac{w_k}{\beta_k}.$
</p>

The generated vectors form the matrix $V_m=[v_1,\ldots,v_m]$ that satisfies

<p align="center">$\displaystyle AV_m=V_mT_m+\beta_mv_{m+1}e_m^T.$</p>

The tridiagonal matrix is

```math
T_m =
\begin{pmatrix}
\alpha_1 & \beta_1 & 0 & \cdots & 0 \\
\beta_1 & \alpha_2 & \beta_2 & \ddots & \vdots \\
0 & \beta_2 & \alpha_3 & \ddots & 0 \\
\vdots & \ddots & \ddots & \ddots & \beta_{m-1} \\
0 & \cdots & 0 & \beta_{m-1} & \alpha_m
\end{pmatrix}
```

It satisfies

<p align="center">
$T_m=V_m^TAV_m.$
</p>

In exact arithmetic, the Lanczos vectors are orthonormal:

<p align="center">
$V_m^TV_m=I_m.$
</p>

#### Ritz values

The eigenvalues of $T_m$ are called Ritz values.

They approximate eigenvalues of the original
matrix $A$.

This allows spectral information to be extracted
from a smaller matrix.

#### Galerkin approximation

For a nonsingular projected matrix $T_m$, an
approximate solution of $Ax=b$ can be constructed as

<p align="center">
$x_m=x_0+V_my_m,$
</p>

where

<p align="center">
$T_my_m=\beta_0e_1.$
</p>

For indefinite $A$, the projected $T_m$ can be singular even when $A$ is
nonsingular. This module reports that case; use MINRES for robust symmetric
indefinite linear solves. For eigenvalue exploration, use tridiagonalization
without requiring the Galerkin linear system to be solvable.


### Algorithm

**Input:** Symmetric matrix $A$, vector $b$,
initial guess $x_0$, Krylov dimension $m$.

1. Compute the initial residual $r_0$.
2. Normalize it to obtain the first Lanczos vector $v_0$.
3. For $k=1,\ldots,m$:
   - Apply the matrix $A$ to the current Lanczos vector $v_{k-1}$.
   - Subtract the contribution of the previous vector to obtain $ w_k$.
   - Compute the diagonal coefficient $\alpha_k$.
   - Remove the component along the current vector in $w_k$.
   - Compute the off-diagonal coefficient $\beta_k$.
   - Normalize to obtain the next Lanczos vector.
4. Construct the tridiagonal matrix $T_m$.
5. Compute Ritz eigenvalues or a Galerkin approximation.
6. Optionally solve the Galerkin system for an approximate solution to $Ax=b$.


### Numerical properties

- Requires a symmetric matrix.
- Uses a three-term recurrence.
- Produces a symmetric tridiagonal projection.
- Useful for approximating eigenvalues.
- May lose orthogonality in finite precision.
- A singular projected matrix can prevent the
  Galerkin linear-system approximation.

### Python usage

From the repository root:

```python
import numpy as np
from krylov_methods.symmetric_lanczos_solver import (
    lanczos_tridiagonalize, lanczos_galerkin,
)

A = np.array([[4., 1.], [1., 3.]])
b = np.array([1., 2.])

projection = lanczos_tridiagonalize(A, b, m=2)
print(projection.tridiagonal, projection.ritz_values)

linear_result = lanczos_galerkin(A, b, m=2)
print(linear_result.converged, linear_result.solution)
```

The module supports dense/sparse matrices, SciPy `LinearOperator`, and
callable matrix-vector products, for real symmetric or complex Hermitian
operators. `reorthogonalize=True` improves orthogonality in finite precision,
at extra memory and computation cost. A generic operator is assumed Hermitian;
automatic verification is not possible without an explicit matrix.

### Numerical tests

Install: `python3 -m pip install numpy scipy`.

From the repository root:

```bash
python3 -m unittest krylov_methods.symmetric_lanczos_solver.test_lanczos -v
python3 -m krylov_methods.symmetric_lanczos_solver.example
```

### References

1. C. Lanczos (1950), *An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators*, Journal of Research of the NBS, 45(4), 255–282.
2. Y. Saad (2011), *Numerical Methods for Large Eigenvalue Problems*, 2nd ed., SIAM.
3. G. H. Golub and C. F. Van Loan (2013), *Matrix Computations*, 4th ed., Johns Hopkins University Press.
4. A. Greenbaum (1997), *Iterative Methods for Solving Linear Systems*, SIAM.
