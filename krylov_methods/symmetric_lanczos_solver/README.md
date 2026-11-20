<div align="center">

# Symmetric Lanczos Algorithm

Numerical Lanczos tridiagonalization for symmetric and complex Hermitian
matrices, with an optional Galerkin approximation to linear systems.

</div>

### Main idea

The **Symmetric Lanczos algorithm** is a Krylov subspace method
that constructs an orthonormal basis associated with a real
symmetric or complex Hermitian matrix.

Its principal objective is to project a large matrix onto a
smaller symmetric tridiagonal matrix.

This projection enables efficient approximation of eigenvalues
and provides the mathematical foundation for iterative solvers
such as Conjugate Gradient (CG) and MINRES.

Lanczos is primarily a **basis construction and tridiagonalization
algorithm**, rather than a general-purpose linear-system solver.

However, the generated Krylov basis can also be used to construct
a Galerkin approximation to the solution of a linear system.

### Files

| File | Purpose |
|:---|:---|
| [Lanczos.py](Lanczos.py) | Symmetric/Hermitian Lanczos tridiagonalization |
| [galerkin.py](galerkin.py) | Optional projected linear-system approximation |
| [test_lanczos.py](test_lanczos.py) | Regression tests for orthogonality and Ritz values |
| [example.py](example.py) | Reproducible experiment |

### Mathematical formulation

Let $A$ be a real symmetric or complex Hermitian matrix,
and let $q$ be a nonzero starting vector.

For eigenvalue approximation, $q$ may be chosen arbitrarily.
For a Galerkin approximation to $Ax=b$, a natural choice is

<p align="center">
$\displaystyle q=r_0=b-Ax_0.$
</p>

Initialize

<p align="center">
$\displaystyle
\gamma=\|q\|_2,
\qquad
v_1=\frac{q}{\gamma},
\qquad
v_0=0,
\qquad
\beta_0=0.$
</p>

The Lanczos recurrence is (For $k=1,2,\ldots,m$):

<p align="center">
$w_k=Av_k-\beta_{k-1}v_{k-1},$
</p>

<p align="center">
$\displaystyle \alpha_k=v_k^Hw_k.$
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
$\displaystyle T_m=V_m^HAV_m.$
</p>

In exact arithmetic, the Lanczos vectors are orthonormal:

<p align="center">
$\displaystyle V_m^HV_m=I_m.$
</p>

> **Note — Complex Hermitian systems:** For complex-valued matrices,
> $V_m^H$ denotes the conjugate transpose. For real-valued matrices,
> the conjugate transpose reduces to the ordinary transpose.

#### Ritz values and Ritz vectors

The eigenvalues of the projected tridiagonal matrix $T_m$
are called **Ritz values**.

Let

<p align="center">
$\displaystyle
T_my_i=\theta_i y_i.
$
</p>

Then $\theta_i$ is a Ritz value, and

<p align="center">
$\displaystyle
u_i=V_my_i
$
</p>

is the corresponding Ritz vector.

The Ritz pair $(\theta_i,u_i)$ approximates an eigenpair
of the original matrix $A$.

The quality of this approximation can be measured using
the eigenvalue residual

<p align="center">
$\displaystyle
\|Au_i-\theta_i u_i\|_2.
$
</p>

Using the Lanczos relation, we obtain

<p align="center">
$\displaystyle
\|Au_i-\theta_i u_i\|_2
=
|\beta_m|\,|e_m^Ty_i|.
$
</p>

This relation holds for the classical Lanczos recurrence
in exact arithmetic. In finite precision, the residual
may differ slightly because of rounding and
reorthogonalization corrections.

A small Ritz-pair residual indicates that the corresponding
Ritz pair is a good approximate eigenpair.

#### Galerkin approximation

For a nonsingular projected matrix $T_m$, an
approximate solution of $Ax=b$ can be constructed as

<p align="center">
$\displaystyle
x_m=x_0+V_my_m.
$
</p>

where

<p align="center">
$\displaystyle
T_my_m=\gamma e_1,
\qquad
\gamma=\|b-Ax_0\|_2.
$
</p>

The Galerkin approximation imposes the orthogonality condition

<p align="center">
$\displaystyle
V_m^H(b-Ax_m)=0.
$
</p>

This means that the residual is orthogonal to the
Krylov subspace used to construct the approximation.

For symmetric positive-definite systems, the projected matrix
$T_m$ is also positive definite.

For indefinite systems, $T_m$ may be singular or ill-conditioned,
even when the original matrix $A$ is nonsingular.

In such cases, the Galerkin approximation may fail or become
numerically inaccurate.

For robust symmetric indefinite linear-system solving,
methods such as MINRES are generally preferable.

For eigenvalue exploration, Lanczos tridiagonalization
can still be used without solving the projected
Galerkin system.

### Algorithm

**Input:** Hermitian matrix or linear operator $A$,
nonzero starting vector $q$, maximum Krylov dimension $m$.

**Output:** Orthonormal Lanczos basis $V_m$,
tridiagonal projection $T_m$, and Ritz values.

1. **Initialization**

   Compute

   <p align="center">
   $\displaystyle
   v_1=\frac{q}{\|q\|_2},
   \qquad
   v_0=0,
   \qquad
   \beta_0=0.
   $
   </p>

2. **Lanczos iterations**

   For $k=1,\ldots,m$:

   - Compute the matrix-vector product

     <p align="center">
     $\displaystyle
     w_k=Av_k-\beta_{k-1}v_{k-1}.
     $
     </p>

   - Compute the diagonal coefficient

     <p align="center">
     $\displaystyle
     \alpha_k=v_k^Hw_k.
     $
     </p>

   - Remove the component along the current vector

     <p align="center">
     $\displaystyle
     w_k\leftarrow w_k-\alpha_kv_k.
     $
     </p>

   - Optionally apply full reorthogonalization.

   - Compute the next coefficient

     <p align="center">
     $\displaystyle
     \beta_k=\|w_k\|_2.
     $
     </p>

   - If $\beta_k$ is numerically zero, terminate
     because the Krylov subspace cannot expand further.

   - Otherwise, construct the next Lanczos vector

     <p align="center">
     $\displaystyle
     v_{k+1}=\frac{w_k}{\beta_k}.
     $
     </p>

3. **Construct the projection**

   Assemble the symmetric tridiagonal matrix $T_m$
   from the coefficients $\alpha_k$ and $\beta_k$.

4. **Compute Ritz values**

   Compute the eigenvalues of $T_m$.

5. **Optional Galerkin approximation**

   If the starting vector is $q=b-Ax_0$ and the
   projected matrix is nonsingular, construct
   an approximate solution of $Ax=b$.


### Numerical properties

- Requires a real symmetric or complex Hermitian matrix.
- Constructs an orthonormal Krylov basis in exact arithmetic.
- Uses a three-term recurrence.
- Produces a real symmetric tridiagonal projection.
- Approximates eigenvalues through Ritz values.
- May lose orthogonality in finite-precision arithmetic.
- Supports optional full reorthogonalization.
- Detects numerical breakdown when the next Lanczos vector
  becomes too small.
- Can provide a Galerkin approximation to linear systems.
- A singular or ill-conditioned projected matrix may prevent
  an accurate Galerkin solution.

**Reorthogonalization**

In finite-precision arithmetic, Lanczos vectors may lose
orthogonality as the iteration progresses.

The implementation provides the option

`reorthogonalize=True`

to apply two passes of full reorthogonalization.

This can improve basis orthogonality but increases
computational cost and memory requirements.

The classical tridiagonal Lanczos relation is exact in
exact arithmetic without additional correction terms.
With full reorthogonalization, small correction terms
may affect this relation in finite precision.

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
