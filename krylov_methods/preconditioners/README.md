<div align="center">

# Preconditioning Methods

Reusable inverse-action preconditioners for Krylov subspace solvers.

This module centralizes factorization and preconditioner construction.
The individual solvers retain their own left-, right-, or PCG recurrences.

</div>

---

## Main idea

Preconditioning replaces an expensive or poorly conditioned system with one that is often easier to solve iteratively. A preconditioner $M$ should approximate $A$ while allowing inexpensive application of its inverse:

<p align="center">
$z=M^{-1}r.$
</p>

**Never form an explicit inverse merely to apply a preconditioner.**
Instead, factor $M$ once (when necessary) and solve for each new residual.

| Method | Identifier | Suitable setting | Caveat |
|:---|:---:|:---|:---|
| Identity | `None` | All | No acceleration |
| Jacobi | `"jacobi"` | Real/complex; CG if SPD | Weak for strongly coupled matrices |
| Exact LU | `"lu"` | GMRES/BiCGSTAB | A direct solve; expensive benchmark |
| ILU | `"ilu"` | GMRES/BiCGSTAB | May break down; not generally SPD |
| Cholesky | `"cholesky"` | HPD systems; CG | Dense factorization in this implementation |
| Incomplete Cholesky IC(0) | `"ic0"` | Real SPD systems | Educational dense-intermediate implementation |
| SSOR | `"ssor"` | Real/complex; CG if SPD | Depends on ordering and relaxation |
| Custom | `"custom"` | Depends on supplied inverse action | User must verify its properties |

---

## Preconditioning Strategies

Preconditioning transforms a linear system into an equivalent
system that is potentially better conditioned and more suitable
for iterative solution.

Depending on how the preconditioner is introduced, three
classical strategies are commonly considered.

<details>
<summary><strong>1. Left Preconditioning</strong></summary>

<br>

The original system

<p align="center">
$\displaystyle Ax=b$
</p>

is transformed into

<p align="center">
$\displaystyle M^{-1}Ax=M^{-1}b.$
</p>

The corresponding preconditioned residual is

<p align="center">
$\displaystyle \widetilde{r}_k=M^{-1}(b-Ax_k).$
</p>

Left-preconditioned GMRES minimizes the norm of the
preconditioned residual over the corresponding Krylov
approximation space:

<p align="center">
$\displaystyle \|M^{-1}(b-Ax_k)\|_2.$
</p>

**Numerical properties**

- Directly modifies the residual used by the iterative method.
- Can improve convergence when the preconditioner is effective.
- The minimized residual differs from the original residual.
- Monitoring the true residual remains important.

**Implementation:** The current GMRES solver supports
left preconditioning.

</details>

<details>
<summary><strong>2. Right Preconditioning</strong></summary>

<br>

Introduce the change of variables

<p align="center">
$\displaystyle x=M^{-1}y.$
</p>

The original system becomes

<p align="center">
$\displaystyle AM^{-1}y=b.$
</p>

After computing the transformed variable $y$, recover
the original solution using

<p align="center">
$\displaystyle x=M^{-1}y.$
</p>

The residual satisfies

<p align="center">
$\displaystyle r=b-AM^{-1}y=b-Ax.$
</p>

Consequently, right-preconditioned GMRES can minimize
the original residual norm directly:

<p align="center">
$\displaystyle \|b-Ax_k\|_2.$
</p>

**Numerical properties**

- Preserves the original residual in the transformed system.
- Allows direct minimization of the true residual in GMRES.
- Requires applying the inverse preconditioner during iterations.
- Is useful for nonsymmetric systems.

**Implementation:** The current BiCGSTAB solver uses
right preconditioning.

</details>

<details>
<summary><strong>3. Split Preconditioning</strong></summary>

<br>

Suppose the preconditioner admits a factorization

<p align="center">
$\displaystyle M=M_LM_R.$
</p>

Introduce the transformed variable

<p align="center">
$\displaystyle x=M_R^{-1}y.$
</p>

The resulting preconditioned system is

<p align="center">
$\displaystyle M_L^{-1}AM_R^{-1}y=M_L^{-1}b.$
</p>

The original solution is recovered through

<p align="center">
$\displaystyle x=M_R^{-1}y.$
</p>

Split preconditioning applies transformations on both
sides of the system matrix.

**Numerical properties**

- Can preserve useful structural properties of the system.
- Is particularly relevant for symmetric positive-definite systems.
- Can improve the spectral properties of the transformed operator.
- Requires compatible left and right factors.

**Example**

For a symmetric positive-definite preconditioner

<p align="center">
$\displaystyle M=CC^H,$
</p>

a symmetric transformation can be constructed using

<p align="center">
$\displaystyle C^{-1}AC^{-H}y=C^{-1}b,$
</p>

with

<p align="center">
$\displaystyle x=C^{-H}y.$
</p>

This provides a useful interpretation of preconditioned
Conjugate Gradient methods.

</details>

---

## Available Preconditioners

<details>
<summary><strong>1. Jacobi (diagonal)</strong></summary>

Take the diagonal part $D=\mathrm{diag}(A)$:

<p align="center">$M=D,\qquad z_i=r_i/D_{ii}.$</p>

Cheap storage and application, but ignores off-diagonal coupling.
The diagonal must be nonzero; for PCG it must be real and positive.

</details>

<details>
<summary><strong>2. Exact LU</strong></summary>

For a nonsingular matrix $A$, compute an LU factorization
with row pivoting:

<p align="center">
$PA=LU,$
</p>

where $P$ is a permutation matrix, $L$ is lower triangular,
and $U$ is upper triangular.

The inverse preconditioner action is obtained by solving

<p align="center">
$Ly=Pr,\qquad Uz=y.$
</p>

The factors are computed once and reused during the
iterative process.

This is generally a direct-solve benchmark rather than
an inexpensive preconditioner.

Generic LU preconditioning is not automatically suitable
for Conjugate Gradient because the required Hermitian
positive-definite structure may not be preserved.

</details>

<details>
<summary><strong>3. Incomplete LU (ILU)</strong></summary>

Approximate $A\approx\widetilde L\widetilde U$ while limiting fill-in.

<p align="center">$\widetilde L y=r,\qquad\widetilde U z=y.$</p>

The `drop_tol` and `fill_factor` parameters control approximation quality,
storage, and construction cost. A generic ILU action need not be SPD.

</details>

<details>
<summary><strong>4. Cholesky</strong></summary>

For Hermitian positive-definite $A$, factor:

<p align="center">$A=LL^H,\qquad Ly=r,\qquad L^Hz=y.$</p>

This implementation factors the *entire* matrix, including sparse matrices
after dense conversion. It is useful as a correctness reference but not a
scalable sparse preconditioner.

</details>

<details>
<summary><strong>5. Incomplete Cholesky IC(0)</strong></summary>

For real SPD matrices, approximate

<p align="center">$A\approx\widetilde L\widetilde L^T.$</p>

IC(0) retains the lower-triangular sparsity pattern during factorization.
The action uses forward and backward substitution. It can fail on an
insufficiently positive pivot; the implementation is educational and uses
dense intermediates, so it is not suited to large production problems.

</details>

<details>
<summary><strong>6. Symmetric successive over-relaxation (SSOR)</strong></summary>

Write $A=D+L+U$ and select $0<\omega<2$. The SSOR preconditioner is

<p align="center">
$\displaystyle M_{\rm SSOR}=\frac{\omega}{2-\omega}
(\frac{D}{\omega}+L)D^{-1}(\frac{D}{\omega}+U).$
</p>

Its inverse action uses two triangular solves and diagonal scaling.
For Hermitian $A$, $U=L^H$; positive definiteness requires the appropriate
assumptions (including a positive diagonal).

</details>

<details>
<summary><strong>7. Custom inverse action</strong></summary>

Provide a callable or `LinearOperator` that already implements

<p align="center">$r\mapsto M^{-1}r.$</p>

Alternatively, provide an explicit preconditioner matrix $M$, which is
factored once. For PCG, the inverse action must be **fixed Hermitian
positive-definite**; this cannot be established automatically for arbitrary
callables.

</details>

## Advanced Preconditioning Techniques

Beyond the preconditioners implemented in this repository,
several advanced techniques are widely used for solving
large-scale sparse linear systems.

| Method | Description |
|:---|:---|
| **SPAI** | Sparse Approximate Inverse |
| **FSAI** | Factorized Sparse Approximate Inverse |
| **GMG** | Geometric Multigrid |
| **AMG** | Algebraic Multigrid |
| **Domain Decomposition** | Preconditioning based on subdomain problems |

These methods are particularly relevant for large sparse
systems arising from the discretization of partial
differential equations.

They are mentioned for completeness and are not currently
implemented in this repository.

## Solver compatibility

The *construction* of the inverse action is shared, but its position in the
algorithm is solver-specific:

- **CG (PCG):** apply a fixed SPD action $z_k=M^{-1}r_k$ in the PCG recurrence.
- **GMRES:** the existing implementation uses **left preconditioning**.
- **BiCGSTAB:** the existing implementation uses **right preconditioning**.

The shared module does **not** rewrite these solver recurrences. Set
`require_spd=True` for CG; this rejects generic LU and ILU, but still requires
the user to supply an HPD system and valid custom actions.

## Python interface

From the repository root:

```python
import numpy as np
from krylov_methods.preconditioners import make_preconditioner

A = np.array([[4., 1.], [1., 3.]])
r = np.array([1., 2.])

apply_inverse = make_preconditioner(A, "jacobi", require_spd=True)
z = apply_inverse(r)

# For a custom inverse action (e.g., fixed diagonal scaling):
my_inverse = lambda residual: residual / np.diag(A)
custom = make_preconditioner(
    A, "custom", M=my_inverse, require_spd=True
)
```

The library exports `make_preconditioner` and its alias
`build_preconditioner`. `options` is a dictionary, e.g.
`options={"drop_tol": 1e-5, "fill_factor": 10}` for ILU or
`options={"omega": 1.2}` for SSOR.

## Tests and demonstration

```bash
python -m unittest krylov_methods.preconditioners.test_preconditioners -v
python -m krylov_methods.preconditioners.example
```

## References

1. Y. Saad (2003), *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM.
2. M. Benzi (2002), *Preconditioning Techniques for Large Linear Systems: A Survey*, Journal of Computational Physics, 182(2), 418–477.
3. A. Greenbaum (1997), *Iterative Methods for Solving Linear Systems*, SIAM.
