<div align="center">

# Conjugate Gradient (CG)

An implementation of Conjugate Gradient and fixed-preconditioner CG,
with convergence diagnostics, tests, and numerical examples.

</div>

### Main idea

The Conjugate Gradient (CG) method improves upon Steepest
Descent by constructing search directions that are
conjugate with respect to the matrix $A$.

Two directions $p_i$ and $p_j$ are $A$-conjugate if

<p align="center">
$p_i^TAp_j=0,
\qquad i\neq j.$
</p>

For a symmetric positive-definite matrix $A$, CG solves $Ax=b$ by minimizing
the quadratic energy $f(x)=\frac12 x^TAx-b^Tx$ over the affine Krylov space
$x_0+\mathcal K_k(A,r_0)$.

Unlike GMRES, which minimizes the Euclidean residual norm,
CG minimizes the error in the energy norm associated with $A$.

For a symmetric positive-definite matrix, this is equivalent
to minimizing the quadratic objective function over the
current affine Krylov subspace.

Instead of repeatedly minimizing along the residual direction,
CG combines the current residual with the previous search
direction.

This avoids repeatedly correcting the same components
of the error.

### Files

| File | Purpose |
|:---|:---|
| [ConjugateGradient.py](ConjugateGradient.py) | CG/PCG implementation and structured result |
| [Preconditioners](../preconditioners/preconditioners.py) | Shared Preconditioner implementations |
| [test_conjugate_gradient.py](test_conjugate_gradient.py) | Numerical regression tests against SciPy |
| [example.py](example.py) | Reproducible numerical demonstration |

### Mathematical formulation

Initialize

<p align="center">
$r_0=b-Ax_0,
\qquad
p_0=r_0.$
</p>

At iteration $k$, compute

<p align="center">
$\alpha_k=
\frac{r_k^Tr_k}{p_k^TAp_k}.$
</p>

Update the approximation:

<p align="center">
$x_{k+1}=x_k+\alpha_kp_k.$
</p>

Update the residual:

<p align="center">
$r_{k+1}=r_k-\alpha_kAp_k.$
</p>

Compute

<p align="center">
$\beta_k=
\frac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k}.$
</p>

Finally, update the search direction:

<p align="center">
$p_{k+1}=r_{k+1}+\beta_kp_k.$
</p>

Stop when the true residual meets the prescribed tolerance.
In exact arithmetic, CG converges in at most $n$ steps.

> **Note — Complex Hermitian systems:** The Conjugate Gradient
> method also applies to complex Hermitian positive-definite
> matrices. In this case, transposes $p_i^T$ are replaced by
> conjugate transposes $p_i^H$, and scalar products are
> interpreted as Hermitian inner products.

### Algorithm

**Input:** SPD matrix $A$, vector $b$, initial guess $x_0$,
tolerance $\varepsilon$, maximum iterations $N$.

1. Initialize

<p align="center">
$r_0=b-Ax_0,\qquad p_0=r_0.$
</p>

2. For $k=0,1,\ldots,N-1$:

   - Check the residual norm.
   - Compute the step length $\alpha_k$.
   - Update $x_{k+1}$.
   - Update $r_{k+1}$.
   - Stop if the residual tolerance is satisfied.
   - Compute $\beta_k$.
   - Construct the new direction $p_{k+1}$.

3. Return the approximate solution and convergence history.

### Preconditioned CG

For descriptions of the available preconditioning techniques,
their mathematical formulations, and their numerical properties,
see the [Preconditioning Methods](../preconditioners/README.md)
documentation.

For a *fixed symmetric positive-definite* preconditioner $M$, replace the
residual search direction with the inverse preconditioner action:

<p align="center">
$z_k=M^{-1}r_k.$
</p>

Initialize $p_0=z_0$. The modified scalar recurrences are:

<p align="center">
$\displaystyle \alpha_k=\frac{r_k^Tz_k}{p_k^TAp_k}.$
</p>

<p align="center">
$x_{k+1}=x_k+\alpha_kp_k,\qquad r_{k+1}=r_k-\alpha_kAp_k.$
</p>

<p align="center">
$z_{k+1}=M^{-1}r_{k+1}.$
</p>

<p align="center">
$\displaystyle \beta_k=\frac{r_{k+1}^Tz_{k+1}}{r_k^Tz_k}.$
</p>

<p align="center">
$p_{k+1}=z_{k+1}+\beta_kp_k.$
</p>

These are the only changes to the classical iteration. Unlike GMRES or
BiCGSTAB, a generic nonsymmetric ILU inverse is not suitable for PCG.
For complex systems, transpose products become Hermitian inner products.

### Numerical properties

- Requires a symmetric positive-definite (SPD) matrix.
- Uses short recurrence relations.
- Requires little storage.
- Converges in at most $n$ iterations in exact arithmetic.
- Often substantially faster than Steepest Descent.

CG minimizes the error in the energy norm

<p align="center">
$\displaystyle
\|e_k\|_A=\sqrt{e_k^HAe_k},
$
</p>

where $e_k=x^\ast-x_k$.

For real-valued systems, the conjugate transpose $e_k^H$
reduces to the ordinary transpose $e_k^T$.

### Python interface

```python
import numpy as np

from krylov_methods.conjugate_gradient_solver import solve_conjugate_gradient

A = np.array([
    [4.0, 1.0],
    [1.0, 3.0],
])

b = np.array([1.0, 2.0])

result = solve_conjugate_gradient(
    A,
    b,
    rtol=1e-10,
    preconditioner="jacobi",
)

print("Converged:", result.converged)
print("Iterations:", result.iterations)
print("Solution:", result.solution)
print("Final residual:", result.residual_norms[-1])
```

The following examples and commands should be executed from the repository root.

The solver accepts dense/sparse arrays, SciPy `LinearOperator`, or a callable
matrix-vector product, and real SPD or complex Hermitian positive-definite
systems. It provides `solution`, `iterations`, `converged`, `residual_norms`,
optional `history`, `breakdown`, and `message`.

The stopping condition is

<p align="center">
$\displaystyle \|b-Ax_k\|_2\leq\max(\mathrm{atol},\mathrm{rtol}\|b\|_2).$
</p>

Available preconditioners: `None`, `"jacobi"`, `"ssor"`,
`"cholesky"`, `"ic0"` (real SPD systems), and `"custom"`.

All preconditioners used with CG must define a fixed Hermitian
positive-definite inverse action.

For mathematical formulations and implementation details, see the
[Preconditioning Methods](../preconditioners/README.md) documentation.

Exact Cholesky and the educational IC(0) implementation use
dense intermediate storage and may be unsuitable for large
sparse systems. For large-scale problems, appropriate sparse
SPD preconditioning techniques are recommended.

Install dependencies: `python -m pip install numpy scipy`.

From the repository root (the directory containing `krylov_methods/`):

```bash
python3 -m unittest krylov_methods.conjugate_gradient_solver.test_conjugate_gradient -v
python3 -m krylov_methods.conjugate_gradient_solver.example
```

### References

1. M. R. Hestenes and E. Stiefel (1952), *Methods of Conjugate Gradients for Solving Linear Systems*, Journal of Research of the NBS, 49(6), 409–436.
2. J. R. Shewchuk (1994), *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*.
3. Y. Saad (2003), *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM.
4. A. Greenbaum (1997), *Iterative Methods for Solving Linear Systems*, SIAM.
