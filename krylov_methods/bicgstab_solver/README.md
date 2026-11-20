<div align="center">

# Bi-Conjugate Gradient Stabilized (BiCGSTAB)

A numerical implementation of the Bi-Conjugate Gradient
Stabilized method for solving large, nonsymmetric linear systems.

The module includes classical and preconditioned BiCGSTAB,
convergence diagnostics, numerical tests, and scientific examples.

</div>

---

### Main idea

BiCGSTAB is an iterative Krylov subspace method for solving
nonsymmetric linear systems of the form

<p align="center">
$\displaystyle Ax=b,$
</p>

where

<p align="center">
$\displaystyle A\in\mathbb{F}^{n\times n},
\qquad b\in\mathbb{F}^{n},
\qquad \mathbb{F}\in\{\mathbb{R},\mathbb{C}\}.$
</p>

It combines ideas from the Bi-Conjugate Gradient method
with a stabilization step designed to improve
residual convergence.

Unlike unrestarted GMRES, BiCGSTAB uses short recurrence
relations and does not store an expanding Krylov basis.

Its main advantages include:

- Applicability to nonsymmetric linear systems.
- Support for dense and sparse matrix representations.
- Relatively low memory requirements.
- Matrix-free implementations.
- Compatibility with preconditioning techniques.
- Efficient iterative solution of large systems.


### Files

| File | Purpose |
|---|---|
| [BiCGSTAB.py](BiCGSTAB.py) | BiCGSTAB iterations, stopping criteria, residual history |
| [Preconditioners](../preconditioners/preconditioners.py) | Shared Preconditioner implementations |
| [test_bicgstab.py](test_bicgstab.py) | Numerical regression tests against SciPy |
| [example.py](example.py) | Executable demonstration |

### Mathematical formulation

Given an initial approximation $x_0$, define

<p align="center">
$\displaystyle r_0=b-Ax_0.$
</p>

Choose a fixed shadow residual $\widehat r_0$
such that

<p align="center">
$\displaystyle \widehat r_0^Tr_0\neq0.$
</p>

A common choice is

<p align="center">
$\displaystyle \widehat{r}_0=r_0.$
</p>

Initialize

<p align="center">
$\displaystyle p_0=r_0.$
</p>

At iteration $k$, define

<p align="center">
$\displaystyle \rho_k=\widehat r_0^Tr_k.$
</p>

For $k>0$, the search direction is updated using

<p align="center">
$\displaystyle p_k=r_k+\beta_{k-1}(p_{k-1}-\omega_{k-1}v_{k-1}).$
</p>

The recurrence coefficient is

<p align="center">
$\displaystyle \beta_{k-1}=\frac{\rho_k}{\rho_{k-1}}\frac{\alpha_{k-1}}{\omega_{k-1}}.$
</p>

Compute the matrix-vector product

<p align="center">
$\displaystyle v_k=Ap_k.$
</p>

The first step length is

<p align="center">
$\displaystyle \alpha_k=
\frac{\rho_k}{\widehat r_0^Tv_k}.$
</p>

Compute the intermediate residual

<p align="center">
$\displaystyle s_k=r_k-\alpha_kv_k.$
</p>

If this residual satisfies the stopping criterion,
the approximate solution can be updated immediately:

<p align="center">
$\displaystyle x_{k+1}=x_k+\alpha_kp_k.$
</p>

Otherwise, compute

<p align="center">
$\displaystyle t_k=As_k$
</p>

and the stabilization parameter

<p align="center">
$\displaystyle \omega_k=
\frac{t_k^Ts_k}{t_k^Tt_k}.$
</p>

This parameter minimizes the Euclidean norm of

<p align="center">
$\displaystyle s_k-\omega t_k$
</p>

over real scalar values of $\omega$.

The approximate solution is updated using

<p align="center">
$\displaystyle x_{k+1}=
x_k+\alpha_kp_k+\omega_ks_k.$
</p>

The corresponding residual becomes

<p align="center">
$\displaystyle r_{k+1}=s_k-\omega_kt_k.$
</p>

The algorithm continues until the residual norm
satisfies the prescribed tolerance or a numerical
breakdown occurs.

The implementation monitors the true residual of the original
linear system and stops when

<p align="center">
$\displaystyle
\|b-Ax_k\|_2
\leq
\max\left(
\mathrm{atol},
\mathrm{rtol}\|b\|_2
\right).$
</p>

> **Note — Complex systems:** BiCGSTAB also applies to complex
> linear systems. In this case, transposes are replaced by
> conjugate transposes, and scalar products are interpreted
> as Hermitian inner products.

### Algorithm

**Input:** Square matrix $A$, vector $b$, initial guess $x_0$,
tolerance $\varepsilon$, maximum iterations $N$.

1. Initialize the residual $r_0$ and shadow residual $\widehat r_0$ (a common choice is $\widehat r_0=r_0$).
2. Construct the search direction $p_0=r_0, \quad \rho_0=\widehat r_0^Tr_0.$
3. *Convergence check*. If $\|r_0\|_2\leq\varepsilon$, return $x_0$.
4. For each iteration $k=0,1,\ldots,N-1$:
   - Compute $v_k=Ap_k$.
   - *Breakdown check:* If $\widehat r_0^Tv_k=0$, terminate.
   - Compute $\alpha_k$.
   - Compute the intermediate residual $s_k$.
   - *Early convergence check:* If $\|s_k\|_2\leq\varepsilon$:
      - Update $x_{k+1}=x_k+\alpha_kp_k$.
      - Recompute the true residual $r_{k+1}=b-Ax_{k+1}$.
      - If $\|r_{k+1}\|_2\leq\varepsilon$, stop: convergence achieved.
      - Otherwise, report a residual-gap failure.
   - Compute $t_k=As_k$.
   - *Breakdown check:* If $t_k^Tt_k=0$, terminate due to numerical breakdown.
   - Compute the stabilization parameter $\omega_k$.
   - Update the solution and residual.
   - *Convergence check*. If $\|r_{k+1}\|_2\leq\varepsilon,$ stop: convergence achieved.
   - *Breakdown check*. If $\omega_k=0$, terminate due to numerical breakdown.
   - Compute the next scalar $\rho_{k+1}$
   - *Breakdown check*. If $\rho_{k+1}=0$, terminate due to numerical breakdown.
   - Compute the recurrence coefficient $\beta_k$.
   - Update the search direction $p_{k+1}$.
5. Return the approximate solution and convergence history.

---

### Preconditioning

The convergence of BiCGSTAB depends strongly on the
spectral and numerical properties of the coefficient matrix $A$.

For ill-conditioned linear systems, convergence may be slow,
irregular, or affected by numerical stagnation.

Preconditioning aims to improve convergence by replacing
the original system with an equivalent system that is
more favorable for iterative solution.

Let $M$ denote a nonsingular preconditioning matrix,
ideally approximating $A$ while remaining inexpensive
to solve with.

The objective is to accelerate convergence without
explicitly computing the inverse of $A$.

#### Right Preconditioning

Consider the original linear system

<p align="center">
$Ax=b.$
</p>

Introduce the transformed variable $y$ through

<p align="center">
$x=M^{-1}y.$
</p>

Substitution gives the right-preconditioned system

<p align="center">
$AM^{-1}y=b.$
</p>

The preconditioned operator is therefore

<p align="center">
$\widetilde A=AM^{-1}.$
</p>

Instead of forming $M^{-1}$ explicitly, the algorithm
applies its action by solving systems involving $M$.

The key advantage is that the residual of the
preconditioned system is identical to the residual
of the original system:

<p align="center">
$r=b-AM^{-1}y=b-Ax.$
</p>

Consequently, convergence can be monitored using
the true residual of the original problem.

#### Preconditioning Methods and Documentation

BiCGSTAB supports Jacobi, LU, ILU, Cholesky, IC(0), SSOR,
and custom inverse actions.

Detailed mathematical formulations, numerical properties,
and implementations are available in the
[Preconditioning Methods](../preconditioners/README.md)
documentation.

The present BiCGSTAB implementation uses right preconditioning.

#### Integration into BiCGSTAB

The integration of right preconditioning can be summarized as follows:

| Operation | Classical BiCGSTAB | Right-Preconditioned BiCGSTAB |
|:---|:---|:---|
| Search direction | $p_k$ | $z_k=M^{-1}p_k$ |
| First matrix-vector product | $v_k=Ap_k$ | $v_k=Az_k$ |
| Intermediate residual | $s_k=r_k-\alpha_kv_k$ | Unchanged |
| Stabilization direction | $s_k$ | $y_k=M^{-1}s_k$ |
| Second matrix-vector product | $t_k=As_k$ | $t_k=Ay_k$ |
| Solution update | $x_{k+1}=x_k+\alpha_kp_k+\omega_ks_k$ | $x_{k+1}=x_k+\alpha_kz_k+\omega_ky_k$ |
| Residual update | $r_{k+1}=s_k-\omega_kt_k$ | Unchanged |

Thus, right preconditioning requires two applications
of the inverse preconditioner per full BiCGSTAB iteration:

<p align="center">
$z_k=M^{-1}p_k,\qquad y_k=M^{-1}s_k.$
</p>

The second application is unnecessary when the
intermediate residual already satisfies the stopping criterion.

The remaining recurrence relations for
$\rho_k$, $\alpha_k$, $\beta_k$, and $\omega_k$
retain their classical form, using the modified
matrix-vector products.

#### Important Numerical Considerations

- The preconditioner should be inexpensive to apply.
- Explicit computation of $M^{-1}$ is generally unnecessary.
- A suitable preconditioner can reduce the number of iterations.
- Preconditioning does not guarantee faster convergence.
- The cost of constructing and applying the preconditioner
  must be considered.
- The implementation assumes a fixed linear preconditioner
  throughout the iteration.
- A singular or poorly conditioned preconditioner
  can cause numerical difficulties.

In finite-precision arithmetic, the implementation checks for numerical
breakdown using smallness thresholds and finite-value checks. These
safeguards help prevent unstable divisions and invalid numerical
operations, although their effectiveness depends on the scaling
and conditioning of the linear system.

#### Numerical Breakdown

BiCGSTAB may encounter numerical breakdown when certain scalar
quantities become zero or numerically too small.

Important quantities include:

- $\rho_k$: measures the interaction between the shadow residual
  and the current residual.
- $\widehat r_0^T v_k$: appears in the denominator of $\alpha_k$.
- $t_k^T t_k$: appears in the denominator of $\omega_k$.
- $\omega_k$: appears in the recurrence defining the next
  search direction.

When these quantities become numerically unsuitable, the algorithm
may terminate before reaching the prescribed tolerance.

A numerical breakdown does not necessarily imply that the original
linear system has no solution.

#### Available Preconditioners

The solver provides several preconditioning options.

| Preconditioner | Identifier | Description |
|:---|:---:|:---|
| None | `None` | Unpreconditioned BiCGSTAB |
| Jacobi | `"jacobi"` | Diagonal preconditioning |
| LU | `"lu"` | LU factorization |
| ILU | `"ilu"` | Incomplete LU factorization |
| Cholesky | `"cholesky"` | Exact Cholesky factorization for Hermitian positive-definite systems |
| IC(0) | `"ic0"` | Incomplete Cholesky with zero fill-in |
| SSOR | `"ssor"` | Symmetric successive over-relaxation |
| Custom | `"custom"` | User-defined inverse action |

### Implementation:

Supports real and complex dense matrices, SciPy sparse matrices,
`LinearOperator`, and callable matrix-vector products.

```python
from krylov_methods.bicgstab_solver.BiCGSTAB import solve_bicgstab

result = solve_bicgstab(A, b, rtol=1e-8, atol=1e-12,
                        preconditioner="ilu",
                        preconditioner_options={"drop_tol": 1e-5})
print(result.converged, result.iterations, result.solution)
```

For custom inverse-action preconditioning:

```python
result = solve_bicgstab(A, b, preconditioner="custom", M=my_inverse_action)
```

Available `preconditioner` values: `None`, `"jacobi"`, `"lu"`,
`"ilu"`, `"cholesky"`, `"ic0"`, `"ssor"`, and `"custom"`.

The `"cholesky"` option requires a Hermitian positive-definite
matrix, while `"ic0"` is an educational implementation
restricted to real symmetric positive-definite matrices.

The `M` argument with `"custom"` may be an inverse-action callable,
`LinearOperator`, or an explicit matrix to be factored and solved.

**Important:** The algorithm is **right-preconditioned**: it computes
`M^{-1}p` and `M^{-1}s` to update the original solution while measuring
the **true residual** `||b-Ax||`.

The solver returns a `BiCGSTABResult` with `solution`, `iterations`,
`converged`, `residual_norms`, optional `history`, `breakdown`, and `message`.
The legacy `bicgstab(...)` function returns a dictionary (and stores iterate
history), while `bicgstab_solver(...)` returns only the solution and raises
on nonconvergence.

The convergence criterion is `||b-Ax|| <= max(atol, rtol*||b||)`.
Numerical breakdown is reported as a nonconverged result, never silently
returned as a successful solution.

To store all approximate solutions, set `store_history=True`.

Otherwise, BiCGSTAB requires O(n) working-vector storage,
excluding the system matrix and any preconditioner factors.

Install: `python -m pip install numpy scipy`.

From the directory containing `bicgstab_solver/`:

```bash
python3 -m unittest krylov_methods.bicgstab_solver.test_bicgstab -v
python3 -m krylov_methods.bicgstab_solver.example
```
The IC(0) preconditioner uses dense intermediate storage and is intended
for educational problems, not high-performance sparse production use.

### References

1. H. A. van der Vorst,
   *Bi-CGSTAB: A Fast and Smoothly Converging Variant of
   Bi-CG for the Solution of Nonsymmetric Linear Systems*,
   SIAM Journal on Scientific and Statistical Computing,
   1992.

2. Y. Saad,
   *Iterative Methods for Sparse Linear Systems*,
   SIAM, 2003.

3. C. T. Kelley,
   *Iterative Methods for Linear and Nonlinear Equations*,
   SIAM, 1995.


