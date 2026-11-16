<div align="center">

# Generalized Minimal Residual (GMRES)

The **Generalized Minimal Residual method (GMRES)** is a Krylov
subspace iterative method for solving large, sparse, and generally
nonsymmetric linear systems.

GMRES constructs an orthonormal basis using the Arnoldi process
and minimizes the Euclidean norm of the residual over successive
Krylov subspaces.

This document presents the mathematical foundations, numerical
algorithms, restarted variants, and preconditioning techniques.

</div>

---

### Modular GMRES Solver

| Module | Responsibility |
|---|---|
| `arnoldi.py` | Modified Gram-Schmidt and optional reorthogonalization |
| `hessenberg.py` | Hessenberg workspace and projection helper |
| `givens.py` | Stable Givens rotations and right-hand-side updates |
| `least_squares.py` | Reduced triangular solve with rank-deficient fallback |
| `preconditioners.py` | Identity, Jacobi, exact LU, ILU, IC(0), SSOR, custom |
| `GMRES.py` | Solver options, convergence, restarting, result metadata |
| `example.py` | Executable usage examples |
| `test_gmres.py` | Regression tests against SciPy |

---

### Mathematical Foundations

<details>
<summary><strong>1. Krylov Subspace</strong></summary>

<br>

Consider the linear system

<p align="center">
$Ax=b,$
</p>

where $A\in\mathbb{R}^{n\times n}$ and $b\in\mathbb{R}^n$.

Given an initial approximation $x_0$, the initial residual is

<p align="center">
$\displaystyle r_0=b-Ax_0.$
</p>

The Krylov subspace of dimension $m$ is

<p align="center">
$\displaystyle \mathcal{K}_m(A,r_0)
=
\mathrm{span}
\left\{
r_0,Ar_0,A^2r_0,\ldots,A^{m-1}r_0
\right\}.$
</p>

GMRES seeks an approximation

<p align="center">
$\displaystyle x_m\in x_0+\mathcal{K}_m(A,r_0)$
</p>

that minimizes the Euclidean residual norm:

<p align="center">
$\displaystyle \|b-Ax_m\|_2
=
\min_{x\in x_0+\mathcal{K}_m(A,r_0)}
\|b-Ax\|_2.$
</p>

</details>

---

<details>
<summary><strong>2. Arnoldi Process</strong></summary>

<br>

The Arnoldi process constructs an orthonormal basis of the
Krylov subspace.

**Initialization**

<p align="center">
$\displaystyle \beta=\|r_0\|_2, \qquad v_1=\frac{r_0}{\beta}.$
</p>

For each iteration $k=1,\ldots,m$:

1. Compute the matrix-vector product:

   <p align="center">
   $\displaystyle w=Av_k.$
   </p>

2. Orthogonalize against the previous Arnoldi vectors:

   <p align="center">
   $\displaystyle h_{i,k}=v_i^Tw, \qquad i=1,\ldots,k.$
   </p>

   <p align="center">
   $\displaystyle w=w-h_{i,k}v_i.$
   </p>

3. Compute the normalization coefficient:

   <p align="center">
   $\displaystyle h_{k+1,k}=\|w\|_2.$
   </p>

4. If $h_{k+1,k}\neq0$, construct:

   <p align="center">
   $\displaystyle v_{k+1}=\frac{w}{h_{k+1,k}}.$
   </p>

The resulting basis $V_m=[v_1,\ldots,v_m].$ satisfies

<p align="center">
$\displaystyle V_m^TV_m=I_m.$
</p>

</details>

---

<details>
<summary><strong>3. Hessenberg Matrix</strong></summary>

<br>

The Arnoldi process generates an upper Hessenberg matrix

<p align="center">
$\displaystyle \overline{H}_m
\in\mathbb{R}^{(m+1)\times m}.$
</p>

Its entries satisfy

<p align="center">
$\displaystyle h_{i,k}=0
\qquad \text{for }i>k+1.$
</p>

The Arnoldi factorization is

<p align="center">
$\displaystyle AV_m=V_{m+1}\overline{H}_m.$
</p>

Equivalently,

<p align="center">
$\displaystyle AV_m
=
V_mH_m
+
h_{m+1,m}v_{m+1}e_m^T.$
</p>

Using the orthogonality of $V_m$, we obtain

<p align="center">
$\displaystyle V_m^TAV_m=H_m.$
</p>

The matrix $H_m$ therefore represents the projection
of $A$ onto the Krylov subspace.

</details>

---

<details>
<summary><strong>4. Least-Squares Problem</strong></summary>

<br>

The approximate solution is expressed as

<p align="center">
$\displaystyle x_m=x_0+V_my_m.$
</p>

The corresponding residual is

<p align="center">
$\displaystyle r_m
=
b-Ax_m
=
r_0-AV_my_m.$
</p>

Using the Arnoldi factorization,

<p align="center">
$\displaystyle r_m
=
V_{m+1}
\left(
\beta e_1-\overline{H}_my_m
\right).$
</p>

Since $V_{m+1}$ has orthonormal columns,

<p align="center">
$\displaystyle \|r_m\|_2
=
\left\|
\beta e_1-\overline{H}_my_m
\right\|_2.$
</p>

Thus, the GMRES minimization problem reduces to

<p align="center">
$\displaystyle y_m
=
\underset{y\in\mathbb{R}^m}
{\mathrm{arg\,min}}
\left\|
\beta e_1-\overline{H}_my
\right\|_2.$
</p>

This reduced least-squares problem can be solved efficiently
using Givens rotations.

</details>

---

<details>
<summary><strong>5. Givens Rotations</strong></summary>

<br>

Givens rotations are used to transform the upper Hessenberg
matrix into an upper-triangular matrix.

For a real-valued problem, the rotation acting on two
consecutive rows is

<p align="center">
  <img
    src="https://latex.codecogs.com/svg.image?G_k%3D%5Cbegin%7Bpmatrix%7Dc_k%26s_k%5C%5C-s_k%26c_k%5Cend%7Bpmatrix%7D"
    alt="G_k = matrix with rows (c_k, s_k) and (-s_k, c_k)"
  >
</p>

where

<p align="center">
$\displaystyle c_k^2+s_k^2=1.$
</p>

The rotation coefficients are computed from the
current Hessenberg entries:

<p align="center">
$\displaystyle \delta_k
=
\sqrt{h_{k,k}^2+h_{k+1,k}^2}.$
</p>

For $\delta_k\neq0$,

<p align="center">
$\displaystyle c_k=\frac{h_{k,k}}{\delta_k},
\qquad
s_k=\frac{h_{k+1,k}}{\delta_k}.$
</p>

Applying successive rotations gives

<p align="center">
  <img
    src="https://latex.codecogs.com/svg.image?Q_m%5Coverline%7BH%7D_m%3D%5Cbegin%7Bpmatrix%7DR_m%5C%5C0%5Cend%7Bpmatrix%7D"
    alt="Q_m H-bar_m = matrix with R_m and 0"
  >
</p>

where $R_m$ is upper triangular.

The transformed right-hand side is

<p align="center">
$\displaystyle g=Q_m(\beta e_1).$
</p>

The reduced problem becomes

<p align="center">
$\displaystyle R_my_m=g_{1:m}.$
</p>

The coefficient vector $y_m$ can then be obtained
through backward substitution.

The residual norm is

<p align="center">
$\displaystyle \|r_m\|_2=|g_{m+1}|.$
</p>

</details>

---

## GMRES Algorithms

The following algorithms share the same Arnoldi orthogonalization,
Hessenberg matrix construction, Givens rotations, and reduced
least-squares solution.

The **unrestarted GMRES algorithm** is presented first in full detail.
The restarted and preconditioned variants refer to its common
computational steps and describe only the necessary modifications.

<details>
<summary><strong>6. Unrestarted GMRES (Without Preconditioning)</strong></summary>

<br>

**Input:**

- Matrix $A\in\mathbb{R}^{n\times n}$.
- Right-hand-side vector $b\in\mathbb{R}^n$.
- Initial approximation $x_0\in\mathbb{R}^n$.
- Convergence tolerance $\varepsilon>0$.
- Maximum number of iterations $N$.

**Output:**

- Approximate solution $x$.
- Residual norm history.
- Number of iterations.
- Convergence status.

### Step 1 — Initialization

**1.1.** Compute the initial residual and its Euclidean norm:

<p align="center">
$\displaystyle r_0=b-Ax_0,\qquad \beta=\|r_0\|_2.$
</p>

**1.2.** If $\beta\leq\varepsilon$, return $x_0$.

**1.3.** Initialize the first Arnoldi vector and
the transformed right-hand side:

<p align="center">
$\displaystyle v_1=\frac{r_0}{\beta},\qquad g=\beta e_1.$
</p>

**1.4.** Initialize the Hessenberg matrix
$\overline H_N$, the Givens rotation coefficients
$(c_k,s_k)$, and the residual norm history.

### Step 2 — Arnoldi–Givens Iterations

**For** $k=1,\ldots,N$:

**2.1. Matrix-vector product**

Compute

<p align="center">
$\displaystyle w=Av_k.$
</p>

**2.2. Arnoldi orthogonalization**

**For** $i=1,\ldots,k$, compute the projection coefficient:

<p align="center">
$\displaystyle h_{i,k}=v_i^Tw.$
</p>

Remove the projection of $w$ onto $v_i$:

<p align="center">
$\displaystyle w\leftarrow w-h_{i,k}v_i.$
</p>

**End For**

**2.3. Hessenberg matrix construction**

Compute the subdiagonal coefficient:

<p align="center">
$\displaystyle h_{k+1,k}=\|w\|_2.$
</p>

The coefficients

<p align="center">
$\displaystyle h_{1,k},h_{2,k},\ldots,h_{k+1,k}$
</p>

form the $k$-th column of the upper Hessenberg matrix
$\overline H_k$.

**2.4. Arnoldi basis extension**

If $h_{k+1,k}\neq0$, construct the next orthonormal vector:

<p align="center">
$\displaystyle v_{k+1}=\frac{w}{h_{k+1,k}}.$
</p>

Otherwise, mark Arnoldi termination and proceed
to the reduced least-squares problem after processing
the current Hessenberg column.

**2.5. Apply previously computed Givens rotations**

**For** $i=1,\ldots,k-1$, store

<p align="center">
$\displaystyle a=h_{i,k},\qquad b=h_{i+1,k}.$
</p>

Apply the previously computed rotation:

<p align="center">
$\displaystyle h_{i,k}=c_i a+s_i b.$
</p>

<p align="center">
$\displaystyle h_{i+1,k}=-s_i a+c_i b.$
</p>

**End For**

These transformations preserve the Euclidean norm and
maintain the upper-triangular structure of the previously
processed Hessenberg columns.

**2.6. Construct the new Givens rotation**

Compute

<p align="center">
$\displaystyle \delta_k=\sqrt{h_{k,k}^2+h_{k+1,k}^2}.$
</p>

If $\delta_k\neq0$, define

<p align="center">
$\displaystyle c_k=\frac{h_{k,k}}{\delta_k},\qquad s_k=\frac{h_{k+1,k}}{\delta_k}.$
</p>

If $\delta_k=0$, handle the degenerate rotation case
before proceeding.

**2.7. Apply the new Givens rotation**

Eliminate the subdiagonal entry:

<p align="center">
$\displaystyle h_{k,k}\leftarrow\delta_k,\qquad h_{k+1,k}\leftarrow0.$
</p>

**2.8. Update the transformed right-hand side**

Store the previous value:

<p align="center">
$\displaystyle g_k^{\mathrm{old}}=g_k.$
</p>

Apply the same rotation:

<p align="center">
$\displaystyle g_k\leftarrow c_k g_k^{\mathrm{old}}.$
</p>

<p align="center">
$\displaystyle g_{k+1}\leftarrow-s_k g_k^{\mathrm{old}}.$
</p>

**2.9. Residual estimation and convergence check**

Compute the residual estimate:

<p align="center">
$\displaystyle \rho_k=|g_{k+1}|.$
</p>

Record $\rho_k$ in the convergence history.

If $\rho_k\leq\varepsilon$, stop the Arnoldi iterations.

If Arnoldi termination occurred, proceed to
solution reconstruction and verify convergence.

**End For**

### Step 3 — Solution Reconstruction

**3.1.** Let $j$ denote the number of completed
Arnoldi iterations.

**3.2.** Solve the upper-triangular system:

<p align="center">
$\displaystyle R_jy_j=g_{1:j}.$
</p>

**3.3.** Construct the approximate solution:

<p align="center">
$\displaystyle x_j=x_0+V_jy_j.$
</p>

### Step 4 — Termination

Return the approximate solution $x_j$, residual norm
history, iteration count, and convergence status.

**Remark:** Unrestarted GMRES allows the Krylov basis to
grow throughout the iterations, increasing memory
requirements and orthogonalization costs.

</details>

---

<details>
<summary><strong>7. Restarted GMRES(m) (Without Preconditioning)</strong></summary>

<br>

Restarted GMRES(m) limits the Krylov subspace dimension
to a prescribed restart parameter $m$.

After at most $m$ Arnoldi iterations, the approximate
solution is updated and a new Krylov cycle begins.

The inner Arnoldi–Givens operations are identical to those
described in **Section 6, Step 2**.

**Input:**

- Matrix $A\in\mathbb{R}^{n\times n}$.
- Right-hand-side vector $b\in\mathbb{R}^n$.
- Initial approximation $x_0\in\mathbb{R}^n$.
- Convergence tolerance $\varepsilon>0$.
- Restart parameter $m$.
- Maximum total number of iterations $N$.

**Output:**

- Approximate solution $x$.
- Residual norm history.
- Total number of iterations.
- Convergence status.

### Step 1 — Initialization

**1.1.** Set the current approximation:

<p align="center">
$\displaystyle x=x_0.$
</p>

**1.2.** Initialize the total iteration counter:

<p align="center">
$\displaystyle n_{\mathrm{iter}}=0.$
</p>

**1.3.** Initialize the convergence status
and residual norm history.

### Step 2 — Restarted Arnoldi–Givens Iterations

**While** convergence has not been achieved and
$n_{\mathrm{iter}}<N$:

**2.1. Initialize a new Krylov cycle**

Compute the residual and its Euclidean norm:

<p align="center">
$\displaystyle r=b-Ax,\qquad \beta=\|r\|_2.$
</p>

If $\beta\leq\varepsilon$, stop.

Initialize

<p align="center">
$\displaystyle v_1=\frac{r}{\beta},\qquad g=\beta e_1.$
</p>

Initialize the Hessenberg matrix and Givens
rotation coefficients for the new cycle.

**2.2. Determine the inner iteration limit**

Set

<p align="center">
$\displaystyle \ell=\min(m,N-n_{\mathrm{iter}}).$
</p>

**2.3. Perform the Arnoldi–Givens iterations**

**For** $k=1,\ldots,\ell$:

Perform **Steps 2.1–2.9 of the Unrestarted GMRES
algorithm (Section 6)**:

- Compute $w=Av_k$.
- Apply Arnoldi orthogonalization.
- Construct the Hessenberg coefficients.
- Extend the Arnoldi basis when possible.
- Apply previous Givens rotations.
- Construct and apply the new Givens rotation.
- Update the transformed right-hand side.
- Compute the residual estimate $\rho_k=|g_{k+1}|$.
- Check convergence and Arnoldi termination.

Increment $n_{\mathrm{iter}}$.

If convergence is detected or Arnoldi terminates,
stop the current inner iteration.

**End For**

### Step 3 — Solution Reconstruction

**3.1.** Let $j\leq\ell$ be the number of completed
Arnoldi iterations in the current cycle.

**3.2.** Solve the reduced upper-triangular system:

<p align="center">
$\displaystyle R_jy_j=g_{1:j}.$
</p>

**3.3.** Update the approximate solution:

<p align="center">
$\displaystyle x\leftarrow x+V_jy_j.$
</p>

### Step 4 — Convergence Check and Restart

**4.1.** Recompute the true residual:

<p align="center">
$\displaystyle r=b-Ax.$
</p>

**4.2.** Check the convergence criterion:

<p align="center">
$\displaystyle \|r\|_2\leq\varepsilon.$
</p>

If satisfied, terminate successfully.

Otherwise, start a new Krylov cycle using
the updated approximation, provided that the
iteration budget remains available.

**End While**

### Step 5 — Termination

Return the approximate solution $x$, residual
history, total iteration count, and convergence status.

**Remark:** Restarting reduces memory requirements
by limiting the number of Arnoldi vectors stored.
However, small restart dimensions may slow convergence
or produce stagnation.

</details>

---

<details>
<summary><strong>8. Preconditioning Techniques</strong></summary>

<br>

Preconditioning aims to improve the convergence of GMRES
by transforming the original linear system into an
equivalent system with more favorable numerical properties.

A preconditioner $M$ approximates $A$ while remaining
significantly cheaper to invert or apply.

**Left preconditioning**

<p align="center">
$\displaystyle M^{-1}Ax=M^{-1}b.$
</p>

The corresponding residual is

<p align="center">
$\displaystyle \widetilde{r}=M^{-1}(b-Ax).$
</p>

Left-preconditioned GMRES minimizes

<p align="center">
$\displaystyle \|M^{-1}(b-Ax)\|_2.$
</p>

**Right preconditioning**

<p align="center">
$\displaystyle AM^{-1}y=b,
\qquad
x=M^{-1}y.$
</p>

Right-preconditioned GMRES can minimize the
original residual norm directly.

**Split preconditioning**

<p align="center">
$\displaystyle M_L^{-1}AM_R^{-1}y=M_L^{-1}b,$
</p>

with

<p align="center">
$\displaystyle x=M_R^{-1}y.$
</p>

**Common preconditioners**

- Incomplete LU factorization (ILU).
- Incomplete Cholesky factorization (IC).
- Sparse Approximate Inverse (SPAI).
- Factorized Sparse Approximate Inverse (FSAI).
- Geometric Multigrid (GMG).
- Algebraic Multigrid (AMG).
- Domain decomposition methods.

</details>

---

<details>
<summary><strong>9. Restarted Preconditioned GMRES</strong></summary>

<br>

Restarted preconditioned GMRES combines restarting
with a preconditioner to improve convergence for
large-scale and ill-conditioned linear systems.

This formulation uses **left preconditioning**:

<p align="center">
$\displaystyle M^{-1}Ax=M^{-1}b.$
</p>

The Arnoldi–Givens procedure remains identical to
that of **Section 6, Step 2**, except that the
matrix-vector product uses the preconditioned operator
$M^{-1}A$.

The restarting strategy follows **Section 7**.

**Input:**

- Matrix $A\in\mathbb{R}^{n\times n}$.
- Right-hand-side vector $b\in\mathbb{R}^n$.
- Initial approximation $x_0\in\mathbb{R}^n$.
- Nonsingular preconditioner $M$.
- Convergence tolerance $\varepsilon>0$.
- Restart parameter $m$.
- Maximum total number of iterations $N$.

**Output:**

- Approximate solution $x$.
- Preconditioned residual norm history.
- Total number of iterations.
- Convergence status.

### Step 1 — Initialization

**1.1.** Set

<p align="center">
$\displaystyle x=x_0.$
</p>

**1.2.** Initialize

<p align="center">
$\displaystyle n_{\mathrm{iter}}=0.$
</p>

**1.3.** Initialize the convergence status
and residual history.

### Step 2 — Preconditioned Arnoldi–Givens Iterations

**While** convergence has not been achieved and
$n_{\mathrm{iter}}<N$:

**2.1. Initialize a new Krylov cycle**

Compute the left-preconditioned residual:

<p align="center">
$\displaystyle r=M^{-1}(b-Ax).$
</p>

Compute its norm:

<p align="center">
$\displaystyle \beta=\|r\|_2.$
</p>

If $\beta\leq\varepsilon$, stop.

Initialize

<p align="center">
$\displaystyle v_1=\frac{r}{\beta},\qquad g=\beta e_1.$
</p>

Initialize the Hessenberg matrix and Givens
rotation coefficients.

**2.2. Determine the inner iteration limit**

Set

<p align="center">
$\displaystyle \ell=\min(m,N-n_{\mathrm{iter}}).$
</p>

**2.3. Perform the preconditioned Arnoldi–Givens iterations**

**For** $k=1,\ldots,\ell$:

Follow **Steps 2.2–2.9 of the Unrestarted GMRES
algorithm (Section 6)**, with the following
modification to Step 2.1.

Replace the standard matrix-vector product

<p align="center">
$\displaystyle w=Av_k$
</p>

with the preconditioned product

<p align="center">
$\displaystyle w=M^{-1}Av_k.$
</p>

All subsequent Arnoldi orthogonalization,
Hessenberg matrix construction, Givens rotations,
and right-hand-side updates remain unchanged.

The residual estimate is

<p align="center">
$\displaystyle \rho_k=|g_{k+1}|.$
</p>

It now represents the estimated norm of the
left-preconditioned residual.

Increment $n_{\mathrm{iter}}$.

If convergence is detected or Arnoldi terminates,
stop the current inner iteration.

**End For**

### Step 3 — Solution Reconstruction

**3.1.** Let $j\leq\ell$ denote the number
of completed Arnoldi iterations.

**3.2.** Solve the reduced upper-triangular system:

<p align="center">
$\displaystyle R_jy_j=g_{1:j}.$
</p>

**3.3.** Update the approximate solution:

<p align="center">
$\displaystyle x\leftarrow x+V_jy_j.$
</p>

### Step 4 — Convergence Check and Restart

**4.1.** Recompute the preconditioned residual:

<p align="center">
$\displaystyle r=M^{-1}(b-Ax).$
</p>

**4.2.** Check the preconditioned convergence criterion:

<p align="center">
$\displaystyle \|r\|_2\leq\varepsilon.$
</p>

If satisfied, terminate successfully.

Otherwise, restart with the updated approximation,
provided that the iteration budget remains available.

The true residual

<p align="center">
$\displaystyle r_{\mathrm{true}}=b-Ax$
</p>

may also be monitored to assess the accuracy
of the solution to the original system.

**End While**

### Step 5 — Termination

Return the approximate solution $x$, preconditioned
residual norm history, total iteration count,
and convergence status.

**Remarks:**

- The preconditioner should be applied by solving
  linear systems involving $M$, rather than explicitly
  constructing $M^{-1}$.

- Left-preconditioned GMRES minimizes the norm

  <p align="center">
  $\displaystyle \|M^{-1}(b-Ax)\|_2.$
  </p>

- This norm differs in general from the true residual
  norm $\|b-Ax\|_2$.

</details>

---

<details>
<summary><strong>10. Numerical Properties</strong></summary>

<br>

**Convergence**

GMRES minimizes the Euclidean residual norm over the
current affine Krylov subspace.

In exact arithmetic, unrestarted GMRES produces
nonincreasing residual norms.

**Computational complexity**

At iteration $k$, Arnoldi orthogonalization requires
inner products and vector updates involving the
previously generated basis vectors.

The total orthogonalization cost increases with
the Krylov dimension.

**Memory requirements**

Unrestarted GMRES stores an expanding orthonormal basis.

Restarted GMRES(m) limits the number of stored basis
vectors to approximately $m+1$.

**Numerical stability**

Loss of orthogonality can affect the accuracy of
the Arnoldi decomposition and residual estimates.

Modified Gram-Schmidt and reorthogonalization
may improve numerical stability.

**Restarting**

Restarting controls memory usage and computational cost,
but may slow convergence or cause stagnation.

**Preconditioning**

An effective preconditioner can substantially reduce
the number of iterations required for convergence.

**Finite-precision arithmetic**

In floating-point arithmetic, explicitly computed
residual norms may differ from the residual estimates
obtained using Givens rotations.

Periodic residual recomputation can help assess
the accuracy of the approximation.

</details>

---

### Python Implementation

**Implementation:** [gmres.py](../gmres.py)

#### Run

Install dependencies: `python -m pip install numpy scipy`.

From the **directory containing** the `gmres_solver` folder:

```bash
python -m gmres_solver.example
python -m unittest gmres_solver.test_gmres -v
```
#### Usage

```python
import numpy as np
from gmres_solver.GMRES import gmres

A = np.array([[4., 1.], [2., 3.]])
b = np.array([1., 2.])

plain = gmres(A, b, restart=None, maxiter=2)
restarted = gmres(A, b, restart=2, maxiter=20)
preconditioned = gmres(A, b, restart=2, maxiter=20,
                       preconditioner="jacobi")
print(preconditioned.x, preconditioned.converged)
```

`restart=None` means **unrestarted**. Otherwise `restart` is the maximum
**inner Krylov dimension per cycle**, not the number of cycles. `maxiter`
limits **total** Arnoldi iterations. Convergence is checked using
`||b-Ax|| <= max(atol, rtol*||b||)` (the **true** residual).

Use `preconditioner="ilu"` with options such as
`preconditioner_options={"drop_tol": 1e-5, "fill_factor": 15}`.
For a custom inverse action, use `preconditioner="custom", M=my_solve`.
An explicit custom matrix `M` is interpreted as the preconditioner matrix
to factor and solve, **not** as the inverse matrix.

`GMRESResult` provides `x`, `converged`, `iterations`, `cycles`,
`residual_norm`, `residual_history`, `preconditioned_history`, and `message`.

#### Notes

- Left preconditioning minimizes the **preconditioned** residual within a
  cycle, but the stopping criterion checks the **true** residual.
- `ic0` is a simple, educational real-SPD IC(0) implementation with dense
  intermediate storage; do not use it on large sparse systems.
- Exact LU is a **direct solve used as a preconditioner** and is primarily
  useful for validation, not for improving GMRES performance.

---

### Further Reading

For a comprehensive mathematical treatment of GMRES, including
the Arnoldi process, Hessenberg matrix construction,
least-squares residual minimization, Givens rotations,
and preconditioning techniques, see:

[![Thesis](https://img.shields.io/badge/Read_PhD_Thesis-HAL-6F42C1)](https://theses.hal.science/tel-05806256/document)

**Chapter I — Literature Review, Section I.3.1: Krylov Subspace Methods**,
particularly **Sections I.3.1.1–I.3.1.5 (pp. 30–36)**.

These sections present detailed mathematical derivations and
the preconditioned GMRES algorithm.

---

### References

1. Y. Saad and M. H. Schultz,
   "GMRES: A Generalized Minimal Residual Algorithm
   for Solving Nonsymmetric Linear Systems,"
   SIAM Journal on Scientific and Statistical Computing,
   7(3), 856–869, 1986.

2. Y. Saad,
   *Iterative Methods for Sparse Linear Systems*,
   2nd edition, SIAM, 2003.
