# Krylov Subspace Methods

Numerical implementations and computational experiments for solving
linear systems using iterative algorithms.

The module includes classical stationary-direction methods, Krylov
subspace solvers, and Lanczos tridiagonalization.

---

## Mathematical Background

We consider the linear system

<p align="center">
$Ax=b,$
</p>

where

<p align="center">
$A\in\mathbb{R}^{n\times n},
\qquad
b\in\mathbb{R}^{n}.$
</p>

Starting from an initial approximation $x_0$, we define the
initial residual

<p align="center">
$r_0=b-Ax_0.$
</p>

The Krylov subspace of dimension $m$ is

<p align="center">
$\mathcal{K}_m(A,r_0)
=
\mathrm{span}
\left\{
r_0,Ar_0,A^2r_0,\ldots,A^{m-1}r_0
\right\}.$
</p>

Krylov methods seek approximate solutions using information
contained in these subspaces.

Unless otherwise specified, we stop when

<p align="center">
$\|r_k\|_2\leq\varepsilon.$
</p>

Here, $\varepsilon>0$ is a prescribed absolute tolerance.

---

## Implemented Algorithms

| Algorithm | Main Idea | Mathematical Description | Python |
|:---|:---|:---:|:---:|
| **Steepest Descent** | Iterative minimization along the residual direction | [Learn more](#1-steepest-descent-method) | [Code](steepest_descent.py) |
| **Conjugate Gradient** | Optimization along conjugate directions | [Learn more](#2-conjugate-gradient-method) | [Code](conjugate_gradient.py) |
| **GMRES** | Residual minimization over Krylov subspaces | [Learn more](#3-generalized-minimal-residual-gmres) | [Code](gmres.py) |
| **BiCGSTAB** | Stabilized bi-conjugate gradient iterations | [Learn more](#4-bi-conjugate-gradient-stabilized-bicgstab) | [Code](bicgstab.py) |
| **Symmetric Lanczos** | Orthogonal projection onto Krylov subspaces | [Learn more](#5-symmetric-lanczos-algorithm) | [Code](symmetric_lanczos.py) |

Steepest Descent is included as a baseline iterative method for
comparison with Krylov subspace solvers.

---

## Mathematical Formulation and Algorithms

The following sections provide mathematical descriptions,
algorithmic steps, and numerical properties.

---

<details>
<summary><strong>1. Steepest Descent Method</strong></summary>

<br>

#### Main idea

The Steepest Descent method solves a symmetric positive-definite (SPD)
linear system by minimizing the quadratic function

<p align="center">
$f(x)=\frac12x^TAx-b^Tx.$
</p>

Its gradient is

<p align="center">
$\nabla f(x)=Ax-b=-r.$
</p>

Therefore, the residual $r_k$ gives the direction of
steepest decrease of the objective function.

At every iteration, the algorithm moves in this direction
using an optimal step length.

#### Mathematical formulation

The search direction is

<p align="center">
$p_k=r_k.$
</p>

The optimal step length is

<p align="center">
$\alpha_k=
\frac{r_k^Tr_k}{r_k^TAr_k}.$
</p>

The solution and residual are updated by

<p align="center">
$x_{k+1}=x_k+\alpha_kr_k,$
</p>

<p align="center">
$r_{k+1}=r_k-\alpha_kAr_k.$
</p>

#### Algorithm

**Input:** SPD matrix $A$, vector $b$, initial guess $x_0$,
tolerance $\varepsilon$, maximum iterations $N$.

1. Compute the initial residual:

   <p align="center">
  $r_0=b-Ax_0.$
   </p>

2. For $k=0,1,\ldots,N-1$:

   - Check whether $\|r_k\|_2\leq\varepsilon$.
   - Compute

     <p align="center">

  $\alpha_k=\frac{r_k^Tr_k}{r_k^TAr_k}.$

</p>

   - Update

     <p align="center">

  $x_{k+1}=x_k+\alpha_kr_k.$

</p>

   - Update the residual

     <p align="center">

  $r_{k+1}=r_k-\alpha_kAr_k.$

</p>

3. Return the approximate solution and convergence history.

#### Numerical properties

- Requires a symmetric positive-definite (SPD) matrix.
- Uses a single residual direction at each iteration.
- Requires relatively little memory.
- Convergence can be slow for ill-conditioned systems.

**Implementation:** [steepest_descent.py](steepest_descent.py)

</details>

---

<details>
<summary><strong>2. Conjugate Gradient Method</strong></summary>

<br>

#### Main idea

The Conjugate Gradient (CG) method improves upon Steepest
Descent by constructing search directions that are
conjugate with respect to the matrix $A$.

Two directions $p_i$ and $p_j$ are $A$-conjugate if

<p align="center">
$p_i^TAp_j=0,
\qquad i\neq j.$
</p>

Instead of repeatedly minimizing along the residual direction,
CG combines the current residual with the previous search
direction.

This avoids repeatedly correcting the same components
of the error.

#### Mathematical formulation

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

#### Algorithm

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

#### Numerical properties

- Requires a symmetric positive-definite (SPD) matrix.
- Uses short recurrence relations.
- Requires little storage.
- Converges in at most $n$ iterations in exact arithmetic.
- Often substantially faster than Steepest Descent.

CG minimizes the error in the energy norm

<p align="center">

  $\|e_k\|_A=\sqrt{e_k^TAe_k},$

</p>

where $e_k=x^\ast-x_k$.

**Implementation:** [conjugate_gradient.py](conjugate_gradient.py)

</details>

---

<details>
<summary><strong>3. Generalized Minimal Residual (GMRES)</strong></summary>

<br>

#### Main idea

The Generalized Minimal Residual method (GMRES) solves
linear systems without requiring the matrix to be symmetric.

The method constructs a sequence of Krylov subspaces

<p align="center">
$\mathcal K_m(A,r_0)
=
\mathrm{span}
\{r_0,Ar_0,\ldots,A^{m-1}r_0\}.$
</p>

At each iteration, GMRES finds the approximation that
minimizes the residual norm over the current affine
Krylov subspace.

#### Arnoldi process

Starting from

<p align="center">
$v_1=\frac{r_0}{\|r_0\|_2},$
</p>

the Arnoldi process constructs orthonormal vectors

<p align="center">
$V_m=[v_1,\ldots,v_m].$
</p>

These vectors satisfy

<p align="center">
$AV_m=V_{m+1}\overline H_m,$
</p>

where $\overline H_m$ is an upper Hessenberg matrix.

#### Residual minimization

The approximate solution has the form

<p align="center">
$x_m=x_0+V_my_m.$
</p>

The vector $y_m$ solves the least-squares problem

<p align="center">
$y_m=
\underset{y\in\mathbb R^m}{\mathrm{argmin}}
\left\|
\beta e_1-\overline H_my
\right\|_2,$
</p>

where

<p align="center">
$\beta=\|r_0\|_2.$
</p>

#### Algorithm

**Input:** Square matrix $A$, vector $b$, initial guess $x_0$,
tolerance $\varepsilon$, maximum iterations $N$.

1. Compute

   <p align="center">

  $r_0=b-Ax_0,\qquad
   \beta=\|r_0\|_2.$

</p>

2. Initialize

   <p align="center">

  $v_1=r_0/\beta.$

</p>

3. For $k=1,2,\ldots,N$:

   - Compute $w=Av_k$.
   - Orthogonalize $w$ against the existing Arnoldi vectors.
   - Normalize the remaining vector.
   - Extend the Hessenberg matrix.
   - Solve the reduced least-squares problem.
   - Compute the approximate solution $x_k$.
   - Check the residual tolerance.

4. Return the approximate solution and convergence history.

#### Numerical properties

- Applicable to general square matrices.
- Minimizes the Euclidean residual norm at each iteration.
- Has a nonincreasing residual norm in exact arithmetic.
- Requires storing an expanding orthonormal basis.
- Orthogonalization costs increase with the iteration count.

The present implementation uses **unrestarted GMRES**.

**Implementation:** [gmres.py](gmres.py)


</details>

---

<details>
<summary><strong>4. Bi-Conjugate Gradient Stabilized (BiCGSTAB)</strong></summary>

<br>

#### Main idea

BiCGSTAB is an iterative method for solving
nonsymmetric linear systems.

It combines ideas from the Bi-Conjugate Gradient method
with a stabilization step designed to improve
residual convergence.

Unlike unrestarted GMRES, BiCGSTAB uses short recurrence
relations and does not store an expanding Krylov basis.

#### Mathematical formulation

Let

<p align="center">
$r_0=b-Ax_0.$
</p>

Choose a fixed shadow residual $\widehat r_0$
such that

<p align="center">
$\widehat r_0^Tr_0\neq0.$
</p>

Define

<p align="center">
$\rho_k=\widehat r_0^Tr_k.$
</p>

At each iteration, construct a search direction $p_k$
and compute

<p align="center">
$v_k=Ap_k.$
</p>

The first step length is

<p align="center">
$\alpha_k=
\frac{\rho_k}{\widehat r_0^Tv_k}.$
</p>

Compute the intermediate residual

<p align="center">
$s_k=r_k-\alpha_kv_k.$
</p>

Then define

<p align="center">
$t_k=As_k$
</p>

and the stabilization parameter

<p align="center">
$\omega_k=
\frac{t_k^Ts_k}{t_k^Tt_k}.$
</p>

Update the solution:

<p align="center">
$x_{k+1}
=
x_k+\alpha_kp_k+\omega_ks_k.$
</p>

Update the residual:

<p align="center">
$r_{k+1}=s_k-\omega_kt_k.$
</p>

The next search direction is obtained using

<p align="center">
$\beta_k
=
\frac{\rho_{k+1}}{\rho_k}
\frac{\alpha_k}{\omega_k},$
</p>

<p align="center">
$p_{k+1}
=
r_{k+1}
+
\beta_k(p_k-\omega_kv_k).$
</p>

#### Algorithm

**Input:** Square matrix $A$, vector $b$, initial guess $x_0$,
tolerance $\varepsilon$, maximum iterations $N$.

1. Initialize the residual and shadow residual.
2. For each iteration:
   - Construct the search direction.
   - Compute $\alpha_k$.
   - Compute the intermediate residual $s_k$.
   - Check for early convergence.
   - Compute the stabilization parameter $\omega_k$.
   - Update the solution and residual.
   - Check convergence and possible numerical breakdown.
3. Return the approximate solution and convergence history.

#### Numerical properties

- Applicable to general nonsymmetric systems.
- Uses short recurrence relations.
- Has relatively low memory requirements.
- Residual convergence may be nonmonotonic.
- May encounter numerical breakdown or stagnation.

**Implementation:** [bicgstab.py](bicgstab.py)

</details>

---

<details>
<summary><strong>5. Symmetric Lanczos Algorithm</strong></summary>

<br>

#### Main idea

The Symmetric Lanczos algorithm constructs an
orthonormal basis of a Krylov subspace associated
with a symmetric matrix.

Its principal objective is to transform a large
symmetric matrix into a smaller symmetric
tridiagonal matrix.

This transformation is useful for eigenvalue
approximation and Krylov projection methods.

#### Mathematical formulation

Let

<p align="center">
$r_0=b-Ax_0,
\qquad
\beta_0=\|r_0\|_2.$
</p>

Initialize

<p align="center">
$v_1=\frac{r_0}{\beta_0},
\qquad
v_0=0.$
</p>

The Lanczos recurrence is

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

The generated vectors form the matrix

<p align="center">
$V_m=[v_1,\ldots,v_m].$
</p>

The tridiagonal matrix is

<p align="center">
$T_m=
\begin{pmatrix}
\alpha_1 & \beta_1 & & \\
\beta_1 & \alpha_2 & \ddots & \\
& \ddots & \ddots & \beta_{m-1}\\
& & \beta_{m-1} & \alpha_m
\end{pmatrix}.$
</p>

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

#### Algorithm

**Input:** Symmetric matrix $A$, vector $b$,
initial guess $x_0$, Krylov dimension $m$.

1. Compute the initial residual.
2. Normalize it to obtain the first Lanczos vector.
3. For $k=1,\ldots,m$:
   - Apply the matrix $A$ to the current Lanczos vector.
   - Subtract the contribution of the previous vector.
   - Compute the diagonal coefficient $\alpha_k$.
   - Remove the component along the current vector.
   - Compute the off-diagonal coefficient $\beta_k$.
   - Normalize to obtain the next Lanczos vector.
4. Construct the tridiagonal matrix $T_m$.
5. Compute Ritz eigenvalues or a Galerkin approximation.

#### Numerical properties

- Requires a symmetric matrix.
- Uses a three-term recurrence.
- Produces a symmetric tridiagonal projection.
- Useful for approximating eigenvalues.
- May lose orthogonality in finite precision.
- A singular projected matrix can prevent the
  Galerkin linear-system approximation.

**Implementation:** [symmetric_lanczos.py](symmetric_lanczos.py)


</details>

---

## Numerical Experiments

The interactive Jupyter notebook provides three numerical studies.

### 1. Symmetric Positive-Definite Systems

Comparison of:

- Steepest Descent
- Conjugate Gradient
- GMRES
- BiCGSTAB

The experiment investigates residual convergence and iteration counts.

### 2. Nonsymmetric Linear Systems

Comparison of GMRES and BiCGSTAB using a finite-difference
discretization of a convection–diffusion equation.

The experiment illustrates monotonic and nonmonotonic residual
convergence behavior.

### 3. Symmetric Lanczos Tridiagonalization

Investigation of:

- Orthogonality of the Lanczos basis.
- Symmetric tridiagonal projection.
- Ritz eigenvalue approximations.
- Galerkin approximation of linear systems.

**Notebook:**
[krylov_comparison.ipynb](notebooks/krylov_comparison.ipynb)

---

## Running the Numerical Experiments

From the repository root, install the required packages:

```bash
python3 -m pip install numpy scipy matplotlib jupyterlab
```

Launch JupyterLab:

```bash
jupyter lab
```

Open:

```text
krylov_methods/notebooks/krylov_comparison.ipynb
```

The notebook imports the numerical algorithms directly from
their corresponding Python modules.

---

## Numerical Considerations

- Steepest Descent and Conjugate Gradient require symmetric
  positive-definite matrices.
- GMRES uses an expanding orthonormal Krylov basis.
- BiCGSTAB uses short recurrences but may exhibit nonmonotonic
  residual convergence.
- Symmetric Lanczos constructs a tridiagonal projection of
  a symmetric matrix.
- Finite-precision arithmetic can affect orthogonality and
  convergence behavior.

For large-scale applications, additional techniques such as
preconditioning, restarted GMRES, and reorthogonalization
may be necessary.

---

## References

1. Y. Saad, *Iterative Methods for Sparse Linear Systems*,
   SIAM, 2003.

2. J. Nocedal and S. J. Wright, *Numerical Optimization*,
   Springer, 1999.

3. H. A. van der Vorst, *Bi-CGSTAB: A Fast and Smoothly
   Converging Variant of Bi-CG for the Solution of
   Nonsymmetric Linear Systems*, SIAM Journal on
   Scientific and Statistical Computing, 1992.

