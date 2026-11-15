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
| **Steepest Descent** | Iterative minimization along the residual direction | [Learn more](#1.steepest-descent-method) | [Code](steepest_descent.py) |
| **Conjugate Gradient** | Optimization along conjugate directions | [Learn more](#2.conjugate-gradient-method) | [Code](conjugate_gradient.py) |
| **GMRES** | Residual minimization over Krylov subspaces | [Learn more](#3.generalized-minimal-residual-gmres) | [Code](gmres.py) |
| **BiCGSTAB** | Stabilized bi-conjugate gradient iterations | [Learn more](#4.bi-conjugate-gradient-stabilized-bicgstab) | [Code](bicgstab.py) |
| **Symmetric Lanczos** | Orthogonal projection onto Krylov subspaces | [Learn more](#5.symmetric-lanczos-algorithm) | [Code](symmetric_lanczos.py) |

Steepest Descent is included as a baseline iterative method for
comparison with Krylov subspace solvers.

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

