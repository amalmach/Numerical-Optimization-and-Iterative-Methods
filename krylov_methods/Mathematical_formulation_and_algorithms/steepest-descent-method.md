## Steepest Descent Method

The following sections provide mathematical descriptions,
algorithmic steps, and numerical properties.

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

