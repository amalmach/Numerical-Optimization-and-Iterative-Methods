## Conjugate Gradient Method

The following sections provide mathematical descriptions,
algorithmic steps, and numerical properties.

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
