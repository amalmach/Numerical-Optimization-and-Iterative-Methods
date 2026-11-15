## Bi-Conjugate Gradient Stabilized (BiCGSTAB)

The following sections provide mathematical descriptions,
algorithmic steps, and numerical properties.

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

