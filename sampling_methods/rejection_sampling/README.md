# Acceptance–Rejection Sampling

A Python implementation of acceptance–rejection sampling for generating random samples from one- and two-dimensional probability distributions.

The implementation provides:

- One-dimensional rejection sampling.
- Two-dimensional rejection sampling.
- Support for unnormalized target densities.
- Reproducible sampling through NumPy random generators.
- Input validation and explicit failure handling.
- Automated tests and reproducible numerical experiments.

## 1. Mathematical Formulation

Suppose we want to generate samples from a target probability density proportional to a nonnegative function

\[
f:\Omega\rightarrow[0,\infty),
\]

where \(\Omega\) is a bounded domain.

We assume that a finite constant \(M>0\) satisfies

\[
0\leq f(x)\leq M,\qquad x\in\Omega.
\]

The normalized target density is

\[
p(x)=\frac{f(x)}{Z},
\]

where

\[
Z=\int_\Omega f(x)\,dx.
\]

The normalization constant \(Z\) does not need to be known.

## 2. Acceptance–Rejection Algorithm

The algorithm generates candidate points uniformly over the sampling domain.

For each iteration:

1. Generate \(X\sim\mathcal U(\Omega)\).
2. Generate \(U\sim\mathcal U(0,M)\), independently of \(X\).
3. Accept \(X\) if \(U\leq f(X)\).
4. Otherwise, reject \(X\).
5. Repeat until the requested sample count is reached.

The accepted samples follow the target density

\[
p(x)=\frac{f(x)}{\int_\Omega f(t)\,dt}.
\]

This method therefore supports unnormalized target functions.

## 3. Acceptance Probability

Let \(|\Omega|\) denote the length, area, or volume of the sampling domain.

The probability that a candidate is accepted is

\[
\boxed{
P_{\mathrm{accept}}
=
\frac{1}{M|\Omega|}
\int_\Omega f(x)\,dx
}.
\]

Equivalently,

\[
P_{\mathrm{accept}}
=
\frac{Z}{M|\Omega|}.
\]

The expected number of proposals required to obtain one accepted sample is

\[
\mathbb E[N_{\mathrm{proposals}}]
=
\frac{1}{P_{\mathrm{accept}}}.
\]

For \(N\) accepted samples, the expected number of proposals is

\[
\mathbb E[N_{\mathrm{total}}]
=
\frac{N}{P_{\mathrm{accept}}}.
\]

Consequently, tighter upper bounds \(M\) generally improve sampling efficiency.

## 4. One-Dimensional Rejection Sampling

Consider the density

\[
p(x)=2x,\qquad 0\leq x\leq1.
\]

Since

\[
0\leq p(x)\leq2,
\]

we use \(M=2\).

The theoretical expectation is

\[
\mathbb E[X]
=
\int_0^1 x(2x)\,dx
=
\frac23.
\]

### Python Example

```python
import numpy as np

from sampling_methods.rejection_sampling import sample_1d

rng = np.random.default_rng(42)

samples = sample_1d(
    pdf=lambda x: 2.0 * x,
    interval=[0.0, 1.0],
    nbr_samples=10000,
    M=2.0,
    rng=rng,
)

print("Empirical mean:", samples.mean())
print("Theoretical mean:", 2.0 / 3.0)
```

With a sufficiently large sample size, the empirical mean should approach \(2/3\).

## 5. Two-Dimensional Rejection Sampling

Consider the unnormalized Gaussian density

\[
f(x,y)
=
\exp\left(
-\frac{x^2+y^2}{2}
\right)
\]

on

\[
\Omega=[-3,3]\times[-3,3].
\]

Since

\[
0<f(x,y)\leq1,
\]

we choose \(M=1\).

### Python Example

```python
import numpy as np

from sampling_methods.rejection_sampling import sample_2d

def gaussian_2d(x):
    return np.exp(-0.5 * np.dot(x, x))

rng = np.random.default_rng(42)

samples = sample_2d(
    pdf=gaussian_2d,
    xy_min=[-3.0, -3.0],
    xy_max=[3.0, 3.0],
    M=1.0,
    nbr_samples=10000,
    rng=rng,
)

print("Empirical mean:", samples.mean(axis=0))
print("Empirical covariance:")
print(np.cov(samples, rowvar=False))
```

Because the distribution is symmetric around the origin,

\[
\mathbb E[X]\approx0,
\qquad
\mathbb E[Y]\approx0.
\]

The domain truncation makes the coordinate variances slightly smaller than those of an untruncated standard Gaussian.

## 6. Reproducibility

The implementation accepts a NumPy random generator:

```python
rng = np.random.default_rng(42)
```

Using the same seed reproduces the same sequence of accepted samples under the same NumPy implementation and numerical environment.

Unlike the original implementation, the functions do not modify NumPy's global random state.

## 7. Computational Complexity

Let \(N\) be the number of requested samples and let \(p_a\) denote the acceptance probability.

The expected number of proposals is

\[
\mathbb E[K]=\frac{N}{p_a}.
\]

If evaluating the target density costs \(C_f\), the expected computational cost is approximately

\[
O\left(\frac{N C_f}{p_a}\right).
\]

The implementation stores the accepted samples, requiring

\[
O(Nd)
\]

memory, where \(d\in\{1,2\}\) is the sampling dimension.

The number of proposals is limited by `n_max`.

If the algorithm fails to obtain the requested number of samples within this limit, it raises a `RuntimeError` instead of silently returning incomplete results.

## 8. Numerical Validation

The implementation is validated using:

- Sample count and output shape.
- Sampling-domain constraints.
- Reproducibility with fixed seeds.
- Statistical moments of known distributions.
- Invalid density and domain handling.
- Maximum-attempt failure handling.

For the density \(p(x)=2x\), a reproducible experiment with 10,000 accepted samples produced:

| Quantity | Value |
|---|---:|
| Empirical mean | 0.666095 |
| Theoretical mean | 0.666667 |
| Absolute error | \(5.719\times10^{-4}\) |

For the truncated two-dimensional Gaussian, the same sample count produced an empirical mean approximately equal to

\[
(-0.004676,\ 0.006498).
\]

These results are consistent with the theoretical properties of the sampled distributions.

## 9. Automated Tests

From the repository root, execute:

```bash
python -m pytest -v sampling_methods/rejection_sampling/test_rejection_sampling.py
```

The initial test suite contains 12 tests covering both sampling functions, reproducibility, statistical properties, input validation, and unsuccessful sampling attempts.

## 10. Reproducible Example

Run:

```bash
python -m sampling_methods.rejection_sampling.example
```

The example generates 10,000 samples from each target distribution and compares their empirical statistics with theoretical expectations.

## 11. Limitations

The current implementation:

- Supports one- and two-dimensional rectangular sampling domains.
- Requires a strictly positive envelope bound \(M\).
- Requires scalar-valued, nonnegative target functions.
- Checks the envelope bound at evaluated points but cannot guarantee it globally.
- Uses scalar proposal loops rather than vectorized sampling.
- Does not report acceptance rates directly.
- Raises an exception if the requested sample count is not reached before `n_max`.

Rejection sampling can become inefficient when the acceptance probability is small, particularly in higher dimensions.

## 12. Relationship to Other Numerical Methods

Rejection sampling is a Monte Carlo method for generating random variables from specified distributions.

It can support numerical integration, uncertainty quantification, stochastic simulation, and probabilistic numerical algorithms.

Although random sampling may also be used in optimization, rejection sampling is primarily a probabilistic sampling technique rather than an optimization algorithm.
