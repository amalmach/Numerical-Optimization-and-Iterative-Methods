# Sampling Methods

This directory contains Python implementations of numerical sampling algorithms used in Monte Carlo simulation, probability, and scientific computing.

The implementations emphasize mathematical clarity, reproducibility, input validation, and numerical experimentation.

## Available Methods

| Method | Description | Documentation |
|---|---|---|
| Acceptance–Rejection Sampling | Generate random samples from bounded, nonnegative target densities in one and two dimensions. | [Explore](rejection_sampling/README.md) |

## Acceptance–Rejection Sampling

Acceptance–rejection sampling generates random variables from a target probability density without requiring its normalization constant.

Given a nonnegative function \(f\) defined on a bounded domain \(\Omega\), assume that

\[
0 \leq f(x) \leq M.
\]

The algorithm generates candidate points uniformly over \(\Omega\) and accepts them with probability

\[
P(\text{accept}\mid X=x)=\frac{f(x)}{M}.
\]

The accepted samples follow the probability density

\[
p(x)=\frac{f(x)}{\int_\Omega f(t)\,dt}.
\]

The current implementation supports one-dimensional intervals and two-dimensional rectangular domains.

## Repository Structure

```text
sampling_methods/
├── __init__.py
├── README.md
├── rejection_sampling/
│   ├── __init__.py
│   ├── README.md
│   ├── rejection_sampling.py
│   ├── example.py
│   └── test_rejection_sampling.py
└── notebooks/
```

## Quick Start

Run the reproducible example from the repository root:

```bash
python -m sampling_methods.rejection_sampling.example
```

Run the automated tests:

```bash
python -m pytest -v sampling_methods/rejection_sampling/test_rejection_sampling.py
```

Import the sampling functions:

```python
import numpy as np

from sampling_methods.rejection_sampling import sample_1d, sample_2d

rng = np.random.default_rng(42)

samples = sample_1d(
    pdf=lambda x: 2.0 * x,
    interval=[0.0, 1.0],
    nbr_samples=1000,
    M=2.0,
    rng=rng,
)

print("Empirical mean:", samples.mean())
```

## Numerical Validation

The implementations are tested for:

- Correct sample counts and output dimensions.
- Sampling-domain constraints.
- Reproducibility with fixed random seeds.
- Agreement with theoretical statistical moments.
- Input validation and error handling.
- Explicit failure when the maximum number of attempts is reached.

The initial automated test suite contains **12 passing tests**.

## Applications

Sampling algorithms are useful in scientific computing, including Monte Carlo integration, stochastic simulation, uncertainty quantification, and probabilistic modeling.

Rejection sampling is especially useful when the target distribution is known up to a normalization constant and a suitable upper bound is available.

## Relationship to Other Methods

Sampling methods complement the optimization, direct linear algebra, and iterative numerical algorithms implemented elsewhere in this repository.

Unlike deterministic optimization algorithms, rejection sampling generates random variables from prescribed probability distributions.

## Documentation

For the mathematical derivation, implementation details, examples, and limitations, see the [Rejection Sampling README](rejection_sampling/README.md).
