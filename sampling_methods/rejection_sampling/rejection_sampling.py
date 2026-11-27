"""
Acceptance-rejection sampling in one and two dimensions.

The target density may be unnormalized, but must be nonnegative
and bounded above by M throughout the sampling domain.
"""

import numpy as np


def _validate_common(nbr_samples, M, n_max):
    """Validate common rejection sampling parameters."""

    if isinstance(nbr_samples, (bool, np.bool_)) or not isinstance(
        nbr_samples, (int, np.integer)
    ):
        raise ValueError("nbr_samples must be an integer.")

    if nbr_samples <= 0:
        raise ValueError("nbr_samples must be positive.")

    if isinstance(n_max, (bool, np.bool_)) or not isinstance(
        n_max, (int, np.integer)
    ):
        raise ValueError("n_max must be an integer.")

    if n_max <= 0:
        raise ValueError("n_max must be positive.")

    if not np.isfinite(M) or M <= 0:
        raise ValueError("M must be positive and finite.")


def _evaluate_density(pdf, x, M):
    """Evaluate and validate the target density."""

    value = np.asarray(pdf(x))

    if value.shape != ():
        raise ValueError("pdf must return a scalar value.")

    if not np.isrealobj(value):
        raise ValueError("pdf must return a real-valued density.")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError("pdf returned a nonfinite value.")

    if value < 0 or value > M:
        raise ValueError(
            "pdf must satisfy 0 <= pdf(x) <= M."
        )

    return value


def sample_1d(
    pdf,
    interval,
    nbr_samples,
    M,
    n_max=1_000_000,
    rng=None,
):
    """
    Generate 1D samples using acceptance-rejection sampling.

    Parameters
    ----------
    pdf : callable
        Nonnegative target function, bounded above by M.
    interval : sequence of two floats
        Sampling interval [a, b], where a < b.
    nbr_samples : int
        Number of accepted samples requested.
    M : float
        Upper bound for pdf throughout the interval.
    n_max : int, optional
        Maximum number of sampling attempts.
    rng : numpy.random.Generator, optional
        Random generator. If omitted, a new generator is created.

    Returns
    -------
    ndarray, shape (nbr_samples,)
        Accepted samples in generation order.

    Raises
    ------
    ValueError
        If inputs or density evaluations are invalid.
    RuntimeError
        If the requested sample count is not reached.
    """
    _validate_common(nbr_samples, M, n_max)

    interval = np.asarray(interval, dtype=float)

    if interval.shape != (2,):
        raise ValueError("interval must contain exactly two values.")

    a, b = interval

    if not np.all(np.isfinite(interval)) or a >= b:
        raise ValueError("interval must satisfy finite a < b.")

    if not callable(pdf):
        raise ValueError("pdf must be callable.")

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator.")

    samples = []

    for _ in range(n_max):
        x = rng.uniform(a, b)

        density = _evaluate_density(pdf, x, M)

        if rng.uniform(0.0, M) <= density:
            samples.append(x)

            if len(samples) == nbr_samples:
                return np.asarray(samples)

    raise RuntimeError(
        f"Only {len(samples)} of {nbr_samples} requested "
        f"samples were accepted after {n_max} attempts."
    )


def sample_2d(
    pdf,
    xy_min,
    xy_max,
    M,
    nbr_samples,
    n_max=1_000_000,
    rng=None,
):
    """
    Generate 2D samples using acceptance-rejection sampling.

    Parameters
    ----------
    pdf : callable
        Target function accepting a two-component vector.
    xy_min : sequence of two floats
        Lower bounds [xmin, ymin].
    xy_max : sequence of two floats
        Upper bounds [xmax, ymax].
    M : float
        Upper bound for pdf throughout the domain.
    nbr_samples : int
        Number of accepted samples requested.
    n_max : int, optional
        Maximum number of sampling attempts.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    ndarray, shape (nbr_samples, 2)
        Accepted samples in generation order.

    Raises
    ------
    ValueError
        If inputs or density evaluations are invalid.
    RuntimeError
        If the requested sample count is not reached.
    """
    _validate_common(nbr_samples, M, n_max)

    xy_min = np.asarray(xy_min, dtype=float)
    xy_max = np.asarray(xy_max, dtype=float)

    if xy_min.shape != (2,) or xy_max.shape != (2,):
        raise ValueError(
            "xy_min and xy_max must contain two coordinates."
        )

    if not (
        np.all(np.isfinite(xy_min))
        and np.all(np.isfinite(xy_max))
    ):
        raise ValueError("Domain bounds must be finite.")

    if np.any(xy_min >= xy_max):
        raise ValueError(
            "Each lower bound must be smaller than its upper bound."
        )

    if not callable(pdf):
        raise ValueError("pdf must be callable.")

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator.")

    samples = []

    for _ in range(n_max):
        x = rng.uniform(xy_min, xy_max)

        density = _evaluate_density(pdf, x, M)

        if rng.uniform(0.0, M) <= density:
            samples.append(x.copy())

            if len(samples) == nbr_samples:
                return np.asarray(samples)

    raise RuntimeError(
        f"Only {len(samples)} of {nbr_samples} requested "
        f"samples were accepted after {n_max} attempts."
    )
