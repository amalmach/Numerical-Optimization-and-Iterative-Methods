"""Tests for one- and two-dimensional rejection sampling."""

import numpy as np
import pytest

from sampling_methods.rejection_sampling.rejection_sampling import (
    sample_1d,
    sample_2d,
)


def test_sample_1d_shape():
    """Return the requested number of samples."""
    samples = sample_1d(
        lambda x: 1.0,
        [0.0, 1.0],
        100,
        1.0,
        rng=np.random.default_rng(42),
    )
    assert samples.shape == (100,)


def test_sample_1d_bounds():
    """All samples must remain inside the interval."""
    samples = sample_1d(
        lambda x: 2.0 * x,
        [0.0, 1.0],
        1000,
        2.0,
        rng=np.random.default_rng(42),
    )
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_sample_1d_reproducibility():
    """Identical seeds produce identical samples."""
    a = sample_1d(
        lambda x: x,
        [0.0, 1.0],
        100,
        1.0,
        rng=np.random.default_rng(42),
    )
    b = sample_1d(
        lambda x: x,
        [0.0, 1.0],
        100,
        1.0,
        rng=np.random.default_rng(42),
    )
    assert np.array_equal(a, b)


def test_sample_1d_mean():
    """Check the mean for the density p(x) = 2x."""
    samples = sample_1d(
        lambda x: 2.0 * x,
        [0.0, 1.0],
        10000,
        2.0,
        rng=np.random.default_rng(42),
    )
    assert abs(samples.mean() - 2.0 / 3.0) < 0.02


def test_sample_2d_shape():
    """Return an array of shape (n, 2)."""
    samples = sample_2d(
        lambda x: 1.0,
        [-1.0, -1.0],
        [1.0, 1.0],
        100.0,
        100,
        rng=np.random.default_rng(42),
    )
    assert samples.shape == (100, 2)


def test_sample_2d_bounds():
    """All coordinates must stay within the domain."""
    samples = sample_2d(
        lambda x: np.exp(-0.5 * np.dot(x, x)),
        [-3.0, -3.0],
        [3.0, 3.0],
        1.0,
        1000,
        rng=np.random.default_rng(42),
    )
    assert np.all(samples >= -3.0)
    assert np.all(samples <= 3.0)


def test_sample_2d_reproducibility():
    """Identical seeds produce identical 2D samples."""
    kwargs = dict(
        pdf=lambda x: np.exp(-0.5 * np.dot(x, x)),
        xy_min=[-2.0, -2.0],
        xy_max=[2.0, 2.0],
        M=1.0,
        nbr_samples=100,
    )
    a = sample_2d(**kwargs, rng=np.random.default_rng(42))
    b = sample_2d(**kwargs, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_sample_2d_gaussian_mean():
    """A symmetric truncated Gaussian has mean near zero."""
    samples = sample_2d(
        lambda x: np.exp(-0.5 * np.dot(x, x)),
        [-3.0, -3.0],
        [3.0, 3.0],
        1.0,
        10000,
        rng=np.random.default_rng(42),
    )
    assert np.all(np.abs(samples.mean(axis=0)) < 0.05)


def test_invalid_sample_count():
    """Reject nonpositive sample counts."""
    with pytest.raises(ValueError):
        sample_1d(lambda x: 1.0, [0, 1], 0, 1.0)


def test_invalid_domain():
    """Reject incorrectly ordered domain bounds."""
    with pytest.raises(ValueError):
        sample_2d(
            lambda x: 1.0,
            [1.0, -1.0],
            [0.0, 1.0],
            1.0,
            10,
        )


def test_invalid_density():
    """Reject densities exceeding the supplied envelope."""
    with pytest.raises(ValueError):
        sample_1d(
            lambda x: 2.0,
            [0.0, 1.0],
            10,
            1.0,
        )


def test_maximum_attempts():
    """Raise an error if no samples can be accepted."""
    with pytest.raises(RuntimeError):
        sample_1d(
            lambda x: 0.0,
            [0.0, 1.0],
            10,
            1.0,
            n_max=20,
            rng=np.random.default_rng(42),
        )
