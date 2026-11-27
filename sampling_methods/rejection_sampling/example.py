"""
Reproducible examples of 1D and 2D rejection sampling.

Run from the repository root:

    python -m sampling_methods.rejection_sampling.example
"""

import numpy as np

from .rejection_sampling import sample_1d, sample_2d


def density_1d(x):
    """Linear probability density on [0, 1]."""
    return 2.0 * x


def density_2d(x):
    """Unnormalized standard Gaussian density in two dimensions."""
    return np.exp(-0.5 * np.dot(x, x))


def main():
    """Run reproducible rejection sampling experiments."""

    print("=" * 60)
    print("ACCEPTANCE-REJECTION SAMPLING")
    print("=" * 60)

    seed = 42
    n_samples = 10_000

    # --------------------------------------------------
    # 1D rejection sampling
    # --------------------------------------------------

    samples_1d = sample_1d(
        pdf=density_1d,
        interval=[0.0, 1.0],
        nbr_samples=n_samples,
        M=2.0,
        rng=np.random.default_rng(seed),
    )

    empirical_mean_1d = np.mean(samples_1d)
    theoretical_mean_1d = 2.0 / 3.0

    print("\n1D REJECTION SAMPLING")
    print("-" * 60)

    print(f"Number of samples: {len(samples_1d)}")
    print(f"Empirical mean:    {empirical_mean_1d:.6f}")
    print(f"Theoretical mean:  {theoretical_mean_1d:.6f}")
    print(
        f"Absolute error:    "
        f"{abs(empirical_mean_1d - theoretical_mean_1d):.6e}"
    )

    # --------------------------------------------------
    # 2D rejection sampling
    # --------------------------------------------------

    samples_2d = sample_2d(
        pdf=density_2d,
        xy_min=[-3.0, -3.0],
        xy_max=[3.0, 3.0],
        M=1.0,
        nbr_samples=n_samples,
        rng=np.random.default_rng(seed),
    )

    empirical_mean_2d = np.mean(samples_2d, axis=0)
    empirical_covariance_2d = np.cov(samples_2d, rowvar=False)

    print("\n2D REJECTION SAMPLING")
    print("-" * 60)

    print(f"Number of samples: {len(samples_2d)}")
    print(f"Empirical mean:    {empirical_mean_2d}")
    print("Empirical covariance:")
    print(empirical_covariance_2d)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print("\nVALIDATION")
    print("-" * 60)

    valid_1d = (
        len(samples_1d) == n_samples
        and abs(empirical_mean_1d - theoretical_mean_1d) < 0.02
    )

    valid_2d = (
        samples_2d.shape == (n_samples, 2)
        and np.all(np.abs(empirical_mean_2d) < 0.05)
    )

    print(f"1D statistical validation: {valid_1d}")
    print(f"2D statistical validation: {valid_2d}")

    assert valid_1d
    assert valid_2d

    print("\nRejection sampling examples completed successfully.")


if __name__ == "__main__":
    main()
