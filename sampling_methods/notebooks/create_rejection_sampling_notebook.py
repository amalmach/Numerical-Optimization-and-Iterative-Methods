"""Generate the interactive rejection sampling notebook."""

from pathlib import Path

import nbformat as nbf


notebook = nbf.v4.new_notebook()

notebook.cells = [
    nbf.v4.new_markdown_cell(
        r"""
# Interactive Acceptance–Rejection Sampling

Explore one- and two-dimensional rejection sampling.

A candidate $X$ is accepted when

$$
U \leq f(X), \qquad U \sim \mathcal{U}(0,M).
$$

The accepted samples follow a distribution proportional to
the target function $f$.

**Experiments**

- **1D:** Linear density $p(x)=2x$ on $[0,1]$.
- **2D:** Gaussian-shaped density on $[-3,3]^2$.

Change the sample count and random seed to explore
the statistical behavior of both algorithms.
"""
    ),

    nbf.v4.new_code_cell(
        """
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

root = Path.cwd().resolve()

while not (
    root / "sampling_methods" / "rejection_sampling"
    / "rejection_sampling.py"
).exists():
    if root == root.parent:
        raise RuntimeError("Repository root not found.")
    root = root.parent

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from sampling_methods.rejection_sampling import (
    sample_1d,
    sample_2d,
)
"""
    ),

    nbf.v4.new_code_cell(
        """
def density_1d(x):
    return 2.0 * x


def density_2d(x):
    return np.exp(-0.5 * np.dot(x, x))


def show_sampling(n_samples=1000, seed=42):

    rng_1d = np.random.default_rng(seed)
    rng_2d = np.random.default_rng(seed)

    samples_1d = sample_1d(
        pdf=density_1d,
        interval=[0.0, 1.0],
        nbr_samples=n_samples,
        M=2.0,
        rng=rng_1d,
    )

    samples_2d = sample_2d(
        pdf=density_2d,
        xy_min=[-3.0, -3.0],
        xy_max=[3.0, 3.0],
        M=1.0,
        nbr_samples=n_samples,
        rng=rng_2d,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fig.suptitle(
        "Acceptance–Rejection Sampling",
        fontsize=16,
        fontweight="bold",
    )

    # 1D: Empirical histogram and theoretical density
    x = np.linspace(0.0, 1.0, 300)

    axes[0].hist(
        samples_1d,
        bins=30,
        density=True,
        alpha=0.65,
        label="Accepted samples",
    )

    axes[0].plot(
        x,
        density_1d(x),
        linewidth=2.5,
        label="Theoretical density",
    )

    axes[0].set_title("1D Rejection Sampling")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Probability density")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # 2D: Gaussian density contours and accepted samples
    grid = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(grid, grid)

    Z = np.exp(-0.5 * (X**2 + Y**2))

    contour = axes[1].contourf(
        X,
        Y,
        Z,
        levels=15,
        cmap="viridis",
        alpha=0.7,
    )

    axes[1].scatter(
        samples_2d[:, 0],
        samples_2d[:, 1],
        s=8,
        color="white",
        alpha=0.5,
        edgecolors="none",
    )

    axes[1].set_title("2D Gaussian Rejection Sampling")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect("equal")

    fig.colorbar(
        contour,
        ax=axes[1],
        label="Unnormalized target density",
    )

    fig.tight_layout()
    plt.show()

    # Numerical statistics
    print("NUMERICAL SUMMARY")
    print("-" * 50)

    print(f"Accepted samples: {n_samples}")

    print(
        f"1D empirical mean:   "
        f"{samples_1d.mean():.6f}"
    )

    print(
        f"1D theoretical mean: "
        f"{2.0 / 3.0:.6f}"
    )

    print(
        f"2D empirical mean:   "
        f"{samples_2d.mean(axis=0)}"
    )

    print("2D empirical covariance:")
    print(np.cov(samples_2d, rowvar=False))
"""
    ),

    nbf.v4.new_code_cell(
        """
widgets.interact(
    show_sampling,
    n_samples=widgets.IntSlider(
        value=1000,
        min=200,
        max=5000,
        step=200,
        description="Samples",
        style={"description_width": "initial"},
    ),
    seed=widgets.IntSlider(
        value=42,
        min=0,
        max=100,
        step=1,
        description="Random seed",
        style={"description_width": "initial"},
    ),
);
"""
    ),
]

notebook.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

output = Path(
    "sampling_methods/notebooks/rejection_sampling_demo.ipynb"
)

nbf.write(notebook, output)

print(f"Created: {output}")
