"""Generate the interactive Rosenbrock benchmark notebook."""

from pathlib import Path

import nbformat as nbf


notebook = nbf.v4.new_notebook()

notebook.cells = [
    nbf.v4.new_markdown_cell(
        r"""
# Interactive Rosenbrock Optimization Benchmark

Explore the two-dimensional Rosenbrock-type function

$$
J_R(x,y)=(y-x^2)^2+(x-1)^2.
$$

This implementation uses a coefficient of **1**, rather than the
conventional Rosenbrock coefficient of 100.

The global minimizer is

$$
(x^*,y^*)=(1,1), \qquad J_R(x^*,y^*)=0.
$$

Use the interactive controls to explore the objective landscape,
its analytical derivatives, and local curvature.
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
    root / "optimization_benchmarks" / "rosenbrock" / "benchmark.py"
).exists():
    if root == root.parent:
        raise RuntimeError("Repository root not found.")
    root = root.parent

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from optimization_benchmarks.rosenbrock import (
    rosenbrock,
    rosenbrock_gradient,
    rosenbrock_hessian,
)
"""
    ),

    nbf.v4.new_code_cell(
        """
def show_rosenbrock(x_point=-1.2, y_point=1.0, view_range=2.0):

    # Include the global minimizer in every visualization.
    x = np.linspace(-view_range, view_range, 160)
    y = np.linspace(-view_range, view_range, 160)

    X, Y = np.meshgrid(x, y)
    Z = (Y - X**2)**2 + (X - 1.0)**2

    point = np.array([x_point, y_point])
    value = rosenbrock(point)
    gradient = rosenbrock_gradient(point)
    hessian = rosenbrock_hessian(point)

    fig = plt.figure(figsize=(15, 5.5))

    fig.suptitle(
        "Rosenbrock Optimization Landscape",
        fontsize=17,
        fontweight="bold",
    )

    # Contour visualization
    ax1 = fig.add_subplot(1, 2, 1)

    levels = np.geomspace(0.01, max(0.02, Z.max()), 25)

    contour = ax1.contourf(
        X, Y, Z,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    ax1.scatter(
        [1.0], [1.0],
        marker="*",
        s=200,
        color="red",
        label="Global minimizer",
        zorder=5,
    )

    ax1.scatter(
        [x_point], [y_point],
        marker="o",
        s=70,
        color="white",
        edgecolors="black",
        label="Selected point",
        zorder=6,
    )

    ax1.set_title("Contour Plot")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper left", fontsize=8)

    fig.colorbar(
        contour,
        ax=ax1,
        label="Objective value",
    )

    # 3D surface visualization
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax2.plot_surface(
        X, Y, Z,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )

    ax2.scatter(
        [1.0], [1.0], [0.0],
        color="red",
        s=80,
        marker="*",
        label="Global minimizer",
    )

    ax2.scatter(
        [x_point], [y_point], [value],
        color="black",
        s=45,
        label="Selected point",
    )

    ax2.set_title("3D Surface")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("J(x, y)")

    fig.tight_layout()
    plt.show()

    eigenvalues = np.linalg.eigvalsh(hessian)

    print("NUMERICAL SUMMARY")
    print("-" * 55)
    print(f"Selected point: ({x_point:.3f}, {y_point:.3f})")
    print(f"Objective value: {value:.6f}")
    print(f"Gradient norm:   {np.linalg.norm(gradient):.6e}")
    print(f"Hessian eigenvalues: {eigenvalues}")

    if np.all(eigenvalues > 0):
        print("Local curvature: positive definite Hessian")
    elif np.all(eigenvalues < 0):
        print("Local curvature: negative definite Hessian")
    else:
        print("Local curvature: indefinite or semidefinite Hessian")
"""
    ),

    nbf.v4.new_code_cell(
        """
widgets.interact(
    show_rosenbrock,
    x_point=widgets.FloatSlider(
        value=-1.2,
        min=-2.0,
        max=2.0,
        step=0.1,
        description="Point x",
        style={"description_width": "initial"},
    ),
    y_point=widgets.FloatSlider(
        value=1.0,
        min=-2.0,
        max=4.0,
        step=0.1,
        description="Point y",
        style={"description_width": "initial"},
    ),
    view_range=widgets.FloatSlider(
        value=2.0,
        min=1.5,
        max=4.0,
        step=0.5,
        description="View range",
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
    "optimization_benchmarks/notebooks/rosenbrock_demo.ipynb"
)

nbf.write(notebook, output)

print(f"Created: {output}")
