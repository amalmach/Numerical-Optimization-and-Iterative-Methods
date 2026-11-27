"""
Generate the interactive Numerical Operations Jupyter notebook.

Run from the repository root:

    python scientific_utilities/notebooks/create_numerical_operations_notebook.py
"""

from pathlib import Path
import textwrap

import nbformat as nbf


def source(text):
    """Remove common indentation from notebook cell source."""
    return textwrap.dedent(text).strip()


def main():
    notebook = nbf.v4.new_notebook()

    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    notebook["metadata"]["language_info"] = {"name": "python"}

    cells = []

    # ---------------------------------------------------------
    # Cell 1 — Introduction
    # ---------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(source(r"""
    # Interactive Numerical Operations Dashboard

    This notebook demonstrates elementary numerical operations
    implemented in the **Scientific Utilities** package.

    The dashboard provides interactive controls for exploring:

    - Matrix multiplication
    - Matrix-vector multiplication
    - Euclidean vector norms
    - Dot products
    - Outer products
    - Matrix transposition
    - Nearest-value search
    - Nested-vector conversion

    The numerical results are computed using the repository's
    implementations and compared against NumPy.

    **Instructions:** Run all cells, then adjust the sliders to
    observe how the numerical results and visualizations change.
    """)))

    # ---------------------------------------------------------
    # Cell 2 — Imports
    # ---------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(source("""
    import sys
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets

    from IPython.display import display

    # Locate the repository root when running from JupyterLab.
    current = Path.cwd().resolve()

    candidates = [current, *current.parents]

    root = next(
        (
            path for path in candidates
            if (path / "scientific_utilities" / "numerical_operations"
                / "operations.py").exists()
        ),
        None,
    )

    if root is None:
        raise RuntimeError(
            "Could not locate the repository root. "
            "Open JupyterLab from the repository directory."
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scientific_utilities.numerical_operations import (
        closest,
        norm_two,
        AxB,
        matvec,
        scalar,
        vecvec,
        matT,
        vec_asarray,
    )

    print("Numerical Operations package imported successfully.")
    """)))

    # ---------------------------------------------------------
    # Cell 3 — Interactive matrix multiplication
    # ---------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(source("""
    matrix_entries = {}

    for matrix_name in ("A", "B"):
        for i in range(2):
            for j in range(2):
                matrix_entries[f"{matrix_name}{i}{j}"] = (
                    widgets.FloatSlider(
                        value=float(1 + i * 2 + j)
                        if matrix_name == "A"
                        else float(5 + i * 2 + j),
                        min=-5.0,
                        max=10.0,
                        step=0.5,
                        description=f"{matrix_name}[{i},{j}]",
                        continuous_update=False,
                        style={"description_width": "initial"},
                    )
                )

    matrix_output = widgets.Output()

    def update_matrix_dashboard(*_):
        A = np.array([
            [matrix_entries[f"A{i}{j}"].value for j in range(2)]
            for i in range(2)
        ])

        B = np.array([
            [matrix_entries[f"B{i}{j}"].value for j in range(2)]
            for i in range(2)
        ])

        C = AxB(A, B)
        reference = A @ B

        with matrix_output:
            matrix_output.clear_output(wait=True)

            print("Matrix A:\\\\n", A)
            print("\\\\nMatrix B:\\\\n", B)
            print("\\\\nEducational implementation A @ B:\\\\n", C)
            print("\\\\nNumPy reference:\\\\n", reference)
            print("\\\\nMaximum absolute error:",
                  np.max(np.abs(C - reference)))

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            for ax, matrix, title in zip(
                axes, (A, B, C), ("Matrix A", "Matrix B", "Product AB")
            ):
                ax.imshow(matrix, cmap="coolwarm", vmin=-50, vmax=50)

                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        ax.text(
                            j, i, f"{matrix[i, j]:.1f}",
                            ha="center", va="center",
                            color="black",
                        )

                ax.set_title(title)
                ax.set_xticks(range(matrix.shape[1]))
                ax.set_yticks(range(matrix.shape[0]))

            fig.suptitle("Interactive Matrix Multiplication")
            plt.tight_layout()
            plt.show()

    for control in matrix_entries.values():
        control.observe(update_matrix_dashboard, names="value")

    display(
        widgets.HTML("<h3>1. Matrix Multiplication</h3>"),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<b>Matrix A</b>"),
                *(matrix_entries[f"A{i}{j}"]
                  for i in range(2) for j in range(2)),
            ]),
            widgets.VBox([
                widgets.HTML("<b>Matrix B</b>"),
                *(matrix_entries[f"B{i}{j}"]
                  for i in range(2) for j in range(2)),
            ]),
        ]),
        matrix_output,
    )

    update_matrix_dashboard()
    """)))

    # ---------------------------------------------------------
    # Cell 4 — Interactive vector operations
    # ---------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(source("""
    vector_controls = {
        name: widgets.FloatSlider(
            value=value,
            min=-5.0,
            max=5.0,
            step=0.25,
            description=name,
            continuous_update=False,
        )
        for name, value in {
            "u1": 1.0,
            "u2": 2.0,
            "v1": 3.0,
            "v2": 4.0,
        }.items()
    }

    vector_output = widgets.Output()

    def update_vector_dashboard(*_):
        u = np.array([
            vector_controls["u1"].value,
            vector_controls["u2"].value,
        ])

        v = np.array([
            vector_controls["v1"].value,
            vector_controls["v2"].value,
        ])

        dot = scalar(u, v)
        outer = vecvec(u, v)
        norm_u = norm_two(u)
        norm_v = norm_two(v)

        with vector_output:
            vector_output.clear_output(wait=True)

            print("Vector u:", u)
            print("Vector v:", v)
            print(f"||u||₂ = {norm_u:.6f}")
            print(f"||v||₂ = {norm_v:.6f}")
            print(f"Dot product u·v = {dot:.6f}")
            print("Outer product:\\\\n", outer)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))

            axes[0].quiver(
                0, 0, u[0], u[1],
                angles="xy", scale_units="xy", scale=1,
                color="tab:blue", label="u",
            )

            axes[0].quiver(
                0, 0, v[0], v[1],
                angles="xy", scale_units="xy", scale=1,
                color="tab:orange", label="v",
            )

            axes[0].set_xlim(-6, 6)
            axes[0].set_ylim(-6, 6)
            axes[0].set_aspect("equal")
            axes[0].axhline(0, color="gray", linewidth=0.5)
            axes[0].axvline(0, color="gray", linewidth=0.5)
            axes[0].grid(alpha=0.3)
            axes[0].legend()
            axes[0].set_title("Vectors in R²")

            axes[1].imshow(outer, cmap="viridis")

            for i in range(outer.shape[0]):
                for j in range(outer.shape[1]):
                    axes[1].text(
                        j, i, f"{outer[i, j]:.2f}",
                        ha="center", va="center",
                        color="white",
                    )

            axes[1].set_title("Outer Product")
            axes[1].set_xticks(range(2))
            axes[1].set_yticks(range(2))

            plt.tight_layout()
            plt.show()

    for control in vector_controls.values():
        control.observe(update_vector_dashboard, names="value")

    display(
        widgets.HTML("<h3>2. Vector Operations</h3>"),
        widgets.HBox([
            widgets.VBox(list(vector_controls.values())),
        ]),
        vector_output,
    )

    update_vector_dashboard()
    """)))

    # ---------------------------------------------------------
    # Cell 5 — Matrix-vector multiplication
    # ---------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(source("""
    matrix_vector_output = widgets.Output()

    def update_matvec_dashboard(*_):
        A = np.array([
            [matrix_entries[f"A{i}{j}"].value for j in range(2)]
            for i in range(2)
        ])

        u = np.array([
            vector_controls["u1"].value,
            vector_controls["u2"].value,
        ])

        result = matvec(A, u)

        with matrix_vector_output:
            matrix_vector_output.clear_output(wait=True)

            print("Matrix A:\\\\n", A)
            print("Vector u:", u)
            print("Matrix-vector product Au:", result)
            print("NumPy reference:", A @ u)

            fig, ax = plt.subplots(figsize=(6, 5))

            ax.quiver(
                0, 0, u[0], u[1],
                angles="xy", scale_units="xy", scale=1,
                color="tab:blue", label="u",
            )

            ax.quiver(
                0, 0, result[0], result[1],
                angles="xy", scale_units="xy", scale=1,
                color="tab:red", label="Au",
            )

            limit = max(
                6.0,
                np.max(np.abs(u)) * 1.2,
                np.max(np.abs(result)) * 1.2,
            )

            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.grid(alpha=0.3)
            ax.legend()
            ax.set_title("Matrix-Vector Transformation")

            plt.tight_layout()
            plt.show()

    for control in [
        *matrix_entries.values(),
        vector_controls["u1"],
        vector_controls["u2"],
    ]:
        control.observe(update_matvec_dashboard, names="value")

    display(
        widgets.HTML("<h3>3. Matrix-Vector Multiplication</h3>"),
        widgets.HTML(
            "This visualization uses Matrix A and vector u "
            "from the controls above."
        ),
        matrix_vector_output,
    )

    update_matvec_dashboard()
    """)))

    # ---------------------------------------------------------
    # Cell 6 — Additional utilities
    # ---------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(source("""
    print("4. Additional Numerical Utilities")
    print("=" * 45)

    A = np.array([[1.0, 2.0], [3.0, 4.0]])

    print("\\\\nMatrix transpose:")
    print(matT(A))

    print("\\\\nNearest value to 0.7:")
    print(closest([0.1, 0.4, 0.8], 0.7))

    print("\\\\nNested-vector conversion:")
    print(vec_asarray([[1, 2], [3, 4]]))

    print("\\\\nAll eight numerical operations are available.")
    """)))

    notebook["cells"] = cells

    output_path = (
        Path(__file__).resolve().parent
        / "numerical_operations_demo.ipynb"
    )

    nbf.write(notebook, output_path)

    print(f"Notebook generated: {output_path}")
    print(f"Number of cells: {len(cells)}")


if __name__ == "__main__":
    main()
