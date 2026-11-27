"""Generate the interactive LU factorization notebook."""

from pathlib import Path

import nbformat as nbf


notebook = nbf.v4.new_notebook()

notebook.cells = [
    nbf.v4.new_markdown_cell(
        "# Interactive LU Factorization\n\n"
        "Explore LU factorization with partial pivoting:\n\n"
        "$$PA=LU$$\n\n"
        "The dashboard visualizes the matrix factors and "
        "verifies the accuracy of solving $Ax=b$."
    ),

    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import ipywidgets as widgets\n"
        "from IPython.display import display\n"
        "\n"
        "root = Path.cwd().resolve()\n"
        "while not (root / 'direct_methods' / 'lu_factorization' / 'lu.py').exists():\n"
        "    if root == root.parent:\n"
        "        raise RuntimeError('Repository root not found.')\n"
        "    root = root.parent\n"
        "sys.path.insert(0, str(root))\n"
        "\n"
        "from direct_methods.lu_factorization import lu_factorization, lu_solve"
    ),

    nbf.v4.new_code_cell(
        "def show_lu(n=4, seed=42, matrix_type='Random'):\n"
        "    rng = np.random.default_rng(seed)\n"
        "\n"
        "    if matrix_type == 'Random':\n"
        "        A = rng.standard_normal((n, n))\n"
        "    elif matrix_type == 'Symmetric positive definite':\n"
        "        B = rng.standard_normal((n, n))\n"
        "        A = B.T @ B + np.eye(n)\n"
        "    else:\n"
        "        A = rng.standard_normal((n, n))\n"
        "        A[0, 0] = 0.0\n"
        "        A[1, 0] = 2.0\n"
        "\n"
        "    b = A @ np.ones(n)\n"
        "\n"
        "    P, L, U = lu_factorization(A)\n"
        "    x = lu_solve(A, b)\n"
        "\n"
        "    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))\n"
        "    fig.suptitle('LU Factorization with Partial Pivoting',\n"
        "                 fontsize=17, fontweight='bold', y=1.04)\n"
        "\n"
        "    titles = ['Original matrix A', 'Permutation P',\n"
        "              'Lower triangular L', 'Upper triangular U']\n"
        "\n"
        "    for ax, matrix, title in zip(axes, [A, P, L, U], titles):\n"
        "        image = ax.imshow(matrix, cmap='coolwarm', aspect='equal')\n"
        "        ax.set_title(title, fontsize=11, pad=12)\n"
        "        ax.set_xticks(range(n))\n"
        "        ax.set_yticks(range(n))\n"
        "        ax.tick_params(labelsize=8)\n"
        "\n"
        "        for i in range(n):\n"
        "            for j in range(n):\n"
        "                ax.text(j, i, f'{matrix[i,j]:.2f}',\n"
        "                        ha='center', va='center', fontsize=7)\n"
        "\n"
        "        fig.colorbar(image, ax=ax, fraction=0.046,\n"
        "                     pad=0.04, shrink=0.75)\n"
        "\n"
        "    fig.tight_layout(w_pad=2.0)\n"
        "    plt.show()\n"
        "\n"
        "    factor_residual = np.linalg.norm(P @ A - L @ U)\n"
        "    solve_residual = np.linalg.norm(A @ x - b)\n"
        "    relative_residual = solve_residual / (\n"
        "        np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b)\n"
        "    )\n"
        "\n"
        "    print(f'Factorization residual: {factor_residual:.3e}')\n"
        "    print(f'Solution residual:      {solve_residual:.3e}')\n"
        "    print(f'Relative residual:      {relative_residual:.3e}')\n"
        "    print(f'Solution error:         {np.linalg.norm(x-np.ones(n)):.3e}')"
    ),

    nbf.v4.new_code_cell(
        "widgets.interact(\n"
        "    show_lu,\n"
        "    n=widgets.IntSlider(value=4, min=2, max=8, step=1,\n"
        "                        description='Dimension'),\n"
        "    seed=widgets.IntSlider(value=42, min=0, max=100, step=1,\n"
        "                           description='Seed'),\n"
        "    matrix_type=widgets.Dropdown(\n"
        "        options=['Random', 'Symmetric positive definite',\n"
        "                 'Zero initial pivot'],\n"
        "        description='Matrix type',\n"
        "    ),\n"
        ");"
    ),
]

notebook.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

output = Path("direct_methods/notebooks/lu_demo.ipynb")
nbf.write(notebook, output)

print(f"Created: {output}")
