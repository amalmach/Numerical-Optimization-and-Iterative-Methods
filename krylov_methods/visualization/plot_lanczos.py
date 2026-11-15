#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lanczos Visualization
=====================

Scientific visualization utilities for
symmetric Lanczos tridiagonalization.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_ritz_values(
    A,
    lanczos_result,
    figsize=(9, 5),
    save_path=None,
):
    """
    Compare Ritz values with eigenvalues of A.

    Parameters
    ----------
    A : ndarray
        Symmetric matrix.

    lanczos_result : dict
        Output of symmetric_lanczos().

    figsize : tuple
        Figure dimensions.

    save_path : str or Path, optional
        Output figure filename.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    A = np.asarray(A, dtype=float)

    eigenvalues = np.linalg.eigvalsh(A)

    ritz_values = lanczos_result["ritz_values"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        eigenvalues,
        np.zeros_like(eigenvalues),
        marker="|",
        s=250,
        linewidths=2,
        label="Eigenvalues of A",
    )

    ax.scatter(
        ritz_values,
        np.ones_like(ritz_values),
        marker="x",
        s=90,
        linewidths=2,
        label="Ritz values",
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels([
        "Eigenvalues",
        "Ritz values",
    ])

    ax.set_xlabel("Eigenvalue")

    ax.set_title(
        "Symmetric Lanczos — Ritz Value Approximation"
    )

    ax.grid(
        True,
        axis="x",
        linestyle="--",
        alpha=0.5,
    )

    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax
