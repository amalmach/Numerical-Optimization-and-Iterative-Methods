#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convergence Visualization
=========================

Scientific visualization utilities for comparing
iterative linear solvers.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(
    results,
    tol=None,
    title="Convergence of Iterative Linear Solvers",
    figsize=(9, 6),
    save_path=None,
    xlim=None,
):
    """
    Plot residual convergence for multiple iterative solvers.

    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm names to result dictionaries.

        Each result must contain:
            residual_norms

    tol : float, optional
        Convergence tolerance displayed as a horizontal line.

    title : str
        Figure title.

    figsize : tuple
        Figure dimensions.

    save_path : str or Path, optional
        Save the figure when a path is provided.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    fig, ax = plt.subplots(figsize=figsize)

    for name, result in results.items():

        residuals = np.asarray(
            result["residual_norms"],
            dtype=float,
        )

        iterations = np.arange(len(residuals))

        # Avoid log(0) when convergence reaches machine precision.
        # Display exact-zero residuals at a meaningful plotting floor.
        # The original residual data remain unchanged.
        plot_floor = max(
           tol * 1e-2 if tol is not None else 1e-14,
           1e-16,
        )

        residuals_plot = np.maximum(
           residuals,
           plot_floor,
        )

        ax.semilogy(
            iterations,
            residuals_plot,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=name,
        )

    if tol is not None:
        ax.axhline(
            tol,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Tolerance",
        )

    ax.set_xlabel(
        "Iteration $k$",
        fontsize=12,
    )

    ax.set_ylabel(
        r"Residual norm $\|b-Ax_k\|_2$",
        fontsize=12,
    )

    ax.set_title(
        title,
        fontsize=14,
    )

    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5,
    )

    ax.legend()

    # ============================================================
    # Automatic logarithmic axis limits
    # ============================================================

    all_residuals = np.concatenate([
        np.asarray(
            result["residual_norms"],
            dtype=float,
        )
        for result in results.values()
    ])

    # Keep only finite nonnegative residuals.
    valid_residuals = all_residuals[
        np.isfinite(all_residuals)
        & (all_residuals >= 0)
    ]

    if valid_residuals.size > 0:

        max_residual = np.max(valid_residuals)

        # Provide sufficient space above the largest residual.
        upper_limit = max(
            max_residual * 2.0,
            plot_floor * 10.0,
        )

        # Avoid clipping residuals close to convergence.
        lower_limit = plot_floor * 0.5

        ax.set_ylim(
            bottom=lower_limit,
            top=upper_limit,
        )

    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax


def plot_krylov_comparison(
    results,
    tol=1e-8,
    title="Convergence of Krylov Subspace Methods",
    figsize=(9, 6),
    save_path=None,
):
    """
    Compare the convergence of Krylov solvers.

    Excludes Steepest Descent from the comparison
    to highlight the convergence behavior of
    CG, GMRES, and BiCGSTAB.
    """

    krylov_names = [
        "Conjugate Gradient",
        "GMRES",
        "BiCGSTAB",
    ]

    krylov_results = {
        name: results[name]
        for name in krylov_names
        if name in results
    }

    if not krylov_results:
        raise ValueError(
            "No supported Krylov solver results were provided."
        )

    max_iterations = max(
        result["iterations"]
        for result in krylov_results.values()
    )

    fig, ax = plot_convergence(
        results=krylov_results,
        tol=tol,
        title=title,
        figsize=figsize,
        save_path=save_path,
        xlim=(0, max_iterations),
    )

    return fig, ax
