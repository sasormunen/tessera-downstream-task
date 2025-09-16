"""
Visualization functions for burned area analysis.

This module contains plotting functions for UMAP and other dimensionality reduction
visualizations used in burned area analysis.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D


def plot_umap_selected_labels(
    umap_2d: np.ndarray,
    labels: np.ndarray,
    valid_values: Optional[List] = None,
    label_map: Optional[Dict] = None,
    color_map: Optional[Dict] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    point_size: int = 6,
    alpha: float = 0.4,
    legend_loc: str = "upper right",
    manual_annotations: Optional[List[Tuple[float, float, str]]] = None,
    legend: bool = True,
) -> None:
    """
    Plot precomputed UMAP embeddings with filtered classes and custom label/color maps.

    Parameters:
    -----------
    umap_2d : np.ndarray
        (N, 2) array of UMAP-reduced embeddings
    labels : np.ndarray
        (N,) array of class labels
    valid_values : list, optional
        List of label values to include (others excluded)
    label_map : dict, optional
        Dictionary mapping label values to display names
    color_map : dict, optional
        Dictionary mapping label values to RGB or hex colors
    title : str, optional
        Optional plot title
    figsize : tuple, optional
        Plot size (width, height) in inches (default: (8, 6))
    point_size : int, optional
        Scatter point size (default: 6)
    alpha : float, optional
        Point transparency (default: 0.4)
    legend_loc : str, optional
        Where to place the legend (default: "upper right")
    manual_annotations : list, optional
        List of (x, y, text) annotations to add to the plot
    legend : bool, optional
        Whether to show the legend (default: True)
    """
    df = pd.DataFrame(umap_2d, columns=["UMAP 1", "UMAP 2"])
    df["label"] = labels

    if valid_values is not None:
        df = df[df["label"].isin(valid_values)]

    unique_labels = sorted(df["label"].unique())

    # Set color palette
    if color_map:
        # Convert any hex strings to RGB
        color_map = {
            val: (
                to_rgb(color_map[val])
                if isinstance(color_map[val], str)
                else color_map[val]
            )
            for val in unique_labels
            if val in color_map
        }
    else:
        palette = sns.color_palette("colorblind", len(unique_labels))
        color_map = dict(zip(unique_labels, palette))

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(
        data=df,
        x="UMAP 1",
        y="UMAP 2",
        hue="label",
        palette=color_map,
        s=point_size,
        alpha=alpha,
        legend=False,
    )

    # Custom legend
    if legend:
        legend_elements = []
        for label in unique_labels:
            label_str = label_map.get(label, str(label)) if label_map else str(label)
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label_str,
                    markerfacecolor=color_map[label],
                    markersize=6,
                )
            )
        ax.legend(handles=legend_elements, title="Class", loc=legend_loc, frameon=True)

    # Manual cluster annotations
    if manual_annotations:
        for x, y, text in manual_annotations:
            ax.text(
                x,
                y,
                text,
                fontsize=10,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )

    # Styling
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )
    plt.grid(False)
    plt.tight_layout()
    plt.title(title or "", fontsize=12)
    plt.show()


def plot_label_efficiency_comparison(ratios, combined_scores_dict, sample_counts):
    """
    Plots a publication-quality comparison of model performance against the
    ratio of training data used, including a second x-axis for absolute sample counts.

    This function creates a dual-axis plot showing both the fraction of training data
    used (bottom x-axis) and the absolute number of training samples (top x-axis).
    It's designed for comparing multiple approaches or models on the same plot.

    Parameters
    ----------
    ratios : list of float
        List of ratios (0.0 to 1.0) representing the fraction of training data used.
        Should be in ascending order.
    combined_scores_dict : dict
        Dictionary where keys are approach/model names (strings) and values are
        lists of F1-scores corresponding to each ratio in `ratios`.
        Example: {"TESSERA": [0.8, 0.85, 0.9], "GSE": [0.75, 0.82, 0.88]}
    sample_counts : list of int
        List of absolute sample counts corresponding to each ratio in `ratios`.
        Should have the same length as `ratios`.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Examples
    --------
    >>> ratios = [0.001, 0.01, 0.1, 1.0]
    >>> sample_counts = [100, 1000, 10000, 100000]
    >>> scores = {
    ...     "TESSERA": [0.8, 0.85, 0.9, 0.95],
    ...     "GSE": [0.75, 0.82, 0.88, 0.93]
    ... }
    >>> plot_label_efficiency_comparison(ratios, scores, sample_counts)
    """

    # --- Helper function for formatting numbers ---
    def format_number(n):
        """Format numbers for display on the top x-axis."""
        if n < 1000:
            return str(n)
        elif n < 10000:
            return f"{n/1000:.1f}k"
        elif n < 1000000:
            return f"{int(n/1000)}k"
        else:
            return f"{n/1000000:.1f}M"

    # --- Plot Styling ---
    plt.rcParams.update({"font.size": 14, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define a color cycle for the lines
    colors = ["#004488", "#D85D1A", "#1E8449"]  # A nice blue, orange, green
    color_cycle = iter(colors)

    # --- Plotting the Data ---
    for approach_name, scores in combined_scores_dict.items():
        ax.plot(
            ratios,
            scores,
            marker="o",
            linestyle="-",
            label=approach_name,
            markersize=8,
            zorder=10,
            color=next(color_cycle),
            linewidth=2.5,
        )

    # --- Aesthetics ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    # --- Bottom X-axis (Percentages) ---
    ax.set_xlabel("Fraction of Training Data Used", fontsize=18, labelpad=10)
    ax.set_ylabel("Average F1-Score", fontsize=18, labelpad=10)
    ax.tick_params(axis="x", which="major", labelsize=14, pad=7, width=1.5)
    ax.tick_params(axis="y", which="major", labelsize=14, width=1.5)
    ax.set_xscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r*100:.3f}%" for r in ratios], rotation=45, ha="right")

    # Ensure y-axis starts at 0
    ax.set_ylim(0.3, 1.00)

    # --- Top X-axis (Sample Counts) ---
    ax2 = ax.twiny()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ratios)
    # Use the helper function to format the labels
    ax2.set_xticklabels(
        [format_number(count) for count in sample_counts], rotation=45, ha="left"
    )
    ax2.set_xlabel("Number of Training Samples", fontsize=18, labelpad=10)
    ax2.tick_params(axis="x", which="major", labelsize=14, pad=7, width=1.5)

    # --- Legend ---
    ax.legend(loc="lower right", fontsize=14, frameon=False)

    plt.tight_layout()
    plt.show()


def plot_comparison_maps(gt_A, pred_A, gt_B, pred_B):
    """
    Plots a 1x4 wide image comparing ground truth vs. prediction for both areas.

    This function creates a side-by-side comparison of ground truth and prediction
    maps for two areas (A and B) in a single row layout. It uses a custom colormap
    to properly display binary classification results with NoData values.

    Parameters
    ----------
    gt_A : np.ndarray
        2D array containing ground truth labels for Area A
    pred_A : np.ndarray
        2D array containing prediction labels for Area A
    gt_B : np.ndarray
        2D array containing ground truth labels for Area B
    pred_B : np.ndarray
        2D array containing prediction labels for Area B

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Examples
    --------
    >>> # Assuming you have ground truth and prediction maps for two areas
    >>> plot_comparison_maps(gt_map_A, pred_map_A, gt_map_B, pred_map_B)
    """
    print("Generating 1x4 comparison map...")

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # --- Correct Color Mapping ---
    # Unburned (0) -> Blue, Burned (1) -> Red, NoData/Other (-1) -> Gray
    cmap_dict = {
        0: "#1f77b4",  # Unburned -> Blue
        1: "#d62728",  # Burned -> Red
        -1: "#cccccc",  # NoData/Other -> Gray
    }
    codes = sorted(cmap_dict.keys())
    colors = [cmap_dict[c] for c in codes]
    cmap = ListedColormap(colors)
    boundaries = [c - 0.5 for c in codes] + [codes[-1] + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)

    # Plot Area A Ground Truth
    axes[0].imshow(gt_A, cmap=cmap, norm=norm)
    axes[0].set_title("Area A: Ground Truth", fontsize=18)
    axes[0].axis("off")

    # Plot Area A Prediction
    axes[1].imshow(pred_A, cmap=cmap, norm=norm)
    axes[1].set_title("Area A: Prediction", fontsize=18)
    axes[1].axis("off")

    # Plot Area B Ground Truth
    axes[2].imshow(gt_B, cmap=cmap, norm=norm)
    axes[2].set_title("Area B: Ground Truth", fontsize=18)
    axes[2].axis("off")

    # Plot Area B Prediction
    axes[3].imshow(pred_B, cmap=cmap, norm=norm)
    axes[3].set_title("Area B: Prediction", fontsize=18)
    axes[3].axis("off")

    legend_patches = [
        mpatches.Patch(color=cmap_dict[0], label="Unburned"),
        mpatches.Patch(color=cmap_dict[1], label="Burned"),
        mpatches.Patch(color=cmap_dict[-1], label="NoData / Other"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        bbox_to_anchor=(0.5, 0.02),
        fontsize=16,
    )

    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.95, bottom=0.1)
    plt.show()
