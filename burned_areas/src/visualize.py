"""
Visualization functions for burned area analysis.

This module contains plotting functions for UMAP and other dimensionality reduction
visualizations used in burned area analysis.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
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
