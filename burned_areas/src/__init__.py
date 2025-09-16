"""
Burned areas analysis module.

This package contains utilities for analyzing and visualizing burned area data,
including functions for creating figures from intermediate UMAP results.
"""

from .create_figures import (
    create_umap_figure,
    get_default_burn_severity_config,
    load_umap_results,
)
from .visualize import plot_umap_selected_labels

__all__ = [
    "load_umap_results",
    "get_default_burn_severity_config",
    "create_umap_figure",
    "plot_umap_selected_labels",
]
