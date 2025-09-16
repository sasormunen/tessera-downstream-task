"""
Module for creating burned area figures from intermediate results.

This module provides functions to load pre-computed UMAP and PCA results
and create publication-ready figures for burned area analysis.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

import visualize

# Local constants for this project
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_umap_results(
    results_path: Union[str, Path], filename: str = "umap_results_burn_scar_1.npz"
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load UMAP results from saved .npz file.

    Parameters:
    -----------
    results_path : str or Path
        Path to the directory containing the results files
    filename : str, optional
        Name of the UMAP results file (default: "umap_results_burn_scar_1.npz")

    Returns:
    --------
    umap_coords : np.ndarray
        2D array of UMAP coordinates (n_samples, 2)
    labels : np.ndarray
        1D array of severity labels (n_samples,)
    """
    filepath = Path(results_path) / filename
    data = np.load(filepath)

    umap_coords = data["umap_coords"]
    labels = data["labels"]

    return umap_coords, labels


def get_default_burn_severity_config() -> Dict:
    """
    Get default configuration for burn severity visualization.

    Returns:
    --------
    config : dict
        Dictionary containing valid_values, label_map, color_map, and manual_annotations
    """
    return {
        "valid_values": [0, 2, 3, 4],
        "label_map": {0: "Unburned", 2: "Low", 3: "Moderate", 4: "High"},
        "color_map": {
            0: "#2166ac",  # Blue
            2: "#66bd63",  # Green
            3: "#fdae61",  # Orange
            4: "#d73027",  # Red
        },
        "manual_annotations": [
            (-6, -1.5, "June Fire"),
            (0, -5.5, "August Fire"),
            (6, 7, "Unburned"),
        ],
    }


def create_umap_figure(
    results_path: Union[str, Path],
    umap_filename: str = "umap_results_burn_scar_1.npz",
    config: Optional[Dict] = None,
    point_size: int = 10,
    title: str = "",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """
    Create and optionally save a UMAP figure from intermediate results.

    Parameters:
    -----------
    results_path : str or Path
        Path to the directory containing the results files
    umap_filename : str, optional
        Name of the UMAP results file
    config : dict, optional
        Configuration dictionary with valid_values, label_map, color_map,
        and manual_annotations.
        If None, uses default burn severity configuration.
    point_size : int, optional
        Size of scatter plot points (default: 10)
    title : str, optional
        Title for the plot (default: "")
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (8, 6))
    save_path : str or Path, optional
        Path to save the figure. If None, only displays the plot.
    dpi : int, optional
        Resolution for saved figure (default: 300)
    """
    # Load UMAP results
    umap_coords, labels = load_umap_results(results_path, umap_filename)

    # Use default config if none provided
    if config is None:
        config = get_default_burn_severity_config()

    # Create the plot
    visualize.plot_umap_selected_labels(
        umap_coords,
        labels,
        valid_values=config["valid_values"],
        label_map=config["label_map"],
        color_map=config["color_map"],
        point_size=point_size,
        title=title,
        figsize=figsize,
        manual_annotations=config["manual_annotations"],
    )

    # Save if path provided
    if save_path is not None:
        import matplotlib.pyplot as plt

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")


if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    FIGURES_DIR.mkdir(exist_ok=True)
    
    create_umap_figure(
        results_path=DATA_DIR,
        save_path=FIGURES_DIR / "umap_burned_areas.png",
        point_size=10,
        figsize=(10, 8),
    )
