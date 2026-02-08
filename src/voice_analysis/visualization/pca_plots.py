"""
PCA visualization module.

This module provides functions for creating PCA and Kernel PCA
visualizations of voice analysis datasets.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class PCAVisualizer:
    """
    Visualizer for PCA projections.

    This class creates 2D and 3D PCA visualizations of datasets,
    supporting both linear PCA and Kernel PCA.

    Parameters
    ----------
    settings : Settings
        Configuration settings.
    """

    COLORS = {"Control": "blue", "Pathological": "red"}

    def __init__(self, settings: Settings):
        self.settings = settings
        self._apply_style()

    def _apply_style(self) -> None:
        """
        Apply matplotlib style settings.

        Configures font family and serif settings from the
        visualization configuration.
        """
        style_config = self.settings.visualization.style
        plt.rcParams["font.family"] = style_config.font_family
        plt.rcParams["font.serif"] = style_config.font_serif

    def plot(
        self,
        dataset: pd.DataFrame,
        save_dir: Path,
        compare_dataset: pd.DataFrame | None = None,
    ) -> None:
        """
        Generate all PCA visualizations.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to visualize.
        save_dir : Path
            Directory to save plots.
        compare_dataset : DataFrame, optional
            Dataset for comparison (e.g., before standardization).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        pca_config = self.settings.visualization.pca
        features = self._get_feature_columns(dataset)

        # Linear PCA
        for n_components in pca_config.n_components:
            self._plot_linear_pca(dataset, features, n_components, save_dir)

        # Kernel PCA
        for kernel in pca_config.kernels:
            if kernel == "linear":
                continue  # Already done above

            for n_components in pca_config.n_components:
                try:
                    self._plot_kernel_pca(dataset, features, kernel, n_components, save_dir)
                except Exception as e:
                    logger.warning(f"Kernel PCA ({kernel}) failed: {e}")

    def _get_feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        """
        Get numeric feature columns.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to extract feature columns from.

        Returns
        -------
        list[str]
            List of numeric feature column names.
        """
        exclude = {"Record", "Diagnosed", "Is_SMOTE", "Gender", "Age"}
        return [
            c
            for c in dataset.columns
            if c not in exclude and dataset[c].dtype in [np.float64, np.int64]
        ]

    def _plot_linear_pca(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        n_components: int,
        save_dir: Path,
    ) -> None:
        """Create linear PCA visualization."""
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(dataset[features])
        explained_var = pca.explained_variance_ratio_ * 100

        self._create_scatter_plot(
            pca_result,
            dataset["Diagnosed"],
            explained_var,
            save_dir / f"pca_linear_{n_components}d.svg",
            f"PCA {n_components}D (Linear)",
        )

    def _plot_kernel_pca(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        kernel: str,
        n_components: int,
        save_dir: Path,
    ) -> None:
        """Create kernel PCA visualization."""
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        kpca_result = kpca.fit_transform(dataset[features])

        # Estimate explained variance for kernel PCA
        explained_var = np.var(kpca_result, axis=0)
        explained_var = (explained_var / np.sum(explained_var)) * 100

        self._create_scatter_plot(
            kpca_result,
            dataset["Diagnosed"],
            explained_var,
            save_dir / f"kpca_{kernel}_{n_components}d.svg",
            f"Kernel PCA {n_components}D ({kernel.capitalize()})",
        )

    def _create_scatter_plot(
        self,
        pca_result: np.ndarray,
        labels: pd.Series,
        explained_var: np.ndarray,
        save_path: Path,
        title: str,
    ) -> None:
        """Create and save scatter plot."""
        n_components = pca_result.shape[1]

        if n_components == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        ax.grid(True)

        # Plot each class
        for label, color in self.COLORS.items():
            mask = labels == label
            if n_components == 3:
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    pca_result[mask, 2],
                    c=color,
                    marker="o",
                    facecolors="none",
                    linewidths=1,
                    label=label,
                    alpha=0.7,
                    s=50,
                )
            else:
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    c=color,
                    marker="o",
                    facecolors="none",
                    linewidths=1,
                    label=label,
                    alpha=0.7,
                    s=50,
                )

        # Labels
        ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}% var.)", fontsize=12)
        ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}% var.)", fontsize=12)
        if n_components == 3:
            ax.set_zlabel(f"PC3 ({explained_var[2]:.2f}% var.)", fontsize=12)

        ax.legend(fontsize=12)
        plt.title(title, fontsize=14)

        plt.savefig(save_path, dpi=self.settings.visualization.style.figure_dpi)
        plt.close()

        logger.debug(f"Saved PCA plot: {save_path}")


def check_pca_conditions(dataset: pd.DataFrame) -> bool:
    """
    Check if dataset meets conditions for PCA.

    Parameters
    ----------
    dataset : DataFrame
        Dataset to check.

    Returns
    -------
    bool
        True if conditions are met.
    """
    # Check for missing values
    if dataset.isnull().values.any():
        logger.warning("Dataset contains missing values")
        return False

    # Check for zero variance rows
    if dataset.nunique(axis=1).eq(1).any():
        logger.warning("Dataset contains rows with zero variance")
        return False

    # Check for sufficient unique rows
    if dataset.drop_duplicates().shape[0] < 2:
        logger.warning("Dataset contains insufficient unique rows")
        return False

    return True
