"""
Distribution visualization module.

This module provides functions for visualizing feature distributions,
class distributions, and SMOTE analysis plots.
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config.settings import Settings

logger = logging.getLogger(__name__)

# Suppress the FutureWarning from seaborn
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


class DistributionVisualizer:
    """
    Visualizer for data distributions.

    This class creates various distribution plots including
    feature distributions, class balance plots, and SMOTE analysis.

    Parameters
    ----------
    settings : Settings
        Configuration settings.
    """

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

    @staticmethod
    def _clean_data(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Replace infinite values with NaN for safe plotting.

        Parameters
        ----------
        data : DataFrame or Series
            Data to clean.

        Returns
        -------
        DataFrame or Series
            Data with infinite values replaced by NaN.
        """
        if isinstance(data, pd.DataFrame):
            return data.replace([np.inf, -np.inf], np.nan)
        elif isinstance(data, pd.Series):
            return data.replace([np.inf, -np.inf], np.nan)
        return data

    def plot_class_distribution(
        self,
        before: pd.Series,
        after: pd.Series,
        save_path: Path,
        title: str = "Class Distribution Before and After SMOTE",
    ) -> None:
        """
        Plot class distribution comparison.

        Parameters
        ----------
        before : Series
            Class labels before resampling.
        after : Series
            Class labels after resampling.
        save_path : Path
            Path to save the plot.
        title : str
            Plot title.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        colors = ["#3498db", "#e74c3c"]

        # Before SMOTE
        before_counts = before.value_counts()
        axes[0].bar(
            before_counts.index,
            before_counts.values,
            color=colors,
        )
        axes[0].set_title("Before SMOTE", fontsize=12)
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Count")
        for i, (label, count) in enumerate(before_counts.items()):
            axes[0].annotate(
                str(count),
                xy=(i, count),
                ha="center",
                va="bottom",
            )

        # After SMOTE
        after_counts = after.value_counts()
        axes[1].bar(
            after_counts.index,
            after_counts.values,
            color=colors,
        )
        axes[1].set_title("After SMOTE", fontsize=12)
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Count")
        for i, (label, count) in enumerate(after_counts.items()):
            axes[1].annotate(
                str(count),
                xy=(i, count),
                ha="center",
                va="bottom",
            )

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.settings.visualization.style.figure_dpi)
        plt.close()

        logger.debug(f"Saved class distribution plot: {save_path}")

    def plot_feature_distributions(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        save_path: Path,
        max_features: int = 30,
    ) -> None:
        """
        Plot feature distributions before and after SMOTE.

        Parameters
        ----------
        before : DataFrame
            Features before resampling.
        after : DataFrame
            Features after resampling.
        save_path : Path
            Path to save the plot.
        max_features : int
            Maximum number of features to plot.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Clean data - replace inf with NaN
        before = self._clean_data(before)
        after = self._clean_data(after)

        # Select numeric columns only
        numeric_cols = before.select_dtypes(include=[np.number]).columns
        features = list(numeric_cols)[:max_features]

        n_features = len(features)
        if n_features == 0:
            logger.warning("No numeric features to plot")
            return

        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            ax = axes[idx]

            # Get data and drop NaN for plotting
            before_data = before[feature].dropna()
            after_data = after[feature].dropna()

            if len(before_data) > 0:
                sns.kdeplot(before_data, ax=ax, color="blue", label="Before", warn_singular=False)
            if len(after_data) > 0:
                sns.kdeplot(after_data, ax=ax, color="red", label="After", warn_singular=False)

            ax.set_title(feature, fontsize=10)
            ax.set_xlabel("")
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Feature Distributions Before and After SMOTE",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.settings.visualization.style.figure_dpi)
        plt.close()

        logger.debug(f"Saved feature distributions plot: {save_path}")

    def plot_correlation_matrix(
        self,
        dataset: pd.DataFrame,
        save_path: Path,
        max_features: int = 50,
    ) -> None:
        """
        Plot correlation matrix heatmap.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features.
        save_path : Path
            Path to save the plot.
        max_features : int
            Maximum number of features to include.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Clean data
        dataset = self._clean_data(dataset)

        # Select numeric columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        features = list(numeric_cols)[:max_features]

        corr_matrix = dataset[features].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap="coolwarm",
            center=0,
            annot=False,
            square=True,
            linewidths=0.5,
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.settings.visualization.style.figure_dpi)
        plt.close()

        logger.debug(f"Saved correlation matrix: {save_path}")

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        save_path: Path,
        top_n: int = 20,
    ) -> None:
        """
        Plot feature importance bar chart.

        Parameters
        ----------
        importance_df : DataFrame
            DataFrame with Feature and Importance columns.
        save_path : Path
            Path to save the plot.
        top_n : int
            Number of top features to display.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Select top features
        top_features = importance_df.nlargest(top_n, "Importance")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=top_features,
            y="Feature",
            x="Importance",
            ax=ax,
            palette="viridis",
        )
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.settings.visualization.style.figure_dpi)
        plt.close()

        logger.debug(f"Saved feature importance plot: {save_path}")


def plot_boxplot_comparison(
    data: pd.DataFrame,
    feature: str,
    group_column: str,
    save_path: Path,
) -> None:
    """
    Create boxplot comparison between groups.

    Parameters
    ----------
    data : DataFrame
        Dataset with feature and group columns.
    feature : str
        Feature column name.
    group_column : str
        Column name for grouping.
    save_path : Path
        Path to save the plot.
    """
    # Clean data
    data = data.replace([np.inf, -np.inf], np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, x=group_column, y=feature, ax=ax)
    ax.set_title(f"{feature} by {group_column}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
