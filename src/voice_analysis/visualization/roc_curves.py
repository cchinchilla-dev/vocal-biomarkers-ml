"""
ROC curve visualization module.

This module provides functions for generating ROC curves
for classification models with cross-validation support.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..config.settings import Settings
from ..models.classifiers import ModelRegistry

logger = logging.getLogger(__name__)


class ROCVisualizer:
    """
    Visualizer for ROC curves.

    This class generates ROC curves for multiple classifiers,
    including cross-validation folds and mean curves.

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

    def plot(
        self,
        dataset,
        save_path: Path,
        seed: int | None = None,
    ) -> None:
        """
        Generate ROC curves for all enabled models.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target.
        save_path : Path
            Path to save the plot.
        seed : int, optional
            Random seed for cross-validation.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if seed is None:
            seed = self.settings.reproducibility.seeds[0]

        # Prepare data
        X, y = self._prepare_data(dataset)

        # Get models
        enabled_models = self.settings.models.enabled_models
        n_models = len(enabled_models)

        # Calculate grid dimensions
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        cv_config = self.settings.models.cross_validation
        cv = StratifiedKFold(
            n_splits=cv_config.n_folds,
            shuffle=cv_config.shuffle,
            random_state=seed,
        )

        for idx, model_name in enumerate(enabled_models):
            if idx >= len(axes):
                break

            ax = axes[idx]
            self._plot_single_model(model_name, X, y, cv, ax, seed)

        # Hide unused subplots
        for idx in range(len(enabled_models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.settings.visualization.style.figure_dpi)
        plt.close()

        logger.info(f"Saved ROC curves to: {save_path}")

    def _prepare_data(self, dataset):
        """
        Prepare data for ROC curve generation.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target.

        Returns
        -------
        tuple
            Scaled features array and encoded labels array.
        """
        # Exclude non-feature columns
        exclude_cols = ["Record", "Diagnosed", "Is_SMOTE", "Gender", "Age"]
        feature_cols = [c for c in dataset.columns if c not in exclude_cols]

        X = dataset[feature_cols]
        y = dataset["Diagnosed"]

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y_encoded

    def _plot_single_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        cv,
        ax,
        seed: int,
    ) -> None:
        """Plot ROC curve for a single model."""
        model = ModelRegistry.get_fresh_instance(model_name)
        model_config = ModelRegistry.get_model(model_name)

        # Set random state if supported
        if model_config.supports_random_state and hasattr(model, "random_state"):
            model.set_params(random_state=seed)

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        colors = sns.color_palette("Set2", cv.get_n_splits())

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)

            viz = RocCurveDisplay.from_estimator(
                model,
                X_test,
                y_test,
                name=f"Fold {fold}",
                alpha=0.5,
                lw=1,
                ax=ax,
                color=colors[fold],
                plot_chance_level=(fold == cv.get_n_splits() - 1),
            )

            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        # Plot mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color="blue",
            label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
            lw=2,
            alpha=0.8,
        )

        # Plot std band
        std_tpr = np.std(tprs, axis=0)
        ax.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            color="grey",
            alpha=0.2,
            label="± 1 std. dev.",
        )

        ax.set_title(f"ROC Curves - {model_name}")
        ax.legend(loc="lower right", fontsize=8)


def plot_roc_comparison(
    results: dict,
    save_path: Path,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot ROC curve comparison for multiple models.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to (fpr, tpr, auc) tuples.
    save_path : Path
        Path to save the plot.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(results))

    for (model_name, (fpr, tpr, roc_auc)), color in zip(results.items(), colors):
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
