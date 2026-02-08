"""
Data cleaning module.

This module provides functions for cleaning datasets by removing
low-variance and highly correlated features.
"""

import logging

import numpy as np
import pandas as pd

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleaner for voice analysis datasets.

    This class removes features with low variance and high correlation
    to improve model performance and reduce dimensionality.

    Parameters
    ----------
    settings : Settings
        Configuration settings.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    variance_threshold : float
        Minimum variance threshold for features.
    correlation_threshold : float
        Maximum correlation threshold between features.
    removed_features : dict
        Dictionary tracking removed features by reason.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        cleaning_config = settings.data_processing.cleaning
        self.variance_threshold = cleaning_config.variance_threshold
        self.correlation_threshold = cleaning_config.correlation_threshold
        self.removed_features: dict[str, list[str]] = {
            "low_variance": [],
            "high_correlation": [],
        }

    def clean(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing problematic features.

        Parameters
        ----------
        dataset : DataFrame
            Input dataset.

        Returns
        -------
        DataFrame
            Cleaned dataset.
        """
        logger.info("Starting data cleaning")
        self.removed_features = {"low_variance": [], "high_correlation": []}

        # Get feature columns (exclude metadata)
        metadata_cols = ["Record", "Diagnosed", "Is_SMOTE", "Gender", "Age"]
        feature_cols = [c for c in dataset.columns if c not in metadata_cols]

        # Remove low variance features
        features_df = dataset[feature_cols]
        low_var_features = self._get_low_variance_features(features_df)
        dataset = self._remove_features(dataset, low_var_features)
        self.removed_features["low_variance"] = low_var_features

        logger.info(f"Removed {len(low_var_features)} features due to low variance")

        # Update feature columns
        feature_cols = [c for c in dataset.columns if c not in metadata_cols]
        features_df = dataset[feature_cols]

        # Remove highly correlated features
        correlated_features = self._get_correlated_features(features_df)
        dataset = self._remove_features(dataset, correlated_features)
        self.removed_features["high_correlation"] = correlated_features

        logger.info(f"Removed {len(correlated_features)} features due to high correlation")

        logger.info(f"Final dataset shape: {dataset.shape}")
        return dataset

    def _get_low_variance_features(self, dataset: pd.DataFrame) -> list[str]:
        """
        Identify features with variance below threshold.

        Parameters
        ----------
        dataset : DataFrame
            Dataset containing only feature columns.

        Returns
        -------
        list
            List of low variance feature names.
        """
        variance = dataset.var(numeric_only=True)
        low_var = variance[variance < self.variance_threshold]
        return low_var.index.tolist()

    def _get_correlated_features(self, dataset: pd.DataFrame) -> list[str]:
        """
        Identify features with correlation above threshold.

        Parameters
        ----------
        dataset : DataFrame
            Dataset containing only feature columns.

        Returns
        -------
        list
            List of highly correlated feature names to remove.
        """
        correlations = dataset.corr(numeric_only=True)

        # Get upper triangle of correlation matrix
        upper_tri = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))

        # Find features to drop
        to_drop = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column].abs() > self.correlation_threshold)
        ]

        return to_drop

    def _remove_features(self, dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """
        Remove specified features from dataset.

        Parameters
        ----------
        dataset : DataFrame
            Input dataset.
        features : list
            Features to remove.

        Returns
        -------
        DataFrame
            Dataset with features removed.
        """
        existing_features = [f for f in features if f in dataset.columns]
        return dataset.drop(columns=existing_features)

    def get_removal_report(self) -> str:
        """
        Generate a report of removed features.

        Returns
        -------
        str
            Formatted report string.
        """
        lines = ["Feature Removal Report", "=" * 40]

        for reason, features in self.removed_features.items():
            lines.append(f"\n{reason.replace('_', ' ').title()} ({len(features)}):")
            if features:
                for f in features[:10]:  # Show first 10
                    lines.append(f"  - {f}")
                if len(features) > 10:
                    lines.append(f"  ... and {len(features) - 10} more")
            else:
                lines.append("  None")

        return "\n".join(lines)


def remove_low_variance_features(
    dataset: pd.DataFrame, threshold: float = 0.1
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features with variance below threshold.

    Parameters
    ----------
    dataset : DataFrame
        Input dataset.
    threshold : float
        Variance threshold.

    Returns
    -------
    tuple
        (cleaned_dataset, removed_features)
    """
    variance = dataset.var(numeric_only=True)
    low_var_features = variance[variance < threshold].index.tolist()
    cleaned = dataset.drop(columns=low_var_features, errors="ignore")
    return cleaned, low_var_features


def remove_correlated_features(
    dataset: pd.DataFrame, threshold: float = 0.9
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove highly correlated features.

    Parameters
    ----------
    dataset : DataFrame
        Input dataset.
    threshold : float
        Correlation threshold.

    Returns
    -------
    tuple
        (cleaned_dataset, removed_features)
    """
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    correlations = dataset[numeric_cols].corr()

    upper_tri = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > threshold)]

    cleaned = dataset.drop(columns=to_drop, errors="ignore")
    return cleaned, to_drop
