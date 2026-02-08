"""
Data resampling module.

This module provides SMOTE and other resampling strategies
for handling class imbalance in voice analysis datasets.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from scipy.stats import ks_2samp

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class DataResampler:
    """
    Resampler for handling class imbalance.

    This class provides various resampling strategies including
    SMOTE, BorderlineSMOTE, SMOTE-Tomek, and SMOTE-ENN.

    Parameters
    ----------
    settings : Settings
        Configuration settings.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    method : str
        Resampling method to use.
    """

    RESAMPLING_METHODS = {
        "smote": SMOTE,
        "borderline_smote": BorderlineSMOTE,
        "smote_tomek": SMOTETomek,
        "smote_enn": SMOTEENN,
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.method = settings.resampling.method
        self._last_analysis: dict | None = None

    def resample(
        self,
        dataset: pd.DataFrame,
        target_column: str = "Diagnosed",
        seed: int = 42,
        analyze: bool = False,
    ) -> pd.DataFrame:
        """
        Resample the dataset to balance classes.

        Parameters
        ----------
        dataset : DataFrame
            Input dataset with imbalanced classes.
        target_column : str
            Name of the target column.
        seed : int
            Random seed for reproducibility.
        analyze : bool
            Whether to perform distribution analysis.

        Returns
        -------
        DataFrame
            Resampled dataset with balanced classes.
        """
        if not self.settings.resampling.enabled:
            logger.info("Resampling is disabled, returning original dataset")
            return dataset

        logger.info(f"Resampling dataset using {self.method}")

        # Shuffle dataset
        dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Separate features and target
        exclude_cols = [target_column, "Is_SMOTE", "Record"]
        X = dataset.drop(columns=[c for c in exclude_cols if c in dataset.columns])
        y = dataset[target_column]

        # Get resampler
        resampler = self._get_resampler(seed)

        # Perform resampling
        X_resampled, y_resampled = resampler.fit_resample(X, y)

        # Create base DataFrame
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)

        # Build auxiliary columns in advance (avoid fragmentation)
        n_original = len(dataset)
        n_total = len(resampled_df)

        aux_data = {
            target_column: y_resampled,
            "Is_SMOTE": np.r_[
                np.zeros(n_original, dtype=bool),
                np.ones(n_total - n_original, dtype=bool),
            ],
        }

        # Preserve Record column where possible
        if "Record" in dataset.columns:
            aux_data["Record"] = np.r_[
                dataset["Record"].values,
                np.full(n_total - n_original, "SMOTE", dtype=object),
            ]

        # Concatenate once → no PerformanceWarning
        resampled_df = pd.concat(
            [resampled_df, pd.DataFrame(aux_data)],
            axis=1,
        )

        # Perform distribution analysis if requested
        if analyze:
            self._last_analysis = self._analyze_distributions(dataset, resampled_df, target_column)

        logger.info(f"Resampled: {len(dataset)} -> {len(resampled_df)} samples")

        return resampled_df

    def _get_resampler(self, seed: int):
        """
        Get the resampler instance based on configuration.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        object
            Configured resampler instance (SMOTE, BorderlineSMOTE, etc.).

        Raises
        ------
        ValueError
            If unknown resampling method is specified.
        """
        smote_config = self.settings.resampling.smote

        if self.method == "smote":
            return SMOTE(
                random_state=seed,
                k_neighbors=smote_config.k_neighbors,
            )
        elif self.method == "borderline_smote":
            return BorderlineSMOTE(
                random_state=seed,
                k_neighbors=smote_config.k_neighbors,
            )
        elif self.method == "smote_tomek":
            return SMOTETomek(
                random_state=seed,
                smote=SMOTE(k_neighbors=smote_config.k_neighbors),
            )
        elif self.method == "smote_enn":
            return SMOTEENN(
                random_state=seed,
                smote=SMOTE(k_neighbors=smote_config.k_neighbors),
            )
        else:
            raise ValueError(f"Unknown resampling method: {self.method}")

    def _analyze_distributions(
        self,
        original: pd.DataFrame,
        resampled: pd.DataFrame,
        target_column: str,
    ) -> dict:
        """
        Analyze distribution changes after resampling.

        Parameters
        ----------
        original : DataFrame
            Original dataset.
        resampled : DataFrame
            Resampled dataset.
        target_column : str
            Name of target column.

        Returns
        -------
        dict
            Analysis results.
        """
        analysis = {
            "class_distribution": {
                "before": original[target_column].value_counts().to_dict(),
                "after": resampled[target_column].value_counts().to_dict(),
            },
            "ks_tests": [],
        }

        # Get numeric columns for KS test
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        exclude = {target_column, "Is_SMOTE"}
        feature_cols = [c for c in numeric_cols if c not in exclude]

        # Perform KS test on pathological samples
        pathological_before = original[original[target_column] == "Pathological"]
        pathological_after = resampled[resampled[target_column] == "Pathological"]

        for col in feature_cols:
            if col in pathological_before.columns and col in pathological_after.columns:
                statistic, pvalue = ks_2samp(
                    pathological_before[col].dropna(),
                    pathological_after[col].dropna(),
                )
                analysis["ks_tests"].append(
                    {
                        "Feature": col,
                        "Statistic": statistic,
                        "P-Value": pvalue,
                        "Significant": pvalue < 0.05,
                    }
                )

        return analysis

    def get_last_analysis(self) -> dict | None:
        """
        Get the last distribution analysis results.

        Returns
        -------
        dict or None
            Analysis results from the last resampling operation,
            or None if no analysis was performed.
        """
        return self._last_analysis


class SMOTEAnalyzer:
    """
    Analyzer for SMOTE resampling effects.

    This class provides methods for analyzing the impact of
    SMOTE on feature distributions and class balance.
    """

    @staticmethod
    def compare_distributions(
        before: pd.DataFrame,
        after: pd.DataFrame,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compare feature distributions before and after SMOTE.

        Parameters
        ----------
        before : DataFrame
            Dataset before SMOTE.
        after : DataFrame
            Dataset after SMOTE.
        alpha : float
            Significance level.

        Returns
        -------
        DataFrame
            Comparison results with KS test statistics.
        """
        results = []

        for column in before.columns:
            if before[column].dtype in [np.float64, np.int64]:
                statistic, pvalue = ks_2samp(before[column], after[column])
                results.append(
                    {
                        "Feature": column,
                        "KS Statistic": statistic,
                        "P-Value": pvalue,
                        "Reject Null": pvalue < alpha,
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def get_class_balance_report(before: pd.Series, after: pd.Series) -> dict:
        """
        Generate class balance report.

        Parameters
        ----------
        before : Series
            Class labels before SMOTE.
        after : Series
            Class labels after SMOTE.

        Returns
        -------
        dict
            Balance statistics.
        """
        before_counts = before.value_counts()
        after_counts = after.value_counts()

        return {
            "before": {
                "counts": before_counts.to_dict(),
                "ratio": before_counts.min() / before_counts.max(),
            },
            "after": {
                "counts": after_counts.to_dict(),
                "ratio": after_counts.min() / after_counts.max(),
            },
            "samples_added": len(after) - len(before),
        }
