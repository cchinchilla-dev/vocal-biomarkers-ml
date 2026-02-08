"""
Data standardization module.

This module provides utilities for standardizing features
in voice analysis datasets using sklearn scalers.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class DataStandardizer:
    """
    Standardizer for voice analysis datasets.

    This class handles feature standardization using various
    scaling methods from sklearn.

    Parameters
    ----------
    settings : Settings
        Configuration settings.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    method : str
        Standardization method to use.
    scaler : object
        Fitted sklearn scaler instance.
    is_fitted : bool
        Whether the scaler has been fitted.
    feature_names : list
        Names of features used during fitting.
    """

    SCALER_METHODS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    EXCLUDE_COLUMNS = {"Record", "Diagnosed", "Is_SMOTE", "Gender", "Age"}

    def __init__(self, settings: Settings, method: str = "standard"):
        self.settings = settings
        self.method = method
        self.scaler = self._create_scaler()
        self.is_fitted = False
        self.feature_names: list[str] = []

    def _create_scaler(self):
        """
        Create scaler instance based on method.

        Returns
        -------
        object
            Sklearn scaler instance.

        Raises
        ------
        ValueError
            If unknown standardization method is specified.
        """
        if self.method not in self.SCALER_METHODS:
            raise ValueError(
                f"Unknown standardization method: {self.method}. "
                f"Available: {list(self.SCALER_METHODS.keys())}"
            )
        return self.SCALER_METHODS[self.method]()

    def _get_feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        """
        Get numeric feature columns excluding metadata.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to extract feature columns from.

        Returns
        -------
        list[str]
            List of numeric feature column names.
        """
        return [
            col
            for col in dataset.columns
            if col not in self.EXCLUDE_COLUMNS
            and dataset[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

    def fit(self, dataset: pd.DataFrame) -> "DataStandardizer":
        """
        Fit the scaler to the dataset.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to fit the scaler on.

        Returns
        -------
        DataStandardizer
            Self for method chaining.
        """
        self.feature_names = self._get_feature_columns(dataset)

        if not self.feature_names:
            logger.warning("No feature columns found for standardization")
            return self

        self.scaler.fit(dataset[self.feature_names])
        self.is_fitted = True

        logger.debug(f"Fitted {self.method} scaler on {len(self.feature_names)} features")
        return self

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset using the fitted scaler.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to transform.

        Returns
        -------
        DataFrame
            Transformed dataset with standardized features.

        Raises
        ------
        RuntimeError
            If scaler has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        if not self.feature_names:
            return dataset.copy()

        result = dataset.copy()
        result[self.feature_names] = self.scaler.transform(dataset[self.feature_names])

        return result

    def fit_transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform dataset in one step.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to fit and transform.

        Returns
        -------
        DataFrame
            Transformed dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the standardization transformation.

        Parameters
        ----------
        dataset : DataFrame
            Standardized dataset.

        Returns
        -------
        DataFrame
            Dataset in original scale.

        Raises
        ------
        RuntimeError
            If scaler has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")

        if not self.feature_names:
            return dataset.copy()

        result = dataset.copy()
        result[self.feature_names] = self.scaler.inverse_transform(dataset[self.feature_names])

        return result

    def standardize(
        self,
        dataset: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Standardize a dataset.

        This is the main method for standardizing datasets,
        providing a simple interface that handles fitting automatically.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to standardize.
        fit : bool
            Whether to fit the scaler. Set to False when transforming
            test data with a scaler fitted on training data.

        Returns
        -------
        DataFrame
            Standardized dataset.
        """
        if fit:
            result = self.fit_transform(dataset)
            logger.info(
                f"Standardized {len(self.feature_names)} features " f"using {self.method} method"
            )
        else:
            result = self.transform(dataset)
            logger.debug(f"Transformed dataset using fitted {self.method} scaler")

        return result

    def standardize_split(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize train and test sets correctly.

        Fits the scaler on training data only to prevent data leakage,
        then transforms both train and test sets.

        Parameters
        ----------
        X_train : DataFrame
            Training features.
        X_test : DataFrame
            Test features.

        Returns
        -------
        tuple
            (X_train_scaled, X_test_scaled)
        """
        # Fit only on training data
        self.feature_names = self._get_feature_columns(X_train)

        if not self.feature_names:
            logger.warning("No feature columns found for standardization")
            return X_train.copy(), X_test.copy()

        # Fit and transform train
        X_train_scaled = X_train.copy()
        X_train_scaled[self.feature_names] = self.scaler.fit_transform(X_train[self.feature_names])
        self.is_fitted = True

        # Transform test (without fitting)
        X_test_scaled = X_test.copy()
        X_test_scaled[self.feature_names] = self.scaler.transform(X_test[self.feature_names])

        logger.info(f"Standardized train/test split with {len(self.feature_names)} features")

        return X_train_scaled, X_test_scaled

    def get_feature_stats(self) -> pd.DataFrame | None:
        """
        Get statistics of fitted scaler.

        Returns
        -------
        DataFrame or None
            DataFrame with mean and scale for each feature,
            or None if not fitted.
        """
        if not self.is_fitted or not self.feature_names:
            return None

        if self.method == "standard":
            return pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Mean": self.scaler.mean_,
                    "Scale": self.scaler.scale_,
                }
            )
        elif self.method == "minmax":
            return pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Min": self.scaler.data_min_,
                    "Max": self.scaler.data_max_,
                    "Scale": self.scaler.scale_,
                }
            )
        elif self.method == "robust":
            return pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Center": self.scaler.center_,
                    "Scale": self.scaler.scale_,
                }
            )

        return None
