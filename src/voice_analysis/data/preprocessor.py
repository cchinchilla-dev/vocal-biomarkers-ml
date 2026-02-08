"""
Data preprocessing module.

This module provides preprocessing utilities for preparing
voice analysis datasets for machine learning.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for voice analysis data.

    This class handles various preprocessing steps including
    label encoding, standardization, and data preparation.

    Parameters
    ----------
    settings : Settings
        Configuration settings.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    label_encoder : LabelEncoder
        Encoder for categorical labels.
    scaler : StandardScaler
        Scaler for feature standardization.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._is_fitted = False

    def prepare_for_training(
        self,
        dataset: pd.DataFrame,
        target_column: str = "Diagnosed",
        fit: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare dataset for model training.

        Parameters
        ----------
        dataset : DataFrame
            Input dataset with features and target.
        target_column : str
            Name of the target column.
        fit : bool
            Whether to fit the transformers or use existing fit.

        Returns
        -------
        tuple
            (X, y, record_ids) - Features, encoded labels, and record IDs.
        """
        logger.debug("Preparing dataset for training")

        # Store record IDs
        record_ids = dataset["Record"].copy() if "Record" in dataset.columns else None

        # Separate features and target
        drop_cols = [target_column, "Record", "Is_SMOTE"]
        X = dataset.drop(
            columns=[c for c in drop_cols if c in dataset.columns]
        )
        y = dataset[target_column]

        # Encode labels
        if fit:
            self.label_encoder.fit(y.unique())

        y_encoded = pd.Series(
            self.label_encoder.transform(y),
            index=y.index,
            name=target_column,
        )

        # Standardize features
        if fit:
            self.scaler.fit(X)
            self._is_fitted = True

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )

        return X_scaled, y_encoded, record_ids

    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original form.

        Parameters
        ----------
        y : array-like
            Encoded labels.

        Returns
        -------
        array
            Original label values.
        """
        return self.label_encoder.inverse_transform(y)

    def get_feature_names(self, dataset: pd.DataFrame) -> list[str]:
        """
        Get list of feature column names.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to extract feature names from.

        Returns
        -------
        list
            List of feature column names.
        """
        exclude_cols = {"Record", "Diagnosed", "Is_SMOTE", "Gender", "Age"}
        return [c for c in dataset.columns if c not in exclude_cols]


class LabelAssigner:
    """
    Utility class for assigning diagnostic labels to datasets.

    This class provides consistent label assignment for control
    and pathological groups.
    """

    LABELS = {
        0: "Control",
        1: "Pathological",
    }

    @classmethod
    def assign_label(cls, dataset: pd.DataFrame, label_code: int) -> pd.DataFrame:
        """
        Assign a diagnostic label to a dataset.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to assign label to.
        label_code : int
            Label code (0 for Control, 1 for Pathological).

        Returns
        -------
        DataFrame
            Dataset with Diagnosed column added.
        """
        label = cls.LABELS.get(label_code)
        if label is None:
            raise ValueError(f"Invalid label code: {label_code}")

        return dataset.assign(Diagnosed=label)

    @classmethod
    def get_label_name(cls, code: int) -> str:
        """
        Get label name from code.

        Parameters
        ----------
        code : int
            Label code (0 or 1).

        Returns
        -------
        str
            Label name ('Control', 'Pathological', or 'Unknown').
        """
        return cls.LABELS.get(code, "Unknown")


class DatasetMerger:
    """
    Utility class for merging multiple datasets.

    This class handles merging of acoustic and biomechanical
    datasets while preserving data integrity.
    """

    @staticmethod
    def merge_on_record(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        how: Literal["inner", "outer", "left", "right"] = "inner",
    ) -> pd.DataFrame:
        """
        Merge two datasets on the Record column.

        Parameters
        ----------
        df1 : DataFrame
            First dataset.
        df2 : DataFrame
            Second dataset.
        how : str
            Type of merge to perform.

        Returns
        -------
        DataFrame
            Merged dataset.
        """
        return pd.merge(df1, df2, on="Record", how=how)

    @staticmethod
    def concatenate_groups(
        control: pd.DataFrame,
        pathological: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Concatenate control and pathological datasets.

        Parameters
        ----------
        control : DataFrame
            Control group dataset.
        pathological : DataFrame
            Pathological group dataset.

        Returns
        -------
        DataFrame
            Combined dataset.
        """
        return pd.concat([control, pathological], ignore_index=True)

    @staticmethod
    def reorder_columns(
        dataset: pd.DataFrame,
        priority_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Reorder columns with priority columns first.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to reorder.
        priority_columns : list, optional
            Columns to place first.

        Returns
        -------
        DataFrame
            Reordered dataset.
        """
        if priority_columns is None:
            priority_columns = ["Record", "Diagnosed", "Gender", "Age"]

        existing_priority = [c for c in priority_columns if c in dataset.columns]
        other_columns = [c for c in dataset.columns if c not in priority_columns]

        return dataset[existing_priority + other_columns]