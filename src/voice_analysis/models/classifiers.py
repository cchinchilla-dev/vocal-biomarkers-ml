"""
Classifier training and management module.

This module provides classes for training, evaluating, and managing
multiple classification models with hyperparameter optimization.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb
import xgboost as xgb

from ..config.settings import Settings
from ..metrics.custom_metrics import MetricsCalculator, create_metrics_dataframe
from .hyperparameter_search import HyperparameterOptimizer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""

    abbreviation: str
    name: str
    instance: BaseEstimator
    supports_random_state: bool = True


class ModelRegistry:
    """
    Registry of available classification models.

    This class provides a centralized registry of all available models
    with their default configurations.
    """

    MODELS = {
        "AdaBoost": ModelConfig(
            abbreviation="ADA",
            name="AdaBoost",
            instance=AdaBoostClassifier(),
        ),
        "KNearestNeighbor": ModelConfig(
            abbreviation="KNN",
            name="KNearestNeighbor",
            instance=KNeighborsClassifier(),
            supports_random_state=False,
        ),
        "SupportVectorMachine": ModelConfig(
            abbreviation="SVM",
            name="SupportVectorMachine",
            instance=SVC(probability=True),
        ),
        "Bagging": ModelConfig(
            abbreviation="BAG",
            name="Bagging",
            instance=BaggingClassifier(),
        ),
        "RandomForest": ModelConfig(
            abbreviation="RF",
            name="RandomForest",
            instance=RandomForestClassifier(),
        ),
        "GradientBoosting": ModelConfig(
            abbreviation="GB",
            name="GradientBoosting",
            instance=GradientBoostingClassifier(),
        ),
        "XGBoost": ModelConfig(
            abbreviation="XGB",
            name="XGBoost",
            instance=xgb.XGBClassifier(n_jobs=1, verbosity=0),
        ),
        "LightGBM": ModelConfig(
            abbreviation="LGB",
            name="LightGBM",
            instance=lgb.LGBMClassifier(verbose=-1),
        ),
        "DecisionTree": ModelConfig(
            abbreviation="DT",
            name="DecisionTree",
            instance=DecisionTreeClassifier(),
        ),
    }

    @classmethod
    def get_model(cls, name: str) -> ModelConfig:
        """Get model configuration by name."""
        if name not in cls.MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls.MODELS.keys())}")
        return cls.MODELS[name]

    @classmethod
    def get_fresh_instance(cls, name: str) -> BaseEstimator:
        """Get a fresh (unfit) model instance."""
        config = cls.get_model(name)
        return clone(config.instance)


class ClassifierTrainer:
    """
    Trainer for classification models.

    This class handles the complete training workflow including
    data preparation, hyperparameter optimization, model training,
    and evaluation.

    Parameters
    ----------
    settings : Settings
        Configuration settings for training.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    metrics_calculator : MetricsCalculator
        Calculator for evaluation metrics.
    hyperparameter_optimizer : HyperparameterOptimizer
        Optimizer for model hyperparameters.
    label_encoder : LabelEncoder
        Encoder for target labels.
    scaler : StandardScaler
        Scaler for feature standardization.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.metrics_calculator = MetricsCalculator()
        self.hyperparameter_optimizer = HyperparameterOptimizer(settings)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def train_all(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        execution_number: int,
        seed: int,
    ) -> dict[str, pd.DataFrame]:
        """
        Train all enabled models on the dataset.

        Parameters
        ----------
        dataset : DataFrame
            Training dataset with features and target.
        target_column : str
            Name of the target column.
        execution_number : int
            Current execution iteration number.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary containing metrics, patient probabilities,
            and leave-one-out results DataFrames.
        """
        np.random.seed(seed)

        # Prepare data - keep track of original records
        X, y, record_series, is_smote_series = self._prepare_data(dataset, target_column)

        # Split data for general metrics
        (
            X_train,
            X_test,
            y_train,
            y_test,
            records_train,
            records_test,
            is_smote_train,
            is_smote_test,
        ) = train_test_split(
            X,
            y,
            record_series,
            is_smote_series,
            test_size=self.settings.models.test_size,
            random_state=seed,
        )

        # Get ALL pathological patients (non-SMOTE) for patient-level evaluation
        pathological_mask = (y == 1) & (~is_smote_series.values)
        pathological_records = record_series[pathological_mask].unique()

        all_metrics = []
        all_patient_probs = []
        all_loo_results = []

        enabled_models = self.settings.models.enabled_models
        method = self.settings.hyperparameter_search.method.upper()

        for model_name in enabled_models:
            logger.info(f"Training model: {model_name}")

            try:
                results = self._train_single_model(
                    model_name=model_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    X_full=X,
                    y_full=y,
                    record_series=record_series,
                    is_smote_series=is_smote_series,
                    pathological_records=pathological_records,
                    execution_number=execution_number,
                    method=method,
                    seed=seed,
                )

                all_metrics.append(results["metrics"])
                all_patient_probs.append(results["patient_probs"])
                all_loo_results.append(results["loo_results"])

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        return {
            "metrics": pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame(),
            "patient_probabilities": (
                pd.concat(all_patient_probs, ignore_index=True)
                if all_patient_probs
                else pd.DataFrame()
            ),
            "loo_results": (
                pd.concat(all_loo_results, ignore_index=True) if all_loo_results else pd.DataFrame()
            ),
        }

    def _prepare_data(
        self, dataset: pd.DataFrame, target_column: str
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Prepare data for training by encoding and scaling."""
        # Store record IDs and SMOTE flag
        record_series = dataset["Record"].copy().reset_index(drop=True)
        is_smote_series = (
            dataset["Is_SMOTE"].copy().reset_index(drop=True)
            if "Is_SMOTE" in dataset.columns
            else pd.Series([False] * len(dataset))
        )

        # Prepare features
        drop_cols = [target_column, "Record", "Is_SMOTE"]
        X = dataset.drop(columns=[c for c in drop_cols if c in dataset.columns])

        # Encode target
        y_original = dataset[target_column]
        self.label_encoder.fit(y_original.unique())
        y = pd.Series(
            self.label_encoder.transform(y_original),
            name=target_column,
        ).reset_index(drop=True)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
        )

        return X_scaled, y, record_series, is_smote_series

    def _train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        record_series: pd.Series,
        is_smote_series: pd.Series,
        pathological_records: np.ndarray,
        execution_number: int,
        method: str,
        seed: int,
    ) -> dict:
        """Train a single model and compute all metrics."""
        # Get fresh model instance
        model = ModelRegistry.get_fresh_instance(model_name)
        model_config = ModelRegistry.get_model(model_name)

        # Set random state if supported
        if model_config.supports_random_state:
            if hasattr(model, "random_state"):
                model.set_params(random_state=seed)

        # Optimize hyperparameters on training set
        best_params = self.hyperparameter_optimizer.optimize(
            model_name=model_name,
            model=model,
            X=X_train,
            y=y_train,
            seed=seed,
        )

        if best_params:
            model.set_params(**best_params)

        # Train model on training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics on test set
        metrics_df = create_metrics_dataframe(
            model_name=model_name,
            method=method,
            execution_number=execution_number,
            y_true=y_test,
            y_pred=y_pred,
        )

        # Patient-level predictions using LOO for ALL pathological patients
        patient_probs = self._compute_patient_probabilities_loo(
            model=model,
            X_full=X_full,
            y_full=y_full,
            record_series=record_series,
            is_smote_series=is_smote_series,
            pathological_records=pathological_records,
            model_name=model_name,
        )

        # Leave-One-Out accuracy for all samples
        loo_results = self._compute_loo_accuracy(
            model=model,
            X=X_full,
            y=y_full,
            record_series=record_series,
            model_name=model_name,
            execution_number=execution_number,
        )

        return {
            "metrics": metrics_df,
            "patient_probs": patient_probs,
            "loo_results": loo_results,
        }

    def _compute_patient_probabilities_loo(
        self,
        model: BaseEstimator,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        record_series: pd.Series,
        is_smote_series: pd.Series,
        pathological_records: np.ndarray,
        model_name: str,
    ) -> pd.DataFrame:
        """
        Compute predictions for ALL pathological patients using Leave-One-Patient-Out.

        This ensures every pathological patient is evaluated, trained on all other data.
        """
        if len(pathological_records) == 0:
            return pd.DataFrame(columns=["Record", "Model", "True_Label", "Predicted"])

        results = []

        for patient_record in pathological_records:
            # Get all indices for this patient (may have multiple recordings)
            patient_mask = record_series == patient_record

            # Train on everything EXCEPT this patient
            train_mask = ~patient_mask
            X_train = X_full[train_mask]
            y_train = y_full[train_mask]
            X_patient = X_full[patient_mask]
            y_patient = y_full[patient_mask]

            # Clone and train model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Predict for this patient
            y_pred = model_clone.predict(X_patient)

            # Convert to original labels
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            y_true_labels = self.label_encoder.inverse_transform(y_patient.values)

            # Store result (one row per recording)
            for pred_label in y_pred_labels:
                results.append(
                    {
                        "Record": patient_record,
                        "Model": model_name,
                        "True_Label": "Pathological",  # We know it's pathological
                        "Predicted": pred_label,
                    }
                )

        return pd.DataFrame(results)

    def _compute_loo_accuracy(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        record_series: pd.Series,
        model_name: str,
        execution_number: int,
    ) -> pd.DataFrame:
        """Compute Leave-One-Out accuracy for sample-level evaluation."""
        if not self.settings.evaluation.leave_one_out.enabled:
            return pd.DataFrame()

        loo = LeaveOneOut()
        results = []

        for train_idx, test_idx in loo.split(X):
            X_train_loo = X.iloc[train_idx]
            X_test_loo = X.iloc[test_idx]
            y_train_loo = y.iloc[train_idx]
            y_test_loo = y.iloc[test_idx]

            # Clone and train model
            model_clone = clone(model)
            model_clone.fit(X_train_loo, y_train_loo)

            # Predict
            prediction = model_clone.predict(X_test_loo)
            is_correct = int(prediction[0] == y_test_loo.values[0])

            results.append(
                {
                    "Record": record_series.iloc[test_idx[0]],
                    "Accuracy": is_correct,
                    "True_Label": y_test_loo.values[0],
                    "Predicted": prediction[0],
                    "Model": model_name,
                    "Execution": execution_number,
                }
            )

        return pd.DataFrame(results)
