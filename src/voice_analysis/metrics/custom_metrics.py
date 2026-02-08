"""
Custom metrics for model evaluation.

This module provides specialized metrics for evaluating classification models,
including sensitivity, specificity, and weighted balanced accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_confusion_matrix_components(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Compute confusion matrix components.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    tuple
        (true_negatives, false_positives, false_negatives, true_positives)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate sensitivity (true positive rate / recall).

    Sensitivity measures the proportion of actual positives that are
    correctly identified.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Sensitivity score between 0 and 1.
    """
    _, _, fn, tp = compute_confusion_matrix_components(y_true, y_pred)
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0.0


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (true negative rate).

    Specificity measures the proportion of actual negatives that are
    correctly identified.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Specificity score between 0 and 1.
    """
    tn, fp, _, _ = compute_confusion_matrix_components(y_true, y_pred)
    denominator = tn + fp
    return tn / denominator if denominator > 0 else 0.0


def weighted_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate weighted balanced accuracy.

    Weighted balanced accuracy is the average of sensitivity and specificity,
    providing a balanced measure that accounts for class imbalance.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Weighted balanced accuracy between 0 and 1.
    """
    sens = sensitivity_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    return (sens + spec) / 2


class MetricsCalculator:
    """
    Calculator for computing multiple evaluation metrics.

    This class provides methods for computing a comprehensive set of
    classification metrics including accuracy, sensitivity, specificity,
    F1 score, and AUC-ROC.

    Examples
    --------
    >>> calculator = MetricsCalculator()
    >>> metrics = calculator.compute_all(y_true, y_pred)
    >>> print(metrics["accuracy"])
    """

    METRIC_FUNCTIONS = {
        "accuracy": accuracy_score,
        "balanced_accuracy": weighted_balanced_accuracy,
        "sensitivity": sensitivity_score,
        "specificity": specificity_score,
        "f1_score": f1_score,
    }

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Compute all available metrics.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels.
        y_prob : array-like, optional
            Predicted probabilities for AUC-ROC calculation.

        Returns
        -------
        dict
            Dictionary mapping metric names to their values.
        """
        results = {}

        for name, func in self.METRIC_FUNCTIONS.items():
            try:
                results[name] = func(y_true, y_pred)
            except Exception:
                results[name] = np.nan

        # Add AUC-ROC if probabilities are available
        if y_prob is not None:
            try:
                results["auc_roc"] = roc_auc_score(y_true, y_prob)
            except Exception:
                results["auc_roc"] = np.nan

        return results

    def compute_patient_level_metrics(
        self,
        predictions_df: pd.DataFrame,
        record_column: str = "Record",
        true_label_column: str = "True_Label",
        predicted_column: str = "Predicted",
    ) -> pd.DataFrame:
        """
        Compute patient-level accuracy metrics.

        Parameters
        ----------
        predictions_df : DataFrame
            DataFrame with predictions for each sample.
        record_column : str
            Column name for patient/record identifier.
        true_label_column : str
            Column name for true labels.
        predicted_column : str
            Column name for predictions.

        Returns
        -------
        DataFrame
            Patient-level accuracy statistics.
        """
        accuracies = predictions_df.groupby(record_column).apply(
            lambda x: (x[true_label_column] == x[predicted_column]).mean()
        )

        return pd.DataFrame(
            {
                "mean": [accuracies.mean()],
                "std": [accuracies.std()],
                "median": [accuracies.median()],
                "iqr": [accuracies.quantile(0.75) - accuracies.quantile(0.25)],
            }
        )

    def compute_metrics_with_confidence(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> dict[str, dict[str, float]]:
        """
        Compute metrics with bootstrap confidence intervals.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels.
        n_bootstrap : int
            Number of bootstrap samples.
        confidence_level : float
            Confidence level for intervals.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary with metric values and confidence intervals.
        """
        rng = np.random.RandomState(seed)
        n_samples = len(y_true)

        bootstrap_metrics = {name: [] for name in self.METRIC_FUNCTIONS}

        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = np.array(y_true)[indices]
            y_pred_boot = np.array(y_pred)[indices]

            for name, func in self.METRIC_FUNCTIONS.items():
                try:
                    bootstrap_metrics[name].append(func(y_true_boot, y_pred_boot))
                except Exception:
                    pass

        alpha = 1 - confidence_level
        results = {}

        for name, values in bootstrap_metrics.items():
            if values:
                values = np.array(values)
                results[name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "ci_lower": np.percentile(values, alpha / 2 * 100),
                    "ci_upper": np.percentile(values, (1 - alpha / 2) * 100),
                }

        return results


def create_metrics_dataframe(
    model_name: str,
    method: str,
    execution_number: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Create a standardized metrics DataFrame for a single evaluation.

    Parameters
    ----------
    model_name : str
        Name of the model.
    method : str
        Optimization method used (e.g., "PSO", "RandomSearch").
    execution_number : int
        Execution iteration number.
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    DataFrame
        Single-row DataFrame with all metrics.
    """
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(y_true, y_pred)

    return pd.DataFrame(
        [
            {
                "Model": model_name,
                "Method": method,
                "Number of execution": execution_number,
                "Accuracy": metrics["accuracy"],
                "Weighted Balanced Accuracy": metrics["balanced_accuracy"],
                "Sensitivity": metrics["sensitivity"],
                "Specificity": metrics["specificity"],
                "F1": metrics["f1_score"],
            }
        ]
    )
