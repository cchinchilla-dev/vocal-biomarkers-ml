"""
Model evaluation module.

This module provides utilities for evaluating machine learning models,
computing aggregated metrics, and performing statistical comparisons.
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for machine learning models.

    This class provides methods for aggregating metrics across
    multiple executions and performing statistical comparisons
    between models.

    Parameters
    ----------
    settings : Settings
        Configuration settings.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def aggregate_metrics(
        self,
        metrics_df: pd.DataFrame,
        loo_results: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate metrics across multiple executions.

        Parameters
        ----------
        metrics_df : DataFrame
            Metrics from all executions.
        loo_results : DataFrame, optional
            Leave-One-Out results for patient-level metrics.

        Returns
        -------
        DataFrame
            Aggregated metrics per model.
        """
        # Select only numeric columns for aggregation (exclude 'Method' which is string)
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_aggregate = ["Model"] + numeric_cols

        # Group by model and calculate mean/std only on numeric columns
        aggregated = (
            metrics_df[cols_to_aggregate].groupby("Model").agg(["mean", "std"]).reset_index()
        )

        # Flatten the MultiIndex columns
        new_columns = ["Model"]
        for col in aggregated.columns[1:]:
            if isinstance(col, tuple):
                new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)

        aggregated.columns = new_columns

        # Add LOO metrics if available
        if loo_results is not None and not loo_results.empty:
            loo_metrics = self._compute_loo_metrics(loo_results)
            aggregated = aggregated.merge(loo_metrics, on="Model", how="left")

        return aggregated

    def _compute_loo_metrics(self, loo_results: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Leave-One-Out metrics per model.

        Parameters
        ----------
        loo_results : DataFrame
            Leave-One-Out results DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with LOO metrics per model.
        """
        results = []

        for model in loo_results["Model"].unique():
            model_data = loo_results[loo_results["Model"] == model]

            # Filter out SMOTE samples if the column exists
            if "Is_SMOTE" in model_data.columns:
                model_data = model_data[~model_data["Is_SMOTE"]]

            accuracies = model_data["Accuracy"]

            if len(accuracies) > 0:
                results.append(
                    {
                        "Model": model,
                        "LOO_Mean": accuracies.mean(),
                        "LOO_Std": accuracies.std(),
                        "LOO_Median": accuracies.median(),
                        "LOO_IQR": (accuracies.quantile(0.75) - accuracies.quantile(0.25)),
                    }
                )

        return pd.DataFrame(results)

    def aggregate_loo_summary(self, loo_results: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated summary of Leave-One-Out results.

        Parameters
        ----------
        loo_results : DataFrame
            Raw Leave-One-Out results with columns:
            [Record, Accuracy, True_Label, Predicted, Model, Execution]

        Returns
        -------
        DataFrame
            Summary with mean, std, median, and IQR per model across all executions.
        """
        if loo_results.empty:
            return pd.DataFrame()

        # Group by Model and Execution, compute mean accuracy per execution
        execution_accuracies = (
            loo_results.groupby(["Model", "Execution"])["Accuracy"].mean().reset_index()
        )

        # Now aggregate across executions
        summary = (
            execution_accuracies.groupby("Model")["Accuracy"]
            .agg(
                [
                    ("Mean", "mean"),
                    ("Std", "std"),
                    ("Median", "median"),
                    ("Min", "min"),
                    ("Max", "max"),
                    ("Q1", lambda x: x.quantile(0.25)),
                    ("Q3", lambda x: x.quantile(0.75)),
                ]
            )
            .reset_index()
        )

        # Add IQR
        summary["IQR"] = summary["Q3"] - summary["Q1"]

        # Add count of executions
        summary["N_Executions"] = (
            execution_accuracies.groupby("Model")["Execution"].nunique().values
        )

        # Reorder columns
        summary = summary[
            ["Model", "Mean", "Std", "Median", "IQR", "Min", "Max", "Q1", "Q3", "N_Executions"]
        ]

        return summary.sort_values("Mean", ascending=False)

    def perform_statistical_comparisons(
        self,
        metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform pairwise statistical comparisons between models.

        Parameters
        ----------
        metrics_df : DataFrame
            Metrics from all executions.

        Returns
        -------
        DataFrame
            Comparison results with t-test statistics.
        """
        logger.info("Performing statistical comparisons between models")

        metrics_to_compare = [
            "Accuracy",
            "Weighted Balanced Accuracy",
            "Sensitivity",
            "Specificity",
            "F1",
        ]

        models = metrics_df["Model"].unique()
        methods = metrics_df["Method"].unique()
        alpha = self.settings.evaluation.statistical_tests.alpha

        results = []

        for metric in metrics_to_compare:
            for method in methods:
                method_data = metrics_df[metrics_df["Method"] == method]

                for model1, model2 in combinations(models, 2):
                    scores1 = method_data[method_data["Model"] == model1][metric]
                    scores2 = method_data[method_data["Model"] == model2][metric]

                    if len(scores1) > 1 and len(scores2) > 1:
                        t_stat, p_value = ttest_ind(scores1, scores2)

                        # Determine better model
                        better_model = model1 if t_stat > 0 else model2
                        if p_value < alpha:
                            better_model += "*"

                        results.append(
                            {
                                "Metric": metric,
                                "Method": method,
                                "Model 1": model1,
                                "Model 2": model2,
                                "T-statistic": t_stat,
                                "P-value": p_value,
                                "Significant": p_value < alpha,
                                "Better Model": better_model,
                            }
                        )

        return pd.DataFrame(results)

    def compute_patient_level_accuracy(
        self,
        predictions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute patient-level accuracy metrics.

        Parameters
        ----------
        predictions_df : DataFrame
            DataFrame with patient predictions.

        Returns
        -------
        DataFrame
            Patient-level accuracy per model.
        """
        results = []

        for model in predictions_df["Model"].unique():
            model_data = predictions_df[predictions_df["Model"] == model]

            # Calculate accuracy per patient
            patient_accuracy = model_data.groupby("Record").apply(
                lambda x: (x["True_Label"] == x["Predicted"]).mean()
            )

            results.append(
                {
                    "Model": model,
                    "Patient_Accuracy_Mean": patient_accuracy.mean(),
                    "Patient_Accuracy_Std": patient_accuracy.std(),
                    "Patient_Accuracy_Median": patient_accuracy.median(),
                    "Patient_Accuracy_IQR": (
                        patient_accuracy.quantile(0.75) - patient_accuracy.quantile(0.25)
                    ),
                }
            )

        return pd.DataFrame(results)

    def compute_misclassification_probability(
        self,
        predictions_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute misclassification probabilities.

        For pathological patients, misclassification means being predicted
        as something other than "Pathological".

        Parameters
        ----------
        predictions_df : DataFrame
            DataFrame with predictions (Record, Model, True_Label, Predicted).

        Returns
        -------
        tuple
            (per_patient_probs, per_model_probs)
        """
        if predictions_df.empty:
            return pd.DataFrame(columns=["Record", "Model", "Misclassified"]), pd.DataFrame(
                columns=["Model", "Misclassified"]
            )

        # Add misclassification flag
        # For pathological patients: misclassified if NOT predicted as Pathological
        df = predictions_df.copy()
        df["Misclassified"] = df["Predicted"] != "Pathological"

        # Per patient and model
        per_patient = df.groupby(["Record", "Model"])["Misclassified"].mean().reset_index()

        # Per model only
        per_model = per_patient.groupby("Model")["Misclassified"].mean().reset_index()

        return per_patient, per_model


class BootstrapEvaluator:
    """
    Evaluator using bootstrap for confidence intervals.

    This class provides methods for computing bootstrap
    confidence intervals for model metrics.
    """

    def __init__(self, n_iterations: int = 100, confidence_level: float = 0.95):
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level

    def compute_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: callable,
        seed: int = 42,
    ) -> dict:
        """
        Compute bootstrap confidence intervals for a metric.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        metric_func : callable
            Function to compute the metric.
        seed : int
            Random seed.

        Returns
        -------
        dict
            Mean, std, and confidence interval bounds.
        """
        rng = np.random.RandomState(seed)
        n_samples = len(y_true)
        bootstrap_scores = []

        for _ in range(self.n_iterations):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)

        scores = np.array(bootstrap_scores)
        alpha = 1 - self.confidence_level

        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "ci_lower": np.percentile(scores, alpha / 2 * 100),
            "ci_upper": np.percentile(scores, (1 - alpha / 2) * 100),
        }
