"""Unit tests for custom_metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voice_analysis.metrics.custom_metrics import (
    MetricsCalculator,
    compute_confusion_matrix_components,
    create_metrics_dataframe,
    sensitivity_score,
    specificity_score,
    weighted_balanced_accuracy,
)


class TestConfusionMatrixComponents:
    """Test suite for compute_confusion_matrix_components function."""

    def test_perfect_predictions(self) -> None:
        """Test with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        tn, fp, fn, tp = compute_confusion_matrix_components(y_true, y_pred)

        assert tn == 2
        assert fp == 0
        assert fn == 0
        assert tp == 2

    def test_all_wrong_predictions(self) -> None:
        """Test with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        tn, fp, fn, tp = compute_confusion_matrix_components(y_true, y_pred)

        assert tn == 0
        assert fp == 2
        assert fn == 2
        assert tp == 0

    def test_mixed_predictions(self) -> None:
        """Test with mixed correct and incorrect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])

        tn, fp, fn, tp = compute_confusion_matrix_components(y_true, y_pred)

        assert tn == 2
        assert fp == 1
        assert fn == 1
        assert tp == 2

    def test_returns_integers(self) -> None:
        """Test that function returns integer values."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])

        tn, fp, fn, tp = compute_confusion_matrix_components(y_true, y_pred)

        assert isinstance(tn, int)
        assert isinstance(fp, int)
        assert isinstance(fn, int)
        assert isinstance(tp, int)


class TestSensitivityScore:
    """Test suite for sensitivity_score function."""

    def test_perfect_sensitivity(self) -> None:
        """Test sensitivity with all positives correctly identified."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1])

        result = sensitivity_score(y_true, y_pred)

        assert result == 1.0

    def test_zero_sensitivity(self) -> None:
        """Test sensitivity when no positives are correctly identified."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])

        result = sensitivity_score(y_true, y_pred)

        assert result == 0.0

    def test_partial_sensitivity(self) -> None:
        """Test sensitivity with partial correct identification."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])

        result = sensitivity_score(y_true, y_pred)

        assert result == 0.5

    def test_no_positive_samples(self) -> None:
        """Test sensitivity when there are no positive samples."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 1])

        result = sensitivity_score(y_true, y_pred)

        assert result == 0.0


class TestSpecificityScore:
    """Test suite for specificity_score function."""

    def test_perfect_specificity(self) -> None:
        """Test specificity with all negatives correctly identified."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0])

        result = specificity_score(y_true, y_pred)

        assert result == 1.0

    def test_zero_specificity(self) -> None:
        """Test specificity when no negatives are correctly identified."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1])

        result = specificity_score(y_true, y_pred)

        assert result == 0.0

    def test_partial_specificity(self) -> None:
        """Test specificity with partial correct identification."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])

        result = specificity_score(y_true, y_pred)

        assert result == 0.5

    def test_no_negative_samples(self) -> None:
        """Test specificity when there are no negative samples."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 0, 1])

        result = specificity_score(y_true, y_pred)

        assert result == 0.0


class TestWeightedBalancedAccuracy:
    """Test suite for weighted_balanced_accuracy function."""

    def test_perfect_balanced_accuracy(self) -> None:
        """Test balanced accuracy with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = weighted_balanced_accuracy(y_true, y_pred)

        assert result == 1.0

    def test_zero_balanced_accuracy(self) -> None:
        """Test balanced accuracy with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        result = weighted_balanced_accuracy(y_true, y_pred)

        assert result == 0.0

    def test_balanced_accuracy_is_average(self) -> None:
        """Test that balanced accuracy is average of sensitivity and specificity."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])

        sens = sensitivity_score(y_true, y_pred)
        spec = specificity_score(y_true, y_pred)
        expected = (sens + spec) / 2

        result = weighted_balanced_accuracy(y_true, y_pred)

        assert result == expected


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""

    @pytest.fixture
    def calculator(self) -> MetricsCalculator:
        """Create a MetricsCalculator instance."""
        return MetricsCalculator()

    def test_compute_all_returns_dict(self, calculator: MetricsCalculator) -> None:
        """Test that compute_all returns a dictionary."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        result = calculator.compute_all(y_true, y_pred)

        assert isinstance(result, dict)

    def test_compute_all_contains_expected_metrics(self, calculator: MetricsCalculator) -> None:
        """Test that compute_all returns all expected metrics."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        result = calculator.compute_all(y_true, y_pred)

        expected_keys = {"accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_score"}
        assert expected_keys.issubset(result.keys())

    def test_compute_all_with_probabilities(self, calculator: MetricsCalculator) -> None:
        """Test compute_all with probability values for AUC-ROC."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        result = calculator.compute_all(y_true, y_pred, y_prob)

        assert "auc_roc" in result
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_compute_all_metrics_in_valid_range(self, calculator: MetricsCalculator) -> None:
        """Test that all metrics are in valid range [0, 1]."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        result = calculator.compute_all(y_true, y_pred)

        for metric_name, value in result.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} out of range: {value}"

    def test_compute_metrics_with_confidence(self, calculator: MetricsCalculator) -> None:
        """Test compute_metrics_with_confidence returns confidence intervals."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])

        result = calculator.compute_metrics_with_confidence(
            y_true, y_pred, n_bootstrap=20, confidence_level=0.95, seed=42
        )

        assert "accuracy" in result
        assert "mean" in result["accuracy"]
        assert "std" in result["accuracy"]
        assert "ci_lower" in result["accuracy"]
        assert "ci_upper" in result["accuracy"]

    def test_confidence_interval_bounds_are_ordered(self, calculator: MetricsCalculator) -> None:
        """Test that CI lower bound <= mean <= CI upper bound."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])

        result = calculator.compute_metrics_with_confidence(y_true, y_pred, n_bootstrap=50, seed=42)

        for metric_name, values in result.items():
            assert (
                values["ci_lower"] <= values["mean"] <= values["ci_upper"]
            ), f"{metric_name}: CI bounds not ordered correctly"


class TestCreateMetricsDataframe:
    """Test suite for create_metrics_dataframe function."""

    def test_returns_dataframe(self) -> None:
        """Test that function returns a DataFrame."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        result = create_metrics_dataframe(
            model_name="TestModel",
            method="PSO",
            execution_number=1,
            y_true=y_true,
            y_pred=y_pred,
        )

        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_single_row(self) -> None:
        """Test that returned DataFrame has exactly one row."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        result = create_metrics_dataframe(
            model_name="TestModel",
            method="PSO",
            execution_number=1,
            y_true=y_true,
            y_pred=y_pred,
        )

        assert len(result) == 1

    def test_dataframe_contains_model_info(self) -> None:
        """Test that DataFrame contains model information columns."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = create_metrics_dataframe(
            model_name="RandomForest",
            method="RandomSearch",
            execution_number=5,
            y_true=y_true,
            y_pred=y_pred,
        )

        assert result["Model"].values[0] == "RandomForest"
        assert result["Method"].values[0] == "RandomSearch"
        assert result["Number of execution"].values[0] == 5

    def test_dataframe_contains_metrics(self) -> None:
        """Test that DataFrame contains expected metric columns."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = create_metrics_dataframe(
            model_name="SVM",
            method="PSO",
            execution_number=1,
            y_true=y_true,
            y_pred=y_pred,
        )

        expected_columns = {"Accuracy", "Sensitivity", "Specificity", "F1"}
        assert expected_columns.issubset(result.columns)

    def test_perfect_predictions_metrics(self) -> None:
        """Test metrics values with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = create_metrics_dataframe(
            model_name="Test",
            method="Test",
            execution_number=1,
            y_true=y_true,
            y_pred=y_pred,
        )

        assert result["Accuracy"].values[0] == 1.0
        assert result["Sensitivity"].values[0] == 1.0
        assert result["Specificity"].values[0] == 1.0
