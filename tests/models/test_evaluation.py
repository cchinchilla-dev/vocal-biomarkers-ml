"""Unit tests for model evaluation module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.evaluation.metrics = ["accuracy", "sensitivity", "specificity"]
        mock.evaluation.bootstrap.n_iterations = 10
        mock.evaluation.bootstrap.confidence_level = 0.95
        mock.evaluation.statistical_tests.method = "t_test"
        mock.evaluation.statistical_tests.alpha = 0.05
        mock.evaluation.leave_one_out.enabled = True
        return mock

    def test_evaluator_initialization(self, mock_settings: MagicMock) -> None:
        """Test ModelEvaluator initialization."""
        from voice_analysis.models.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(mock_settings)

        assert evaluator.settings == mock_settings

    def test_perform_statistical_comparisons(self, mock_settings: MagicMock) -> None:
        """Test statistical comparisons between models."""
        from voice_analysis.models.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(mock_settings)

        metrics_df = pd.DataFrame(
            {
                "Model": ["RF", "RF", "RF", "SVM", "SVM", "SVM"],
                "Method": ["cv", "cv", "cv", "cv", "cv", "cv"],
                "Accuracy": [0.85, 0.87, 0.86, 0.82, 0.84, 0.83],
                "Weighted Balanced Accuracy": [0.84, 0.86, 0.85, 0.81, 0.83, 0.82],
                "Sensitivity": [0.80, 0.82, 0.81, 0.78, 0.80, 0.79],
                "Specificity": [0.90, 0.92, 0.91, 0.86, 0.88, 0.87],
                "F1": [0.83, 0.85, 0.84, 0.80, 0.82, 0.81],
                "Number of execution": [1, 2, 3, 1, 2, 3],
            }
        )

        result = evaluator.perform_statistical_comparisons(metrics_df)

        assert isinstance(result, pd.DataFrame)

    def test_aggregate_metrics(self, mock_settings: MagicMock) -> None:
        """Test metrics aggregation."""
        from voice_analysis.models.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(mock_settings)

        metrics_df = pd.DataFrame(
            {
                "Model": ["RF", "RF"],
                "Method": ["cv", "cv"],
                "Accuracy": [0.85, 0.87],
                "Number of execution": [1, 2],
            }
        )

        loo_df = pd.DataFrame(
            {
                "Model": ["RF", "RF"],
                "Accuracy": [0.8, 0.9],
                "Execution": [1, 2],
            }
        )

        result = evaluator.aggregate_metrics(metrics_df, loo_df)

        assert isinstance(result, pd.DataFrame)

    def test_aggregate_loo_summary(self, mock_settings: MagicMock) -> None:
        """Test LOO summary aggregation."""
        from voice_analysis.models.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(mock_settings)

        loo_results = pd.DataFrame(
            {
                "Model": ["RF"] * 10,
                "Record": [f"r_{i}" for i in range(10)],
                "Accuracy": np.random.uniform(0.7, 0.9, 10),
                "Execution": [1] * 5 + [2] * 5,
                "True_Label": [0, 1] * 5,
                "Predicted": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        result = evaluator.aggregate_loo_summary(loo_results)

        assert "Mean" in result.columns
        assert "Std" in result.columns
        assert "Median" in result.columns

    def test_aggregate_loo_summary_empty(self, mock_settings: MagicMock) -> None:
        """Test LOO summary with empty DataFrame."""
        from voice_analysis.models.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(mock_settings)

        result = evaluator.aggregate_loo_summary(pd.DataFrame())

        assert result.empty


class TestBootstrapConfidenceIntervals:
    """Test suite for bootstrap confidence intervals."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.evaluation.bootstrap.n_iterations = 100
        mock.evaluation.bootstrap.confidence_level = 0.95
        return mock

    def test_bootstrap_ci_calculation(self, mock_settings: MagicMock) -> None:
        """Test bootstrap confidence interval calculation."""
        from voice_analysis.models.evaluation import BootstrapEvaluator
        from sklearn.metrics import accuracy_score

        evaluator = BootstrapEvaluator(
            n_iterations=mock_settings.evaluation.bootstrap.n_iterations,
            confidence_level=mock_settings.evaluation.bootstrap.confidence_level,
        )

        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 1, 1])

        ci = evaluator.compute_confidence_intervals(y_true, y_pred, accuracy_score)

        assert "ci_lower" in ci
        assert "ci_upper" in ci
        assert ci["ci_lower"] <= ci["ci_upper"]
