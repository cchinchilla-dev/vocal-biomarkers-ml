"""Unit tests for hyperparameter search module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier


class TestHyperparameterOptimizer:
    """Test suite for HyperparameterOptimizer class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.hyperparameter_search.method = "pso"
        mock.hyperparameter_search.pso.swarm_size = 5
        mock.hyperparameter_search.pso.max_iterations = 3
        mock.hyperparameter_search.pso.min_func = 1e-6
        mock.hyperparameter_search.pso.min_step = 1e-6
        mock.hyperparameter_search.random_search.n_iterations = 5
        mock.hyperparameter_search.random_search.scoring = "accuracy"
        mock.hyperparameter_search.param_ranges = {
            "RandomForest": {
                "n_estimators": [10, 100],
                "max_depth": [5, 20],
            }
        }
        mock.models.cross_validation.n_folds = 2
        return mock

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.array([0] * 25 + [1] * 25)
        return X, y

    def test_optimizer_initialization(self, mock_settings: MagicMock) -> None:
        """Test HyperparameterOptimizer initialization."""
        from voice_analysis.models.hyperparameter_search import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(mock_settings)

        assert optimizer.method == "pso"

    def test_optimize_returns_dict_or_none(
        self, mock_settings: MagicMock, sample_data: tuple
    ) -> None:
        """Test optimize returns dict or None."""
        from voice_analysis.models.hyperparameter_search import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(mock_settings)
        X, y = sample_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        with patch.object(optimizer, "_pso_search", return_value={"n_estimators": 50}):
            result = optimizer.optimize("RandomForest", model, X, y, seed=42)

        assert result is None or isinstance(result, dict)

    def test_optimize_unknown_model(self, mock_settings: MagicMock, sample_data: tuple) -> None:
        """Test optimize with unknown model returns None."""
        mock_settings.hyperparameter_search.param_ranges = {}

        from voice_analysis.models.hyperparameter_search import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(mock_settings)
        X, y = sample_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = optimizer.optimize("UnknownModel", model, X, y, seed=42)

        assert result is None


class TestPSOOptimization:
    """Test suite for PSO optimization."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for PSO."""
        mock = MagicMock()
        mock.hyperparameter_search.method = "pso"
        mock.hyperparameter_search.pso.swarm_size = 3
        mock.hyperparameter_search.pso.max_iterations = 2
        mock.hyperparameter_search.pso.min_func = 1e-6
        mock.hyperparameter_search.pso.min_step = 1e-6
        mock.hyperparameter_search.param_ranges = {
            "RandomForest": {
                "n_estimators": [10, 50],
                "max_depth": [2, 10],
            }
        }
        mock.models.cross_validation.n_folds = 2
        return mock

    def test_pso_optimization_runs(self, mock_settings: MagicMock) -> None:
        """Test PSO optimization executes without error."""
        from voice_analysis.models.hyperparameter_search import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(mock_settings)

        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.array([0] * 15 + [1] * 15)

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        with patch("voice_analysis.models.hyperparameter_search.pso") as mock_pso:
            mock_pso.return_value = ([20, 5], 0.8)

            param_ranges = mock_settings.hyperparameter_search.param_ranges.get("RandomForest", {})
            result = optimizer._pso_search("RandomForest", model, X, y, param_ranges)

            mock_pso.assert_called_once()


class TestCategoricalParameterMappings:
    """Test suite for categorical parameter mappings."""

    def test_categorical_mappings_exist(self) -> None:
        """Test that categorical mappings are defined."""
        from voice_analysis.models.hyperparameter_search import CATEGORICAL_PARAM_MAPPINGS

        assert "KNearestNeighbor" in CATEGORICAL_PARAM_MAPPINGS
        assert "SupportVectorMachine" in CATEGORICAL_PARAM_MAPPINGS
        assert "RandomForest" in CATEGORICAL_PARAM_MAPPINGS

    def test_knn_categorical_params(self) -> None:
        """Test KNN categorical parameter mappings."""
        from voice_analysis.models.hyperparameter_search import CATEGORICAL_PARAM_MAPPINGS

        knn_params = CATEGORICAL_PARAM_MAPPINGS["KNearestNeighbor"]

        assert "weights" in knn_params
        assert "uniform" in knn_params["weights"]
        assert "distance" in knn_params["weights"]

    def test_svm_categorical_params(self) -> None:
        """Test SVM categorical parameter mappings."""
        from voice_analysis.models.hyperparameter_search import CATEGORICAL_PARAM_MAPPINGS

        svm_params = CATEGORICAL_PARAM_MAPPINGS["SupportVectorMachine"]

        assert "kernel" in svm_params
        assert "rbf" in svm_params["kernel"]
