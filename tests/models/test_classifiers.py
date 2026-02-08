"""Unit tests for classifiers module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from voice_analysis.models.classifiers import (
    ClassifierTrainer,
    ModelConfig,
    ModelRegistry,
)


class TestModelConfig:
    """Test suite for ModelConfig dataclass."""

    def test_model_config_creation(self) -> None:
        """Test creating ModelConfig with all fields."""
        config = ModelConfig(
            abbreviation="RF",
            name="RandomForest",
            instance=RandomForestClassifier(),
            supports_random_state=True,
        )

        assert config.abbreviation == "RF"
        assert config.name == "RandomForest"
        assert isinstance(config.instance, RandomForestClassifier)
        assert config.supports_random_state is True

    def test_model_config_default_random_state(self) -> None:
        """Test that supports_random_state defaults to True."""
        config = ModelConfig(
            abbreviation="TEST",
            name="Test",
            instance=MagicMock(),
        )

        assert config.supports_random_state is True


class TestModelRegistry:
    """Test suite for ModelRegistry class."""

    def test_registry_contains_expected_models(self) -> None:
        """Test that registry contains all expected models."""
        expected_models = {
            "AdaBoost",
            "KNearestNeighbor",
            "SupportVectorMachine",
            "Bagging",
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
            "LightGBM",
            "DecisionTree",
        }

        assert expected_models.issubset(ModelRegistry.MODELS.keys())

    def test_get_model_returns_config(self) -> None:
        """Test that get_model returns ModelConfig."""
        config = ModelRegistry.get_model("RandomForest")

        assert isinstance(config, ModelConfig)
        assert config.name == "RandomForest"
        assert config.abbreviation == "RF"

    def test_get_model_unknown_raises(self) -> None:
        """Test that get_model raises for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.get_model("UnknownModel")

    def test_get_fresh_instance(self) -> None:
        """Test that get_fresh_instance returns unfitted model."""
        instance1 = ModelRegistry.get_fresh_instance("RandomForest")
        instance2 = ModelRegistry.get_fresh_instance("RandomForest")

        assert isinstance(instance1, RandomForestClassifier)
        assert instance1 is not instance2  # Different instances

    def test_knn_does_not_support_random_state(self) -> None:
        """Test that KNN is marked as not supporting random_state."""
        config = ModelRegistry.get_model("KNearestNeighbor")

        assert config.supports_random_state is False

    def test_svm_has_probability_enabled(self) -> None:
        """Test that SVM instance has probability=True."""
        config = ModelRegistry.get_model("SupportVectorMachine")

        assert isinstance(config.instance, SVC)
        assert config.instance.probability is True

    def test_all_models_have_required_fields(self) -> None:
        """Test that all registered models have required fields."""
        for name, config in ModelRegistry.MODELS.items():
            assert config.abbreviation, f"{name} missing abbreviation"
            assert config.name, f"{name} missing name"
            assert config.instance is not None, f"{name} missing instance"


class TestClassifierTrainer:
    """Test suite for ClassifierTrainer class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for ClassifierTrainer."""
        mock = MagicMock()
        mock.models.enabled_models = ["RandomForest", "KNearestNeighbor"]
        mock.models.test_size = 0.2
        mock.models.cross_validation.n_folds = 3
        mock.models.cross_validation.stratified = True
        mock.models.cross_validation.shuffle = True
        mock.hyperparameter_search.method = "pso"
        mock.hyperparameter_search.pso.swarm_size = 5
        mock.hyperparameter_search.pso.max_iterations = 5
        mock.hyperparameter_search.param_ranges = {}
        mock.evaluation.leave_one_out.enabled = False
        return mock

    @pytest.fixture
    def trainer(self, mock_settings: MagicMock) -> ClassifierTrainer:
        """Create a ClassifierTrainer instance."""
        return ClassifierTrainer(mock_settings)

    @pytest.fixture
    def sample_dataset(self) -> pd.DataFrame:
        """Create sample dataset for training."""
        np.random.seed(42)
        n_samples = 60

        return pd.DataFrame(
            {
                "Record": [f"r_{i}" for i in range(n_samples)],
                "Diagnosed": ["Control"] * 30 + ["Pathological"] * 30,
                "Feature1": np.concatenate(
                    [
                        np.random.randn(30) - 1,
                        np.random.randn(30) + 1,
                    ]
                ),
                "Feature2": np.concatenate(
                    [
                        np.random.randn(30) + 0.5,
                        np.random.randn(30) - 0.5,
                    ]
                ),
            }
        )

    def test_trainer_initialization(
        self, trainer: ClassifierTrainer, mock_settings: MagicMock
    ) -> None:
        """Test ClassifierTrainer initialization."""
        assert trainer.settings == mock_settings
        assert trainer.metrics_calculator is not None
        assert trainer.hyperparameter_optimizer is not None
        assert trainer.label_encoder is not None
        assert trainer.scaler is not None

    def test_train_all_returns_dict(
        self, trainer: ClassifierTrainer, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that train_all returns a dictionary."""
        with patch.object(trainer.hyperparameter_optimizer, "optimize", return_value=None):
            result = trainer.train_all(
                sample_dataset,
                target_column="Diagnosed",
                execution_number=1,
                seed=42,
            )

        assert isinstance(result, dict)
        assert "metrics" in result
        assert "patient_probabilities" in result

    def test_train_all_metrics_dataframe(
        self, trainer: ClassifierTrainer, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that train_all returns metrics DataFrame."""
        with patch.object(trainer.hyperparameter_optimizer, "optimize", return_value=None):
            result = trainer.train_all(
                sample_dataset,
                target_column="Diagnosed",
                execution_number=1,
                seed=42,
            )

        metrics_df = result["metrics"]
        assert isinstance(metrics_df, pd.DataFrame)
        assert "Model" in metrics_df.columns
        assert "Accuracy" in metrics_df.columns

    def test_train_all_trains_enabled_models(
        self, trainer: ClassifierTrainer, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that train_all trains all enabled models."""
        with patch.object(trainer.hyperparameter_optimizer, "optimize", return_value=None):
            result = trainer.train_all(
                sample_dataset,
                target_column="Diagnosed",
                execution_number=1,
                seed=42,
            )

        metrics_df = result["metrics"]
        trained_models = set(metrics_df["Model"].unique())

        assert "RandomForest" in trained_models
        assert "KNearestNeighbor" in trained_models

    def test_train_all_with_hyperparameter_optimization(
        self, trainer: ClassifierTrainer, sample_dataset: pd.DataFrame
    ) -> None:
        """Test train_all with hyperparameter optimization."""
        mock_params = {"n_estimators": 50, "max_depth": 10}

        with patch.object(trainer.hyperparameter_optimizer, "optimize", return_value=mock_params):
            result = trainer.train_all(
                sample_dataset,
                target_column="Diagnosed",
                execution_number=1,
                seed=42,
            )

        assert "metrics" in result

    def test_train_all_execution_number_in_results(
        self, trainer: ClassifierTrainer, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that execution number is included in results."""
        with patch.object(trainer.hyperparameter_optimizer, "optimize", return_value=None):
            result = trainer.train_all(
                sample_dataset,
                target_column="Diagnosed",
                execution_number=5,
                seed=42,
            )

        metrics_df = result["metrics"]
        assert all(metrics_df["Number of execution"] == 5)

    def test_train_all_handles_missing_model(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that train_all handles unknown models gracefully."""
        mock_settings.models.enabled_models = ["RandomForest", "UnknownModel"]
        trainer = ClassifierTrainer(mock_settings)

        with patch.object(trainer.hyperparameter_optimizer, "optimize", return_value=None):
            result = trainer.train_all(
                sample_dataset,
                target_column="Diagnosed",
                execution_number=1,
                seed=42,
            )

        assert "metrics" in result

    def test_patient_probabilities_dataframe(
        self, trainer: ClassifierTrainer, sample_dataset: pd.DataFrame
    ) -> None:
        """Test patient probabilities DataFrame structure."""
        with patch.object(trainer.hyperparameter_optimizer, "optimize", return_value=None):
            result = trainer.train_all(
                sample_dataset,
                target_column="Diagnosed",
                execution_number=1,
                seed=42,
            )

        probs_df = result["patient_probabilities"]
        assert isinstance(probs_df, pd.DataFrame)

    def test_metrics_values_in_valid_range(
        self, trainer: ClassifierTrainer, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that metric values are in valid range [0, 1]."""
        with patch.object(trainer.hyperparameter_optimizer, "optimize", return_value=None):
            result = trainer.train_all(
                sample_dataset,
                target_column="Diagnosed",
                execution_number=1,
                seed=42,
            )

        metrics_df = result["metrics"]
        numeric_cols = ["Accuracy", "Sensitivity", "Specificity"]

        for col in numeric_cols:
            if col in metrics_df.columns:
                assert all(metrics_df[col] >= 0), f"{col} has negative values"
                assert all(metrics_df[col] <= 1), f"{col} > 1"
