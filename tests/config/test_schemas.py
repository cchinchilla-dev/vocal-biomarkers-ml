"""Unit tests for configuration schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestProjectConfig:
    """Test suite for ProjectConfig schema."""

    def test_valid_project_config(self) -> None:
        """Test creating valid ProjectConfig."""
        from voice_analysis.config.schemas import ProjectConfig

        config = ProjectConfig(
            name="test-project",
            version="1.0.0",
            description="Test description",
        )

        assert config.name == "test-project"
        assert config.version == "1.0.0"

    def test_project_config_defaults(self) -> None:
        """Test ProjectConfig default values."""
        from voice_analysis.config.schemas import ProjectConfig

        config = ProjectConfig(name="test", version="1.0.0")

        assert config.description == ""


class TestAudioConfig:
    """Test suite for AudioConfig schema."""

    def test_valid_audio_config(self) -> None:
        """Test creating valid AudioConfig."""
        from voice_analysis.config.schemas import AudioConfig

        config = AudioConfig(
            f0_min=75,
            f0_max=500,
            unit="Hertz",
            sample_rate=44100,
            n_mfcc=13,
        )

        assert config.f0_min == 75
        assert config.f0_max == 500

    def test_audio_config_defaults(self) -> None:
        """Test AudioConfig default values."""
        from voice_analysis.config.schemas import AudioConfig

        config = AudioConfig()

        assert config.f0_min == 75
        assert config.f0_max == 500
        assert config.n_mfcc == 13

    def test_audio_config_validation(self) -> None:
        """Test AudioConfig validation."""
        from voice_analysis.config.schemas import AudioConfig

        with pytest.raises(ValidationError):
            AudioConfig(f0_min=-1)  # Must be positive

        with pytest.raises(ValidationError):
            AudioConfig(n_mfcc=0)  # Must be >= 1


class TestCleaningConfig:
    """Test suite for CleaningConfig schema."""

    def test_valid_cleaning_config(self) -> None:
        """Test creating valid CleaningConfig."""
        from voice_analysis.config.schemas import CleaningConfig

        config = CleaningConfig(
            variance_threshold=0.1,
            correlation_threshold=0.9,
        )

        assert config.variance_threshold == 0.1
        assert config.correlation_threshold == 0.9

    def test_cleaning_config_threshold_bounds(self) -> None:
        """Test CleaningConfig threshold bounds validation."""
        from voice_analysis.config.schemas import CleaningConfig

        with pytest.raises(ValidationError):
            CleaningConfig(variance_threshold=-0.1)

        with pytest.raises(ValidationError):
            CleaningConfig(correlation_threshold=1.5)


class TestRFEConfig:
    """Test suite for RFEConfig schema."""

    def test_valid_rfe_config(self) -> None:
        """Test creating valid RFEConfig."""
        from voice_analysis.config.schemas import RFEConfig

        config = RFEConfig(
            stability_threshold=0.4,
            n_bootstrap_iterations=20,
            step=1,
        )

        assert config.stability_threshold == 0.4
        assert config.n_bootstrap_iterations == 20

    def test_rfe_config_validation(self) -> None:
        """Test RFEConfig validation."""
        from voice_analysis.config.schemas import RFEConfig

        with pytest.raises(ValidationError):
            RFEConfig(stability_threshold=1.5)  # Must be <= 1.0

        with pytest.raises(ValidationError):
            RFEConfig(n_bootstrap_iterations=0)  # Must be >= 1


class TestResamplingConfig:
    """Test suite for ResamplingConfig schema."""

    def test_valid_resampling_config(self) -> None:
        """Test creating valid ResamplingConfig."""
        from voice_analysis.config.schemas import ResamplingConfig

        config = ResamplingConfig(
            enabled=True,
            method="smote",
        )

        assert config.enabled is True
        assert config.method == "smote"

    def test_resampling_config_valid_methods(self) -> None:
        """Test ResamplingConfig accepts valid methods."""
        from voice_analysis.config.schemas import ResamplingConfig

        valid_methods = ["smote", "borderline_smote", "smote_tomek", "smote_enn"]

        for method in valid_methods:
            config = ResamplingConfig(method=method)
            assert config.method == method

    def test_resampling_config_invalid_method(self) -> None:
        """Test ResamplingConfig rejects invalid methods."""
        from voice_analysis.config.schemas import ResamplingConfig

        with pytest.raises(ValidationError):
            ResamplingConfig(method="invalid_method")


class TestModelsConfig:
    """Test suite for ModelsConfig schema."""

    def test_valid_models_config(self) -> None:
        """Test creating valid ModelsConfig."""
        from voice_analysis.config.schemas import ModelsConfig

        config = ModelsConfig(
            enabled_models=["RandomForest", "SupportVectorMachine"],
            test_size=0.2,
        )

        assert "RandomForest" in config.enabled_models
        assert config.test_size == 0.2

    def test_models_config_test_size_bounds(self) -> None:
        """Test ModelsConfig test_size bounds."""
        from voice_analysis.config.schemas import ModelsConfig

        with pytest.raises(ValidationError):
            ModelsConfig(test_size=0.05)  # Must be >= 0.1

        with pytest.raises(ValidationError):
            ModelsConfig(test_size=0.6)  # Must be <= 0.5


class TestCrossValidationConfig:
    """Test suite for CrossValidationConfig schema."""

    def test_valid_cv_config(self) -> None:
        """Test creating valid CrossValidationConfig."""
        from voice_analysis.config.schemas import CrossValidationConfig

        config = CrossValidationConfig(
            n_folds=5,
            stratified=True,
            shuffle=True,
        )

        assert config.n_folds == 5
        assert config.stratified is True

    def test_cv_config_min_folds(self) -> None:
        """Test CrossValidationConfig minimum folds."""
        from voice_analysis.config.schemas import CrossValidationConfig

        with pytest.raises(ValidationError):
            CrossValidationConfig(n_folds=1)  # Must be >= 2


class TestEvaluationConfig:
    """Test suite for EvaluationConfig schema."""

    def test_valid_evaluation_config(self) -> None:
        """Test creating valid EvaluationConfig."""
        from voice_analysis.config.schemas import EvaluationConfig

        config = EvaluationConfig(
            metrics=["accuracy", "sensitivity", "specificity"],
        )

        assert "accuracy" in config.metrics

    def test_evaluation_config_default_metrics(self) -> None:
        """Test EvaluationConfig default metrics."""
        from voice_analysis.config.schemas import EvaluationConfig

        config = EvaluationConfig()

        assert "accuracy" in config.metrics
        assert "balanced_accuracy" in config.metrics


class TestVisualizationConfig:
    """Test suite for VisualizationConfig schema."""

    def test_valid_visualization_config(self) -> None:
        """Test creating valid VisualizationConfig."""
        from voice_analysis.config.schemas import VisualizationConfig

        config = VisualizationConfig(enabled=True)

        assert config.enabled is True

    def test_visualization_config_defaults(self) -> None:
        """Test VisualizationConfig default values."""
        from voice_analysis.config.schemas import VisualizationConfig

        config = VisualizationConfig()

        assert config.style.font_family == "serif"
        assert config.style.figure_dpi == 300
