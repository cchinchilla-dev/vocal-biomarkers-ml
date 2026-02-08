"""Unit tests for pipeline module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestVoiceAnalysisPipeline:
    """Test suite for VoiceAnalysisPipeline class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for pipeline testing."""
        mock = MagicMock()

        # Project config
        mock.project.name = "test-project"
        mock.project.version = "1.0.0"

        # Paths config
        mock.paths.data_dir = "data"
        mock.paths.results_dir = "data/results"
        mock.paths.outputs.datasets = "datasets"
        mock.paths.outputs.features = "features"
        mock.paths.outputs.metrics = "metrics"
        mock.paths.outputs.visualizations = "visualizations"

        # Data processing
        mock.data_processing.cleaning.variance_threshold = 0.1
        mock.data_processing.cleaning.correlation_threshold = 0.9

        # Feature selection
        mock.feature_selection.enabled = False

        # Resampling
        mock.resampling.enabled = True
        mock.resampling.method = "smote"
        mock.resampling.smote.k_neighbors = 3
        mock.resampling.smote.sampling_strategy = "auto"
        mock.resampling.analyze_distribution = False

        # Models
        mock.models.enabled_models = ["RandomForest"]
        mock.models.test_size = 0.2
        mock.models.cross_validation.n_folds = 2
        mock.models.cross_validation.stratified = True
        mock.models.cross_validation.shuffle = True

        # Hyperparameter search
        mock.hyperparameter_search.method = "pso"
        mock.hyperparameter_search.pso.swarm_size = 3
        mock.hyperparameter_search.pso.max_iterations = 3
        mock.hyperparameter_search.param_ranges = {}

        # Evaluation
        mock.evaluation.metrics = ["accuracy"]
        mock.evaluation.bootstrap.n_iterations = 5
        mock.evaluation.bootstrap.confidence_level = 0.95
        mock.evaluation.leave_one_out.enabled = False

        # Visualization
        mock.visualization.enabled = False

        # Pipeline
        mock.pipeline.stages.analyze_recordings = False
        mock.pipeline.stages.feature_selection = False
        mock.pipeline.stages.train_classifiers = True
        mock.pipeline.stages.evaluate_models = True
        mock.pipeline.stages.save_intermediate = False
        mock.visualization.enabled = False
        mock.pipeline.verbosity = 0

        # Logging
        mock.logging.level = "INFO"
        mock.logging.format = None
        mock.logging.file = None
        mock.logging.console = True

        # Reproducibility
        mock.reproducibility.seeds = [42]
        mock.reproducibility.n_executions = 1
        mock.get_seeds.return_value = [42]

        return mock

    @pytest.fixture
    def sample_dataset(self) -> pd.DataFrame:
        """Create sample dataset for pipeline testing."""
        np.random.seed(42)
        n_samples = 50

        return pd.DataFrame(
            {
                "Record": [f"s_{i}" for i in range(n_samples)],
                "Diagnosed": ["Control"] * 25 + ["Pathological"] * 25,
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
            }
        )

    def test_pipeline_initialization(self, mock_settings: MagicMock) -> None:
        """Test pipeline initialization."""
        with patch("voice_analysis.pipeline.ResultsManager"):
            with patch("voice_analysis.pipeline.DataLoader"):
                from voice_analysis.pipeline import VoiceAnalysisPipeline

                pipeline = VoiceAnalysisPipeline(mock_settings)

                assert pipeline.settings == mock_settings

    def test_pipeline_run_loads_dataset(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that pipeline run loads dataset correctly."""
        with patch("voice_analysis.pipeline.ResultsManager") as mock_rm:
            with patch("voice_analysis.pipeline.DataLoader"):
                mock_rm_instance = MagicMock()
                mock_rm_instance.load_dataset.return_value = sample_dataset
                mock_rm.return_value = mock_rm_instance

                from voice_analysis.pipeline import VoiceAnalysisPipeline

                pipeline = VoiceAnalysisPipeline(mock_settings)

                with patch.object(pipeline, "_clean_data", return_value=sample_dataset):
                    with patch.object(
                        pipeline,
                        "_load_existing_features",
                        return_value=["feature1", "feature2", "feature3"],
                    ):
                        with patch.object(
                            pipeline,
                            "_train_and_evaluate",
                            return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
                        ):
                            with patch.object(pipeline, "_save_results"):
                                pipeline.run()

                mock_rm_instance.load_dataset.assert_called()

    def test_pipeline_clean_dataset_removes_low_variance(self, mock_settings: MagicMock) -> None:
        """Test that pipeline cleaning removes low variance features."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c", "d", "e"],
                "Diagnosed": ["Control", "Control", "Pathological", "Pathological", "Control"],
                "constant": [1, 1, 1, 1, 1],  # Zero variance
                "varying": [1, 2, 3, 4, 5],
            }
        )

        with patch("voice_analysis.pipeline.ResultsManager"):
            with patch("voice_analysis.pipeline.DataLoader"):
                from voice_analysis.pipeline import VoiceAnalysisPipeline

                pipeline = VoiceAnalysisPipeline(mock_settings)
                cleaned = pipeline._clean_data(df)

                assert "constant" not in cleaned.columns
                assert "varying" in cleaned.columns


class TestPipelineFeatureSelection:
    """Test suite for pipeline feature selection stage."""

    @pytest.fixture
    def mock_settings_with_fs(self) -> MagicMock:
        mock = MagicMock()
        mock.feature_selection.enabled = True
        mock.pipeline.stages.feature_selection = True
        mock.pipeline.stages.save_intermediate = False
        mock.visualization.enabled = False
        mock.logging.level = "INFO"
        mock.logging.format = None
        mock.logging.file = None
        mock.logging.console = True
        return mock

    def test_feature_selection_returns_filtered_dataset(
        self, mock_settings_with_fs: MagicMock
    ) -> None:
        """Test that feature selection filters dataset."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c"],
                "Diagnosed": ["Control", "Control", "Pathological"],
                "important_feature": [1, 2, 3],
                "unimportant_feature": [0.1, 0.1, 0.1],
            }
        )

        with patch("voice_analysis.pipeline.ResultsManager"):
            with patch("voice_analysis.pipeline.FeatureSelector") as mock_fs:
                mock_fs_instance = MagicMock()
                mock_fs_instance.select_features.return_value = {
                    "selected_features": ["important_feature"],
                    "bonferroni": pd.DataFrame(),
                    "rfe": pd.DataFrame(),
                    "mutual_information": pd.DataFrame(),
                    "optimal_n_features": 1,
                    "rf_importance": pd.DataFrame(),
                    "lasso": pd.DataFrame(),
                    "cfs": ["important_feature"],
                }
                mock_fs.return_value = mock_fs_instance

                from voice_analysis.pipeline import VoiceAnalysisPipeline

                pipeline = VoiceAnalysisPipeline(mock_settings_with_fs)

                with patch("voice_analysis.pipeline.DataLoader"):
                    filtered, features = pipeline._select_features(df)

                assert "important_feature" in filtered.columns
