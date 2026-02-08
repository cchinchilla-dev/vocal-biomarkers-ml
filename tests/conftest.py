"""Shared pytest fixtures for voice analysis tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from tests.voice_analysis import Settings


# Configuration Fixtures


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock Settings object with default test values."""
    mock = MagicMock()

    # Project config
    mock.project.name = "test-project"
    mock.project.version = "1.0.0"

    # Paths config
    mock.paths.data_dir = "data"
    mock.paths.results_dir = "data/results"
    mock.paths.recordings.control = "raw/control"
    mock.paths.recordings.pathological = "raw/pathological"
    mock.paths.biomechanical.control = "raw/biomechanical/control.csv"
    mock.paths.biomechanical.pathological = "raw/biomechanical/pathological.csv"
    mock.paths.outputs.datasets = "datasets"
    mock.paths.outputs.features = "features"
    mock.paths.outputs.metrics = "metrics"
    mock.paths.outputs.visualizations = "visualizations"

    # Data processing config
    mock.data_processing.audio.f0_min = 75
    mock.data_processing.audio.f0_max = 500
    mock.data_processing.audio.unit = "Hertz"
    mock.data_processing.audio.sample_rate = 44100
    mock.data_processing.audio.n_mfcc = 13
    mock.data_processing.cleaning.variance_threshold = 0.1
    mock.data_processing.cleaning.correlation_threshold = 0.9
    mock.data_processing.records.control = ["all"]
    mock.data_processing.records.pathological = ["all"]

    # Feature selection config
    mock.feature_selection.enabled = True
    mock.feature_selection.rfe.stability_threshold = 0.4
    mock.feature_selection.rfe.n_bootstrap_iterations = 5
    mock.feature_selection.rfe.step = 1
    mock.feature_selection.statistics.alpha = 0.05
    mock.feature_selection.statistics.correction_method = "bonferroni"
    mock.feature_selection.cfs.threshold = 0.8

    # Resampling config
    mock.resampling.enabled = True
    mock.resampling.method = "smote"
    mock.resampling.smote.k_neighbors = 5
    mock.resampling.smote.sampling_strategy = "auto"

    # Models config
    mock.models.enabled_models = ["RandomForest", "SupportVectorMachine"]
    mock.models.test_size = 0.2
    mock.models.cross_validation.n_folds = 3
    mock.models.cross_validation.stratified = True
    mock.models.cross_validation.shuffle = True

    # Hyperparameter search config
    mock.hyperparameter_search.method = "pso"
    mock.hyperparameter_search.pso.swarm_size = 5
    mock.hyperparameter_search.pso.max_iterations = 10
    mock.hyperparameter_search.param_ranges = {}

    # Evaluation config
    mock.evaluation.metrics = ["accuracy", "sensitivity", "specificity"]
    mock.evaluation.bootstrap.n_iterations = 10
    mock.evaluation.bootstrap.confidence_level = 0.95

    # Visualization config
    mock.visualization.enabled = False
    mock.visualization.style.font_family = "serif"
    mock.visualization.style.figure_dpi = 100

    # Pipeline config
    mock.pipeline.stages.analyze_recordings = False
    mock.pipeline.stages.save_intermediate = False
    mock.pipeline.stages.feature_selection = True
    mock.pipeline.stages.train_classifiers = True
    mock.pipeline.verbosity = 0

    # Logging config
    mock.logging.level = "WARNING"
    mock.logging.console = False

    # Reproducibility
    mock.reproducibility.seeds = [42, 123]
    mock.reproducibility.n_executions = 2
    mock.get_seeds.return_value = [42, 123]

    return mock


@pytest.fixture
def sample_config_dict() -> dict:
    """Return a valid configuration dictionary for Settings."""
    return {
        "project": {
            "name": "test-project",
            "version": "1.0.0",
            "description": "Test description",
        },
        "paths": {
            "data_dir": "data",
            "raw_dir": "data/raw",
            "results_dir": "data/results",
            "recordings": {
                "control": "raw/control",
                "pathological": "raw/pathological",
            },
            "biomechanical": {
                "control": "raw/biomechanical/control.csv",
                "pathological": "raw/biomechanical/pathological.csv",
            },
            "outputs": {
                "datasets": "datasets",
                "features": "features",
                "metrics": "metrics",
                "visualizations": "visualizations",
            },
        },
        "reproducibility": {
            "seeds": [42],
            "n_executions": 1,
            "min_executions": 1,
        },
        "data_processing": {
            "audio": {"f0_min": 75, "f0_max": 500},
            "records": {"control": ["all"], "pathological": ["all"]},
            "cleaning": {"variance_threshold": 0.1, "correlation_threshold": 0.9},
        },
        "feature_selection": {"enabled": True},
        "resampling": {"enabled": True},
        "models": {"enabled_models": ["RandomForest"]},
        "hyperparameter_search": {"method": "pso"},
        "evaluation": {"metrics": ["accuracy"]},
        "visualization": {"enabled": False},
        "pipeline": {"stages": {"train_classifiers": True}},
        "logging": {"level": "WARNING"},
    }


# Data Fixtures


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n_samples = 50

    data = {
        "Record": [f"sample_{i}" for i in range(n_samples)],
        "Diagnosed": ["Control"] * 30 + ["Pathological"] * 20,
        "Feature1": np.random.randn(n_samples),
        "Feature2": np.random.randn(n_samples),
        "Feature3": np.random.randn(n_samples),
        "Feature4": np.random.randn(n_samples),
        "Feature5": np.random.randn(n_samples),
        "Pr01": np.random.randn(n_samples) * 100 + 150,  # Biomechanical
        "Pr02": np.random.randn(n_samples) * 0.1 + 0.5,
        "MFCC_01": np.random.randn(n_samples) * 10,
        "MFCC_02": np.random.randn(n_samples) * 10,
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_dataset_with_target() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample features and target for classification."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame(
        {
            "feature1": np.concatenate(
                [
                    np.random.randn(50) - 1,
                    np.random.randn(50) + 1,
                ]
            ),
            "feature2": np.concatenate(
                [
                    np.random.randn(50) + 0.5,
                    np.random.randn(50) - 0.5,
                ]
            ),
            "feature3": np.random.randn(n_samples),  # Noise
        }
    )

    y = pd.Series([0] * 50 + [1] * 50, name="target")

    return X, y


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Create sample true and predicted labels."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    return y_true, y_pred


@pytest.fixture
def sample_biomechanical_df() -> pd.DataFrame:
    """Create sample biomechanical marker data."""
    np.random.seed(42)
    n_samples = 20

    data = {"Record": [f"record_{i}" for i in range(n_samples)]}

    for i in range(1, 23):
        data[f"Pr{i:02d}"] = np.random.randn(n_samples) * 10 + 50

    return pd.DataFrame(data)


@pytest.fixture
def low_variance_dataset() -> pd.DataFrame:
    """Create dataset with some low variance features."""
    np.random.seed(42)
    n_samples = 50

    return pd.DataFrame(
        {
            "Record": [f"s_{i}" for i in range(n_samples)],
            "Diagnosed": ["Control"] * 25 + ["Pathological"] * 25,
            "high_var": np.random.randn(n_samples) * 10,
            "low_var": np.ones(n_samples) * 5 + np.random.randn(n_samples) * 0.01,
            "zero_var": np.ones(n_samples) * 3,
        }
    )


@pytest.fixture
def correlated_dataset() -> pd.DataFrame:
    """Create dataset with correlated features."""
    np.random.seed(42)
    n_samples = 50

    base = np.random.randn(n_samples)

    return pd.DataFrame(
        {
            "Record": [f"s_{i}" for i in range(n_samples)],
            "Diagnosed": ["Control"] * 25 + ["Pathological"] * 25,
            "feature_a": base,
            "feature_b": base + np.random.randn(n_samples) * 0.01,
            "feature_c": np.random.randn(n_samples),
        }
    )


# Path Fixtures


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw" / "control").mkdir(parents=True)
    (data_dir / "raw" / "pathological").mkdir(parents=True)
    (data_dir / "raw" / "biomechanical").mkdir(parents=True)
    (data_dir / "results" / "datasets").mkdir(parents=True)
    (data_dir / "results" / "features").mkdir(parents=True)
    (data_dir / "results" / "metrics").mkdir(parents=True)
    (data_dir / "results" / "visualizations").mkdir(parents=True)
    return data_dir


@pytest.fixture
def tmp_config_file(tmp_path: Path, sample_config_dict: dict) -> Path:
    """Create temporary config file."""
    import yaml

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


# Model Fixtures


@pytest.fixture
def fitted_scaler() -> MagicMock:
    """Create a mock fitted StandardScaler."""
    mock = MagicMock()
    mock.transform.side_effect = lambda x: x
    mock.fit_transform.side_effect = lambda x: x
    mock.inverse_transform.side_effect = lambda x: x
    return mock


@pytest.fixture
def mock_classifier() -> MagicMock:
    """Create a mock classifier."""
    mock = MagicMock()
    mock.fit.return_value = mock
    mock.predict.return_value = np.array([0, 1, 0, 1, 1])
    mock.predict_proba.return_value = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.2, 0.8],
        ]
    )
    mock.get_params.return_value = {"param1": 1, "param2": "value"}
    return mock
