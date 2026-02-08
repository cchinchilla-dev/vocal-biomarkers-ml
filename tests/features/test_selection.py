"""Unit tests for feature selection module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestFeatureSelector:
    """Test suite for FeatureSelector class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for feature selector."""
        mock = MagicMock()
        mock.feature_selection.enabled = True
        mock.feature_selection.rfe.stability_threshold = 0.4
        mock.feature_selection.rfe.n_bootstrap_iterations = 5
        mock.feature_selection.rfe.step = 1
        mock.feature_selection.statistics.alpha = 0.05
        mock.feature_selection.statistics.correction_method = "bonferroni"
        mock.feature_selection.cfs.threshold = 0.8
        mock.feature_selection.mutual_information.thresholds = [0.01, 0.05]
        mock.get_seeds = MagicMock(return_value=[42])
        mock.paths.results_dir = "/tmp/results"
        return mock

    @pytest.fixture
    def sample_dataset(self) -> pd.DataFrame:
        """Create sample dataset for feature selection."""
        np.random.seed(42)
        n_samples = 50

        control = np.random.randn(25)
        pathological = np.random.randn(25) + 2

        return pd.DataFrame(
            {
                "Record": [f"s_{i}" for i in range(n_samples)],
                "Diagnosed": ["Control"] * 25 + ["Pathological"] * 25,
                "good_feature": np.concatenate([control, pathological]),
                "random_feature": np.random.randn(n_samples),
                "constant_feature": np.ones(n_samples),
            }
        )

    def test_selector_initialization(self, mock_settings: MagicMock) -> None:
        """Test FeatureSelector initialization."""
        from voice_analysis.features.selection import FeatureSelector

        selector = FeatureSelector(mock_settings)

        assert selector.settings == mock_settings

    def test_select_features_returns_dict(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test select_features returns expected dictionary structure."""
        from voice_analysis.features.selection import FeatureSelector

        selector = FeatureSelector(mock_settings)

        rfe_results = {
            "rankings": pd.DataFrame(
                {"Feature": ["good_feature"], "Average Ranking": [1], "Stability Index": [0.8]}
            ),
            "optimal_n": 1,
            "analysis": pd.DataFrame(),
        }

        with patch.object(selector, "_rfe_bootstrap_selection", return_value=rfe_results):
            with patch.object(selector, "_statistical_selection", return_value=pd.DataFrame()):
                with patch.object(
                    selector, "_mutual_information_selection", return_value=pd.DataFrame()
                ):
                    with patch.object(
                        selector, "_correlation_based_selection", return_value=["good_feature"]
                    ):
                        with patch.object(
                            selector, "_rf_importance_selection", return_value=pd.DataFrame()
                        ):
                            with patch.object(
                                selector, "_lasso_selection", return_value=pd.DataFrame()
                            ):
                                with patch.object(
                                    selector,
                                    "_get_optimal_features",
                                    return_value=pd.Series(["good_feature"]),
                                ):
                                    result = selector.select_features(sample_dataset)

        assert "selected_features" in result
        assert "bonferroni" in result
        assert "rfe" in result
        assert "mutual_information" in result

    def test_statistical_test_bonferroni_correction(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test Bonferroni correction in statistical tests."""
        from voice_analysis.features.selection import FeatureSelector

        selector = FeatureSelector(mock_settings)

        feature_cols = [c for c in sample_dataset.columns if c not in {"Record", "Diagnosed"}]

        result = selector._statistical_selection(sample_dataset, feature_cols)

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "P-Value" in result.columns or "Adjusted P-Value" in result.columns


class TestRFEBootstrap:
    """Test suite for RFE with bootstrap stability."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.feature_selection.rfe.stability_threshold = 0.4
        mock.feature_selection.rfe.n_bootstrap_iterations = 3
        mock.feature_selection.rfe.step = 1
        mock.get_seeds = MagicMock(return_value=[42, 123, 456])
        mock.paths.results_dir = "/tmp/results"
        return mock

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample data for RFE testing."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.array([0] * 25 + [1] * 25)
        return X, y

    def test_rfe_bootstrap_returns_dataframe(
        self, mock_settings: MagicMock, sample_data: tuple
    ) -> None:
        """Test RFE bootstrap returns DataFrame with stability scores."""
        from voice_analysis.features.selection import FeatureSelector

        selector = FeatureSelector(mock_settings)
        X, y = sample_data

        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        dataset = X_df.copy()
        dataset["Diagnosed"] = ["Control"] * 25 + ["Pathological"] * 25
        dataset["Record"] = [f"r_{i}" for i in range(50)]

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir)
            result = selector._rfe_bootstrap_selection(dataset, [42, 123], results_path)

        assert isinstance(result, dict)
        assert "rankings" in result
        assert isinstance(result["rankings"], pd.DataFrame)
        if len(result["rankings"]) > 0:
            assert (
                "Stability Index" in result["rankings"].columns
                or "Average Ranking" in result["rankings"].columns
            )


class TestMutualInformation:
    """Test suite for mutual information feature selection."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.feature_selection.mutual_information.thresholds = [0.01, 0.05, 0.1]
        return mock

    def test_mutual_information_scores(self, mock_settings: MagicMock) -> None:
        """Test mutual information score computation."""
        from voice_analysis.features.selection import FeatureSelector

        selector = FeatureSelector(mock_settings)

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.array([0] * 25 + [1] * 25)

        X[:25, 0] = X[:25, 0] - 2
        X[25:, 0] = X[25:, 0] + 2

        dataset = pd.DataFrame(X, columns=["f0", "f1", "f2"])
        dataset["Diagnosed"] = ["Control"] * 25 + ["Pathological"] * 25
        dataset["Record"] = [f"r_{i}" for i in range(50)]

        result = selector._mutual_information_selection(dataset)

        assert isinstance(result, pd.DataFrame)
