"""Unit tests for settings module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from voice_analysis.config.settings import Settings, get_settings, reset_settings


class TestSettings:
    """Test suite for Settings class."""

    @pytest.fixture
    def valid_config(self) -> dict:
        """Return a minimal valid configuration dictionary."""
        return {
            "project": {
                "name": "test-project",
                "version": "1.0.0",
                "description": "Test",
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
                    "control": "raw/bio/control.csv",
                    "pathological": "raw/bio/pathological.csv",
                },
                "outputs": {
                    "datasets": "datasets",
                    "features": "features",
                    "metrics": "metrics",
                    "visualizations": "visualizations",
                },
            },
            "reproducibility": {
                "seeds": [42, 123],
                "n_executions": 2,
                "min_executions": 2,
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
            "logging": {"level": "INFO"},
        }

    def test_settings_from_dict(self, valid_config: dict) -> None:
        """Test creating Settings from dictionary."""
        settings = Settings(**valid_config)

        assert settings.project.name == "test-project"
        assert settings.project.version == "1.0.0"
        assert settings.paths.data_dir == "data"

    def test_settings_from_yaml(self, tmp_path: Path, valid_config: dict) -> None:
        """Test loading Settings from YAML file."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(valid_config, f)

        settings = Settings.from_yaml(config_path)

        assert settings.project.name == "test-project"
        assert settings.reproducibility.seeds == [42, 123]

    def test_settings_from_yaml_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing config file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Settings.from_yaml("nonexistent/config.yaml")

    def test_settings_get_data_path(self, valid_config: dict) -> None:
        """Test get_data_path method."""
        settings = Settings(**valid_config)

        result = settings.get_data_path()

        assert isinstance(result, Path)
        assert str(result) == "data"

    def test_settings_get_results_path(self, valid_config: dict) -> None:
        """Test get_results_path method."""
        settings = Settings(**valid_config)

        result = settings.get_results_path()

        assert isinstance(result, Path)
        assert str(result) == "data/results"

    def test_settings_get_seeds_default(self, valid_config: dict) -> None:
        """Test get_seeds returns configured seeds."""
        settings = Settings(**valid_config)

        result = settings.get_seeds()

        assert result == [42, 123]

    def test_settings_get_seeds_with_n_required(self, valid_config: dict) -> None:
        """Test get_seeds with n_required parameter."""
        settings = Settings(**valid_config)

        result = settings.get_seeds(n_required=5)

        assert len(result) == 5
        assert result[:2] == [42, 123]

    def test_settings_get_seeds_extends_if_needed(self, valid_config: dict) -> None:
        """Test that get_seeds extends seed list if needed."""
        valid_config["reproducibility"]["seeds"] = [42]
        settings = Settings(**valid_config)

        result = settings.get_seeds(n_required=3)

        assert len(result) == 3
        assert result[0] == 42

    def test_settings_extra_fields_forbidden(self, valid_config: dict) -> None:
        """Test that extra fields raise validation error."""
        valid_config["unknown_field"] = "value"

        with pytest.raises(Exception):  # Pydantic ValidationError
            Settings(**valid_config)

    def test_settings_missing_required_fields(self) -> None:
        """Test that missing required fields raise validation error."""
        incomplete_config = {"project": {"name": "test"}}

        with pytest.raises(Exception):  # Pydantic ValidationError
            Settings(**incomplete_config)


class TestGetSettings:
    """Test suite for get_settings function."""

    @pytest.fixture(autouse=True)
    def reset_global_settings(self) -> None:
        """Reset global settings before each test."""
        reset_settings()

    def test_get_settings_with_path(self, tmp_path: Path) -> None:
        """Test get_settings with explicit path."""
        config = {
            "project": {"name": "test", "version": "1.0.0", "description": ""},
            "paths": {
                "data_dir": "data",
                "raw_dir": "data/raw",
                "results_dir": "data/results",
                "recordings": {"control": "c", "pathological": "p"},
                "biomechanical": {"control": "c.csv", "pathological": "p.csv"},
                "outputs": {
                    "datasets": "d",
                    "features": "f",
                    "metrics": "m",
                    "visualizations": "v",
                },
            },
            "reproducibility": {"seeds": [42], "n_executions": 1, "min_executions": 1},
            "data_processing": {
                "audio": {"f0_min": 75, "f0_max": 500},
                "records": {"control": ["all"], "pathological": ["all"]},
                "cleaning": {"variance_threshold": 0.1, "correlation_threshold": 0.9},
            },
            "feature_selection": {"enabled": True},
            "resampling": {"enabled": True},
            "models": {"enabled_models": ["RF"]},
            "hyperparameter_search": {"method": "pso"},
            "evaluation": {"metrics": ["accuracy"]},
            "visualization": {"enabled": False},
            "pipeline": {"stages": {"train_classifiers": True}},
            "logging": {"level": "INFO"},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        settings = get_settings(config_path)

        assert settings.project.name == "test"

    def test_get_settings_caches_result(self, tmp_path: Path) -> None:
        """Test that get_settings caches the settings instance."""
        config = {
            "project": {"name": "cached", "version": "1.0.0", "description": ""},
            "paths": {
                "data_dir": "data",
                "raw_dir": "data/raw",
                "results_dir": "data/results",
                "recordings": {"control": "c", "pathological": "p"},
                "biomechanical": {"control": "c.csv", "pathological": "p.csv"},
                "outputs": {
                    "datasets": "d",
                    "features": "f",
                    "metrics": "m",
                    "visualizations": "v",
                },
            },
            "reproducibility": {"seeds": [42], "n_executions": 1, "min_executions": 1},
            "data_processing": {
                "audio": {"f0_min": 75, "f0_max": 500},
                "records": {"control": ["all"], "pathological": ["all"]},
                "cleaning": {"variance_threshold": 0.1, "correlation_threshold": 0.9},
            },
            "feature_selection": {"enabled": True},
            "resampling": {"enabled": True},
            "models": {"enabled_models": ["RF"]},
            "hyperparameter_search": {"method": "pso"},
            "evaluation": {"metrics": ["accuracy"]},
            "visualization": {"enabled": False},
            "pipeline": {"stages": {"train_classifiers": True}},
            "logging": {"level": "INFO"},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        settings1 = get_settings(config_path)
        settings2 = get_settings()  # Should return cached

        assert settings1 is settings2

    def test_get_settings_no_path_no_cache_raises(self) -> None:
        """Test that get_settings raises error when no path and no cache."""
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            with pytest.raises(FileNotFoundError):
                get_settings()
        finally:
            os.chdir(original_cwd)


class TestResetSettings:
    """Test suite for reset_settings function."""

    def test_reset_clears_cache(self, tmp_path: Path) -> None:
        """Test that reset_settings clears the cached settings."""
        config = {
            "project": {"name": "first", "version": "1.0.0", "description": ""},
            "paths": {
                "data_dir": "data",
                "raw_dir": "data/raw",
                "results_dir": "data/results",
                "recordings": {"control": "c", "pathological": "p"},
                "biomechanical": {"control": "c.csv", "pathological": "p.csv"},
                "outputs": {
                    "datasets": "d",
                    "features": "f",
                    "metrics": "m",
                    "visualizations": "v",
                },
            },
            "reproducibility": {"seeds": [42], "n_executions": 1, "min_executions": 1},
            "data_processing": {
                "audio": {"f0_min": 75, "f0_max": 500},
                "records": {"control": ["all"], "pathological": ["all"]},
                "cleaning": {"variance_threshold": 0.1, "correlation_threshold": 0.9},
            },
            "feature_selection": {"enabled": True},
            "resampling": {"enabled": True},
            "models": {"enabled_models": ["RF"]},
            "hyperparameter_search": {"method": "pso"},
            "evaluation": {"metrics": ["accuracy"]},
            "visualization": {"enabled": False},
            "pipeline": {"stages": {"train_classifiers": True}},
            "logging": {"level": "INFO"},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        settings1 = get_settings(config_path)
        reset_settings()

        config["project"]["name"] = "second"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        settings2 = get_settings(config_path)

        assert settings1.project.name == "first"
        assert settings2.project.name == "second"
        assert settings1 is not settings2
