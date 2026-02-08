"""
Configuration settings management using Pydantic for validation.

This module provides type-safe configuration loading and validation
for the voice analysis pipeline.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .schemas import (
    DataProcessingConfig,
    EvaluationConfig,
    FeatureSelectionConfig,
    HyperparameterSearchConfig,
    LoggingConfig,
    ModelsConfig,
    PathsConfig,
    PipelineConfig,
    ProjectConfig,
    ReproducibilityConfig,
    ResamplingConfig,
    VisualizationConfig,
)


class Settings(BaseModel):
    """Main configuration container with all settings."""

    project: ProjectConfig
    paths: PathsConfig
    reproducibility: ReproducibilityConfig
    data_processing: DataProcessingConfig
    feature_selection: FeatureSelectionConfig
    resampling: ResamplingConfig
    models: ModelsConfig
    hyperparameter_search: HyperparameterSearchConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    pipeline: PipelineConfig
    logging: LoggingConfig

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """
        Load settings from a YAML configuration file.

        Parameters
        ----------
        config_path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        Settings
            Validated settings instance.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the configuration is invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def get_data_path(self) -> Path:
        """Get the base data directory path."""
        return Path(self.paths.data_dir)

    def get_results_path(self) -> Path:
        """Get the results directory path."""
        return Path(self.paths.results_dir)

    def get_seeds(self, n_required: int | None = None) -> list[int]:
        """
        Get seeds for reproducibility, extending if necessary.

        Parameters
        ----------
        n_required : int, optional
            Number of seeds required. If None, uses n_executions.

        Returns
        -------
        list[int]
            List of seed values.
        """
        n_required = n_required or self.reproducibility.n_executions
        seeds = self.reproducibility.seeds.copy()

        while len(seeds) < n_required:
            seeds.append(seeds[-1] + 1)

        return seeds[:n_required]


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings(config_path: str | Path | None = None) -> Settings:
    """
    Get the global settings instance.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file. If not provided, searches for
        config/config.yaml in the project root.

    Returns
    -------
    Settings
        The global settings instance.
    """
    global _settings

    if _settings is None or config_path is not None:
        if config_path is None:
            # Search for config file in common locations
            search_paths = [
                Path("config/config.yaml"),
                Path("../config/config.yaml"),
                Path.cwd() / "config" / "config.yaml",
            ]
            for path in search_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                raise FileNotFoundError("Could not find config.yaml. Please specify config_path.")

        _settings = Settings.from_yaml(config_path)

    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
