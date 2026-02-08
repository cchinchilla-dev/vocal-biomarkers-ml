"""
Input/Output utilities module.

This module provides utilities for reading and writing data files,
managing results, and handling file operations.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ResultsManager:
    """
    Manager for saving and loading analysis results.

    This class provides a centralized interface for managing
    all output files from the analysis pipeline.

    Parameters
    ----------
    base_path : Path
        Base directory for results.
    output_config : object
        Configuration object with output subdirectory names.

    Attributes
    ----------
    base_path : Path
        Base results directory.
    datasets_path : Path
        Directory for dataset files.
    features_path : Path
        Directory for feature selection results.
    metrics_path : Path
        Directory for model metrics.
    visualization_path : Path
        Directory for visualizations.
    """

    def __init__(self, base_path: Path, output_config):
        self.base_path = Path(base_path)
        self.output_config = output_config

        # Create subdirectories
        self.datasets_path = self.base_path / output_config.datasets
        self.features_path = self.base_path / output_config.features
        self.metrics_path = self.base_path / output_config.metrics
        self.visualization_path = self.base_path / output_config.visualizations

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """
        Create all required directories.

        Creates the base path and all subdirectories for datasets,
        features, metrics, and visualizations if they don't exist.
        """
        for path in [
            self.base_path,
            self.datasets_path,
            self.features_path,
            self.metrics_path,
            self.visualization_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_dataset(
        self,
        data: pd.DataFrame,
        filename: str,
        float_format: str = "%.4f",
    ) -> Path:
        """
        Save a dataset to CSV.

        Parameters
        ----------
        data : DataFrame
            Dataset to save.
        filename : str
            Output filename.
        float_format : str
            Format string for floating point numbers.

        Returns
        -------
        Path
            Path to saved file.
        """
        filepath = self.datasets_path / filename
        data.to_csv(filepath, index=False, float_format=float_format)
        logger.info(f"Saved dataset: {filepath}")
        return filepath

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load a dataset from CSV.

        Parameters
        ----------
        filename : str
            Filename to load.

        Returns
        -------
        DataFrame
            Loaded dataset.
        """
        filepath = self.datasets_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        logger.info(f"Loading dataset: {filepath}")
        return pd.read_csv(filepath)

    def save_features(
        self,
        data: pd.DataFrame,
        filename: str,
        float_format: str = "%.4f",
    ) -> Path:
        """Save feature selection results."""
        filepath = self.features_path / filename
        data.to_csv(filepath, index=False, float_format=float_format)
        logger.debug(f"Saved features: {filepath}")
        return filepath

    def load_features(self, filename: str) -> pd.DataFrame:
        """Load feature selection results."""
        filepath = self.features_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Features file not found: {filepath}")
        return pd.read_csv(filepath)

    def save_metrics(
        self,
        data: pd.DataFrame,
        filename: str,
        float_format: str = "%.4f",
    ) -> Path:
        """Save model metrics."""
        filepath = self.metrics_path / filename
        data.to_csv(filepath, index=False, float_format=float_format)
        logger.info(f"Saved metrics: {filepath}")
        return filepath

    def load_metrics(self, filename: str) -> pd.DataFrame:
        """Load model metrics."""
        filepath = self.metrics_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")
        return pd.read_csv(filepath)

    def get_visualization_path(self) -> Path:
        """Get the visualization directory path."""
        return self.visualization_path


def save_dataframe(
    data: pd.DataFrame,
    filepath: str | Path,
    float_format: str = "%.4f",
) -> None:
    """
    Save DataFrame to CSV file.

    Parameters
    ----------
    data : DataFrame
        Data to save.
    filepath : str or Path
        Output file path.
    float_format : str
        Format for floating point numbers.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False, float_format=float_format)
    logger.debug(f"Saved: {filepath}")


def load_dataframe(filepath: str | Path) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    DataFrame
        Loaded data.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    Path
        The directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
