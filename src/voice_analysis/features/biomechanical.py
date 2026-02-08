"""
Biomechanical features module.

This module provides utilities for handling biomechanical markers
extracted from the Voice Clinical Systems App Online Lab.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class BiomechanicalMarker:
    """Definition of a biomechanical marker."""

    code: str
    name: str
    category: str
    description: str
    unit: str = ""

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class BiomechanicalMarkerRegistry:
    """
    Registry of biomechanical markers from Voice Clinical Systems.

    This class provides definitions and metadata for all 22 biomechanical
    markers analyzed by the App Online Lab.
    """

    # Category definitions
    CATEGORIES = {
        "A": "Fundamental Frequency",
        "B": "Harmony in the movement of the vocal folds",
        "C": "Phase characteristics",
        "D": "Muscle tension",
        "E": "Sufficiency of the seal",
        "F": "Tension and instability",
        "G": "Edge spacing",
        "H": "Mucosal wave and edema correlates",
        "I": "Mass correlates",
    }

    # All markers
    MARKERS = {
        "Pr01": BiomechanicalMarker(
            code="Pr01",
            name="Fundamental Frequency",
            category="A",
            description="Basic frequency of vocal fold vibration",
            unit="Hz",
        ),
        "Pr02": BiomechanicalMarker(
            code="Pr02",
            name="Ratio of cycles in closing",
            category="B",
            description="Ratio of cycles during the closing phase",
            unit="ratio",
        ),
        "Pr03": BiomechanicalMarker(
            code="Pr03",
            name="Asymmetry",
            category="C",
            description="Asymmetry between opening and closing phases",
            unit="",
        ),
        "Pr04": BiomechanicalMarker(
            code="Pr04",
            name="Duration of closed phase",
            category="C",
            description="Percentage of cycle in closed phase",
            unit="%",
        ),
        "Pr05": BiomechanicalMarker(
            code="Pr05",
            name="Duration of open phase",
            category="C",
            description="Percentage of cycle in open phase",
            unit="%",
        ),
        "Pr06": BiomechanicalMarker(
            code="Pr06",
            name="Duration opening",
            category="C",
            description="Duration of the opening movement",
            unit="%",
        ),
        "Pr07": BiomechanicalMarker(
            code="Pr07",
            name="Duration closing",
            category="C",
            description="Duration of the closing movement",
            unit="%",
        ),
        "Pr08": BiomechanicalMarker(
            code="Pr08",
            name="Stress Ratio",
            category="D",
            description="Ratio indicating muscle tension during phonation",
            unit="UR",
        ),
        "Pr09": BiomechanicalMarker(
            code="Pr09",
            name="Glottic closure force",
            category="D",
            description="Force of glottal closure",
            unit="",
        ),
        "Pr10": BiomechanicalMarker(
            code="Pr10",
            name="Efficiency index",
            category="E",
            description="Efficiency of vocal fold closure",
            unit="",
        ),
        "Pr11": BiomechanicalMarker(
            code="Pr11",
            name="Gap width",
            category="E",
            description="Width of glottal gap during closure",
            unit="",
        ),
        "Pr12": BiomechanicalMarker(
            code="Pr12",
            name="Gap size",
            category="E",
            description="Size of glottal gap as percentage",
            unit="%",
        ),
        "Pr13": BiomechanicalMarker(
            code="Pr13",
            name="Instability index",
            category="F",
            description="Index of vibration instability during phonation",
            unit="",
        ),
        "Pr14": BiomechanicalMarker(
            code="Pr14",
            name="Index variation in amplitude",
            category="F",
            description="Variation in amplitude across the voice sample",
            unit="",
        ),
        "Pr15": BiomechanicalMarker(
            code="Pr15",
            name="Vibration locking index",
            category="G",
            description="Index of vibration locking",
            unit="",
        ),
        "Pr16": BiomechanicalMarker(
            code="Pr16",
            name="Amplitude index",
            category="G",
            description="Index of vibration amplitude",
            unit="",
        ),
        "Pr17": BiomechanicalMarker(
            code="Pr17",
            name="Index OM closed phase",
            category="H",
            description="Mucosal wave index during closed phase",
            unit="",
        ),
        "Pr18": BiomechanicalMarker(
            code="Pr18",
            name="Index OM opening phase",
            category="H",
            description="Mucosal wave index during opening phase",
            unit="",
        ),
        "Pr19": BiomechanicalMarker(
            code="Pr19",
            name="Adec. OM Closed",
            category="H",
            description="Adequacy of mucosal wave in closed phase",
            unit="",
        ),
        "Pr20": BiomechanicalMarker(
            code="Pr20",
            name="Adec OM Opening",
            category="H",
            description="Adequacy of mucosal wave in opening phase",
            unit="",
        ),
        "Pr21": BiomechanicalMarker(
            code="Pr21",
            name="Structural imbalance index",
            category="I",
            description="Index of structural imbalance between vocal folds",
            unit="",
        ),
        "Pr22": BiomechanicalMarker(
            code="Pr22",
            name="Mass alteration index",
            category="I",
            description="Index indicating mass alterations in vocal folds",
            unit="",
        ),
    }

    # Markers identified as significant for COVID-19 in the paper
    COVID_SIGNIFICANT_MARKERS = ["Pr06", "Pr07", "Pr13", "Pr14", "Pr17"]

    @classmethod
    def get_marker(cls, code: str) -> BiomechanicalMarker | None:
        """
        Get marker definition by code.

        Parameters
        ----------
        code : str
            Marker code (e.g., 'Pr01').

        Returns
        -------
        BiomechanicalMarker or None
            Marker definition if found, None otherwise.
        """
        return cls.MARKERS.get(code)

    @classmethod
    def get_all_codes(cls) -> list[str]:
        """
        Get all marker codes.

        Returns
        -------
        list[str]
            List of all marker codes.
        """
        return list(cls.MARKERS.keys())

    @classmethod
    def get_markers_by_category(cls, category: str) -> list[BiomechanicalMarker]:
        """
        Get all markers in a category.

        Parameters
        ----------
        category : str
            Category identifier (e.g., 'A', 'B').

        Returns
        -------
        list[BiomechanicalMarker]
            List of markers in the specified category.
        """
        return [m for m in cls.MARKERS.values() if m.category == category]

    @classmethod
    def get_covid_significant_markers(cls) -> list[BiomechanicalMarker]:
        """
        Get markers identified as significant for COVID-19.

        Returns
        -------
        list[BiomechanicalMarker]
            List of COVID-19 significant markers.
        """
        return [cls.MARKERS[code] for code in cls.COVID_SIGNIFICANT_MARKERS]

    @classmethod
    def get_description(cls, code: str) -> str:
        """
        Get description for a marker code.

        Parameters
        ----------
        code : str
            Marker code (e.g., 'Pr01').

        Returns
        -------
        str
            Description of the marker or 'Unknown marker' if not found.
        """
        marker = cls.MARKERS.get(code)
        return marker.description if marker else "Unknown marker"


class BiomechanicalFeatureExtractor:
    """
    Extractor for biomechanical features from Voice Clinical Systems data.

    This class handles loading and processing of biomechanical marker
    data from CSV files exported from the App Online Lab.

    Parameters
    ----------
    settings : Settings
        Configuration settings.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry = BiomechanicalMarkerRegistry()

    def load_from_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load biomechanical data from CSV file.

        Parameters
        ----------
        filepath : Path
            Path to CSV file.

        Returns
        -------
        DataFrame
            Biomechanical marker data.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Biomechanical data not found: {filepath}")

        df = pd.read_csv(filepath)

        # Validate expected columns
        self._validate_columns(df)

        logger.info(f"Loaded biomechanical data: {df.shape[0]} records, {df.shape[1]} columns")

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that expected columns are present.

        Parameters
        ----------
        df : DataFrame
            DataFrame to validate.

        Raises
        ------
        ValueError
            If required 'Record' column is missing.
        """
        if "Record" not in df.columns:
            raise ValueError("Missing required 'Record' column")

        # Check for biomechanical marker columns
        marker_cols = [c for c in df.columns if c.startswith("Pr")]
        if not marker_cols:
            logger.warning("No biomechanical marker columns (Pr*) found")

    def get_marker_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Get list of biomechanical marker columns in dataset.

        Parameters
        ----------
        df : DataFrame
            Dataset to analyze.

        Returns
        -------
        list[str]
            List of column names starting with 'Pr'.
        """
        return [c for c in df.columns if c.startswith("Pr")]

    def extract_significant_markers(
        self,
        df: pd.DataFrame,
        include_record: bool = True,
    ) -> pd.DataFrame:
        """
        Extract only the COVID-19 significant markers.

        Parameters
        ----------
        df : DataFrame
            Full biomechanical dataset.
        include_record : bool
            Whether to include the Record column.

        Returns
        -------
        DataFrame
            Dataset with only significant markers.
        """
        cols = list(self.registry.COVID_SIGNIFICANT_MARKERS)

        if include_record and "Record" in df.columns:
            cols = ["Record"] + cols

        # Filter to existing columns
        cols = [c for c in cols if c in df.columns]

        return df[cols]

    def generate_marker_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a report of marker statistics.

        Parameters
        ----------
        df : DataFrame
            Biomechanical dataset.

        Returns
        -------
        DataFrame
            Report with statistics for each marker.
        """
        marker_cols = self.get_marker_columns(df)

        report_data = []
        for col in marker_cols:
            marker = self.registry.get_marker(col)
            stats = df[col].describe()

            report_data.append(
                {
                    "Marker": col,
                    "Name": marker.name if marker else "Unknown",
                    "Category": marker.category if marker else "Unknown",
                    "Mean": stats["mean"],
                    "Std": stats["std"],
                    "Min": stats["min"],
                    "Max": stats["max"],
                    "Missing": df[col].isna().sum(),
                }
            )

        return pd.DataFrame(report_data)


def load_biomechanical_data(
    control_path: Path,
    pathological_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load biomechanical data for both groups.

    Parameters
    ----------
    control_path : Path
        Path to control group biomechanical data.
    pathological_path : Path
        Path to pathological group biomechanical data.

    Returns
    -------
    tuple
        (control_df, pathological_df)
    """
    control_df = pd.read_csv(control_path)
    pathological_df = pd.read_csv(pathological_path)

    logger.info(f"Loaded control biomechanical data: {control_df.shape}")
    logger.info(f"Loaded pathological biomechanical data: {pathological_df.shape}")

    return control_df, pathological_df
