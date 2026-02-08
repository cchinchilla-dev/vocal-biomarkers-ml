"""Unit tests for biomechanical features module."""

from __future__ import annotations

import pytest


class TestBiomechanicalMarker:
    """Test suite for BiomechanicalMarker dataclass."""

    def test_marker_creation(self) -> None:
        """Test creating a BiomechanicalMarker."""
        from voice_analysis.features.biomechanical import BiomechanicalMarker

        marker = BiomechanicalMarker(
            code="Pr01",
            name="Fundamental Frequency",
            category="A",
            description="Basic frequency",
            unit="Hz",
        )

        assert marker.code == "Pr01"
        assert marker.name == "Fundamental Frequency"
        assert marker.unit == "Hz"

    def test_marker_str_representation(self) -> None:
        """Test marker string representation."""
        from voice_analysis.features.biomechanical import BiomechanicalMarker

        marker = BiomechanicalMarker(
            code="Pr01",
            name="Fundamental Frequency",
            category="A",
            description="Basic frequency",
        )

        assert str(marker) == "Pr01: Fundamental Frequency"

    def test_marker_default_unit(self) -> None:
        """Test marker default unit."""
        from voice_analysis.features.biomechanical import BiomechanicalMarker

        marker = BiomechanicalMarker(
            code="Pr03",
            name="Asymmetry",
            category="C",
            description="Asymmetry measure",
        )

        assert marker.unit == ""


class TestBiomechanicalMarkerRegistry:
    """Test suite for BiomechanicalMarkerRegistry class."""

    def test_registry_contains_all_markers(self) -> None:
        """Test that registry contains all 22 markers."""
        from voice_analysis.features.biomechanical import BiomechanicalMarkerRegistry

        assert len(BiomechanicalMarkerRegistry.MARKERS) == 22

    def test_registry_marker_codes(self) -> None:
        """Test marker codes format."""
        from voice_analysis.features.biomechanical import BiomechanicalMarkerRegistry

        for code in BiomechanicalMarkerRegistry.MARKERS.keys():
            assert code.startswith("Pr")
            assert code[2:].isdigit()

    def test_registry_categories(self) -> None:
        """Test category definitions."""
        from voice_analysis.features.biomechanical import BiomechanicalMarkerRegistry

        expected_categories = {"A", "B", "C", "D", "E", "F", "G", "H", "I"}
        actual_categories = set(BiomechanicalMarkerRegistry.CATEGORIES.keys())

        assert expected_categories == actual_categories

    def test_each_marker_has_valid_category(self) -> None:
        """Test each marker has a valid category."""
        from voice_analysis.features.biomechanical import BiomechanicalMarkerRegistry

        valid_categories = set(BiomechanicalMarkerRegistry.CATEGORIES.keys())

        for marker in BiomechanicalMarkerRegistry.MARKERS.values():
            assert marker.category in valid_categories

    def test_get_marker(self) -> None:
        """Test getting a specific marker."""
        from voice_analysis.features.biomechanical import BiomechanicalMarkerRegistry

        marker = BiomechanicalMarkerRegistry.MARKERS["Pr01"]

        assert marker.code == "Pr01"
        assert marker.category == "A"

    def test_get_markers_by_category(self) -> None:
        """Test filtering markers by category."""
        from voice_analysis.features.biomechanical import BiomechanicalMarkerRegistry

        category_a_markers = [
            m for m in BiomechanicalMarkerRegistry.MARKERS.values() if m.category == "A"
        ]

        assert len(category_a_markers) > 0
        assert all(m.category == "A" for m in category_a_markers)
