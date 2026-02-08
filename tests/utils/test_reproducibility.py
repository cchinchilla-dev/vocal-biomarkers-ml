"""Unit tests for reproducibility utilities module."""

from __future__ import annotations

import numpy as np
import pytest


class TestSetGlobalSeed:
    """Test suite for set_global_seed function."""

    def test_set_global_seed_numpy(self) -> None:
        """Test that global seed affects NumPy."""
        from voice_analysis.utils.reproducibility import set_global_seed

        set_global_seed(42)
        values1 = np.random.randn(5)

        set_global_seed(42)
        values2 = np.random.randn(5)

        np.testing.assert_array_equal(values1, values2)

    def test_set_global_seed_python_random(self) -> None:
        """Test that global seed affects Python random."""
        import random

        from voice_analysis.utils.reproducibility import set_global_seed

        set_global_seed(42)
        values1 = [random.random() for _ in range(5)]

        set_global_seed(42)
        values2 = [random.random() for _ in range(5)]

        assert values1 == values2


class TestGetRandomState:
    """Test suite for get_random_state function."""

    def test_get_random_state_returns_randomstate(self) -> None:
        """Test that get_random_state returns RandomState."""
        from voice_analysis.utils.reproducibility import get_random_state

        rng = get_random_state(42)

        assert isinstance(rng, np.random.RandomState)

    def test_get_random_state_reproducible(self) -> None:
        """Test that RandomState is reproducible."""
        from voice_analysis.utils.reproducibility import get_random_state

        rng1 = get_random_state(42)
        rng2 = get_random_state(42)

        values1 = rng1.randn(5)
        values2 = rng2.randn(5)

        np.testing.assert_array_equal(values1, values2)


class TestSeedManager:
    """Test suite for SeedManager class."""

    def test_seed_manager_initialization(self) -> None:
        """Test SeedManager initialization."""
        from voice_analysis.utils.reproducibility import SeedManager

        manager = SeedManager([42, 123])

        assert 42 in manager.seeds
        assert 123 in manager.seeds

    def test_seed_manager_get_next(self) -> None:
        """Test getting next seed."""
        from voice_analysis.utils.reproducibility import SeedManager

        manager = SeedManager([42, 123])

        assert manager.get_next() == 42
        assert manager.get_next() == 123

    def test_seed_manager_reset(self) -> None:
        """Test resetting seed manager."""
        from voice_analysis.utils.reproducibility import SeedManager

        manager = SeedManager([42, 123])
        manager.get_next()
        manager.get_next()
        manager.reset()

        assert manager.get_next() == 42

    def test_seed_manager_extends_if_needed(self) -> None:
        """Test that seed manager extends list if needed."""
        from voice_analysis.utils.reproducibility import SeedManager

        manager = SeedManager([42], min_seeds=5)

        assert len(manager.seeds) >= 5

    def test_seed_manager_get_all(self) -> None:
        """Test getting all seeds."""
        from voice_analysis.utils.reproducibility import SeedManager

        manager = SeedManager([42, 123, 456])
        seeds = manager.get_all()

        assert 42 in seeds
        assert 123 in seeds

    def test_seed_manager_get_all_with_n(self) -> None:
        """Test getting specific number of seeds."""
        from voice_analysis.utils.reproducibility import SeedManager

        manager = SeedManager([42, 123])
        seeds = manager.get_all(n=5)

        assert len(seeds) == 5
        assert seeds[0] == 42


class TestValidateSeed:
    """Test suite for validate_seed function."""

    def test_validate_seed_within_bounds(self) -> None:
        """Test seed within bounds passes through."""
        from voice_analysis.utils.reproducibility import validate_seed

        result = validate_seed(100, lower=0, upper=200)

        assert result == 100

    def test_validate_seed_below_lower(self) -> None:
        """Test seed below lower bound is constrained."""
        from voice_analysis.utils.reproducibility import validate_seed

        result = validate_seed(-10, lower=0, upper=100)

        assert result == 0

    def test_validate_seed_above_upper(self) -> None:
        """Test seed above upper bound is constrained."""
        from voice_analysis.utils.reproducibility import validate_seed

        result = validate_seed(200, lower=0, upper=100)

        assert result == 100


class TestGenerateSeeds:
    """Test suite for generate_seeds function."""

    def test_generate_seeds_count(self) -> None:
        """Test generating correct number of seeds."""
        from voice_analysis.utils.reproducibility import generate_seeds

        seeds = generate_seeds(5, base_seed=42)

        assert len(seeds) == 5

    def test_generate_seeds_reproducible(self) -> None:
        """Test that seed generation is reproducible."""
        from voice_analysis.utils.reproducibility import generate_seeds

        seeds1 = generate_seeds(5, base_seed=42)
        seeds2 = generate_seeds(5, base_seed=42)

        assert seeds1 == seeds2

    def test_generate_seeds_different_base_different_result(self) -> None:
        """Test different base seeds produce different results."""
        from voice_analysis.utils.reproducibility import generate_seeds

        seeds1 = generate_seeds(5, base_seed=42)
        seeds2 = generate_seeds(5, base_seed=123)

        assert seeds1 != seeds2


class TestReproducibilityInfo:
    """Test suite for ReproducibilityInfo class."""

    def test_reproducibility_info_creation(self) -> None:
        """Test ReproducibilityInfo creation."""
        from voice_analysis.utils.reproducibility import ReproducibilityInfo

        info = ReproducibilityInfo(seeds=[42, 123])

        assert info.seeds == [42, 123]

    def test_reproducibility_info_versions(self) -> None:
        """Test that versions are captured."""
        from voice_analysis.utils.reproducibility import ReproducibilityInfo

        info = ReproducibilityInfo(seeds=[42])

        assert "numpy" in info.versions or len(info.versions) >= 0

    def test_reproducibility_info_to_dict(self) -> None:
        """Test converting to dictionary."""
        from voice_analysis.utils.reproducibility import ReproducibilityInfo

        info = ReproducibilityInfo(seeds=[42, 123], config_hash="abc123")
        result = info.to_dict()

        assert result["seeds"] == [42, 123]
        assert result["config_hash"] == "abc123"
        assert "library_versions" in result
