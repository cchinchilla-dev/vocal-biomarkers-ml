"""
Reproducibility utilities module.

This module provides utilities for ensuring reproducible results
across different runs of the analysis pipeline.
"""

import logging
import os
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Set random seed for all relevant libraries.

    This function sets the seed for Python's random module,
    NumPy, and environment variables to ensure reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.debug(f"Set global random seed: {seed}")


def get_random_state(seed: int) -> np.random.RandomState:
    """
    Get a NumPy RandomState object.

    Parameters
    ----------
    seed : int
        Random seed value.

    Returns
    -------
    RandomState
        NumPy RandomState instance.
    """
    return np.random.RandomState(seed)


class SeedManager:
    """
    Manager for handling multiple seeds across executions.

    This class provides a centralized way to manage seeds
    for multiple experimental runs.

    Parameters
    ----------
    base_seeds : list[int]
        Initial list of seed values.
    min_seeds : int
        Minimum number of seeds to maintain.

    Attributes
    ----------
    seeds : list[int]
        List of available seeds.
    current_index : int
        Current position in seed list.

    Examples
    --------
    >>> manager = SeedManager([42, 123])
    >>> seed1 = manager.get_next()
    >>> seed2 = manager.get_next()
    """

    def __init__(self, base_seeds: list[int], min_seeds: int = 10):
        self.seeds = list(base_seeds)
        self.min_seeds = min_seeds
        self.current_index = 0

        # Ensure minimum number of seeds
        self._extend_seeds()

    def _extend_seeds(self) -> None:
        """
        Extend seed list to meet minimum requirement.

        Generates additional sequential seeds until the list
        meets the minimum seeds requirement.
        """
        while len(self.seeds) < self.min_seeds:
            new_seed = self.seeds[-1] + 1
            self.seeds.append(new_seed)

    def get_next(self) -> int:
        """
        Get the next seed value.

        Returns
        -------
        int
            Next seed value from the list.
        """
        if self.current_index >= len(self.seeds):
            # Generate new seed
            new_seed = self.seeds[-1] + 1
            self.seeds.append(new_seed)

        seed = self.seeds[self.current_index]
        self.current_index += 1
        return seed

    def reset(self) -> None:
        """
        Reset to the beginning of the seed list.

        Sets the current index back to zero, allowing the seed
        sequence to be reused from the start.
        """
        self.current_index = 0

    def get_all(self, n: int | None = None) -> list[int]:
        """
        Get multiple seeds.

        Parameters
        ----------
        n : int, optional
            Number of seeds to return. If None, returns all.

        Returns
        -------
        list[int]
            List of seed values.
        """
        if n is None:
            return list(self.seeds)

        # Extend if needed
        while len(self.seeds) < n:
            self._extend_seeds()

        return self.seeds[:n]


def validate_seed(seed: int, lower: int = 0, upper: int = 2**32 - 1) -> int:
    """
    Validate and constrain a seed value.

    Parameters
    ----------
    seed : int
        Seed value to validate.
    lower : int
        Lower bound (inclusive).
    upper : int
        Upper bound (inclusive).

    Returns
    -------
    int
        Validated seed value.
    """
    return max(lower, min(seed, upper))


def generate_seeds(n: int, base_seed: int = 42) -> list[int]:
    """
    Generate a list of deterministic seeds.

    Parameters
    ----------
    n : int
        Number of seeds to generate.
    base_seed : int
        Starting seed for generation.

    Returns
    -------
    list[int]
        List of generated seeds.
    """
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31)) for _ in range(n)]


class ReproducibilityInfo:
    """
    Container for reproducibility metadata.

    This class stores information needed to reproduce
    an experiment, including seeds, library versions,
    and configuration hashes.

    Parameters
    ----------
    seeds : list[int]
        Seeds used in the experiment.
    config_hash : str, optional
        Hash of the configuration.
    """

    def __init__(self, seeds: list[int], config_hash: str | None = None):
        self.seeds = seeds
        self.config_hash = config_hash
        self.versions = self._get_versions()

    def _get_versions(self) -> dict[str, str]:
        """Get versions of key libraries."""
        versions = {}

        try:
            import sklearn

            versions["scikit-learn"] = sklearn.__version__
        except ImportError:
            pass

        try:
            versions["numpy"] = np.__version__
        except Exception:
            pass

        try:
            import pandas

            versions["pandas"] = pandas.__version__
        except ImportError:
            pass

        return versions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seeds": self.seeds,
            "config_hash": self.config_hash,
            "library_versions": self.versions,
        }

    def __repr__(self) -> str:
        return f"ReproducibilityInfo(seeds={len(self.seeds)}, versions={self.versions})"
