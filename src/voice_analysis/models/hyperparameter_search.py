"""
Hyperparameter optimization module.

This module provides implementations for various hyperparameter search
strategies including Particle Swarm Optimization (PSO) and RandomizedSearchCV.
"""

import logging
from typing import Any

import numpy as np
from joblib import parallel_backend
from pyswarm import pso
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from ..config.settings import Settings

logger = logging.getLogger(__name__)


# Mapping from integer indices to categorical parameter values
CATEGORICAL_PARAM_MAPPINGS = {
    "KNearestNeighbor": {
        "weights": ["uniform", "distance"],
        "metric": ["manhattan", "euclidean", "minkowski"],
    },
    "SupportVectorMachine": {
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
    },
    "RandomForest": {
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
    },
    "GradientBoosting": {
        "loss": ["exponential", "log_loss"],
    },
    "DecisionTree": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_features": ["sqrt", "log2"],
    },
}


class HyperparameterOptimizer:
    """
    Optimizer for model hyperparameters.

    This class supports multiple optimization strategies including
    Particle Swarm Optimization (PSO) and RandomizedSearchCV.

    Parameters
    ----------
    settings : Settings
        Configuration settings containing optimization parameters.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    method : str
        Optimization method to use.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.method = settings.hyperparameter_search.method

    def optimize(
        self,
        model_name: str,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
    ) -> dict[str, Any] | None:
        """
        Optimize hyperparameters for a given model.

        Parameters
        ----------
        model_name : str
            Name of the model for parameter lookup.
        model : BaseEstimator
            The model instance to optimize.
        X : array-like
            Training features.
        y : array-like
            Training labels.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict or None
            Optimal hyperparameters, or None if optimization fails.
        """
        param_ranges = self._get_param_ranges(model_name)
        if not param_ranges:
            logger.debug(f"No parameter ranges defined for {model_name}")
            return None

        try:
            if self.method == "pso":
                return self._pso_search(model_name, model, X, y, param_ranges)
            elif self.method == "random_search":
                return self._random_search(model, X, y, param_ranges, seed)
            else:
                logger.warning(f"Unknown optimization method: {self.method}")
                return None
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed for {model_name}: {e}")
            return None

    def _get_param_ranges(self, model_name: str) -> dict[str, Any]:
        """
        Get parameter ranges for a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model.

        Returns
        -------
        dict[str, Any]
            Parameter ranges for the model.
        """
        all_ranges = self.settings.hyperparameter_search.param_ranges
        return all_ranges.get(model_name, {})

    def _pso_search(
        self,
        model_name: str,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_ranges: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform Particle Swarm Optimization for hyperparameters.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model : BaseEstimator
            Model instance to optimize.
        X : array-like
            Training features.
        y : array-like
            Training labels.
        param_ranges : dict
            Parameter ranges for optimization.

        Returns
        -------
        dict
            Optimal hyperparameters.
        """
        pso_config = self.settings.hyperparameter_search.pso

        # Build bounds for continuous optimization
        bounds, param_names = self._build_pso_bounds(param_ranges)

        def fitness_function(params: np.ndarray) -> float:
            """Objective function for PSO (minimizes 1 - accuracy)."""
            param_dict = self._decode_pso_params(params, param_names, param_ranges, model_name)

            try:
                model_copy = model.__class__(**model.get_params())
                model_copy.set_params(**param_dict)
                scores = cross_val_score(model_copy, X, y, cv=5, scoring="accuracy")
                return 1 - scores.mean()
            except Exception:
                return 1.0  # Return worst case on error

        # Run PSO
        xopt, fopt = pso(
            fitness_function,
            lb=[b[0] for b in bounds],
            ub=[b[1] for b in bounds],
            swarmsize=pso_config.swarm_size,
            maxiter=pso_config.max_iterations,
            minfunc=pso_config.min_func,
            minstep=pso_config.min_step,
        )

        # Decode optimal parameters
        return self._decode_pso_params(xopt, param_names, param_ranges, model_name)

    def _build_pso_bounds(
        self, param_ranges: dict[str, Any]
    ) -> tuple[list[tuple[float, float]], list[str]]:
        """
        Build bounds array for PSO from parameter ranges.

        Parameters
        ----------
        param_ranges : dict[str, Any]
            Parameter ranges dictionary.

        Returns
        -------
        tuple[list[tuple[float, float]], list[str]]
            Bounds list and parameter names list.
        """
        bounds = []
        param_names = []

        for param, values in param_ranges.items():
            param_names.append(param)

            if isinstance(values, list):
                # Categorical parameter: use index bounds
                bounds.append((0, len(values) - 1))
            elif isinstance(values, (list, tuple)) and len(values) == 2:
                # Continuous parameter: use min/max bounds
                bounds.append((values[0], values[1]))
            else:
                raise ValueError(f"Invalid parameter range for {param}: {values}")

        return bounds, param_names

    def _decode_pso_params(
        self,
        params: np.ndarray,
        param_names: list[str],
        param_ranges: dict[str, Any],
        model_name: str,
    ) -> dict[str, Any]:
        """
        Decode PSO continuous values to actual parameter values.

        Parameters
        ----------
        params : ndarray
            Array of continuous parameter values from PSO.
        param_names : list[str]
            List of parameter names.
        param_ranges : dict[str, Any]
            Original parameter ranges.
        model_name : str
            Name of the model for categorical mappings.

        Returns
        -------
        dict[str, Any]
            Decoded parameter dictionary.
        """
        decoded = {}
        categorical_mappings = CATEGORICAL_PARAM_MAPPINGS.get(model_name, {})

        for i, param in enumerate(param_names):
            value = params[i]

            # Check if this is a categorical parameter with mapping
            if param in categorical_mappings:
                idx = int(round(value))
                idx = max(0, min(idx, len(categorical_mappings[param]) - 1))
                decoded[param] = categorical_mappings[param][idx]
            elif isinstance(param_ranges[param], list):
                # Categorical without explicit mapping
                idx = int(round(value))
                idx = max(0, min(idx, len(param_ranges[param]) - 1))
                decoded[param] = param_ranges[param][idx]
            else:
                # Check if it should be an integer
                original_range = param_ranges[param]
                if all(isinstance(v, int) for v in original_range):
                    decoded[param] = int(round(value))
                else:
                    decoded[param] = value

        return decoded

    def _random_search(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_ranges: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """
        Perform RandomizedSearchCV for hyperparameters.

        Parameters
        ----------
        model : BaseEstimator
            Model instance to optimize.
        X : array-like
            Training features.
        y : array-like
            Training labels.
        param_ranges : dict
            Parameter distributions/ranges for search.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        rs_config = self.settings.hyperparameter_search.random_search

        # Convert ranges to distributions for RandomizedSearchCV
        param_distributions = self._convert_to_distributions(param_ranges)

        with parallel_backend("threading"):
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=rs_config.n_iterations,
                scoring=rs_config.scoring,
                cv=5,
                random_state=seed,
                n_jobs=-1,
            )
            search.fit(X, y)

        return search.best_params_

    def _convert_to_distributions(self, param_ranges: dict[str, Any]) -> dict[str, Any]:
        """
        Convert parameter ranges to distributions for RandomizedSearchCV.

        Parameters
        ----------
        param_ranges : dict[str, Any]
            Parameter ranges dictionary.

        Returns
        -------
        dict[str, Any]
            Parameter distributions for RandomizedSearchCV.
        """
        distributions = {}

        for param, values in param_ranges.items():
            if isinstance(values, list):
                # Categorical parameter
                distributions[param] = values
            elif isinstance(values, (list, tuple)) and len(values) == 2:
                # Continuous range: create list of values
                min_val, max_val = values
                if isinstance(min_val, int) and isinstance(max_val, int):
                    distributions[param] = list(range(min_val, max_val + 1))
                else:
                    # Create discrete samples for continuous range
                    distributions[param] = np.linspace(min_val, max_val, 20).tolist()

        return distributions
