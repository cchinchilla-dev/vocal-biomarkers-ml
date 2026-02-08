"""
Feature selection module.

This module provides various feature selection methods including
RFE, mutual information, statistical tests, and correlation-based selection.
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

from ..config.settings import Settings

logger = logging.getLogger(__name__)

# Plot style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


class FeatureSelector:
    """
    Feature selector for voice analysis data.

    This class implements multiple feature selection strategies
    including RFE with bootstrap, mutual information, statistical
    tests, and LASSO regularization.

    Parameters
    ----------
    settings : Settings
        Configuration settings.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    scaler : StandardScaler
        Scaler for feature standardization.
    label_encoder : LabelEncoder
        Encoder for target labels.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def select_features(self, dataset: pd.DataFrame) -> dict:
        """
        Perform comprehensive feature selection.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target.

        Returns
        -------
        dict
            Dictionary containing results from all selection methods
            and the final selected features.
        """
        logger.info("Starting feature selection process")

        # Get feature columns
        feature_cols = self._get_feature_columns(dataset)
        seeds = self.settings.get_seeds()

        # Get results path for saving plots
        results_path = Path(self.settings.paths.results_dir) / "features"
        results_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Statistical test (Mann-Whitney U with Bonferroni correction)
        results["bonferroni"] = self._statistical_selection(dataset, feature_cols)

        # RFE with bootstrap (includes RFE analysis plot)
        rfe_results = self._rfe_bootstrap_selection(dataset, seeds, results_path)
        results["rfe"] = rfe_results["rankings"]
        results["optimal_n_features"] = rfe_results["optimal_n"]
        results["rfe_analysis"] = rfe_results["analysis"]  # For n_features_analysis.csv

        # Mutual Information
        results["mutual_information"] = self._mutual_information_selection(dataset)

        # Random Forest importance
        results["rf_importance"] = self._rf_importance_selection(dataset, seeds)

        # LASSO selection
        results["lasso"] = self._lasso_selection(dataset, seeds)

        # Correlation-based selection
        results["cfs"] = self._correlation_based_selection(dataset)

        # Get final selected features from RFE
        results["selected_features"] = self._get_optimal_features(
            rfe_results["rankings"],
            rfe_results["optimal_n"],
        )

        logger.info(f"Selected {len(results['selected_features'])} features")
        return results

    def _get_feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        """
        Get list of feature column names.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to extract feature columns from.

        Returns
        -------
        list[str]
            List of feature column names.
        """
        exclude = {"Record", "Diagnosed", "Is_SMOTE", "Gender", "Age"}
        return [c for c in dataset.columns if c not in exclude]

    def _prepare_data(
        self, dataset: pd.DataFrame, standardize: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for feature selection.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target.
        standardize : bool, optional
            Whether to standardize features.

        Returns
        -------
        tuple[DataFrame, Series]
            Features and target.
        """
        dataset_copy = dataset.drop(columns=["Record"], errors="ignore")

        X = dataset_copy.drop(columns=["Diagnosed"])
        y = dataset_copy["Diagnosed"]

        if standardize:
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )

        return X, y

    def _statistical_selection(
        self, dataset: pd.DataFrame, feature_cols: list[str]
    ) -> pd.DataFrame:
        """Perform Mann-Whitney U test with Bonferroni correction."""
        logger.info("Performing statistical feature selection")

        p_values = []
        for feature in feature_cols:
            control = dataset[dataset["Diagnosed"] == "Control"][feature]
            pathological = dataset[dataset["Diagnosed"] == "Pathological"][feature]

            if len(control) > 0 and len(pathological) > 0:
                stat, pval = mannwhitneyu(control, pathological)
                p_values.append({"Feature": feature, "P-Value": pval})

        df = pd.DataFrame(p_values)

        # Apply Bonferroni correction
        alpha = self.settings.feature_selection.statistics.alpha
        reject, adjusted_pvals, _, _ = multipletests(
            df["P-Value"],
            alpha=alpha,
            method="bonferroni",
        )

        df["Adjusted P-Value"] = adjusted_pvals
        df["Reject Null Hypothesis"] = reject

        return df.sort_values("Adjusted P-Value")

    def _rfe_bootstrap_selection(
        self, dataset: pd.DataFrame, seeds: list[int], results_path: Path
    ) -> dict:
        """Perform RFE with bootstrap for stability analysis."""
        logger.info("Performing RFE with bootstrap")

        X, y = self._prepare_data(dataset)
        rfe_config = self.settings.feature_selection.rfe

        # Find optimal number of features and get analysis results
        optimal_n, analysis_df = self._find_optimal_n_features(X, y, seeds[0], results_path)

        # Bootstrap RFE
        n_iterations = min(rfe_config.n_bootstrap_iterations, len(seeds))
        all_rankings = []
        selection_counts = {f: 0 for f in X.columns}

        for i, seed in enumerate(seeds[:n_iterations]):
            logger.debug(f"RFE bootstrap iteration {i + 1}/{n_iterations}")

            try:
                estimator = RandomForestClassifier(n_jobs=-1, random_state=seed)
                selector = RFE(
                    estimator,
                    n_features_to_select=optimal_n,
                    step=rfe_config.step,
                )
                selector.fit(X, y)

                all_rankings.append(selector.ranking_)

                for feature in X.columns[selector.support_]:
                    selection_counts[feature] += 1

            except Exception as e:
                logger.warning(f"RFE iteration {i + 1} failed: {e}")
                continue

        # Compute statistics
        avg_rankings = np.mean(all_rankings, axis=0) if all_rankings else np.ones(len(X.columns))
        stability_indices = {f: c / n_iterations for f, c in selection_counts.items()}

        rankings_df = pd.DataFrame(
            {
                "Feature": X.columns,
                "Average Ranking": avg_rankings,
                "Stability Index": [stability_indices[f] for f in X.columns],
            }
        ).sort_values("Average Ranking")

        return {
            "rankings": rankings_df,
            "optimal_n": optimal_n,
            "analysis": analysis_df,
        }

    def _find_optimal_n_features(
        self, X: pd.DataFrame, y: pd.Series, seed: int, results_path: Path
    ) -> tuple[int, pd.DataFrame]:
        """
        Find optimal number of features using RFE and plot results.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : Series
            Target variable.
        seed : int
            Random seed for reproducibility.
        results_path : Path
            Path to save results.

        Returns
        -------
        tuple[int, DataFrame]
            Optimal number of features and analysis DataFrame.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        results = []
        for n in range(1, len(X.columns)):
            logger.debug(f"Analyzing RFE with {n} features")
            estimator = RandomForestClassifier(n_jobs=-1, random_state=seed)
            selector = RFE(estimator, n_features_to_select=n, step=1)
            selector.fit(X_train, y_train)
            score = 1 - selector.score(X_test, y_test)
            results.append({"Number of Features": n, "Score": score})

        df = pd.DataFrame(results)
        best_error = df["Score"].min()

        # Select smallest n within 5% of best error
        threshold = best_error * 1.05
        optimal_n = df[df["Score"] <= threshold]["Number of Features"].min()

        logger.info(f"Optimal number of features: {optimal_n}")

        # Plot RFE analysis
        self._plot_rfe_analysis(df, results_path)

        return int(optimal_n), df

    def _plot_rfe_analysis(self, results: pd.DataFrame, results_path: Path) -> None:
        """
        Plot RFE analysis results.

        Parameters
        ----------
        results : DataFrame
            RFE analysis results with Number of Features and Score columns.
        results_path : Path
            Path to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.ylim(0, 0.6)
        plt.plot(results["Number of Features"], results["Score"], linestyle="-")
        plt.xlabel("Number of Features", fontsize=14)
        plt.ylabel("Error", fontsize=14)
        plt.title("RFE Analysis", fontsize=16)
        plt.tight_layout()

        save_path = results_path / "rfe_analysis.svg"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Saved RFE analysis plot: {save_path}")

    def _mutual_information_selection(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mutual information scores.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target.

        Returns
        -------
        DataFrame
            DataFrame with features and their mutual information scores.
        """
        logger.info("Calculating mutual information")

        X, y = self._prepare_data(dataset, standardize=False)
        mi_scores = mutual_info_classif(X, y)

        return pd.DataFrame(
            {
                "Feature": X.columns,
                "Mutual Information": mi_scores,
            }
        ).sort_values("Mutual Information", ascending=False)

    def _rf_importance_selection(self, dataset: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
        """
        Calculate Random Forest feature importance.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target.
        seeds : list[int]
            List of random seeds for multiple iterations.

        Returns
        -------
        DataFrame
            DataFrame with features and their importance scores.
        """
        logger.info("Calculating Random Forest importance")

        X, y = self._prepare_data(dataset)
        all_importances = []

        for i, seed in enumerate(seeds[:20]):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )

            clf = RandomForestClassifier(n_jobs=-1, random_state=seed)
            clf.fit(X_train, y_train)
            all_importances.append(clf.feature_importances_)

        avg_importance = np.mean(all_importances, axis=0)

        return pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": avg_importance,
            }
        ).sort_values("Importance", ascending=False)

    def _lasso_selection(self, dataset: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
        """
        Perform LASSO-based feature selection.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target.
        seeds : list[int]
            List of random seeds for reproducibility.

        Returns
        -------
        DataFrame
            DataFrame with features, coefficients, and selection status.
        """
        logger.info("Performing LASSO selection")

        X, y = self._prepare_data(dataset)
        y_encoded = self.label_encoder.fit_transform(y)

        alpha_values = np.logspace(-6, 1, 100)
        lasso = LassoCV(alphas=alpha_values, cv=5, random_state=seeds[0])
        lasso.fit(X, y_encoded)

        coefficients = pd.Series(lasso.coef_, index=X.columns)

        df = pd.DataFrame(
            {
                "Feature": X.columns,
                "Coefficient": coefficients.values,
                "Selected": coefficients.abs() > 0,
            }
        )

        df["AbsCoefficient"] = df["Coefficient"].abs()

        return df.sort_values("AbsCoefficient", ascending=False).drop(columns="AbsCoefficient")

    def _correlation_based_selection(
        self, dataset: pd.DataFrame, threshold: float = 0.8
    ) -> list[str]:
        """
        Perform correlation-based feature selection.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target.
        threshold : float, optional
            Correlation threshold for feature selection.

        Returns
        -------
        list[str]
            List of selected feature names.
        """
        logger.info("Performing correlation-based selection")

        threshold = self.settings.feature_selection.cfs.threshold
        dataset_encoded = dataset.drop(columns=["Record"], errors="ignore").copy()
        dataset_encoded["Diagnosed"] = self.label_encoder.fit_transform(
            dataset_encoded["Diagnosed"]
        )

        corr_matrix = dataset_encoded.corr()
        target_corr = corr_matrix["Diagnosed"].abs().sort_values(ascending=False)

        selected = []
        for feature in target_corr.index:
            if feature == "Diagnosed":
                continue
            if not selected:
                selected.append(feature)
            elif not any(abs(corr_matrix.loc[feature, s]) > threshold for s in selected):
                selected.append(feature)

        return selected

    def _get_optimal_features(self, rfe_rankings: pd.DataFrame, n_optimal: int) -> pd.Series:
        """
        Get optimal features from RFE rankings.

        Parameters
        ----------
        rfe_rankings : DataFrame
            RFE rankings DataFrame with Feature, Average Ranking,
            and Stability Index columns.
        n_optimal : int
            Number of optimal features to select.

        Returns
        -------
        Series
            Series of selected feature names.
        """
        threshold = self.settings.feature_selection.rfe.stability_threshold

        # Filter by stability threshold
        stable_features = rfe_rankings[rfe_rankings["Stability Index"] >= threshold]

        # Sort by ranking and select top n
        optimal_df = stable_features.nsmallest(n_optimal, "Average Ranking")

        return optimal_df["Feature"]
