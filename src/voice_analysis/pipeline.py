"""
Main pipeline orchestrator for the voice analysis workflow.

This module coordinates all stages of the analysis pipeline, from data loading
through feature extraction, selection, model training, and evaluation.
"""

import logging
from pathlib import Path

import pandas as pd

from .config.settings import Settings, get_settings
from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor
from .features.selection import FeatureSelector
from .metrics.custom_metrics import MetricsCalculator
from .models.classifiers import ClassifierTrainer
from .models.evaluation import ModelEvaluator
from .preprocessing.cleaning import DataCleaner
from .preprocessing.resampling import DataResampler
from .preprocessing.standardization import DataStandardizer
from .utils.io import ResultsManager
from .utils.logging import setup_logging
from .utils.reproducibility import set_global_seed
from .visualization.pca_plots import PCAVisualizer
from .visualization.roc_curves import ROCVisualizer
from .visualization.distribution_plots import DistributionVisualizer

logger = logging.getLogger(__name__)


class VoiceAnalysisPipeline:
    """
    Main pipeline for COVID-19 voice analysis.

    This class orchestrates the entire analysis workflow, coordinating
    data loading, preprocessing, feature extraction and selection,
    model training, and evaluation.

    Parameters
    ----------
    settings : Settings, optional
        Configuration settings. If not provided, loads from default config file.

    Attributes
    ----------
    settings : Settings
        Pipeline configuration settings.
    results_manager : ResultsManager
        Manager for saving and loading results.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._setup_logging()
        self._initialize_components()

    def _setup_logging(self) -> None:
        """
        Configure logging based on settings.

        Sets up logging level, format, file output, and console output
        according to the pipeline configuration.
        """
        setup_logging(
            level=self.settings.logging.level,
            log_format=self.settings.logging.format,
            log_file=self.settings.logging.file,
            console=self.settings.logging.console,
        )

    def _initialize_components(self) -> None:
        """
        Initialize pipeline components.

        Creates instances of all required components including data loaders,
        preprocessors, feature selectors, classifiers, and visualizers.
        """
        self.results_manager = ResultsManager(
            base_path=Path(self.settings.paths.results_dir),
            output_config=self.settings.paths.outputs,
        )
        self.data_loader = DataLoader(self.settings)
        self.data_preprocessor = DataPreprocessor(self.settings)
        self.data_cleaner = DataCleaner(self.settings)
        self.data_standardizer = DataStandardizer(self.settings)
        self.feature_selector = FeatureSelector(self.settings)
        self.resampler = DataResampler(self.settings)
        self.classifier_trainer = ClassifierTrainer(self.settings)
        self.model_evaluator = ModelEvaluator(self.settings)
        self.metrics_calculator = MetricsCalculator()

        if self.settings.visualization.enabled:
            self.pca_visualizer = PCAVisualizer(self.settings)
            self.roc_visualizer = ROCVisualizer(self.settings)
            self.distribution_visualizer = DistributionVisualizer(self.settings)

    def run(self) -> dict:
        """
        Execute the complete analysis pipeline.

        Returns
        -------
        dict
            Dictionary containing all results including metrics,
            selected features, and trained models.
        """
        logger.info("Starting voice analysis pipeline")
        results = {}

        # Stage 1: Data Loading and Preparation
        dataset = self._load_or_analyze_data()
        results["raw_dataset_shape"] = dataset.shape
        logger.info(f"Dataset loaded with shape: {dataset.shape}")

        # Stage 2: Data Cleaning
        dataset = self._clean_data(dataset)
        results["cleaned_dataset_shape"] = dataset.shape

        # Stage 3: Feature Selection (if enabled)
        if self.settings.pipeline.stages.feature_selection:
            dataset, selected_features = self._select_features(dataset)
            results["selected_features"] = selected_features
            results["n_selected_features"] = len(selected_features)
        else:
            selected_features = self._load_existing_features(dataset)
            # Filter dataset to selected features
            columns_to_keep = ["Diagnosed", "Record"] + list(selected_features)
            dataset = dataset[columns_to_keep]

        # Stage 4: Model Training and Evaluation
        if self.settings.pipeline.stages.train_classifiers:
            metrics, patient_probs, loo_results = self._train_and_evaluate(dataset)
            results["metrics"] = metrics
            results["patient_probabilities"] = patient_probs
            results["loo_results"] = loo_results

            # Save results
            self._save_results(results)

        logger.info("Pipeline execution completed successfully")
        return results

    def _load_or_analyze_data(self) -> pd.DataFrame:
        """
        Load existing data or analyze recordings.

        Returns
        -------
        DataFrame
            Loaded or analyzed dataset.
        """
        stages = self.settings.pipeline.stages

        if stages.analyze_recordings:
            logger.info("Analyzing audio recordings...")
            dataset = self.data_loader.analyze_recordings()

            if stages.save_intermediate:
                self.results_manager.save_dataset(dataset, "dataset.csv")

            if self.settings.visualization.enabled:
                self._generate_pca_visualization(dataset, "unclean")
        else:
            logger.info("Loading pre-processed dataset...")
            dataset = self.results_manager.load_dataset("dataset_clean.csv")

        return dataset

    def _clean_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing low-variance and correlated features.

        Parameters
        ----------
        dataset : DataFrame
            Input dataset to clean.

        Returns
        -------
        DataFrame
            Cleaned dataset.
        """
        logger.info("Cleaning dataset...")

        cleaned = self.data_cleaner.clean(dataset)

        if self.settings.pipeline.stages.save_intermediate:
            self.results_manager.save_dataset(cleaned, "dataset_clean.csv")

        if self.settings.visualization.enabled:
            self._generate_pca_visualization(cleaned, "clean")

        return cleaned

    def _select_features(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Perform feature selection.

        Parameters
        ----------
        dataset : DataFrame
            Input dataset with all features.

        Returns
        -------
        tuple[DataFrame, Series]
            Filtered dataset and selected feature names.
        """
        logger.info("Performing feature selection...")

        selection_results = self.feature_selector.select_features(dataset)

        # Save feature selection results
        self.results_manager.save_features(selection_results["bonferroni"], "bonferroni.csv")
        self.results_manager.save_features(selection_results["rfe"], "rfe.csv")
        self.results_manager.save_features(
            selection_results["mutual_information"], "mutual_information.csv"
        )

        # Save additional feature selection results
        self.results_manager.save_features(
            pd.DataFrame({"Optimal Number of Features": [selection_results["optimal_n_features"]]}),
            "n_optimal.csv",
        )
        self.results_manager.save_features(selection_results["rf_importance"], "rf_importance.csv")
        self.results_manager.save_features(selection_results["lasso"], "lasso_coefficients.csv")
        self.results_manager.save_features(
            pd.DataFrame({"Feature": selection_results["cfs"]}), "selected_features_cfs.csv"
        )

        # Save RFE analysis results if available
        if "rfe_analysis" in selection_results:
            self.results_manager.save_features(
                selection_results["rfe_analysis"], "n_features_analysis.csv"
            )

        selected_features = selection_results["selected_features"]
        self.results_manager.save_features(
            pd.DataFrame({"Feature": selected_features}), "features_to_use.csv"
        )

        # Filter dataset to selected features
        columns_to_keep = ["Diagnosed", "Record"] + list(selected_features)
        filtered_dataset = dataset[columns_to_keep]

        if self.settings.pipeline.stages.save_intermediate:
            self.results_manager.save_dataset(filtered_dataset, "dataset_features.csv")

        if self.settings.visualization.enabled:
            self._generate_pca_visualization(filtered_dataset, "features")

        return filtered_dataset, selected_features

    def _load_existing_features(self, dataset: pd.DataFrame) -> list[str]:
        """
        Load previously selected features.

        Parameters
        ----------
        dataset : DataFrame
            Current dataset (unused, kept for interface consistency).

        Returns
        -------
        list[str]
            List of previously selected feature names.
        """
        features_df = self.results_manager.load_features("features_to_use.csv")
        return features_df["Feature"].tolist()

    def _train_and_evaluate(
        self, dataset: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Train classifiers and evaluate performance.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target for training.

        Returns
        -------
        tuple[DataFrame, DataFrame, DataFrame]
            Metrics DataFrame, patient probabilities DataFrame,
            and Leave-One-Out results DataFrame.
        """
        logger.info("Training and evaluating classifiers...")

        seeds = self.settings.get_seeds()
        n_executions = self.settings.reproducibility.n_executions

        all_metrics = []
        all_patient_probs = []
        all_loo_results = []

        for i, seed in enumerate(seeds[:n_executions]):
            logger.info(f"Execution {i + 1}/{n_executions} (seed={seed})")
            set_global_seed(seed)

            # Prepare dataset with SMOTE
            dataset_with_smote, dataset_before_smote = self._prepare_dataset_for_training(
                dataset, seed, is_first=(i == 0)
            )

            # Train and evaluate classifiers
            execution_results = self.classifier_trainer.train_all(
                dataset_with_smote,
                target_column="Diagnosed",
                execution_number=i + 1,
                seed=seed,
            )

            all_metrics.append(execution_results["metrics"])
            all_patient_probs.append(execution_results["patient_probabilities"])
            all_loo_results.append(execution_results["loo_results"])

        # Aggregate results
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        patient_probs_df = pd.concat(all_patient_probs, ignore_index=True)
        loo_results_df = pd.concat(all_loo_results, ignore_index=True)

        # Generate ROC curves (first execution only)
        if self.settings.visualization.enabled and self.settings.visualization.roc.enabled:
            self._generate_roc_curves(dataset)

        return metrics_df, patient_probs_df, loo_results_df

    def _prepare_dataset_for_training(
        self, dataset: pd.DataFrame, seed: int, is_first: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare dataset with SMOTE resampling.

        Parameters
        ----------
        dataset : DataFrame
            Input dataset to prepare.
        seed : int
            Random seed for reproducibility.
        is_first : bool, optional
            Whether this is the first execution (for saving intermediate results).

        Returns
        -------
        tuple[DataFrame, DataFrame]
            Resampled dataset and original dataset before resampling.
        """
        dataset_before = dataset.copy()
        dataset_before["Is_SMOTE"] = False

        if self.settings.resampling.enabled:
            resampled = self.resampler.resample(
                dataset=dataset_before,
                target_column="Diagnosed",
                seed=seed,
                analyze=is_first and self.settings.resampling.analyze_distribution,
            )

            # Save SMOTE dataset and generate visualizations on first execution
            if is_first:
                if self.settings.pipeline.stages.save_intermediate:
                    self.results_manager.save_dataset(resampled, "dataset_smote.csv")

                # Generate PCA for SMOTE dataset
                if self.settings.visualization.enabled:
                    self._generate_pca_visualization(resampled, "smote")

                    # Generate SMOTE distribution analysis plots
                    if self.settings.visualization.smote_analysis.enabled:
                        self._generate_smote_analysis(dataset_before, resampled)

            return resampled, dataset_before

        return dataset_before, dataset_before

    def _generate_smote_analysis(self, before: pd.DataFrame, after: pd.DataFrame) -> None:
        """
        Generate SMOTE distribution analysis visualizations.

        Parameters
        ----------
        before : DataFrame
            Dataset before SMOTE resampling.
        after : DataFrame
            Dataset after SMOTE resampling.
        """
        logger.info("Generating SMOTE analysis visualizations...")

        vis_path = self.results_manager.get_visualization_path() / "smote_analysis"
        vis_path.mkdir(parents=True, exist_ok=True)

        # Class distribution plot
        if self.settings.visualization.smote_analysis.plot_class_distribution:
            self.distribution_visualizer.plot_class_distribution(
                before=before["Diagnosed"],
                after=after["Diagnosed"],
                save_path=vis_path / "class_distribution.svg",
            )

        # Feature distributions plot (for pathological samples)
        if self.settings.visualization.smote_analysis.plot_feature_distributions:
            pathological_before = before[before["Diagnosed"] == "Pathological"]
            pathological_after = after[after["Diagnosed"] == "Pathological"]

            # Get numeric columns only
            exclude_cols = ["Record", "Diagnosed", "Is_SMOTE"]
            numeric_cols = [c for c in pathological_before.columns if c not in exclude_cols]

            self.distribution_visualizer.plot_feature_distributions(
                before=pathological_before[numeric_cols],
                after=pathological_after[numeric_cols],
                save_path=vis_path / "feature_distributions.svg",
            )

        # Save distribution comparison (KS test results)
        analysis = self.resampler.get_last_analysis()
        if analysis and "ks_tests" in analysis:
            ks_df = pd.DataFrame(analysis["ks_tests"])
            self.results_manager.save_metrics(ks_df, "smote_distribution_comparison.csv")

    def _generate_pca_visualization(
        self, dataset: pd.DataFrame, stage: str, compare_dataset: pd.DataFrame | None = None
    ) -> None:
        """
        Generate PCA visualization for dataset.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to visualize.
        stage : str
            Pipeline stage identifier (e.g., 'clean', 'features', 'smote').
        compare_dataset : DataFrame, optional
            Optional dataset for comparison visualization.
        """
        logger.info(f"Generating PCA visualization for stage: {stage}")

        # Standardize for PCA if needed
        if stage in ["clean", "features", "smote"]:
            pca_standardizer = DataStandardizer(self.settings)
            dataset_for_pca = pca_standardizer.standardize(dataset)

            # Save standardized dataset on first standardization
            if stage == "smote" and self.settings.pipeline.stages.save_intermediate:
                self.results_manager.save_dataset(dataset_for_pca, "dataset_standardized.csv")
        else:
            dataset_for_pca = dataset

        save_dir = self.results_manager.get_visualization_path() / "pca" / stage
        self.pca_visualizer.plot(dataset_for_pca, save_dir, compare_dataset)

    def _generate_roc_curves(self, dataset: pd.DataFrame) -> None:
        """
        Generate ROC curve visualizations.

        Parameters
        ----------
        dataset : DataFrame
            Dataset with features and target for ROC analysis.
        """
        logger.info("Generating ROC curves...")
        self.roc_visualizer.plot(
            dataset=dataset,
            save_path=self.results_manager.get_visualization_path() / "roc_curves.svg",
        )

    def _save_results(self, results: dict) -> None:
        """
        Save all results to disk.

        Parameters
        ----------
        results : dict
            Dictionary containing metrics, patient probabilities,
            and LOO results to save.
        """
        logger.info("Saving results...")

        if "metrics" in results:
            # Save raw metrics
            self.results_manager.save_metrics(results["metrics"], "metrics.csv")

            # Calculate and save summary statistics
            metrics_mean = (
                results["metrics"]
                .groupby(["Model", "Method"])
                .mean(numeric_only=True)
                .reset_index()
            )
            metrics_std = (
                results["metrics"].groupby(["Model", "Method"]).std(numeric_only=True).reset_index()
            )

            self.results_manager.save_metrics(metrics_mean, "metrics_mean.csv")
            self.results_manager.save_metrics(metrics_std, "metrics_std.csv")

            # Calculate and save aggregated metrics with LOO
            if "loo_results" in results:
                aggregated = self.model_evaluator.aggregate_metrics(
                    results["metrics"], results["loo_results"]
                )
                self.results_manager.save_metrics(aggregated, "metrics_aggregated.csv")

            # Perform and save statistical comparisons
            comparisons = self.model_evaluator.perform_statistical_comparisons(results["metrics"])
            self.results_manager.save_metrics(comparisons, "comparison_results.csv")

        if "patient_probabilities" in results:
            self.results_manager.save_metrics(results["patient_probabilities"], "patients_prob.csv")

            # Calculate and save misclassification probabilities
            self._save_misclassification_analysis(results["patient_probabilities"])

        if "loo_results" in results:
            # Save raw LOO results
            self.results_manager.save_metrics(results["loo_results"], "one_out.csv")

            # Save aggregated LOO summary
            loo_summary = self.model_evaluator.aggregate_loo_summary(results["loo_results"])
            self.results_manager.save_metrics(loo_summary, "one_out_summary.csv")
            logger.info("Saved Leave-One-Out summary")

    def _save_misclassification_analysis(self, patient_probs: pd.DataFrame) -> None:
        """
        Calculate and save misclassification probability analysis.

        Parameters
        ----------
        patient_probs : DataFrame
            Patient probabilities DataFrame.
        """
        per_patient, per_model = self.model_evaluator.compute_misclassification_probability(
            patient_probs
        )
        self.results_manager.save_metrics(per_patient, "prob_misclassification.csv")
        self.results_manager.save_metrics(per_model, "prob_misclassification_model.csv")


def run_pipeline(config_path: str | Path | None = None) -> dict:
    """
    Convenience function to run the complete pipeline.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file.

    Returns
    -------
    dict
        Pipeline results.
    """
    settings = get_settings(config_path) if config_path else get_settings()
    pipeline = VoiceAnalysisPipeline(settings)
    return pipeline.run()
