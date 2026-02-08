"""
Pydantic schemas for configuration validation.

This module defines all configuration schemas used throughout the pipeline,
ensuring type safety and validation of configuration parameters.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = "COVID-19 Voice Analysis"
    version: str = "1.0.0"
    description: str = ""


class RecordingsPathsConfig(BaseModel):
    """Paths to recording directories."""

    control: str
    pathological: str


class BiomechanicalPathsConfig(BaseModel):
    """Paths to biomechanical data files."""

    control: str
    pathological: str


class OutputPathsConfig(BaseModel):
    """Paths for output directories."""

    datasets: str = "datasets"
    features: str = "features"
    metrics: str = "metrics"
    visualizations: str = "visualizations"


class PathsConfig(BaseModel):
    """All path configurations."""

    data_dir: str = "data"
    raw_dir: str = "data/raw"
    results_dir: str = "data/results"
    recordings: RecordingsPathsConfig
    biomechanical: BiomechanicalPathsConfig
    outputs: OutputPathsConfig


class ReproducibilityConfig(BaseModel):
    """Reproducibility settings."""

    seeds: list[int] = Field(default=[42, 123, 1234, 12345, 0, 1])
    n_executions: int = Field(default=10, ge=1)
    min_executions: int = Field(default=10, ge=1)

    @field_validator("n_executions")
    @classmethod
    def validate_n_executions(cls, v: int, info) -> int:
        """Ensure n_executions meets minimum requirement."""
        min_exec = info.data.get("min_executions", 2)
        return max(v, min_exec)


class AudioConfig(BaseModel):
    """Audio processing parameters."""

    f0_min: int = Field(default=75, ge=50, le=200)
    f0_max: int = Field(default=500, ge=300, le=800)
    unit: str = "Hertz"
    sample_rate: int = Field(default=44100, ge=8000)
    n_mfcc: int = Field(default=13, ge=1, le=40)


class RecordsConfig(BaseModel):
    """Records to process configuration."""

    control: list[str] = ["all"]
    pathological: list[str] = ["all"]


class CleaningConfig(BaseModel):
    """Data cleaning thresholds."""

    variance_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    correlation_threshold: float = Field(default=0.9, ge=0.5, le=1.0)


class DataProcessingConfig(BaseModel):
    """Data processing configuration."""

    audio: AudioConfig = AudioConfig()
    records: RecordsConfig = RecordsConfig()
    cleaning: CleaningConfig = CleaningConfig()


class RFEConfig(BaseModel):
    """Recursive Feature Elimination configuration."""

    stability_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    n_bootstrap_iterations: int = Field(default=20, ge=1)
    step: int = Field(default=1, ge=1)


class MutualInformationConfig(BaseModel):
    """Mutual Information configuration."""

    thresholds: list[float] = Field(default_factory=list)


class CFSConfig(BaseModel):
    """Correlation-based Feature Selection configuration."""

    threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class StatisticsConfig(BaseModel):
    """Statistical testing configuration."""

    alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    correction_method: str = "bonferroni"


class FeatureSelectionConfig(BaseModel):
    """Feature selection configuration."""

    enabled: bool = True
    rfe: RFEConfig = RFEConfig()
    mutual_information: MutualInformationConfig = MutualInformationConfig()
    cfs: CFSConfig = CFSConfig()
    statistics: StatisticsConfig = StatisticsConfig()


class SMOTEConfig(BaseModel):
    """SMOTE algorithm configuration."""

    k_neighbors: int = Field(default=7, ge=1)
    sampling_strategy: str = "auto"


class ResamplingConfig(BaseModel):
    """Resampling configuration."""

    enabled: bool = True
    method: Literal["smote", "borderline_smote", "smote_tomek", "smote_enn"] = "smote"
    smote: SMOTEConfig = SMOTEConfig()
    analyze_distribution: bool = True


class CrossValidationConfig(BaseModel):
    """Cross-validation configuration."""

    n_folds: int = Field(default=5, ge=2)
    stratified: bool = True
    shuffle: bool = True


class ModelsConfig(BaseModel):
    """Model training configuration."""

    enabled_models: list[str] = Field(
        default=[
            "AdaBoost",
            "KNearestNeighbor",
            "SupportVectorMachine",
            "Bagging",
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
            "LightGBM",
            "DecisionTree",
        ]
    )
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cross_validation: CrossValidationConfig = CrossValidationConfig()


class PSOConfig(BaseModel):
    """Particle Swarm Optimization configuration."""

    swarm_size: int = Field(default=10, ge=5)
    max_iterations: int = Field(default=30, ge=10)
    min_func: float = Field(default=1e-6, ge=0)
    min_step: float = Field(default=1e-6, ge=0)


class RandomSearchConfig(BaseModel):
    """Random Search configuration."""

    n_iterations: int = Field(default=10, ge=1)
    scoring: str = "accuracy"


class HyperparameterSearchConfig(BaseModel):
    """Hyperparameter search configuration."""

    method: Literal["pso", "random_search", "grid_search"] = "pso"
    pso: PSOConfig = PSOConfig()
    random_search: RandomSearchConfig = RandomSearchConfig()
    param_ranges: dict[str, dict] = Field(default_factory=dict)


class BootstrapConfig(BaseModel):
    """Bootstrap configuration for confidence intervals."""

    n_iterations: int = Field(default=100, ge=10)
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)


class StatisticalTestsConfig(BaseModel):
    """Statistical tests configuration."""

    method: str = "t_test"
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)


class LeaveOneOutConfig(BaseModel):
    """Leave-One-Out evaluation configuration."""

    enabled: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: list[str] = Field(
        default=[
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1_score",
            "auc_roc",
        ]
    )
    bootstrap: BootstrapConfig = BootstrapConfig()
    statistical_tests: StatisticalTestsConfig = StatisticalTestsConfig()
    leave_one_out: LeaveOneOutConfig = LeaveOneOutConfig()


class PlotStyleConfig(BaseModel):
    """Plot styling configuration."""

    font_family: str = "serif"
    font_serif: list[str] = ["Times New Roman"]
    figure_dpi: int = Field(default=300, ge=72)
    save_format: str = "svg"


class PCAVisualizationConfig(BaseModel):
    """PCA visualization configuration."""

    enabled: bool = True
    n_components: list[int] = [2, 3]
    kernels: list[str] = ["linear", "rbf", "sigmoid", "poly"]


class ROCVisualizationConfig(BaseModel):
    """ROC curve visualization configuration."""

    enabled: bool = True
    plot_folds: bool = True
    plot_mean: bool = True
    plot_std: bool = True


class SMOTEAnalysisConfig(BaseModel):
    """SMOTE analysis visualization configuration."""

    enabled: bool = True
    plot_class_distribution: bool = True
    plot_feature_distributions: bool = True


class VisualizationConfig(BaseModel):
    """Visualization configuration."""

    enabled: bool = True
    style: PlotStyleConfig = PlotStyleConfig()
    pca: PCAVisualizationConfig = PCAVisualizationConfig()
    roc: ROCVisualizationConfig = ROCVisualizationConfig()
    smote_analysis: SMOTEAnalysisConfig = SMOTEAnalysisConfig()


class PipelineStagesConfig(BaseModel):
    """Pipeline execution stages configuration."""

    analyze_recordings: bool = False
    save_intermediate: bool = True
    generate_visualizations: bool = True
    feature_selection: bool = True
    train_classifiers: bool = True
    evaluate_models: bool = True


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    stages: PipelineStagesConfig = PipelineStagesConfig()
    verbosity: int = Field(default=2, ge=0, le=2)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/voice_analysis.log"
    console: bool = True
