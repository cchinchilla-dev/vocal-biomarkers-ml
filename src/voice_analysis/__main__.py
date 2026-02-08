"""
Main entry point for the COVID-19 voice analysis pipeline.

This module provides the command-line interface for running the analysis.
"""

import argparse
import sys
from pathlib import Path

from .config.settings import get_settings, reset_settings
from .pipeline import VoiceAnalysisPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="voice-analysis",
        description="COVID-19 Voice Analysis Pipeline - Machine learning for respiratory disease detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  voice-analysis

  # Run with custom configuration file
  voice-analysis --config path/to/config.yaml

  # Run only feature extraction stage
  voice-analysis --stage extract

  # Run with verbose output
  voice-analysis --verbose

For more information, visit: https://github.com/cchinchilla-dot/Voice-Analysis-for-Respiratory-Disease-Detection
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration YAML file (default: config/config.yaml)",
    )

    parser.add_argument(
        "-s",
        "--stage",
        choices=["extract", "features", "train", "evaluate", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the pipeline",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser.parse_args()


def configure_stages(settings, stage: str) -> None:
    """Configure pipeline stages based on command-line argument."""
    stages = settings.pipeline.stages

    if stage == "extract":
        stages.analyze_recordings = True
        stages.feature_selection = False
        stages.train_classifiers = False
        stages.evaluate_models = False
    elif stage == "features":
        stages.analyze_recordings = False
        stages.feature_selection = True
        stages.train_classifiers = False
        stages.evaluate_models = False
    elif stage == "train":
        stages.analyze_recordings = False
        stages.feature_selection = False
        stages.train_classifiers = True
        stages.evaluate_models = True
    elif stage == "evaluate":
        stages.analyze_recordings = False
        stages.feature_selection = False
        stages.train_classifiers = False
        stages.evaluate_models = True


def main() -> int:
    """
    Main entry point for the voice analysis pipeline.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_args()

    try:
        # Load and validate configuration
        reset_settings()  # Clear any cached settings
        settings = get_settings(args.config)

        # Apply verbosity setting
        if args.verbose:
            settings.pipeline.verbosity = 2
            settings.logging.level = "DEBUG"

        # Configure stages
        if args.stage != "all":
            configure_stages(settings, args.stage)

        # Dry run: just validate configuration
        if args.dry_run:
            print("Configuration validated successfully!")
            print(f"  Project: {settings.project.name} v{settings.project.version}")
            print(f"  Data directory: {settings.paths.data_dir}")
            print(f"  Results directory: {settings.paths.results_dir}")
            print(f"  Enabled models: {', '.join(settings.models.enabled_models)}")
            print(f"  Number of executions: {settings.reproducibility.n_executions}")
            return 0

        # Run the pipeline
        print("\n" + "=" * 60)
        print(f"  {settings.project.name} v{settings.project.version}")
        print("=" * 60 + "\n")

        pipeline = VoiceAnalysisPipeline(settings)
        results = pipeline.run()

        # Print summary
        print("\n" + "=" * 60)
        print("  Pipeline Execution Summary")
        print("=" * 60)

        if "raw_dataset_shape" in results:
            print(f"  Original dataset shape: {results['raw_dataset_shape']}")

        if "cleaned_dataset_shape" in results:
            print(f"  Cleaned dataset shape: {results['cleaned_dataset_shape']}")

        if "n_selected_features" in results:
            print(f"  Selected features: {results['n_selected_features']}")

        if "metrics" in results:
            metrics = results["metrics"]
            best_model = metrics.loc[metrics["Accuracy"].idxmax()]
            print(f"\n  Best performing model: {best_model['Model']}")
            print(f"    Accuracy: {best_model['Accuracy']:.4f}")
            print(f"    Sensitivity: {best_model['Sensitivity']:.4f}")
            print(f"    Specificity: {best_model['Specificity']:.4f}")

        print("\n" + "=" * 60)
        print("  Execution completed successfully!")
        print("=" * 60 + "\n")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure the configuration file exists.", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
