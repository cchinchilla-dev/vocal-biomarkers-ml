# Vocal Biomarkers for Respiratory Disease Detection using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2024.3487773-green.svg)](https://doi.org/10.1109/ACCESS.2024.3487773)

A comprehensive machine learning pipeline for early and cost-effective diagnosis of COVID-19 and other respiratory diseases through voice analysis, combining acoustic and biomechanical features.

## 📋 Table of Contents

- [Overview](#overview)
- [Paper Reference](#paper-reference)
- [Key Findings](#key-findings)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Output Files](#output-files)
- [Methodology](#methodology)
- [Code, Data Availability, and Reproducibility](#code-data-availability-and-reproducibility)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)

## 🔬 Overview

This repository implements a robust methodology for respiratory disease diagnosis based on vocal features and machine learning techniques. In contrast to existing methodologies that rely solely on acoustic attributes of the voice (such as intensity or frequency), our approach represents a **pioneering investigation that incorporates biomechanical aspects of vocal production**, including:

- Muscle tension
- Coordination of articulatory movements  
- Respiration patterns
- Glottal closure characteristics
- Vocal fold vibratory patterns

## 📄 Paper Reference

This code accompanies the research published in:

> A. J. L. Rivero, C. C. Corbacho, T. R. Arias, M. Martín-Merino and P. Vaz, **"Application of Machine Learning Techniques for the Characterization and Early Diagnosis of Respiratory Diseases Such as COVID-19,"** in *IEEE Access*, vol. 12, pp. 160516-160528, 2024.
> 
> DOI: [10.1109/ACCESS.2024.3487773](https://doi.org/10.1109/ACCESS.2024.3487773)

### Abstract

This paper presents a robust methodology for the early and cost-effective diagnosis of COVID-19 based on vocal features and machine learning techniques. The proposed methodology addresses all challenges inherent to the prediction of COVID-19, including those related to feature extraction and selection, the imbalance problem, and predictor training. In contrast to existing methodologies that rely solely on acoustic attributes of the voice, such as intensity or frequency, our approach represents a pioneering investigation that incorporates biomechanical aspects of vocal production. These include muscle tension, the coordination of articulatory movements, and respiration.

The relationship between these characteristics and the presence of the virus is investigated rigorously using robust feature selection techniques. To this end, we have constructed an original dataset comprising patients with confirmed cases of COVID-19 infection and a control group, incorporating both acoustic and biomechanical features using Voice Clinical Software. The robustness and reproducibility of the experimental results have been enhanced through the rigorous comparison of several classifiers and feature selection algorithms, as well as the employment of resampling strategies.

The application of random forests for feature selection has revealed that a limited set of biomechanical markers are significantly associated with the presence of COVID-19 infection. Moreover, a random forest classifier based on a subset of biomechanical and acoustic features demonstrates high efficacy in predicting cases of COVID-19 infection, achieving a sensitivity of  
**S = (0.9212 ± 0.0775)** while maintaining a specificity of  
**Sp = (0.9150 ± 0.0649)**.

Considering these findings, the proposed methodology can be regarded as a non-invasive and cost-effective alternative for the diagnosis of COVID-19 infection. Furthermore, it can be extended to the diagnosis of other respiratory diseases, provided that the vocal cords are affected.

## 📈 Key Findings

Our best-performing models achieve:

| Model | Accuracy | Balanced Accuracy | Sensitivity | Specificity | F1-Score |
|-------|----------|-------------------|-------------|-------------|----------|
| **SVM** | 0.9370 ± 0.0581 | 0.9571 ± 0.0771 | 0.9216 ± 0.0804 | 0.9394 ± 0.0566 | 0.9351 ± 0.0604 |
| **Random Forest** | 0.9135 ± 0.0647 | 0.9089 ± 0.0960 | 0.9212 ± 0.0775 | 0.9150 ± 0.0649 | 0.9071 ± 0.0820 |
| **KNN** | 0.9165 ± 0.0614 | 0.9342 ± 0.0761 | 0.9029 ± 0.0959 | 0.9185 ± 0.0600 | 0.9132 ± 0.0689 |

### Significant Biomechanical Markers

The following biomechanical markers were identified as significantly associated with COVID-19:

| Marker | Description | Clinical Interpretation |
|--------|-------------|------------------------|
| **Pr13** | Instability index | Voice instability in resisting vibration stress |
| **Pr14** | Index variation in amplitude | Inability to maintain amplitude |
| **Pr06** | Duration opening (%) | Challenges with vocal fold separation during opening |
| **Pr07** | Duration closing (%) | Challenges with vocal fold separation during closing |
| **Pr17** | Index OM closed phase | Presence of mucosal waves during closing phase |

## ✨ Features

### Feature Extraction
- **Acoustic Features**: F0 (fundamental frequency), formants (F1, F2), jitter, shimmer, HNR, MFCCs with deltas and delta-deltas
- **Biomechanical Markers**: 22 markers from Voice Clinical Systems App Online Lab (see Table 1 in paper)

### Feature Selection Methods
- **RFE with Bootstrap**: Recursive Feature Elimination with stability analysis (20 bootstrap iterations)
- **Statistical Tests**: Mann-Whitney U test with Bonferroni correction (α = 0.05)
- **Mutual Information**: Information-theoretic feature ranking
- **LASSO**: L1-regularized feature selection
- **CFS**: Correlation-based Feature Selection
- **Random Forest Importance**: Feature importance ranking

### Machine Learning Models
| Model | Abbreviation | Notes |
|-------|--------------|-------|
| AdaBoost | ADA | Ensemble boosting |
| K-Nearest Neighbors | KNN | Instance-based learning |
| Support Vector Machine | SVM | With RBF kernel, optimized via PSO |
| Bagging | BAG | Bootstrap aggregating |
| Random Forest | RF | Interpretable via decision rules |
| Gradient Boosting | GB | Sequential boosting |
| XGBoost | XGB | Optimized gradient boosting |
| LightGBM | LGB | Fast gradient boosting |
| Decision Tree | DT | Single tree classifier |

### Optimization & Evaluation
- **Particle Swarm Optimization (PSO)** for hyperparameter tuning
- **SMOTE** for class imbalance handling (generates synthetic samples)
- **Stratified 5-Fold Cross-Validation**
- **Bootstrap Confidence Intervals** (100 iterations)
- **Leave-One-Out Evaluation**

## 📁 Project Structure

```
vocal-biomarkers-ml/
├── config/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── raw/
│   │   ├── control/             # Control group audio recordings (.wav)
│   │   ├── pathological/        # COVID-19 positive audio recordings (.wav)
│   │   └── biomechanical/       # Biomechanical marker CSV files
│   └── results/
│       ├── datasets/            # Processed datasets
│       ├── features/            # Feature selection results
│       ├── metrics/             # Model performance metrics
│       └── visualizations/      # Generated plots (PCA, ROC curves)
├── logs/
│   └── voice_analysis.log       # Execution logs
├── src/
│   └── voice_analysis/
│       ├── __init__.py
│       ├── __main__.py          # CLI entry point
│       ├── pipeline.py          # Main pipeline orchestrator
│       ├── config/              # Configuration management (Pydantic)
│       ├── data/                # Data loading and preprocessing
│       ├── features/            # Feature extraction (acoustic, biomechanical)
│       ├── models/              # Classifiers and hyperparameter optimization
│       ├── metrics/             # Custom evaluation metrics
│       ├── preprocessing/       # Cleaning, resampling, standardization
│       ├── utils/               # I/O, logging, reproducibility
│       └── visualization/       # PCA plots, ROC curves
├── tests/                       # Unit and integration tests
├── pyproject.toml              # Project dependencies (Poetry)
└── README.md
```

## 💻 Requirements

### System Requirements
- Python 3.10, 3.11, or 3.12
- 8GB RAM minimum (16GB recommended)
- Unix-like OS (Linux/macOS) or Windows 10+

### Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3.2 | Machine learning algorithms |
| imbalanced-learn | 0.12.3 | SMOTE resampling |
| praat-parselmouth | ^0.4.3 | Acoustic feature extraction via Praat |
| librosa | ^0.10.0 | Audio analysis and MFCCs |
| xgboost | 2.0.3 | XGBoost classifier |
| lightgbm | 4.5.0 | LightGBM classifier |
| pyswarm | ^0.6 | Particle Swarm Optimization |
| pydantic | ^2.0.0 | Configuration validation |

## 🚀 Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/cchinchilla-dev/vocal-biomarkers-ml.git
cd vocal-biomarkers-ml

# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip

```bash
git clone https://github.com/cchinchilla-dev/vocal-biomarkers-ml.git
cd vocal-biomarkers-ml

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

## ⚙️ Configuration

All parameters are configured via `config/config.yaml`:

```yaml
# Key parameters (as used in the paper)
data_processing:
  cleaning:
    variance_threshold: 0.1      # Remove features with variance < 0.1
    correlation_threshold: 0.9   # Remove features with correlation > 0.9

feature_selection:
  rfe:
    stability_threshold: 0.4     # Minimum stability index
    n_bootstrap_iterations: 20   # Bootstrap iterations for RFE
  statistics:
    alpha: 0.05                  # Significance level
    correction_method: "bonferroni"

models:
  test_size: 0.2                 # 20% test split
  cross_validation:
    n_folds: 5                   # 5-fold CV
    stratified: true

hyperparameter_search:
  method: "pso"                  # Particle Swarm Optimization
  pso:
    swarm_size: 10
    max_iterations: 30

evaluation:
  bootstrap:
    n_iterations: 100            # Bootstrap iterations for metrics
    confidence_level: 0.95
```

## 📖 Usage

### Command Line

```bash
# Run complete pipeline
voice-analysis

# Custom configuration
voice-analysis --config path/to/config.yaml

# Run specific stage
voice-analysis --stage extract    # Feature extraction only
voice-analysis --stage features   # Feature selection only
voice-analysis --stage train      # Model training only

# Validate configuration
voice-analysis --dry-run

# Verbose output
voice-analysis --verbose
```

### Python API

```python
from voice_analysis.config.settings import get_settings
from voice_analysis.pipeline import VoiceAnalysisPipeline

# Load configuration
settings = get_settings("config/config.yaml")

# Run pipeline
pipeline = VoiceAnalysisPipeline(settings)
results = pipeline.run()

# Access results
print(f"Selected features: {results['n_selected_features']}")
print(f"Best model: {results['metrics'].loc[results['metrics']['Accuracy'].idxmax(), 'Model']}")
```

## 🔄 Pipeline Stages

### Stage 1: Data Loading
- Load audio recordings (Spanish vowel /a/, 4 seconds, 44.1 kHz)
- Load biomechanical markers from Voice Clinical Systems CSV exports
- Merge acoustic and biomechanical features

### Stage 2: Data Cleaning  
- Remove features with variance < 0.1
- Remove features with correlation > 0.9

### Stage 3: Feature Selection
1. Statistical tests (Mann-Whitney U + Bonferroni)
2. RFE with bootstrap stability analysis
3. Mutual Information scoring
4. LASSO regularization
5. Correlation-based Feature Selection
6. Ensemble combination

### Stage 4: SMOTE Resampling
- Balance classes using SMOTE (k_neighbors=7)
- Validate distribution preservation via PCA and Kolmogorov-Smirnov tests

### Stage 5: Model Training
- Train all classifiers with PSO-optimized hyperparameters
- 5-fold stratified cross-validation

### Stage 6: Evaluation
- Compute metrics with bootstrap confidence intervals
- Generate ROC curves
- Perform Leave-One-Out analysis
- Calculate per-patient misclassification probabilities

## 📊 Output Files

| Directory | File | Description |
|-----------|------|-------------|
| `datasets/` | `dataset.csv` | Raw combined dataset |
| `datasets/` | `dataset_clean.csv` | After cleaning |
| `datasets/` | `dataset_smote.csv` | After SMOTE |
| `features/` | `bonferroni.csv` | Statistical test results |
| `features/` | `rfe.csv` | RFE rankings and stability indices |
| `features/` | `features_to_use.csv` | Final selected features |
| `metrics/` | `classifier_metrics.csv` | All model metrics |
| `metrics/` | `prob_misclassification.csv` | Misclassification probabilities |
| `visualizations/` | `pca_*.svg` | PCA plots |
| `visualizations/` | `roc_*.svg` | ROC curves |

## 🧪 Methodology

### Participants (as described in paper)
- **74 adult participants** total
- **15 COVID-19 positive** (7 women, 8 men, ages 23-55)
- **59 Control group** (44 women, 15 men, ages 18-58)
- Voice samples collected June-September 2020 (pre-vaccination)
- Recording: Spanish vowel /a/ for 4 seconds using App Online Lab

### Acoustic Features
| Category | Features | Functionals |
|----------|----------|-------------|
| Phonation | F0, F1, F2 | Mean, Std, Skew, Kurtosis, Δ1, Δ2 |
| Perturbation | Jitter (5 types), Shimmer (6 types) | Direct measures |
| Noise | HNR | Direct measure |
| Spectral | MFCC[1-13] | Mean, Std, Skew, Kurtosis, Δ1, Δ2 |

### Biomechanical Markers (22 total)
See Table 1 in the paper for complete descriptions. Categories include:
- Fundamental Frequency (Pr01)
- Harmony in vocal fold movement (Pr02)
- Phase characteristics (Pr03-Pr07)
- Muscle tension (Pr08-Pr09)
- Sufficiency of seal (Pr10-Pr12)
- Tension and instability (Pr13-Pr14)
- Edge spacing (Pr15-Pr16)
- Mucosal wave correlates (Pr17-Pr20)
- Mass correlates (Pr21-Pr22)

---

## 📚 Code, Data Availability, and Reproducibility

This repository contains the complete codebase developed throughout the research process, including both the code used to produce the results reported in the published article and additional experimental scripts and analyses that were not included in the final manuscript or its supplementary materials.

The additional code reflects intermediate iterations, alternative methodological approaches, and exploratory analyses conducted during model development. Consequently, some scripts may be experimental in nature and may include incomplete configurations, provisional implementations, or minor inconsistencies that were not part of the final validated pipeline.

### Known Differences Between Code and Paper

Minor discrepancies between the published article and this codebase may exist as a result of the natural evolution of the code during the research process, including experimentation, refactoring, and optimization. **These discrepancies do not affect the core methodology or the main conclusions of the study.** The published article should be considered the authoritative reference for the final experimental design and results.

### Data Availability

Due to data privacy and ethical constraints, the dataset used in this research cannot be made publicly available. Access to the data may be granted by the authors upon reasonable request, subject to applicable regulations and approvals.

### Reproducibility

To support methodological transparency and reproducibility, this repository provides all preprocessing, feature selection, and modeling code required to replicate the experimental pipeline. Full reproduction of the reported results requires access to the original dataset.

To match paper results exactly:
1. Set `reproducibility.n_executions: 100` in config.yaml
2. Set `evaluation.bootstrap.n_iterations: 100`
3. Use the same random seeds: `[42, 123, 1234, 12345, 0, 1]`

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@ARTICLE{10737357,
  author={Rivero, Alfonso José López and Corbacho, Carlos Chinchilla and 
          Arias, Tatiana Romero and Martín-Merino, Manuel and Vaz, Paulo},
  journal={IEEE Access}, 
  title={Application of Machine Learning Techniques for the Characterization 
         and Early Diagnosis of Respiratory Diseases Such as COVID-19}, 
  year={2024},
  volume={12},
  pages={160516-160528},
  doi={10.1109/ACCESS.2024.3487773}
}
```

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License** (CC BY-NC-ND 4.0).

## 👥 Authors

### Core Development Team

- **Carlos Chinchilla Corbacho**
  Universidad Pontificia de Salamanca  
  📧 [cchinchilla@usal.es](mailto:cchinchilla@usal.es)  
  🔗 [GitHub](https://github.com/cchinchilla-dev)  
  Primary developer and maintainer of the repository.

### Research Team

- **Alfonso José López Rivero** - *Principal Investigator* - Universidad Pontificia de Salamanca
- **Tatiana Romero Arias** - *Clinical Research & Data Collection* - Universidad Europea de Canarias
- **Manuel Martín-Merino** - *Machine Learning Advisor* - Universidad Pontificia de Salamanca
- **Paulo Vaz** - *Research Collaboration* - Polytechnic Institute of Viseu

---

For questions, issues, or contributions, please open an issue on GitHub or contact the authors directly.