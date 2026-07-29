# LANTERN-XGB: An Interpretable Multi-Modal Machine Learning for Improving Clinical Decision-Making 

LANTERN-XGB is a machine learning pipeline designed for building and evaluating predictive models using XGBoost. It is tailored for integrating clinical and genomic data from multiple modalities to perform binary classification, multiclass classification, and survival analysis. The pipeline includes features for stable feature selection, hyperparameter optimization, and comprehensive model evaluation.

## Features

- **Data Integration:** Combines data from multiple sources (modalities) like clinical records, radiomics, and genomics.
- **Flexible Analysis:** Supports binary, multiclass, and survival analysis tasks.
- **XGBoost Core:** Utilizes the powerful XGBoost library for modeling.
- **Automated Workflow:**
    - Nested cross-validation for robust performance estimation.
    - Bayesian hyperparameter search for optimal model tuning.
    - Stable feature selection to identify the most predictive variables.
- **Comprehensive Evaluation:** Generates ROC curves, confusion matrices, and other relevant metrics.
- **Model Interpretability:** Uses SHAP (SHapley Additive exPlanations) to explain model predictions.
- **Configuration-Driven:** All aspects of the analysis are controlled through a central `config.yml` file.

## Installation

To set up the environment for this project, you will need to have Conda installed. Then you can create and activate the environment using the provided `environment.yml` file (you will need to create this file).

```bash
# Create the conda environment
conda env create -n lantern-xgb -f environment.yml

# Activate the environment
conda activate lantern-xgb
```

## Usage

The main entry point for the pipeline is `training.py`. The entire workflow is controlled by the `scripts/config.yml` file.

### 1. Configure the Analysis

Before running the pipeline, edit `scripts/config.yml` to define your analysis. This is where you specify:

- **`ANALYSIS_TYPE`**: The type of machine learning task (`binary`, `multiclass`, or `survival`).
- **`MODEL_TYPE`**: The algorithm to use (currently `xgboost`).
- **`PIPELINES`**: Define one or more models to build. Each pipeline has a `name` and a list of `modalities` (data sources) to use. For example:
  ```yaml
  PIPELINES:
    - name: Clinical_Model
      modalities:
        - PT
    - name: Combined_Model
      modalities:
        - PT
        - RAD
  ```
- **`TARGET_COLUMN`**: The name of the outcome variable you want to predict.
- **Data Paths**: The locations of your data files (`CLINICAL_DATA_DIR`, `MUTATION_FILE_PATH`).
- **And many other parameters** for cross-validation, feature selection, etc.

### 2. Run the Pipeline

The workflow is split into three main execution scripts depending on what stage of analysis you are in:

## Step A: Model Training & Internal Validation

Run this script to train your models, perform stable feature selection, optimize hyperparameters, and perform internal cross-validation.

```bash
python training.py
```

The script will execute the following steps:
1.  Load and preprocess the data according to the configuration.
2.  For each defined pipeline:
    - Perform nested cross-validation.
    - Tune hyperparameters using Bayesian optimization.
    - Select stable features.
    - Train a final consensus model.
Outputs: Saved model bundles (.joblib), internal CV ROC/DCA curves, global SHAP summary plots, and feature selection matrices.

## Step B: External Cohort Validation
Run this script to test your trained models on independent external datasets (defined under EXTERNAL_VALIDATION in the config).

```bash
python validation.py
```

Outputs: Evaluates standard (Z-scored) and ComBat-harmonized data, generating baseline vs. harmonized performance metrics, confusion matrices, and comparison ROC curves.

## Step C: Leave-One-Out (LOO) & Clinical Reports

Run this script specifically for micro-cohorts (e.g., n=20 Radiogenomic cohorts). It safely standardizes small datasets, calculates individualized SHAP trajectories, and generates the PDF reports.

```bash
python validation_loo.py
```

Outputs: Patient-specific LOO predictions, order-independent SHAP decision plots, comprehensive LOO ROC comparison plots, and individual Clinical Risk Assessment PDFs located in the reports/ subfolders.


## Project Structure

```
LANTERN-XGB/
├── training.py            # Core script for CV, feature selection, and model building
├── validation.py          # Script for large-scale external cohort evaluation
├── validation_loo.py      # Script for Leave-One-Out analysis & PDF report generation
├── environment.yml        # Conda environment dependency file
├── README.md              # This file
├── data/                  # Directory containing raw clinical and genomic datasets
│   └── dataset_Ouyang/    
├── scripts/
│   ├── config.yml         # Main configuration file
│   └── lib/               # Underlying pipeline modules
│       ├── loadData.py               # Data merging, cleaning, and strict type-casting
│       ├── train.py                  # Core training and cross-validation logic
│       ├── externalValidation.py     # External validation and ComBat logic
│       ├── clinical_report_generator.py # FPDF biological narrative builder
│       ├── clinical_metrics.py       # Calibration, DCA, and Bootstrapping utilities
│       ├── shap.py                   # Global SHAP plot generation
│       ├── paths.py                  # Directory path management
│       ├── saveModels.py             # Model bundle saving
│       └── savePlots.py              # Visualization formatting and saving
└── results/               # Auto-generated output directory for models, plots, and CSVs
```

### Citation
If you use this workflow in a paper, please cite:

Dalfovo, D.; Sassorossi, C.; De Paolis, E.; Campanella, A.; Nachira, D.; Petracca Ciavarella, L.; Boldrini, L.; Troost, E.G.C.; Ádány, R.; Farré, N.; et al. LANTERN-XGB: An Interpretable Multi-Modal Machine Learning for Improving Clinical Decision-Making in Lung Cancer. Int. J. Mol. Sci. 2026, 27, 3128. https://doi.org/10.3390/ijms27073128
