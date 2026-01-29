# LANTERN-XGB: A Machine Learning Pipeline for Clinical and Genomic Data

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
conda env create -f environment.yml

# Activate the environment
conda activate lantern-xgb
```
*Note: The `environment.yml` file is not yet created. You will need to create it with the necessary dependencies (e.g., pandas, scikit-learn, xgboost, shap, pyyaml, matplotlib, seaborn, lifelines, skopt, imbalanced-learn).*

## Usage

The main entry point for the pipeline is `main.py`. The entire workflow is controlled by the `scripts/config.yml` file.

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

Once the configuration is set, run the main script from the root directory of the project:

```bash
python main.py
```

The script will execute the following steps:
1.  Load and preprocess the data according to the configuration.
2.  For each defined pipeline:
    - Perform nested cross-validation.
    - Tune hyperparameters using Bayesian optimization.
    - Select stable features.
    - Train a final consensus model.
3.  Save the trained models, evaluation plots (like ROC curves), and performance metrics to the `results/` directory (or the directory specified in `BASE_DIR` in the config).
4.  Run external validation if configured.

## Project Structure

```
├── main.py                # Main script to run the pipeline
├── README.md              # This file
├── data/                  # Directory for datasets
│   └── dataset_Ouyang/
├── scripts/
│   ├── config.yml         # Main configuration file for the project
│   └── lib/               # Python modules for the pipeline
│       ├── loadData.py        # Handles data loading and preprocessing
│       ├── train.py           # Core training and evaluation logic
│       ├── externalValidation.py # External validation logic
│       ├── saveModels.py      # Saves trained models
│       ├── savePlots.py       # Saves evaluation plots
│       ├── shap.py            # SHAP value computation
│       └── paths.py           # Defines output paths
└── results/                 # Output directory for models, plots, and metrics
```
