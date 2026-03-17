from pathlib import Path
from scripts.lib.loadData import load_and_prepare_data
from scripts.lib.train import train_evaluate_cv
from scripts.lib.saveModels import save_model_package
from scripts.lib.externalValidation import run_external_validation, run_external_decision_plots
import joblib

# CONFIG FILE
import yaml
config = yaml.safe_load(open("./scripts/config.yml"))

# to define output folders
from scripts.lib.paths import *

# --- Create output folders and remove all files into individual features ---
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("--- Loading data ---")

X, X_display, y, class_names, mappings, modality_features = load_and_prepare_data(config['MUTATION_FILE_PATH'], config['TARGET_COLUMN'], config['POSITIVE_CLASS'], config['ANALYSIS_TYPE'])

print("--- Loading complete ---")
print("--- Training ---")

pipelines = train_evaluate_cv(X, y, class_names, modality_features, config['ANALYSIS_TYPE'], config['MODEL_TYPE'], config['HANDLE_IMBALANCE'], config['FEATURE_SELECTION'], mappings)

print("--- Training complete ---")
print("--- Saving the models ---")
save_model_package(pipelines, class_names, RESULTS_DIR)
context_bundle = {
    'X_train': X,
    'y_train': y,
    'mappings': mappings,
    'modality_features': modality_features
}
joblib.dump(context_bundle, RESULTS_DIR / "training_context.joblib")
print("Saved training context for later validation.")
