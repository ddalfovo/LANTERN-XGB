import joblib
from pathlib import Path
from scripts.lib.externalValidation import run_external_validation

from scripts.lib.paths import *

# 1. Load the context
context = joblib.load(RESULTS_DIR / "training_context.joblib")

# 2. Load your model bundles
bundle = joblib.load(RESULTS_DIR / "saved_models/Combined_Model_bundle.joblib")
pipelines = {bundle['analysis_metadata']['pipeline']: bundle}

print("--- Run external validation ---")
validation_performance = run_external_validation(
    pipelines, 
    X_train=context['X_train'], 
    y_train=context['y_train'], 
    mappings=context['mappings'], 
    modality_features=context['modality_features']
)
print("--- Analysis complete ---")





# validation_performance = run_external_validation(pipelines, X_train=X, y_train=y, mappings=mappings, modality_features=modality_features)
# if config['RUN_DECISION_PLOTS']:
#     run_external_decision_plots(pipelines, validation_performance)

