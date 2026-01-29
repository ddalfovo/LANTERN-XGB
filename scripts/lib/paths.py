"""
Results Directory Management
----------------------------
This script initializes the output directory structure for model results. 
It dynamically generates folder names based on the experimental parameters 
defined in the configuration file to ensure results are organized and 
not overwritten during different runs.
"""

from pathlib import Path
import yaml

# --- CONFIGURATION LOADING ---
# Load the master configuration file that dictates paths and experiment settings.
config = yaml.safe_load(open("./scripts/config.yml"))

# --- DYNAMIC RESULTS DIRECTORY SETUP ---

# 1. Define the base storage path from the config.
base_dir = Path(config['BASE_DIR'])

# 2. Determine the 'Outcome Label' for the folder name.
# Depending on the analysis type, we use different identifier values:
# - Survival: Uses the specified 'SURVIVAL_CUTOFF'.
# - Binary: Uses the specified 'POSITIVE_CLASS'.
# - Multiclass: Simply labels the folder as 'multiclass'.
outcome_label = (
    config['SURVIVAL_CUTOFF'] if config['ANALYSIS_TYPE'] == 'survival' 
    else config['POSITIVE_CLASS'] if config['ANALYSIS_TYPE'] == 'binary' 
    else 'multiclass'
)

# 3. Handle optional suffixing.
# If 'ADD_SUFF' is provided in the config, it adds a custom tag (e.g., "_experiment_v2").
suffix = f"_{config['ADD_SUFF']}" if config['ADD_SUFF'] != '' else ""

# 4. Construct the Final Folder Name.
# Format: AnalysisOf.{Target}_Class.{Label}_Model.{ModelType}_{Suffix}
# Example: "AnalysisOf.pdl1_Class.High_Model.XGBoost_v1"
folder_name = (
    f"AnalysisOf.{config['TARGET_COLUMN']}_"
    f"Class.{outcome_label}_"
    f"Model.{config['MODEL_TYPE']}"
    f"{suffix}"
)

# 5. Create the full Path object.
RESULTS_DIR = base_dir / folder_name
