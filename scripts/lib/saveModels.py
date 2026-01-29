"""
Model Serialization & Deployment Module
---------------------------------------
This module handles the persistent storage of trained machine learning 
pipelines. It bundles the model with its metadata, feature lists, and 
evaluation metrics to ensure full reproducibility and ease of deployment 
for future clinical data.
"""

import joblib
import datetime

def save_model_package(all_results, class_names, RESULTS_DIR):
    """
    Exports trained models and their associated metadata as compressed bundles.

    For each pipeline executed, this function creates a dictionary (bundle) 
    containing everything required to reload the model and understand its 
    context without needing the original configuration files.

    Args:
        all_results (dict): A dictionary containing pipeline results, where keys 
                            are pipeline names and values are dicts containing 
                            the trained 'pipeline', 'selected_features', etc.
        class_names (list): List of human-readable labels for the target classes 
                            (e.g., ['Healthy', 'Diseased']).
        RESULTS_DIR (Path): Path object pointing to the main results directory.

    Returns:
        None
    """
    # Generate a timestamp for version control and to prevent overwriting previous runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Define and create the sub-directory for saved models
    save_path = RESULTS_DIR / 'saved_models'
    save_path.mkdir(parents=True, exist_ok=True)

    # Iterate through each trained pipeline (e.g., 'PT_Model', 'Combined_Model')
    for pipeline_name, data in all_results.items():
        
        # --- BUNDLE CONSTRUCTION ---
        # We wrap the model in a dictionary to keep metadata attached to the binary
        bundle = {
            'model': data['pipeline'],              # The actual trained Scikit-Learn/XGBoost pipeline
            'features': data['selected_features'],  # List of features used (important for future data alignment)
            'best_params': data['best_params'],     # Hyperparameters found during BayesSearchCV
            'class_names': class_names,             # Mapping for the prediction output
            'auc_verification': data['verification_auc'], # Quick reference for model quality
            'curve_data': data.get('curve_data'),   # Stored ROC/PR curve points for future visualization
            'analysis_metadata': {
                'date': timestamp,
                'pipeline': pipeline_name,
                'python_version': '3.x',            # Tracks environment context
            }
        }
        
        # --- FILE SERIALIZATION ---
        # Save as a .joblib file. Joblib is preferred over Pickle for models 
        # containing large NumPy arrays (typical for ML models).
        filename = save_path / f"{pipeline_name}_bundle_{timestamp}.joblib"
        
        try:
            joblib.dump(bundle, filename)
            print(f"Successfully saved complete bundle to: {filename}")
        except Exception as e:
            print(f"Failed to save model {pipeline_name}: {e}")