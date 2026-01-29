"""
External Validation Module
--------------------------
This module provides functions to evaluate trained models on independent 
external datasets. It ensures feature consistency between training and 
testing sets and generates comparative ROC visualization plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import cycle
import yaml
import re
import shap

# Internal library imports for standardized plot saving and path management
from scripts.lib.savePlots import save_plot
from scripts.lib.paths import *

# --- CONFIGURATION & PATH SETUP ---
config = yaml.safe_load(open("./scripts/config.yml"))
RESULTS_DIR = Path(RESULTS_DIR)

def sigmoid(x):
    """Converts logit (log-odds) to probability."""
    return 1 / (1 + np.exp(-x))


def run_external_validation(all_results):
    """
    Evaluates every trained pipeline against all external datasets defined in the config.

    This function performs the following for each external dataset:
    1. Loads the raw data and sanitizes column names.
    2. Identifies and aligns features (handling missing columns).
    3. Generates predictions using the stored model bundles.
    4. Calculates AUC and plots an ROC curve comparing all pipelines.

    Args:
        all_results (dict): Dictionary of model bundles from the training phase.

    Returns:
        dict: A nested dictionary containing performance metrics (FPR, TPR, AUC) 
              indexed by dataset and pipeline name.
    """
    external_configs = config.get('EXTERNAL_VALIDATION', [])
    if not external_configs:
        print("No external validation sets found in config.")
        return

    # To store data for the final "Combined" plot
    # Structure: {dataset_name: {pipeline_name: {fpr, tpr, auc}}}
    validation_performance = {}

    for ext_conf in external_configs:
        dataset_name = ext_conf['name']
        print(f"\n--- Validating on Dataset: {dataset_name} ---")
        
        # 1. Load external data and sanitize headers (matching XGBoost training names)
        ext_df = pd.read_csv(ext_conf['path'], sep='\t')
        ext_df.columns = [re.sub(r'[\[\]<>()\s,]', '_', col) for col in ext_df.columns]
        
        y_true = ext_df[ext_conf['label_col']]
        validation_performance[dataset_name] = {}

        # Initialize plot for this specific dataset
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for pipeline_name, bundle in all_results.items():
            model = bundle['pipeline']
            features = bundle['selected_features']
            
            # --- CRITICAL: FEATURE ALIGNMENT ---
            # Models fail if columns are missing or in the wrong order.
            X_ext = ext_df.copy()
            for col in features:
                if col not in X_ext.columns:
                    # Fill missing features required by the model with 0
                    X_ext[col] = 0 
            
            # Force the external dataframe to have the exact column order as training
            X_ext_aligned = X_ext[features]
            
            # 2. Prediction and Metric Calculation
            # Get probability scores for the positive class (column 1)
            y_proba = model.predict_proba(X_ext_aligned)[:, 1]
            auc_val = roc_auc_score(y_true, y_proba)
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            
            # Store results for the multi-dataset summary plot
            validation_performance[dataset_name][pipeline_name] = {
                'fpr': fpr, 'tpr': tpr, 'auc': auc_val
            }
            
            n_bootstraps = 1000
            bootstrapped_aucs = [roc_auc_score(y_true[indices], y_proba[indices])
                                     for _ in range(n_bootstraps)
                                     if len(np.unique(y_true[(indices := np.random.RandomState(config['RANDOM_STATE']+_).choice(len(y_true), len(y_true), replace=True))])) > 1]
            lower_global, upper_global = np.percentile(bootstrapped_aucs, [2.5, 97.5])

            # Add this pipeline's curve to the plot
            ax.plot(fpr, tpr, label=f"{pipeline_name} (AUC = {auc_val:.2f}) [95% CI: {lower_global:.2f} - {upper_global:.2f}]")
            print(f"  Pipeline {pipeline_name} -> AUC: {auc_val:.4f}")

        # --- PLOT FORMATTING ---
        ax.plot([0, 1], [0, 1], 'k--', label="Random Chance")
        ax.set_title(f"External Validation: {dataset_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        
        # Export the visual result
        save_plot(fig, f"Ext_Val_{dataset_name}_Comparison", RESULTS_DIR / "external_validation")
        plt.show()

    return validation_performance


def plot_combined_external_results(validation_performance):
    """
    Generates a master plot comparing performance across all datasets.
    
    This visualizes the 'robustness' of the models—if the curves are grouped
    together, the model is stable across different populations.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    for dataset_name, pipelines in validation_performance.items():
        color = next(colors)
        for pipeline_name, metrics in pipelines.items():
            ax.plot(metrics['fpr'], metrics['tpr'], 
                    label=f"{dataset_name} - {pipeline_name} (AUC: {metrics['auc']:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_title("Combined External Validation Results")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_plot(fig, "Combined_External_Validation", RESULTS_DIR / "external_validation")
    plt.show()


def run_external_decision_plots(all_results, validation_performance):
    """
    Generates SHAP decision plots for every individual sample in the external datasets.
    """
    external_configs = config.get('EXTERNAL_VALIDATION', [])
    
    for ext_conf in external_configs:
        dataset_name = ext_conf['name']
        # Load and clean data (re-using your existing logic)
        ext_df = pd.read_csv(ext_conf['path'], sep='\t')
        # ext_df.columns = [re.sub(r'[\[\]<>()\s,]', '_', col) for col in ext_df.columns]
        
        # Create output directory for this specific dataset's samples
        sample_dir = RESULTS_DIR / "external_validation" / f"samples_{dataset_name}"
        sample_dir.mkdir(parents=True, exist_ok=True)

    for pipeline_name, bundle in all_results.items():
        pipeline = bundle['pipeline']
        selected_features = bundle['selected_features']
        
        # 1. Extract the actual model
        try:
            model = pipeline.named_steps['model']
        except (AttributeError, KeyError):
            model = pipeline

        # 2. Align features
        X_ext = ext_df.copy()
        for col in selected_features:
            if col not in X_ext.columns:
                X_ext[col] = 0
        X_ext_aligned = X_ext[selected_features]

        # 3. Initialize SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values_obj = explainer(X_ext_aligned)
        
        # 4. Handle base_values
        base_value = explainer.expected_value
        # For multi-class/multi-output, we take the target class index
        # Usually 1 for binary classification (the positive class)
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            base_value = base_value[1]

        print(f" - Processing {len(X_ext_aligned)} samples for pipeline: {pipeline_name}")
        
        for i in range(len(X_ext_aligned)):
            # ID Retrieval
            clinical_id_col = config.get('CLINICAL_ID', 'ID')
            sample_id = X_ext_aligned.iloc[i][clinical_id_col] if clinical_id_col in X_ext_aligned.columns else X_ext_aligned.index[i]
            
            # --- START PERCENTAGE CALCULATION LOGIC ---
            sample_shaps = shap_values_obj.values[i]
            # Handle multiclass indexing if necessary
            if len(sample_shaps.shape) > 1:
                sample_shaps = sample_shaps[:, 1]

            sample_features = X_ext_aligned.iloc[i]
            
            # Bundle data for sorting: (original_index, shap_val, feature_val, name)
            features_data = []
            for idx, name in enumerate(selected_features):
                features_data.append((idx, sample_shaps[idx], sample_features.values[idx], name))

            # Sort by SHAP value magnitude to calculate incremental probability change
            sorted_features_data = sorted(features_data, key=lambda x: x[1])

            new_feature_labels = [None] * len(selected_features)
            current_logit = base_value

            for data_tuple in sorted_features_data:
                orig_idx, shap_val, feat_val, feat_name = data_tuple
                
                prob_before = sigmoid(current_logit)
                current_logit += shap_val
                prob_after = sigmoid(current_logit)
                
                prob_change_percent = (prob_after - prob_before) * 100
                
                # Format name (removing prefixes as per your snippet)
                clean_name = feat_name.replace('spiro.', '').replace("ph.", "")
                new_label = f"{clean_name} = {feat_val} ({prob_change_percent:+.1f}%)"
                
                new_feature_labels[orig_idx] = new_label
            # --- END PERCENTAGE CALCULATION LOGIC ---

            # Create the Plot
            plt.figure(figsize=(6, 16))
            
            shap.decision_plot(
                base_value, 
                sample_shaps, 
                feature_names=new_feature_labels,
                show=False,
                link='logit' # This keeps the X-axis in probability space (0 to 1)
            )
            
            plt.title(f"Decision Path: {pipeline_name} | Patient: {sample_id}\n(Values in parentheses show % change in Probability)", 
                      fontsize=14, pad=20)
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"Decision_{pipeline_name}_Patient_{sample_id}"
            save_plot(plt.gcf(), plot_filename, sample_dir)
            plt.close()

    print("Individual sample decision plots completed.")
