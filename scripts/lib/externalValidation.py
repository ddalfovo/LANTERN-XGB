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
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
from neuroCombat import neuroCombat
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import cycle
import yaml
import re
import shap
import joblib # Added to save parameters for N=1

from scripts.lib.savePlots import save_plot
from scripts.lib.clinical_metrics import evaluate_clinical_utility
from scripts.lib.paths import *

config = yaml.safe_load(open("./scripts/config.yml"))
RESULTS_DIR = Path(RESULTS_DIR)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_clinical_metrics(y_true, y_pred, y_proba, pipeline_name, dataset_name, out_dir, suffix=""):
    """Helper function to compute realistic metrics and plot confusion matrix."""
    from sklearn.metrics import confusion_matrix, roc_auc_score
    import seaborn as sns
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    # Calculate Metrics
    metrics_dict = {
        'Dataset': dataset_name,
        'Pipeline': f"{pipeline_name}{suffix}",
        'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'Sensitivity (TPR)': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity (TNR)': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'PPV (Precision)': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1 Score': (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'ROC AUC': roc_auc_score(y_true, y_proba)
    }
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = out_dir / f"{pipeline_name}_{dataset_name}{suffix}_Metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set_title(f"Confusion Matrix: {pipeline_name}{suffix}\nDataset: {dataset_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_plot(fig, f"Confusion_Matrix_{pipeline_name}_{dataset_name}{suffix}", out_dir)
    plt.close(fig)


def run_external_validation(all_results, X_train=None, y_train=None, mappings=None ,modality_features=None):
    external_configs = config.get('EXTERNAL_VALIDATION', [])
    if not external_configs:
        print("No external validation sets found in config.")
        return

    ext_base_dir = RESULTS_DIR / "external_validation"
    ext_base_dir.mkdir(parents=True, exist_ok=True)
    validation_performance = {}
    
    # Extract numerical AND categorical mappings
    categorical_mappings = mappings[0] if mappings else {}
    numerical_mappings = mappings[1] if mappings else {}

    # Define a color cycle to stay consistent across different plots
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    pipeline_colors = {name: colors[i % len(colors)] for i, name in enumerate(all_results.keys())}

    for ext_conf in external_configs:
        dataset_name = ext_conf['name']
        
        # Create a specific folder for this dataset
        dataset_ext_dir = ext_base_dir / dataset_name
        dataset_ext_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Validating on Dataset: {dataset_name} ---")
        
        ext_df = pd.read_csv(ext_conf['path'], sep='\t')
        ext_df.columns = [re.sub(r'[\[\]<>()\s,]', '_', col) for col in ext_df.columns]
        y_true = ext_df[ext_conf['label_col']]
        validation_performance[dataset_name] = {}

        fig, ax = plt.subplots(figsize=(8, 8))
        
        for pipeline_name, bundle in all_results.items():
            color = pipeline_colors[pipeline_name]

            model = bundle['model']
            features = bundle['features']
            
            # --- NEW: Identify harmonized features for THIS specific pipeline ---
            pipe_conf = next((p for p in config.get('PIPELINES', []) if p['name'] == pipeline_name), {})
            harmonization_rule = pipe_conf.get('harmonization', [])
            
            harmonized_feature_set = set()
            if isinstance(harmonization_rule, list) and modality_features:
                for mod in harmonization_rule:
                    if mod.lower() in modality_features:
                        harmonized_feature_set.update(modality_features[mod.lower()])

            # Create a raw copy for ComBat later
            X_ext_raw = ext_df.copy()
            for col in features:
                if col not in X_ext_raw.columns:
                    X_ext_raw[col] = np.nan
            X_ext_aligned_raw = X_ext_raw[features].copy()
            
            # Create aligned scaled copy for baseline model
            X_ext_aligned = X_ext_aligned_raw.copy()

            # try:
            #     booster = model.get_booster()
            #     f_types = booster.feature_types if hasattr(booster, 'feature_types') else None
            # except Exception:
            #     f_types = None

            for col in features:
                # 1. Use the mappings from training to know exactly what is categorical!
                is_cat = col in categorical_mappings

                if is_cat:
                    # 2. Safely cast to category AND enforce the same categories as training
                    X_ext_aligned[col] = X_ext_aligned[col].astype('category')
                    X_ext_aligned[col] = X_ext_aligned[col].cat.set_categories(categorical_mappings[col])
                    
                    X_ext_aligned_raw[col] = X_ext_aligned_raw[col].astype('category')
                    X_ext_aligned_raw[col] = X_ext_aligned_raw[col].cat.set_categories(categorical_mappings[col])
                else:
                    X_ext_aligned[col] = pd.to_numeric(X_ext_aligned[col], errors='coerce')
                    X_ext_aligned_raw[col] = pd.to_numeric(X_ext_aligned_raw[col], errors='coerce')
                    
                    if col in numerical_mappings and col in harmonized_feature_set:
                        train_mean = numerical_mappings[col]['mean']
                        train_std = numerical_mappings[col]['std']
                        if train_std > 0:
                            X_ext_aligned[col] = (X_ext_aligned[col] - train_mean) / train_std
            
            # 1. BASELINE PREDICTION & CI
            y_proba = model.predict_proba(X_ext_aligned)[:, 1]
            
            # --- YOUDEN INDEX & PREDICTION ---
            youden_threshold = 0.5
            if X_train is not None and y_train is not None:
                X_train_aligned = X_train[features].copy()
                y_train_proba = model.predict_proba(X_train_aligned)[:, 1]
                y_train_true = y_train.values if isinstance(y_train, pd.Series) else y_train.iloc[:, 0].values
                
                fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, y_train_proba)
                youden_index = tpr_train - fpr_train
                youden_threshold = thresholds_train[np.argmax(youden_index)]
            
            y_pred_pseudo = (y_proba >= youden_threshold).astype(int)
            
            # Calculate Baseline Metrics & CM
            save_clinical_metrics(y_true, y_pred_pseudo, y_proba, pipeline_name, dataset_name, dataset_ext_dir, suffix="_Baseline")

            auc_val = roc_auc_score(y_true, y_proba)
            fpr, tpr, _ = roc_curve(y_true, y_proba)

            evaluate_clinical_utility(
                y_true=y_true, y_proba=y_proba, pipeline_name=pipeline_name, 
                dataset_name=dataset_name, out_dir=dataset_ext_dir
            )

            validation_performance[dataset_name][pipeline_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val}
            
            # Bootstrap for CI
            n_bootstraps = 1000
            rng = np.random.RandomState(config['RANDOM_STATE'])
            bootstrapped_aucs = []
            for i in range(n_bootstraps):
                indices = rng.choice(len(y_true), len(y_true), replace=True)
                if len(np.unique(y_true.iloc[indices])) < 2:
                    continue
                bootstrapped_aucs.append(roc_auc_score(y_true.iloc[indices], y_proba[indices]))
            
            lower, upper = np.percentile(bootstrapped_aucs, [2.5, 97.5])
            
            ax.plot(fpr, tpr, color=color, linestyle='--', alpha=0.7,
                    label=f"{pipeline_name} Baseline (AUC: {auc_val:.2f} [{lower:.2f}-{upper:.2f}])")
            print(f"  Pipeline {pipeline_name} (Baseline) -> AUC: {auc_val:.4f}")

            # 2. HARMONIZATION (ComBat) PREDICTION & CI
            if X_train is not None and y_train is not None and mappings is not None:
                try:
                    import warnings
                    print(f"  --- Running ComBat Harmonization for {dataset_name} ---")
                    
                    # 1. Identify continuous harmonized features (Radiomics only)
                    harmonized_continuous_features = [
                        col for col in features 
                        if X_ext_aligned_raw[col].dtype.name != 'category' and col in harmonized_feature_set
                    ]
                    
                    # 2. Reconstruct RAW Training Data FOR HARMONIZED FEATURES ONLY
                    X_train_raw = X_train.copy()
                    valid_cont_features = []
                    for col in harmonized_continuous_features:
                        if col in numerical_mappings and col in X_train_raw.columns:
                            train_mean = numerical_mappings[col]['mean']
                            train_std = numerical_mappings[col]['std']
                            # Reverse the Z-score for training data
                            X_train_raw[col] = (X_train_raw[col] * train_std) + train_mean
                            
                            if X_train_raw[col].var() > 1e-6 and X_ext_aligned_raw[col].var() > 1e-6:
                                valid_cont_features.append(col)
                    
                    if valid_cont_features:
                        X_train_cont_raw = X_train_raw[valid_cont_features].copy()
                        X_ext_cont_raw = X_ext_aligned_raw[valid_cont_features].copy()
                        
                        # 3. Combine raw data
                        combined_data = pd.concat([X_train_cont_raw, X_ext_cont_raw], axis=0).T
                        combined_data = combined_data.fillna(combined_data.mean(axis=1))
                        
                        y_train_labels = y_train.values if isinstance(y_train, pd.Series) else y_train.iloc[:, 0].values
                        y_ext_labels = y_true.values
                        
                        batch_labels = [0] * len(X_train_cont_raw) + [1] * len(X_ext_cont_raw)
                        # Use True labels for Training, and True for External
                        # disease_labels = list(y_train_labels) + list(y_ext_labels)
                        # Use True labels for Training, and Pseudo-Labels for External
                        disease_labels = list(y_train_true) + list(y_pred_pseudo)
                        covars = pd.DataFrame({'batch': batch_labels, 'disease': disease_labels})
                        
                        # 4. Run ComBat on the RAW data
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            combat_res = neuroCombat(
                                dat=combined_data,
                                covars=covars,
                                batch_col='batch', 
                                # categorical_cols=['disease'],
                                ref_batch=0
                            )
                        
                        # Extract the harmonized RAW external data
                        harmonized_combined = combat_res['data'].T
                        X_ext_harmonized_cont_raw = harmonized_combined[len(X_train_cont_raw):] 
                        
                        # 5. Apply the TRAINING Z-Score to the Harmonized Data
                        # We start with the aligned dataset which safely retains the RAW clinical variables untouched
                        X_ext_harmonized = X_ext_aligned.copy() 
                        
                        for i, col in enumerate(valid_cont_features):
                            harm_raw_vals = X_ext_harmonized_cont_raw[:, i]
                            train_mean = numerical_mappings[col]['mean']
                            train_std = numerical_mappings[col]['std']
                            
                            if X_ext_harmonized[col].dtype != 'float64':
                                X_ext_harmonized[col] = X_ext_harmonized[col].astype(float)
                            # Standardize using training statistics
                            if train_std > 0:
                                X_ext_harmonized.loc[:, col] = (harm_raw_vals - train_mean) / train_std
                            else:
                                X_ext_harmonized.loc[:, col] = harm_raw_vals

                        # 6. Re-Predict using properly scaled harmonized data
                        y_proba_harm = model.predict_proba(X_ext_harmonized)[:, 1]
                        
                        # Apply the EXACT SAME Youden Threshold from training!
                        y_pred_harm = (y_proba_harm >= youden_threshold).astype(int)
                        
                        # Calculate Harmonized Metrics & CM
                        save_clinical_metrics(y_true, y_pred_harm, y_proba_harm, pipeline_name, dataset_name, dataset_ext_dir, suffix="_Harmonized")

                        auc_harm = roc_auc_score(y_true, y_proba_harm)
                        fpr_harm, tpr_harm, _ = roc_curve(y_true, y_proba_harm)
                        
                        # Bootstrap for Harmonized CI
                        boot_harm = []
                        for i in range(n_bootstraps):
                            indices = rng.choice(len(y_true), len(y_true), replace=True)
                            if len(np.unique(y_true.iloc[indices])) < 2:
                                continue
                            boot_harm.append(roc_auc_score(y_true.iloc[indices], y_proba_harm[indices]))
                        
                        l_h, u_h = np.percentile(boot_harm, [2.5, 97.5])

                        ax.plot(fpr_harm, tpr_harm, color=color, linestyle='-', linewidth=2,
                                label=f"{pipeline_name} + ComBat (AUC: {auc_harm:.2f} [{l_h:.2f}-{u_h:.2f}])")
                        
                        print(f"  Pipeline {pipeline_name} (ComBat Harmonized) -> AUC: {auc_harm:.4f}")

                        # Generate new Calibration and DCA plots specifically for the Harmonized data!
                        evaluate_clinical_utility(
                            y_true=y_true, 
                            y_proba=y_proba_harm, 
                            pipeline_name=f"{pipeline_name}_Harmonized", 
                            dataset_name=dataset_name, 
                            out_dir=dataset_ext_dir,
                            precalculated_threshold=youden_threshold
                        )
                        # --- UPDATE 4: SAVING ESTIMATES FOR N=1 FUTURE SAMPLES ---
                        # neuroCombat outputs an 'estimates' dict. Save this mapping for later!
                        deployment_mapping = {
                            'combat_estimates': combat_res['estimates'],
                            'training_mean_std': numerical_mappings,
                            'features': valid_cont_features
                        }
                        joblib.dump(deployment_mapping, dataset_ext_dir / f"{dataset_name}_harmonization_mapping.joblib")
                        print(f"  [Saved] Harmonization mapping for future N=1 samples from {dataset_name}.")

                    else:
                        print("  [Warning] No valid continuous features found for ComBat.")
                except Exception as e:
                    print(f"  [Error] ComBat harmonization failed: {e}")
            # ==========================================================

        ax.plot([0, 1], [0, 1], 'k--', label="Random Chance")
        ax.set_title(f"External Validation: {dataset_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        
        save_plot(fig, f"Ext_Val_{dataset_name}_Comparison", dataset_ext_dir)
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
        ext_df = pd.read_csv(ext_conf['path'], sep='\t')
        
        # Create dataset-specific subfolder for samples
        dataset_ext_dir = RESULTS_DIR / "external_validation" / dataset_name
        sample_dir = dataset_ext_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # BUG FIX: Indent the pipeline loop so it runs for EVERY dataset
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
                X_ext[col] = np.nan

        X_ext_aligned = X_ext[selected_features].copy()
        
        try:
            booster = model.get_booster()
            if hasattr(booster, 'feature_types') and booster.feature_types:
                for col, f_type in zip(selected_features, booster.feature_types):
                    if f_type == 'c':
                        X_ext_aligned[col] = X_ext_aligned[col].astype('category')
                    else:
                        X_ext_aligned[col] = pd.to_numeric(X_ext_aligned[col], errors='coerce')
                        # Standardize (Z-score) the external numeric feature
                        col_mean = X_ext_aligned[col].mean()
                        col_std = X_ext_aligned[col].std()
                        if col_std > 0:
                            X_ext_aligned[col] = (X_ext_aligned[col] - col_mean) / col_std
        except Exception:
            for col in selected_features:
                if X_ext_aligned[col].dtype == 'object' or X_ext_aligned[col].dtype.name == 'string':
                    X_ext_aligned[col] = X_ext_aligned[col].astype('category')
                else:
                    X_ext_aligned[col] = pd.to_numeric(X_ext_aligned[col], errors='coerce')

        X_ext_aligned = X_ext[selected_features]

        # 1. Extract the actual model
        try:
            model = pipeline.named_steps['model']
        except (AttributeError, KeyError):
            model = pipeline

        # --- EXTRACT BASE MODEL FOR SHAP ---
        if hasattr(model, 'calibrated_classifiers_'):
            model = model.calibrated_classifiers_[0].estimator
        else:
            model = model

        # 3. Initialize SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values_obj = explainer(X_ext_aligned)
        
        # 4. Handle base_values
        base_value = explainer.expected_value
        # For multi-class/multi-output, we take the target class index
        # Usually 1 for binary classification (the positive class)
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            base_value = base_value[1]

        print(f" - Processing {len(X_ext_aligned)} samples for pipeline: {pipeline_name} on {dataset_name}")

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
