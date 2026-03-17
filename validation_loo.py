import joblib
import pandas as pd
import numpy as np
import yaml
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from neuroCombat import neuroCombat
import warnings
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from itertools import cycle

from scripts.lib.paths import *
from scripts.lib.clinical_metrics import evaluate_clinical_utility

# --- HOTFIX FOR neuroCombat NUMPY 1.24+ COMPATIBILITY ---
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
# --------------------------------------------------------

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_bootstrapped_ci(y_true, y_pred, y_proba, n_bootstraps=1000, random_seed=42):
    """Calculates 95% Confidence Intervals for clinical metrics using bootstrapping."""
    np.random.seed(random_seed)
    metrics = {'Accuracy': [], 'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': [], 'F1': [], 'AUC': []}
    
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_proba_arr = np.array(y_proba)
    n_samples = len(y_true_arr)
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        y_t = y_true_arr[idx]
        y_p = y_pred_arr[idx]
        y_prob = y_proba_arr[idx]
        
        cm = confusion_matrix(y_t, y_p)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics['Accuracy'].append((tp + tn) / n_samples if n_samples > 0 else np.nan)
        metrics['Sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        metrics['Specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
        metrics['PPV'].append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
        metrics['NPV'].append(tn / (tn + fn) if (tn + fn) > 0 else np.nan)
        metrics['F1'].append((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan)
        
        if len(np.unique(y_t)) > 1:
            metrics['AUC'].append(roc_auc_score(y_t, y_prob))
        else:
            metrics['AUC'].append(np.nan)
            
    ci_results = {}
    for k, v in metrics.items():
        valid_vals = [x for x in v if not np.isnan(x)]
        if len(valid_vals) > 0:
            lower = np.percentile(valid_vals, 2.5)
            upper = np.percentile(valid_vals, 97.5)
            ci_results[k] = (lower, upper)
        else:
            ci_results[k] = (np.nan, np.nan)
    return ci_results

# ==========================================
# 1. LOAD CONFIG AND SHARED CONTEXT
# ==========================================
config = yaml.safe_load(open("./scripts/config.yml"))

# Identify which model should get the heavy SHAP and PDF generation
shap_model_name = config.get('SHAP_ANALYSIS_MODEL', 'Combined_Model')

print("Loading shared training context...")
context = joblib.load(RESULTS_DIR / "training_context.joblib")

mappings = context['mappings']
categorical_mappings = mappings[0]
numerical_mappings = mappings[1]

X_train = context['X_train']
y_train = context['y_train']
modality_features = context['modality_features']
y_train_true = y_train.values if isinstance(y_train, pd.Series) else y_train.iloc[:, 0].values

# ==========================================
# 2. ITERATE OVER DATASETS
# ==========================================
external_configs = config.get('EXTERNAL_VALIDATION', [])

if not external_configs:
    print("No external validation sets found in config.")
    exit()

for ext_conf in external_configs:
    dataset_name = ext_conf['name']
    
    print(f"\n{'='*60}")
    print(f" STARTING LOO ANALYSIS: {dataset_name} ")
    print(f"{'='*60}")
    
    ext_df = pd.read_csv(ext_conf['path'], sep='\t')
    clinical_id_col = config.get('CLINICAL_ID', 'ID')
    y_true_all = ext_df[ext_conf['label_col']]
    
    # Dictionary to hold data for the final comparison ROC plot
    roc_comparison_data = {}

    # ==========================================
    # 3. ITERATE OVER ALL PIPELINES (MODELS)
    # ==========================================
    pipelines = config.get('PIPELINES', [])
    for pipe_conf in pipelines:
        pipeline_name = pipe_conf['name']
        print(f"\n>>> Evaluating Pipeline: {pipeline_name} <<<")
        
        run_shap = (pipeline_name == shap_model_name)
        if run_shap:
            print(f" -> [SHAP ANALYSIS ENABLED] PDF Reports will be generated for {pipeline_name}.")

        bundle_path = RESULTS_DIR / f"saved_models/{pipeline_name}_bundle.joblib"
        if not bundle_path.exists():
            print(f" -> Skipping {pipeline_name}: Bundle not found at {bundle_path}")
            continue
            
        bundle = joblib.load(bundle_path)
        model = bundle['model']
        features = bundle['features']
        
        for col in features:
            if col not in ext_df.columns:
                ext_df[col] = np.nan

        X_train_aligned = X_train[features].copy()
        y_train_proba = model.predict_proba(X_train_aligned)[:, 1]
        
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, y_train_proba)
        youden_index = tpr_train - fpr_train
        youden_threshold = thresholds_train[np.argmax(youden_index)]

        harmonization_rule = pipe_conf.get('harmonization', [])
        harmonized_feature_set = set()
        if isinstance(harmonization_rule, list) and modality_features:
            for mod in harmonization_rule:
                if mod.lower() in modality_features:
                    harmonized_feature_set.update(modality_features[mod.lower()])

        X_train_raw = X_train.copy()
        valid_cont_features = []

        for col in features:
            if col in numerical_mappings and col in X_train_raw.columns and col in harmonized_feature_set:
                train_mean = numerical_mappings[col]['mean']
                train_std = numerical_mappings[col]['std']
                X_train_raw[col] = (X_train_raw[col] * train_std) + train_mean
                
                if X_train_raw[col].var() > 1e-6:
                    valid_cont_features.append(col)

        X_train_cont_raw = X_train_raw[valid_cont_features].copy() if valid_cont_features else None

        output_dir = RESULTS_DIR / f"loo_analysis/{dataset_name}/{pipeline_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        reports_dir = output_dir / "reports"
        if run_shap:
            reports_dir.mkdir(parents=True, exist_ok=True)

        all_predictions = []

        explainer_kernel = None
        background_data = None
        final_feature_names = features
        base_value = 0.0
        
        if run_shap:
            print(f" -> Initializing KernelExplainer for {pipeline_name}...")
            X_train_for_shap = X_train[features].copy()
            
            if hasattr(model, 'steps'):
                for name, step in model.steps[:-1]:
                    X_train_for_shap = step.transform(X_train_for_shap)
                    
                if 'selector' in model.named_steps:
                    try:
                        final_feature_names = model.named_steps['selector'].get_feature_names_out()
                    except AttributeError:
                        mask = model.named_steps['selector'].get_support()
                        final_feature_names = np.array(features)[mask].tolist()

            background_data = shap.kmeans(X_train_for_shap, 50)

            def calibrated_predict(data):
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data, columns=final_feature_names)
                final_estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
                return final_estimator.predict_proba(data)[:, 1]

            explainer_kernel = shap.KernelExplainer(calibrated_predict, background_data)
            base_value_raw = explainer_kernel.expected_value
            base_value = float(base_value_raw[0] if isinstance(base_value_raw, (list, np.ndarray)) else base_value_raw)

        # ==========================================
        # 4. LEAVE-ONE-OUT (LOO) PATIENT LOOP
        # ==========================================
        print(f" -> Running LOO across {len(ext_df)} patients...")

        for i in range(len(ext_df)):
            patient_id = ext_df.iloc[i][clinical_id_col] if clinical_id_col in ext_df.columns else ext_df.index[i]
            
            X_patient_raw = ext_df.iloc[[i]][features].copy()
            X_bg_raw = ext_df.drop(index=i)[features].copy()
            
            combat_estimates = None
            
            if valid_cont_features and X_train_cont_raw is not None:
                X_ext_cont_raw = X_bg_raw[valid_cont_features].copy()
                for col in valid_cont_features:
                    X_ext_cont_raw[col] = pd.to_numeric(X_ext_cont_raw[col], errors='coerce')
                    
                combined_data = pd.concat([X_train_cont_raw, X_ext_cont_raw], axis=0)
                combined_data = combined_data.fillna(combined_data.mean(axis=0)) 
                combined_data = combined_data.T 
                
                np.random.seed(42)
                combined_data = combined_data + np.random.normal(0, 1e-6, combined_data.shape)
                
                batch_labels = [0] * len(X_train_cont_raw) + [1] * len(X_ext_cont_raw)
                covars = pd.DataFrame({'batch': batch_labels})
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        combat_res = neuroCombat(dat=combined_data, covars=covars, batch_col='batch', ref_batch=0)
                        combat_estimates = combat_res['estimates']
                    except Exception as e:
                        combat_estimates = None
            
            X_patient_aligned = X_patient_raw[features].copy()
            
            for col in features:
                raw_val = X_patient_aligned[col].iloc[0]
                
                if col in categorical_mappings:
                    X_patient_aligned[col] = X_patient_aligned[col].astype('category').cat.set_categories(categorical_mappings[col])
                    
                elif col in numerical_mappings and col in harmonized_feature_set:
                    if pd.isna(raw_val):
                        X_patient_aligned[col] = np.nan
                        continue
                    
                    val = float(raw_val)
                    tm = numerical_mappings[col]['mean']
                    ts = numerical_mappings[col]['std']
                    
                    if col in valid_cont_features and combat_estimates is not None:
                        idx = valid_cont_features.index(col)
                        grand_mean = combat_estimates['stand.mean'][idx][0]
                        var_pooled = combat_estimates['var.pooled'][idx][0]  
                        gamma_star = combat_estimates['gamma.star'][1][idx] 
                        delta_star = combat_estimates['delta.star'][1][idx]
                        
                        z = (val - grand_mean) / np.sqrt(var_pooled)
                        z_adj = (z - gamma_star) / np.sqrt(delta_star)
                        harmonized_raw_val = (z_adj * np.sqrt(var_pooled)) + grand_mean
                        
                        if ts > 0:
                            X_patient_aligned.loc[X_patient_aligned.index[0], col] = (harmonized_raw_val - tm) / ts
                        else:
                            X_patient_aligned.loc[X_patient_aligned.index[0], col] = harmonized_raw_val
                    else:
                        if ts > 0:
                            X_patient_aligned.loc[X_patient_aligned.index[0], col] = (val - tm) / ts
                        else:
                            X_patient_aligned.loc[X_patient_aligned.index[0], col] = val
                            
            for col in features:
                if col not in categorical_mappings:
                    X_patient_aligned[col] = X_patient_aligned[col].astype(float)
                    
            proba = model.predict_proba(X_patient_aligned)[0, 1]
            pred = int(proba >= youden_threshold)
            true_label = y_true_all.iloc[i]
            
            all_predictions.append({
                'Patient_ID': patient_id,
                'True_Label': true_label,
                'Predicted_Label': pred,
                'Probability': proba
            })
            
            if run_shap and explainer_kernel is not None:
                X_for_shap = X_patient_aligned.copy()
                
                if hasattr(model, 'steps'):
                    for step_name, step in model.steps[:-1]:
                        X_for_shap = step.transform(X_for_shap)

                if not isinstance(X_for_shap, pd.DataFrame):
                    X_for_shap = pd.DataFrame(X_for_shap, columns=final_feature_names, index=X_patient_aligned.index)

                sample_shaps = explainer_kernel.shap_values(X_for_shap, nsamples=100)
                if isinstance(sample_shaps, list):
                    sample_shaps = sample_shaps[0] 
                sample_shaps_1d = sample_shaps[0] if len(sample_shaps.shape) > 1 else sample_shaps

                # GRAB THE FINAL STANDARDIZED/HARMONIZED VALUES
                sample_features_vals_scaled = X_for_shap.iloc[0].values
                
                final_prob_calibrated = base_value + np.sum(sample_shaps_1d)
                new_feature_labels = [None] * len(final_feature_names)
                feature_impact_pct = {}  
                
                for idx, name in enumerate(final_feature_names):
                    shap_val = float(sample_shaps_1d[idx])
                    
                    # USE SCALED VALUES FOR LABELS
                    feat_val = sample_features_vals_scaled[idx]
                    
                    prob_change_percent = shap_val * 100
                    feature_impact_pct[name] = prob_change_percent

                    clean_name = name
                    if isinstance(feat_val, (float, np.floating)) and not pd.isna(feat_val):
                        new_label = f"{clean_name} = {feat_val:.2f} ({prob_change_percent:+.1f}%)"
                    else:
                        new_label = f"{clean_name} = {feat_val} ({prob_change_percent:+.1f}%)"
                    new_feature_labels[idx] = new_label
                    
                plt.figure(figsize=(10, 6))
                shap.decision_plot(
                    base_value,
                    sample_shaps_1d, 
                    feature_names=new_feature_labels,
                    show=False,
                    link='identity', 
                    auto_size_plot=False,
                    # xlim=(0,1),
                )

                # --- ADD THE TRAINING THRESHOLD LINE ---
                # This draws a vertical dashed line at the Youden Index calculated from training
                plt.axvline(x=youden_threshold, color='red', linestyle='--', label=f'Decision Threshold ({youden_threshold:.2f})')

                plt.title(f"LOO Decision Path | Patient: {patient_id}\nCalibrated Prob: {proba:.3f} | Pred: {pred} | True: {true_label}", fontsize=12, pad=20)
                plt.tight_layout()
                
                plot_image_path = reports_dir / f"Decision_LOO_Patient_{patient_id}.png"
                plt.savefig(plot_image_path)
                plt.close()

                if config['ICE_PLOTS']:
                    ice_narratives = {}
                    # ==========================================
                    # 5. MULTI-FEATURE ICE PLOTS (Top 5 Features)
                    # ==========================================
                    # Create the specific subfolder for ICE plots
                    ice_dir = reports_dir / "ice"
                    ice_dir.mkdir(parents=True, exist_ok=True)

                    # Identify the indices of the top 5 features by absolute SHAP value
                    # sample_shaps_1d contains the SHAP values for the current patient
                    top_5_indices = np.argsort(np.abs(sample_shaps_1d))[-config['ICE_FEATS_N']:][::-1]

                    for idx in top_5_indices:
                        feat_name = final_feature_names[idx]
                        
                        # 1. Define the range based on training data distribution (X_train_for_shap)
                        # This ensures the 'What-If' scenarios stay within realistic bounds
                        f_min = X_train_for_shap[feat_name].min()
                        f_max = X_train_for_shap[feat_name].max()
                        feat_values = np.linspace(f_min, f_max, 100)
                        
                        # 2. Create synthetic dataset: 100 copies of the current patient
                        temp_ice_df = pd.concat([X_for_shap] * 100, ignore_index=True)
                        temp_ice_df[feat_name] = feat_values
                        
                        # 3. Predict probabilities for the variations
                        # We use the final estimator because X_for_shap is already pre-processed
                        final_estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
                        ice_probas = final_estimator.predict_proba(temp_ice_df)[:, 1]
                        
                        # 4. Plotting
                        plt.figure(figsize=(8, 5))
                        plt.plot(feat_values, ice_probas, color='#2c7bb6', lw=2.5, label='Risk Path', zorder=2)
                        
                        # Mark the actual current clinical state of the patient
                        current_feat_val = X_for_shap[feat_name].iloc[0]
                        plt.scatter([current_feat_val], [proba], color='red', s=80, edgecolors='black', 
                                    label='Current State', zorder=5)
                        
                        # Add the Training Threshold as a horizontal reference
                        plt.axhline(y=youden_threshold, color='#d7191c', linestyle='--', alpha=0.7, 
                                    label=f'Decision Threshold ({youden_threshold:.2f})')
                        
                        plt.title(f"ICE What-If: {feat_name}\nPatient: {patient_id}", fontsize=11)
                        plt.xlabel(f"Feature Value (Standardized/Scaled)")
                        plt.ylabel("Probability of Positive Label")
                        plt.ylim(-0.05, 1.05)
                        plt.grid(True, alpha=0.2)
                        plt.legend(loc='best', fontsize=9)
                        
                        # Save in the dedicated 'ice' subfolder
                        clean_feat_name = str(feat_name).replace("/", "_").replace("\\", "_")
                        plt.savefig(ice_dir / f"ICE_{patient_id}_{clean_feat_name}.png", dpi=150)
                        plt.close()

                        # 5. Find where the curve crosses the threshold
                        crossing_indices = np.where(np.diff(np.sign(ice_probas - youden_threshold)))[0]
                        if len(crossing_indices) > 0:
                            c_idx = crossing_indices[0]
                            x1, x2 = feat_values[c_idx], feat_values[c_idx+1]
                            y1, y2 = ice_probas[c_idx], ice_probas[c_idx+1]
                            
                            tipping_point_scaled = x1 + ((youden_threshold - y1) * (x2 - x1)) / (y2 - y1)
                            current_feat_scaled = X_for_shap[feat_name].iloc[0]
                            
                            # FIX: Grab the guaranteed raw value from before ANY pipeline processing
                            current_feat_raw = X_patient_raw[feat_name].iloc[0]
                            
                            # Determine if the feature in SHAP space is actually scaled
                            # If they differ significantly, the pipeline scaled it, so we inverse the tipping point.
                            if abs(current_feat_scaled - current_feat_raw) > 1e-4 and feat_name in numerical_mappings:
                                tm = numerical_mappings[feat_name]['mean']
                                ts = numerical_mappings[feat_name]['std']
                                tipping_point_raw = (tipping_point_scaled * ts) + tm
                            else:
                                # Feature was NEVER scaled (like Age). Use identical space.
                                tipping_point_raw = tipping_point_scaled
                                
                            delta = tipping_point_raw - current_feat_raw
                            
                            if current_feat_raw != 0:
                                pct_change = (abs(delta) / abs(current_feat_raw)) * 100
                            else:
                                pct_change = 0.0
                            
                            ice_narratives[feat_name] = {
                                'current_raw': current_feat_raw,
                                'tipping_point_raw': tipping_point_raw,
                                'delta': delta,
                                'pct_change': pct_change,
                                'direction': 'A decrease' if delta < 0 else 'An increase'
                            }

                try:
                    from scripts.lib.clinical_report_generator import generate_biological_narrative_data, create_patient_pdf_report
                    
                    # USE SCALED VALUES FOR THE PDF NARRATIVE
                    sample_features_series = pd.Series(sample_features_vals_scaled, index=final_feature_names)
                    base_probability = float(base_value)
                    
                    report_data = generate_biological_narrative_data(
                        patient_features=sample_features_series, 
                        patient_shap=sample_shaps_1d, 
                        X_train=X_train,  
                        patient_proba=proba,                  
                        base_proba=base_probability,          
                        feature_impacts=feature_impact_pct,
                        pred=pred,                  
                        threshold=youden_threshold, 
                        top_k=8,
                        ice_narratives=ice_narratives           
                    )
                    
                    create_patient_pdf_report(
                        patient_id=patient_id,
                        report_data=report_data,  
                        plot_image_path=str(plot_image_path),
                        output_dir=str(reports_dir)
                    )
                except Exception as e:
                    pass

        # ==========================================
        # 5. PIPELINE METRICS & PLOTS (MATCHING TRAINING.PY STYLE)
        # ==========================================
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_csv(output_dir / f"{pipeline_name}_LOO_Predictions.csv", index=False)
        
        y_true = pred_df['True_Label']
        y_pred = pred_df['Predicted_Label']
        y_proba = pred_df['Probability']
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate Base Metrics
        roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
        acc_val = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        sens_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1_val = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # Bootstrapped 95% Confidence Intervals
        ci = get_bootstrapped_ci(y_true, y_pred, y_proba)
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_comparison_data[pipeline_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'auc_ci': ci['AUC']}

        # Save Metrics with CI ranges
        metrics_dict = {
            'Dataset': dataset_name,
            'Pipeline': pipeline_name,
            'Accuracy': acc_val,
            'Accuracy_95CI': f"[{ci['Accuracy'][0]:.3f}-{ci['Accuracy'][1]:.3f}]",
            'Sensitivity (TPR)': sens_val,
            'Sensitivity_95CI': f"[{ci['Sensitivity'][0]:.3f}-{ci['Sensitivity'][1]:.3f}]",
            'Specificity (TNR)': spec_val,
            'Specificity_95CI': f"[{ci['Specificity'][0]:.3f}-{ci['Specificity'][1]:.3f}]",
        }
        pd.DataFrame([metrics_dict]).to_csv(output_dir / f"{pipeline_name}_LOO_Metrics.csv", index=False)

        # Individual CM Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        ax.set_title(f"LOO Confusion Matrix: {pipeline_name}\nDataset: {dataset_name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        plt.tight_layout()
        plt.savefig(output_dir / f"Confusion_Matrix_LOO_{pipeline_name}.png")
        plt.close(fig)

        # Individual ROC Curve
        if len(np.unique(y_true)) > 1:
            fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
            
            auc_label = f'{pipeline_name} LOO (AUC = {roc_auc:.2f} [{ci["AUC"][0]:.2f}-{ci["AUC"][1]:.2f}])'
            ax_roc.plot(fpr, tpr, color='b', label=auc_label, lw=2, alpha=.8)
            ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
            
            ax_roc.set_xlim([-0.05, 1.05])
            ax_roc.set_ylim([-0.05, 1.05])
            ax_roc.set_xlabel('False Positive Rate', fontsize=12)
            ax_roc.set_ylabel('True Positive Rate', fontsize=12)
            ax_roc.set_title(f"LOO ROC Curve - {dataset_name}", fontsize=14)
            ax_roc.legend(loc="lower right", fontsize=10)
            ax_roc.grid(alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"ROC_Curve_LOO_{pipeline_name}.png", dpi=300)
            plt.savefig(output_dir / f"ROC_Curve_LOO_{pipeline_name}.svg")
            plt.close(fig_roc)

        # DCA/Calibration Plots
        try:
            evaluate_clinical_utility(
                y_true=y_true, 
                y_proba=y_proba, 
                pipeline_name=f"{pipeline_name}_LOO", 
                dataset_name=dataset_name, 
                out_dir=output_dir,
                precalculated_threshold=youden_threshold
            )
        except Exception as e:
            pass

    # ==========================================
    # 6. DATASET-LEVEL COMPARISON PLOTS
    # ==========================================
    if roc_comparison_data:
        comp_dir = RESULTS_DIR / f"loo_analysis/{dataset_name}/Comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        # Match training.py size and styling
        fig_comp, ax_comp = plt.subplots(figsize=(10, 10))
        
        # Define colors identically to training.py
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        
        for pipe_name, data in roc_comparison_data.items():
            color = next(colors)
            auc_str = f"{pipe_name} (AUC = {data['auc']:.2f} [{data['auc_ci'][0]:.2f}-{data['auc_ci'][1]:.2f}])"
            ax_comp.plot(data['fpr'], data['tpr'], color=color, lw=2.5, label=auc_str)

        # Standard Plot Formatting from training.py
        ax_comp.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.5, label='Chance')
        ax_comp.set_xlim([-0.02, 1.02])
        ax_comp.set_ylim([-0.02, 1.02])
        ax_comp.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax_comp.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax_comp.set_title(f'Pipeline Comparison: LOO ROC Curves - {dataset_name}', fontsize=14)
        ax_comp.legend(loc="lower right", frameon=True, fontsize=10)
        ax_comp.grid(alpha=0.2)
        plt.tight_layout()

        # Save standard and vector images
        plt.savefig(comp_dir / f"Pipeline_Comparison_ROC_LOO_{dataset_name}.png", dpi=300)
        plt.savefig(comp_dir / f"Pipeline_Comparison_ROC_LOO_{dataset_name}.svg")
        plt.close(fig_comp)
        
        print(f" -> Generated Comparison plots in {comp_dir}")

print("\nAll LOO analyses complete!")