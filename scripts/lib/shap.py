import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path


from scripts.lib.savePlots import save_plot
# CONFIG FILE
import yaml
try:
    config = yaml.safe_load(open("./scripts/config.yml"))
except FileNotFoundError:
    print("Error: config.yml not found. Using default values.")
    exit()
    
# to define output folders
from scripts.lib.paths import *

def sigmoid(x):
    """Converts logit (log-odds) to probability."""
    return 1 / (1 + np.exp(-x))

def analyze_shap_impact(X_display, shap_values, feature_importances, shap_feature_columns, analysis_type, class_names, pipeline_name, out_dir, mappings):
    """
    Analyzes SHAP values to find optimal cutoffs for numerical features and plots
    the aggregated impact on the model's prediction.
    
    Args:
        X_display (pd.DataFrame): The original dataframe with feature values.
        shap_values (np.array): The raw SHAP values from the explainer.
        feature_importances (pd.DataFrame): DataFrame with 'feature' and 'importance' columns.
        shap_feature_columns (list): Column names corresponding to the shap_values array.
        analysis_type (str): 'binary', 'multiclass', or 'survival'.
        class_names (list): List of class names for the plot legend.
        categorical_mappings (dict): Dictionary mapping categorical features to their encodings.
    """
    print("--- Step 6: Analyzing SHAP Impacts for Important Features ---")

    # Get a list of features with non-zero importance
    important_features = feature_importances[feature_importances['importance'] > 0]['feature'].tolist()

    if not important_features:
        print(" - No features with importance > 0 found. Skipping plot.")
        return

    # Create a list of SHAP DataFrames, one for each class
    if analysis_type == 'multiclass':
        shap_dfs = [pd.DataFrame(shap_values[:,:,i], columns=shap_feature_columns, index=X_display.index) for i in range(len(class_names))]
    else: # binary or survival
        shap_dfs = [pd.DataFrame(shap_values, columns=shap_feature_columns, index=X_display.index)]
        # Assign a default class name if not multiclass
        if analysis_type == 'binary':
            class_names = [config.get('POSITIVE_CLASS', 'Positive')]
        else: # survival
            class_names = ['Risk']

    plot_data = []
    # Loop through each important feature to calculate its impact
    for feature in important_features:
        # Safety checks to ensure the feature exists in both dataframes
        if feature not in X_display.columns or feature not in shap_feature_columns:
            print(f" - Feature '{feature}' not found in both X_display and SHAP columns. Skipping.")
            continue
            
        # Process each class's SHAP values for the given feature
        for i, class_name in enumerate(class_names):
            shap_df = shap_dfs[i]
            
            # --- THIS IS THE CORE LOGIC FOR NUMERICAL FEATURES ---
            if feature in mappings[1].keys():
                try:
                    # Prepare data: feature values (X) and their SHAP values (y)
                    feature_series = pd.to_numeric(X_display[feature], errors='coerce')
                    temp_df = pd.DataFrame({
                        'feature_val': feature_series,
                        'shap_val': shap_df[feature]
                    }).dropna()

                    # Ensure there's more than one unique value to split on
                    if temp_df['feature_val'].nunique() < 2:
                        continue

                    # Train a "decision stump" to find the best single split point
                    # The tree is learning to predict the SHAP value based on the feature's value
                    tree_stump = DecisionTreeRegressor(max_depth=1, random_state=config.get('RANDOM_STATE', 42))
                    tree_stump.fit(temp_df[['feature_val']], temp_df['shap_val'])
                    
                    # The threshold is the optimal cutoff found by the tree
                    cutoff = tree_stump.tree_.threshold[0]

                    # Calculate the average SHAP impact for data points above and below the cutoff
                    mean_shap_below = temp_df['shap_val'][temp_df['feature_val'] < cutoff].mean()
                    mean_shap_above = temp_df['shap_val'][temp_df['feature_val'] >= cutoff].mean()

                    # Append the results for plotting
                    plot_data.append({
                        'Feature': feature, 
                        'Feature_Group': f"{feature} < {round(cutoff, 2)} |-> {class_name}", 
                        'Avg_SHAP': mean_shap_below
                    })
                    plot_data.append({
                        'Feature': feature, 
                        'Feature_Group': f"{feature} >= {round(cutoff, 2)} |-> {class_name}", 
                        'Avg_SHAP': mean_shap_above
                    })
                except Exception as e:
                    print(f" - Could not process numerical feature '{feature}': {e}")
            
            # --- LOGIC FOR CATEGORICAL AND BINARY FEATURES ---
            else:
                # Group by each category and find the mean SHAP value
                for category in X_display[feature].unique():
                    if pd.isna(category):
                        continue
                    
                    mask = (X_display[feature] == category)
                    if mask.sum() > 0:
                        mean_shap = shap_df.loc[mask, feature].mean()
                        # Use a more descriptive label for binary (0/1) features
                        if X_display[feature].nunique() <= 2 and str(category) in ['0', '1', '0.0', '1.0']:
                            label = f"{feature} (Present)" if category in [1, 1.0] else f"{feature} (Absent)"
                        else:
                            label = f"{feature}: {category}"
                        
                        plot_data.append({
                            'Feature': feature, 
                            'Feature_Group': f"{label} |-> {class_name}", 
                            'Avg_SHAP': mean_shap
                        })
    
    # --- PLOTTING LOGIC ---
    if not plot_data:
        print(" - No data available for SHAP impact plot.")
        return
    
    # Convert to DataFrame and remove entries with no impact
    summary_df = pd.DataFrame(plot_data).dropna(subset=['Avg_SHAP'])
    summary_df = summary_df[summary_df['Avg_SHAP'] != 0]

    if summary_df.empty:
        print(" - SHAP impact plot is empty after processing (all impacts were zero or NaN).")
        return
    
    # Sort values for a cleaner plot
    summary_df_sorted = summary_df.sort_values(by='Avg_SHAP', ascending=False)
    
    print("--- Generating Aggregated SHAP Impact Plot ---")
    fig, ax = plt.subplots(figsize=(12, max(8, len(summary_df_sorted) * 0.35)))
    
    # Use red for positive impact (pushes prediction higher) and blue for negative
    colors = ['#ff4d4d' if x > 0 else '#4d4dff' for x in summary_df_sorted['Avg_SHAP']]
    
    sns.barplot(x='Avg_SHAP', y='Feature_Group', data=summary_df_sorted, palette=colors, ax=ax)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Aggregated SHAP Impact of Important Features', fontsize=16, pad=20)
    ax.set_xlabel('Average SHAP Value (Impact on Prediction)', fontsize=12)
    ax.set_ylabel('Feature Group', fontsize=12)
    plt.tight_layout()
    
    save_plot(fig, f"{pipeline_name}_aggregated_shap_impact",out_dir)
    print("--- SHAP Impact Plot Saved ---")

def run_decision_plots(all_results, X_display):
    """
    Generates SHAP decision plots with percentage probability contributions 
    for every individual sample.
    """
    # Create output directory
    dataset_sample_dir = RESULTS_DIR / "decision_plots"
    dataset_sample_dir.mkdir(parents=True, exist_ok=True)

    for pipeline_name, bundle in all_results.items():
        pipeline = bundle['pipeline']
        selected_features = bundle['selected_features']
        
        # 1. Extract the actual model
        try:
            model = pipeline.named_steps['model']
        except (AttributeError, KeyError):
            model = pipeline

        # 2. Align features
        X_ext = X_display.copy()
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
            save_plot(plt.gcf(), plot_filename, dataset_sample_dir)
            plt.close()

    print("--- Individual Decision Plots with % Contributions Completed ---")

def explain_with_shap(pipelines, X_raw, analysis_type, class_names, out_dir, mappings):
    """
    Generates and saves SHAP summary plots for a scikit-learn pipeline.
    Returns SHAP values (array), feature importances (df), and selected feature names (list).
    """
    print("--- Explaining model with SHAP values ---")
    shap_folder = out_dir
    shap_individual = out_dir / "individualFeatures"

    for pipeline_name in pipelines.keys():
        pipeline_dict = pipelines[pipeline_name]
        selected_features = pipeline_dict['selected_features']
        X_transformed = X_raw[selected_features]
        X_display = X_raw[selected_features]
        
        pipeline = pipeline_dict['pipeline']

        # 1. Extract the final trained model (e.g., XGBClassifier)
        try:
            model = pipeline.named_steps['model']
        except Exception as e:
            print(f"Warning: Could not find 'model' step in pipeline: {e}. Assuming pipeline *is* the model.")
            model = pipeline # Fallback if no pipeline was used

        # 2. Transform the data
        X_display_transformed = X_display
        selected_feature_names = list(X_transformed.columns) # Default to all columns

        if 'selector' in getattr(pipeline, 'named_steps', {}):
            print(" - Applying feature selector from pipeline...")
            selector = pipeline.named_steps['selector']
            X_transformed = selector.transform(X_display_transformed)
            X_display_transformed = selector.transform(X_display)
            selected_feature_names = selector.selected_features_
            
            # Ensure they are DataFrames with correct columns for SHAP
            if not isinstance(X_transformed, pd.DataFrame):
                X_transformed = pd.DataFrame(X_transformed, columns=selected_feature_names, index=X_display_transformed.index)
            if not isinstance(X_display_transformed, pd.DataFrame):
                X_display_transformed = pd.DataFrame(X_display_transformed, columns=selected_feature_names, index=X_display.index)
        else:
            print(" - No 'selector' step found. Using original features.")
        
        print(f" - Explaining model on {len(selected_feature_names)} features.")

        # 3. Create SHAP Explainer
        # Pass background data (X_transformed) to the explainer
        # Note: Using X_transformed as background. Could also use shap.maskers.Independent
        explainer = shap.TreeExplainer(model)
        
        # Get SHAP values for the (potentially different) display data
        shap_values_obj = explainer(X_display_transformed)
        
        # Calculate feature importances from SHAP values
        shap_values_arr = shap_values_obj.values
        if analysis_type == 'multiclass':
            # shap_values_arr shape: (n_samples, n_features, n_classes)
            # We average the absolute SHAP values across all samples (axis 0)
            # then sum the importance across all classes (axis 1)
            mean_abs_shap = np.abs(shap_values_arr).mean(axis=0).sum(axis=1)
        else:
            # shap_values_arr shape: (n_samples, n_features)
            mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)

        # Use the *transformed* columns
        feature_importance_df = pd.DataFrame({'feature': X_display_transformed.columns, 'importance': mean_abs_shap})
        feature_importance_df = feature_importance_df[feature_importance_df['importance'] > 0]
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(config.get('N_TOP_FEATURES', 20))

        print("Generating SHAP summary plots...")
        
        plot_args = {
            'max_display': len(feature_importance_df),
            'show': False
        }
        if analysis_type == 'multiclass':
            plot_args['class_names'] = class_names

        # Bar plot
        # Use X_display_transformed for plotting
        shap.summary_plot(shap_values_obj, X_display_transformed, plot_type="bar", **plot_args)
        plt.title('SHAP Feature Importance (Bar)')
        save_plot(plt.gcf(), f"shap_{pipeline_name}_shap_summary_bar",shap_folder)
        
        # Beeswarm plot
        # Use X_display_transformed for plotting
        shap.summary_plot(shap_values_obj, X_display_transformed, **plot_args)
        plt.title('SHAP Feature Summary (Beeswarm)')
        save_plot(plt.gcf(), f"shap_{pipeline_name}_shap_summary_beeswarm",shap_folder)

        # Individual feature plots
        print("Generating individual SHAP feature plots...")
        for feature in feature_importance_df['feature']:
            if feature in mappings[0].keys():
                # Categorical feature logic based on user's example
                shap_values_feat = shap_values_obj[:, feature]
                feat_values = shap_values_feat.data
                if pd.api.types.is_categorical_dtype(feat_values):
                    feat_values = feat_values.astype(str)
                
                if analysis_type == 'multiclass':
                    for i, class_name in enumerate(class_names):
                        shap_values_feat_class = shap_values_feat[:, i]
                        
                        not_nan_mask = ~pd.isnull(shap_values_feat_class.data)
                        shap_values_feat_class.values = shap_values_feat_class.values[not_nan_mask]
                        shap_values_feat_class.data = shap_values_feat_class.data[not_nan_mask]
                        shap_values_feat_class.base_values = shap_values_feat_class.base_values[not_nan_mask]

                        shap.plots.scatter(shap_values_feat_class, show=False)
                        plt.title(f'SHAP Beeswarm for {feature} - Class {class_name}')
                        save_plot(plt.gcf(), f"{pipeline_name}_shap_beeswarm_{feature.replace('/', '_')}_class_{class_name}", shap_individual)
                else: # Binary or survival
                    shap_obj = shap_values_obj[:, feature]
                    not_nan_mask = ~pd.isnull(shap_obj.data)
                    shap_obj.values = shap_obj.values[not_nan_mask]
                    shap_obj.data = shap_obj.data[not_nan_mask]
                    shap_obj.base_values = shap_obj.base_values[not_nan_mask]
                    shap.plots.scatter(shap_obj, show=False)
                    plt.title(f'SHAP Scatter for {feature}')
                    save_plot(plt.gcf(), f"{pipeline_name}_shap_scatter_{feature.replace('/', '_')}", shap_individual)
            else: # Numerical or binary mutation feature
                shap_values_feat = shap_values_obj[:, feature]
                if analysis_type == 'multiclass':
                    for i, class_name in enumerate(class_names):
                        shap_values_feat_class = shap_values_feat[:, i]
                        not_nan_mask = ~pd.isnull(shap_values_feat_class.data)
                        shap_values_feat_class.values = shap_values_feat_class.values[not_nan_mask]
                        if feature not in mappings[1].keys():
                            shap_values_feat_class.data = np.array(shap_values_feat_class.data[not_nan_mask],dtype=str)
                        else:
                            shap_values_feat_class.data = shap_values_feat_class.data[not_nan_mask]
                        shap_values_feat_class.base_values = shap_values_feat_class.base_values[not_nan_mask]
                        shap.plots.scatter(shap_values_feat_class, show=False)
                        plt.title(f'SHAP Beeswarm for {feature} - Class {class_name}')
                        save_plot(plt.gcf(), f"{pipeline_name}_shap_beeswarm_{feature.replace('/', '_')}_class_{class_name}", shap_individual)
                else:
                    shap_obj = shap_values_feat
                    not_nan_mask = ~pd.isnull(shap_obj.data)
                    shap_obj.values = shap_obj.values[not_nan_mask]
                    if feature not in mappings[1].keys():
                        shap_obj.data = np.array(shap_obj.data[not_nan_mask],dtype=str)
                    else:
                        shap_obj.data = shap_obj.data[not_nan_mask]
                    # shap_obj.data = shap_obj.data[not_nan_mask]
                    shap_obj.base_values = shap_obj.base_values[not_nan_mask]
                    shap.plots.scatter(shap_obj, show=False)
                    plt.title(f'SHAP Scatter for {feature}')
                    save_plot(plt.gcf(), f"{pipeline_name}_shap_scatter_{feature.replace('/', '_')}", shap_individual)
                    # shap.plots.scatter(shap_values_obj[:, feature], show=False)
                    # plt.title(f'SHAP Scatter for {feature}')
                    # save_plot(plt.gcf(), f"shap_scatter_{feature.replace('/', '_')}")

        if config['RUN_DECISION_PLOTS']:
            run_decision_plots(pipelines, X_display)
        
        analyze_shap_impact(X_display, shap_values_arr, feature_importance_df, selected_feature_names, config['ANALYSIS_TYPE'], class_names, pipeline_name, out_dir, mappings)
    # Return the raw array, the importance DF, and the list of features
    return shap_values_arr, feature_importance_df, selected_feature_names
