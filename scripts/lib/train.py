import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import auc
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from itertools import cycle
import shap
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats

from scripts.lib.savePlots import save_plot
from scripts.lib.shap import explain_with_shap

import yaml
try:
    config = yaml.safe_load(open("./scripts/config.yml"))
except FileNotFoundError:
    print("Error: config.yml not found. Using default values.")
    exit()

from scripts.lib.paths import RESULTS_DIR
RESULTS_DIR = Path(RESULTS_DIR)

def select_stable_features(X, y, modalities_to_use, modality_features, best_params, num_classes, analysis_type, random_state, pipeline_name="default", outer_fold=None):
    """
    Performs stable feature selection using 10-fold CV for each modality,
    inspired by the logic in specificAnalysis.py.
    """
    print("--- Starting Stable Feature Selection (10-fold CV per modality) ---")
    
    N_SPLITS = 3 # 10


    all_stable_features = []
    
    exp_details = config.get('EXP_DETAILS', False)

    for modality in modalities_to_use:
        print(f"--- Selecting features for modality: {modality} ---")
        # modality_cols = _get_modal_cols(X, modality)
        if modality == 'ALL':
            modality_cols = modality_features
        else:
            modality_cols = modality_features[modality.lower()]
        if not modality_cols:
            print(f"  No columns found for modality {modality}. Skipping.")
            continue
            
        X_modality = X[modality_cols]
        
        stability_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        
        y_stratify = y
        if analysis_type == 'survival':
            y_stratify = y['OS']
        elif isinstance(y, pd.DataFrame):
            y_stratify = y.iloc[:, 0]

        stable_feature_collector = []
        feature_importance_collector = [] # For exploration mode
        
        # Setup directory for this modality
        explore_dir = None
        results_exploration = RESULTS_DIR / 'exploration'
        folder_suffix = f"OuterFold_{outer_fold}" if outer_fold is not None else "Final_Model"
        explore_dir = results_exploration / pipeline_name / folder_suffix / modality
        explore_dir.mkdir(parents=True, exist_ok=True)

        X_modality = X_modality[X_modality.isna().sum(axis=1)<X_modality.shape[1]*0.90]
        y_stratify = y_stratify.loc[X_modality.index]
        for fold, (train_idx, val_idx) in enumerate(stability_cv.split(X_modality, y_stratify)):
            X_train_fold, X_val_fold = X_modality.iloc[train_idx], X_modality.iloc[val_idx]
            y_train_fold = y_stratify.iloc[train_idx]

            model_fold = None
            y_train_fit = y_train_fold
            
            model_params = {**best_params, 'random_state': random_state, 'enable_categorical': True}
            if analysis_type == 'binary':
                # Assuming y_train is a Series or array of 0s and 1s
                num_neg = (y_train_fit == 0).sum()
                num_pos = (y_train_fit == 1).sum()
                scale_pos_weight = num_neg / num_pos
            if analysis_type == 'survival':
                 model_fold = xgb.XGBRegressor(objective='survival:cox', eval_metric='cox-nloglik', **model_params)
                 y_train_fit = np.where(y_train_fold['OS'] == 1, y_train_fold['OS_time'], -y_train_fold['OS_time'])
            else:
                # ratio = (y_train_fit==1).sum() / (y_train_fit==0).sum()
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_fit)
                if analysis_type == 'binary':
                    model_fold = xgb.XGBClassifier(objective='multi:softprob' if analysis_type == 'multiclass' else 'binary:logistic',
                        eval_metric='mlogloss' if analysis_type == 'multiclass' else 'logloss',
                        num_class=num_classes if analysis_type == 'multiclass' and num_classes > 1 else None,
                        scale_pos_weight=scale_pos_weight,
                        **model_params
                    )
                else:
                    model_fold = xgb.XGBClassifier(objective='multi:softprob' if analysis_type == 'multiclass' else 'binary:logistic',
                        eval_metric='mlogloss' if analysis_type == 'multiclass' else 'logloss',
                        num_class=num_classes if analysis_type == 'multiclass' and num_classes > 1 else None,
                        **model_params
                    )

            if analysis_type == 'binary':
                model_fold.fit(X_train_fold, y_train_fit)
            else:
                model_fold.fit(X_train_fold, y_train_fit, sample_weight=sample_weights)

            explainer = shap.TreeExplainer(model_fold)
            shap_values = explainer.shap_values(X_val_fold)

            if num_classes > 2: # multiclass
                global_shap_importance = np.abs(np.array(shap_values)).mean(axis=0).sum(axis=1)
            else: # binary or survival
                global_shap_importance = np.abs(shap_values).mean(axis=0)
            
            feature_importance_df = pd.DataFrame({
                'feature': X_val_fold.columns,
                'importance': global_shap_importance
            })
            
            # top_features = feature_importance_df[feature_importance_df['importance'] > 0]['feature'].tolist()
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            
            # Select top features (positive importance, up to 50% of features)
            top_features_df = feature_importance_df.head(min((feature_importance_df['importance'] > 0).sum(), int(len(feature_importance_df) / 2)))
            top_features = top_features_df['feature'].tolist()
            
            stable_feature_collector.extend(top_features)
            feature_importance_collector.append(top_features_df)
            
            if exp_details:
                 # 2. Save Fold-Specific Data (All features for context)
                 fold_csv_path = explore_dir / f'fold_{fold+1}_importance.csv'
                 feature_importance_df.to_csv(fold_csv_path, index=False)
                 
                 # 3. Save Fold-Specific Plot
                 plt.figure(figsize=(10, 8))
                 # Plot top 20 or all selected if > 20
                 n_plot = max(20, len(top_features_df))
                 sns.barplot(x='importance', y='feature', data=feature_importance_df.head(n_plot))
                 plt.title(f'Fold {fold+1} Feature Importance - {modality}')
                 plt.xlabel('SHAP Importance')
                 plt.tight_layout()
                 plt.savefig(explore_dir / f'fold_{fold+1}_importance.png')
                 plt.close()

        # --- OLD METHOD: Select features in 10/10 folds ---
        # feature_stability = Counter(stable_feature_collector)
        # modality_stable_features = [
        #     f for f, count in feature_stability.items() if count >= STABILITY_THRESHOLD
        # ]
        
        # --- NEW METHOD: Threshold on Median of Mean SHAP of stable features ---

        # outer_fold is None, means analysis is on ALL, final model building.
        # All features important in more than 5 folds are selected
        if outer_fold is None:
            all_fold_importance = pd.concat(feature_importance_collector)
            stats = all_fold_importance.groupby('feature')['importance'].agg(['count', 'mean'])

            selected_mask = (stats['count'] >= N_SPLITS/2-1)

            stable_5 = stats[selected_mask]
            threshold = stable_5['mean'].median()
            selected_mask2 = (stats['count'] >= N_SPLITS/2-1) & (stats['mean'] > threshold)

            # Use selected_mask to select all features in more than 5 folds.
            # use selected_mask2 to use features in more than 5 folds AND above threshlod
            modality_stable_features = stats[selected_mask2].index.tolist()
            
            print(f"  Found {len(modality_stable_features)} features for {modality} (Frequency >= 8)")
        # If some features are selected, then compute median for the features selected in all folds. then select features above this threshold for all features at least important in 8
        elif feature_importance_collector:
            all_fold_importance = pd.concat(feature_importance_collector)
            stats = all_fold_importance.groupby('feature')['importance'].agg(['count', 'mean'])
            
            # 1. Identify features stable in all folds (10/10)
            # stable_10 = stats[stats['count'] == N_SPLITS]
            stable_10 = stats[stats['count'] > 0]
            
            # 2. Calculate threshold (median of means of stable features)
            if not stable_10.empty:
                threshold = stable_10['mean'].median()
                print(f"  [Selection] Threshold calculated from {len(stable_10)} stable features: {threshold:.6f}")
            else:
                threshold = 0
                print(f"  [Selection] No features stable in {N_SPLITS} folds. Threshold set to 0.")
            
            # 3. Select features: Count >= 8 AND Mean > Threshold
            selected_mask = (stats['count'] >= N_SPLITS-2) & (stats['mean'] > threshold)
            modality_stable_features = stats[selected_mask].index.tolist()
            
            print(f"  Found {len(modality_stable_features)} features for {modality} (Frequency >= 8, Mean Importance > {threshold:.6f})")
        else:
            modality_stable_features = []
            print(f"  Warning: No features selected for {modality}.")

        all_stable_features.extend(modality_stable_features)
        
        # --- EXPLORING MODE: Global Summary ---
        if feature_importance_collector:
            # Combine all fold data (selected features only)
            all_fold_importance = pd.concat(feature_importance_collector)
            
            # Summary stats
            summary_stats = all_fold_importance.groupby('feature').agg(
                Selection_Count=('importance', 'count'),
                Avg_Importance=('importance', 'mean')
            ).sort_values(['Selection_Count', 'Avg_Importance'], ascending=[False, False])
            
            summary_path = explore_dir / 'feature_selection_summary.csv'
            summary_stats.to_csv(summary_path)
            print(f"  [Exploration] Saved global feature selection summary to {summary_path}")
            
            # Global Plot
            top_n_plot = 30
            plot_data = summary_stats.head(top_n_plot)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x=plot_data['Avg_Importance'], y=plot_data.index, hue=plot_data['Selection_Count'], dodge=False)
            plt.title(f'Global Top {top_n_plot} Selected Features for {modality} (Color=Selection Freq)')
            plt.xlabel('Average SHAP Importance')
            plt.tight_layout()
            plot_path = explore_dir / 'feature_selection_plot.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"  [Exploration] Saved global feature selection plot to {plot_path}")

    master_feature_list = sorted(list(set(all_stable_features)))
    print(f"\n--- Total unique stable features from all modalities: {len(master_feature_list)} ---")
    
    return master_feature_list

def train_evaluate_cv(X, y, class_names, modality_features, analysis_type, model_type, handle_imbalance, feature_selection_default, mappings):
    """
    Performs nested cross-validation and builds a final consensus model.
    """
    
    pipeline_configs = config.get('PIPELINES')

    all_results = {}

    for pipeline_config in pipeline_configs:
        pipeline_name = pipeline_config.get('name', 'unnamed_pipeline')
        modalities_to_use = pipeline_config.get('modalities', [])
        feature_selection = pipeline_config.get('feature_selection', False)
        
        results_pipeline = Path(RESULTS_DIR, 'models')
        results_pipeline_feats = results_pipeline / 'individualFeatures'
        results_pipeline_feats.mkdir(parents=True, exist_ok=True)

        results_exploration = Path(RESULTS_DIR, 'exploration')
        results_exploration.mkdir(parents=True, exist_ok=True)

        print(f"\n\n{'='*20} RUNNING PIPELINE: {pipeline_name} {'='*20}")
        print(f"--- Active modalities: {modalities_to_use} ---")
        print(f"--- Feature Selection: {'ENABLED' if feature_selection else 'DISABLED'} ---")

        # --- Filter X based on active modalities ---
        active_features = []
        for modality in modalities_to_use:
            #  active_features.extend(_get_modal_cols(X, modality))
            active_features.extend(modality_features[modality.lower()])
        active_features = sorted(list(set(active_features)))

        if not active_features:
             print("  Warning: No features found for selected modalities. Using all features.")
             active_features = X.columns.tolist()
        
        X_pipeline = X[active_features].copy() # Work on a copy
        print(f"  Using {len(active_features)} features for this pipeline (filtered by modalities).")

        # --- EXPLORING MODE: Missing Value Analysis & Imputation ---
        if config.get('EXPLORING_MISSING', False):
            print("\n--- [Exploration Mode] Checking for missing values ---")
            missing_threshold = config.get('MISSING_THRESHOLD', 0.50)
            cols_imputed = []
            
            for col in X_pipeline.columns:
                missing_rate = X_pipeline[col].isna().mean()
                if missing_rate > missing_threshold:
                    # Create flag column
                    flag_col_name = f"{col}_ismissing"
                    X_pipeline[flag_col_name] = X_pipeline[col].isna().astype(int) # 1 if missing, 0 otherwise
                    X_pipeline[flag_col_name] = X_pipeline[flag_col_name].astype('category') # Treat as cat for XGB
                    
                    # Impute original column
                    if pd.api.types.is_numeric_dtype(X_pipeline[col]):
                         impute_val = X_pipeline[col].mean() # Or median
                         X_pipeline[col] = X_pipeline[col].fillna(impute_val)
                    else:
                         # For categorical, impute with mode
                         if not X_pipeline[col].mode().empty:
                             impute_val = X_pipeline[col].mode()[0]
                             X_pipeline[col] = X_pipeline[col].fillna(impute_val)
                    
                    cols_imputed.append(f"{col} ({missing_rate:.1%} missing)")
            
            if cols_imputed:
                print(f"  Created '_ismissing' flags and imputed values for {len(cols_imputed)} columns > {missing_threshold*100}% missing:")
                # print(cols_imputed) # Optional: print list
            else:
                print("  No columns exceeded the missing threshold.")
        # -----------------------------------------------------------

        cv_outer = StratifiedKFold(n_splits=config['CROSS_VALIDATION_FOLDS'], shuffle=True, random_state=config['RANDOM_STATE'])
        y_stratify = y['OS'] if analysis_type == 'survival' else (y.iloc[:,0] if isinstance(y, pd.DataFrame) else y)

        # Collectors for CV evaluation and consensus building
        all_y_test_true, all_y_pred, all_y_pred_proba, tprs, aucs = [], [], [], [], []
        fold_feature_lists = []
        final_fold_feature_lists = []
        fold_best_params = []
        mean_fpr = np.linspace(0, 1, 100)

        print(f"--- Step 1: Performing {config['CROSS_VALIDATION_FOLDS']}-fold nested cross-validation for evaluation ---")
        for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X_pipeline, y_stratify)):
            X_train, X_test = X_pipeline.iloc[train_idx], X_pipeline.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f"\n--- Fold {fold+1}/{config['CROSS_VALIDATION_FOLDS']} ---")
            # 1. Hyperparameter Optimization
            print("  Starting hyperparameter search...")
            num_class = len(class_names) if analysis_type == 'multiclass' and len(class_names) > 2 else 1
            model = xgb.XGBClassifier(objective='multi:softprob' if analysis_type == 'multiclass' else 'binary:logistic',
                                      eval_metric='mlogloss' if analysis_type == 'multiclass' else 'logloss',
                                      num_class=num_class if num_class > 1 else None,
                                      enable_categorical=True, random_state=config['RANDOM_STATE'])
            model_search_spaces = {
                'learning_rate': Real(0.001, 1.0, 'log-uniform'), 'n_estimators': Integer(50, 200),
                'max_depth': Integer(3, 15), 'subsample': Real(0.6, 1.0, 'uniform'),
                'colsample_bytree': Real(0.6, 1.0, 'uniform'),
            }
            cv_inner = StratifiedKFold(n_splits=config['CROSS_VALIDATION_FOLDS'], shuffle=True, random_state=config['RANDOM_STATE'])
            y_train_stratify = y_train['OS'] if analysis_type == 'survival' else (y_train.iloc[:,0] if isinstance(y_train, pd.DataFrame) else y_train)
            bayes_search = BayesSearchCV(
                estimator=model, search_spaces=model_search_spaces, n_iter=50, cv=cv_inner,
                n_jobs=-1, verbose=0, random_state=config['RANDOM_STATE'],
                scoring='roc_auc_ovr' if analysis_type == 'multiclass' else 'roc_auc'
            )
            y_train_fit = y_train
            sample_weights = None
            if analysis_type == 'survival':
                y_train_fit = np.where(y_train['OS'] == 1, y_train['OS_time'], -y_train['OS_time'])
            else: # for binary and multiclass
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_fit.tolist())

            bayes_search.fit(X_train, y_train_fit.tolist(), sample_weight=sample_weights)
            best_params = bayes_search.best_params_
            fold_best_params.append(best_params)
            print(f"  Best params found: {best_params}, Score: {bayes_search.best_score_:.4f}")

            # 2. Stable Feature Selection
            master_feature_list = X_train.columns.tolist()
            if feature_selection:
                master_feature_list = select_stable_features(
                    X_train, y_train, modalities_to_use, modality_features, best_params,
                    len(class_names), analysis_type, config['RANDOM_STATE'],
                    pipeline_name=pipeline_name, # Pass pipeline name for exploration mode
                    outer_fold=fold + 1
                )
            if not master_feature_list:
                print("  Warning: No stable features found. Using all features for this fold.")
                master_feature_list = X_train.columns.tolist()
            fold_feature_lists.append(master_feature_list)

            # 3. Train & Evaluate on Test Set
            print(f"  Training final model on {len(master_feature_list)} features and evaluating.")
            X_train_final, X_test_final = X_train[master_feature_list], X_test[master_feature_list]
            model_params = {**best_params, 'random_state': config['RANDOM_STATE'], 'enable_categorical': True}
            final_model_fold = xgb.XGBClassifier(objective='multi:softprob' if analysis_type == 'multiclass' else 'binary:logistic',
                                               eval_metric='mlogloss' if analysis_type == 'multiclass' else 'logloss',
                                               num_class=num_class if num_class > 1 else None, **model_params)
            if analysis_type == 'survival':
                final_model_fold = xgb.XGBRegressor(objective='survival:cox', eval_metric='cox-nloglik', **model_params)
            final_model_fold.fit(X_train_final, y_train_fit, sample_weight=sample_weights)
            
            # --- Evaluation data collection ---
            if analysis_type == 'survival':
                all_y_pred.extend(final_model_fold.predict(X_test_final))
                all_y_test_true.append(y_test)
            else:
                y_pred, y_pred_proba = final_model_fold.predict(X_test_final), final_model_fold.predict_proba(X_test_final)
                all_y_test_true.extend(y_test.iloc[:,0] if isinstance(y_test, pd.DataFrame) else y_test)
                all_y_pred.extend(y_pred)
                all_y_pred_proba.append(y_pred_proba)
                if analysis_type == 'binary':
                    y_test_binary = y_test.iloc[:,0] if isinstance(y_test, pd.DataFrame) else y_test
                    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, 1])
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    aucs.append(roc_auc_score(y_test_binary, y_pred_proba[:, 1]))
                    print(f"  AUC on test set {roc_auc_score(y_test_binary, y_pred_proba[:, 1]):.4f}")


            explainer = shap.TreeExplainer(final_model_fold)
            shap_values = explainer.shap_values(X_test_final)

            if analysis_type == 'multiclass': # multiclass
                global_shap_importance = np.abs(np.array(shap_values)).mean(axis=0).sum(axis=1)
            else: # binary or survival
                global_shap_importance = np.abs(shap_values).mean(axis=0)

            feature_importance_df = pd.DataFrame({
                'feature': X_test_final.columns,
                'importance': global_shap_importance
            })

            # top_features = feature_importance_df[feature_importance_df['importance'] > 0]['feature'].tolist()
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            top_features = feature_importance_df['feature'].head(min((feature_importance_df['importance'] > 0).sum(),int(len(feature_importance_df) / 2)))
            # stable_feature_collector.extend(top_features)
            final_fold_feature_lists.append(top_features.tolist())


        # --- Step 2: Overall Evaluation Visualization (after all folds) ---
        # ... [Plotting and printing overall CV metrics - this logic is correct and remains] ...
        print(f"\n--- Step 2: Overall evaluation for pipeline {pipeline_name} completed. ---")

        # --- CV-level Evaluation (ROC, CM, etc.) ---
        if analysis_type == 'survival':
            if all_y_test_true:
                all_y_test_true_df = pd.concat(all_y_test_true)
                c_index = concordance_index(all_y_test_true_df['OS_time'], -np.array(all_y_pred), all_y_test_true_df['OS'])
                print(f"Overall C-Index from CV for {pipeline_name}: {c_index:.4f}")
        else:
            all_y_pred_proba_flat = np.concatenate(all_y_pred_proba, axis=0)
            fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
            y_true_total = np.array(all_y_test_true)

            if analysis_type == 'binary':
                # Individual fold ROCs
                for i in range(len(tprs)):
                     ax_roc.plot(mean_fpr, tprs[i], alpha=0.3, label=f'ROC Fold {i+1} (AUC = {aucs[i]:.2f})')
                ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
                
                # Global ROC with CI
                y_pred_proba_total = all_y_pred_proba_flat[:, 1]
                global_auc = roc_auc_score(y_true_total, y_pred_proba_total)
                mean_auc = np.mean(aucs)

                n_bootstraps = 1000
                bootstrapped_aucs = [roc_auc_score(y_true_total[indices], y_pred_proba_total[indices])
                                     for _ in range(n_bootstraps)
                                     if len(np.unique(y_true_total[(indices := np.random.RandomState(config['RANDOM_STATE']+_).choice(len(y_true_total), len(y_true_total), replace=True))])) > 1]
                
                # std_auc = np.std(aucs, ddof=1) # ddof=1 for sample standard deviation
                mean_auc = np.mean(aucs)
                n_folds = len(aucs)
                confidence = 0.95
                se = stats.sem(aucs) # Standard Error of the Mean
                h = se * stats.t.ppf((1 + confidence) / 2., n_folds-1)

                lower = mean_auc - h
                upper = mean_auc + h
                label = f'Mean-fold ROC (AUC = {mean_auc:.2f} [95% CI: {lower:.2f}-{upper:.2f}])'

                label_global = f'Global ROC (AUC = {global_auc:.2f})'
                if bootstrapped_aucs:
                    lower_global, upper_global = np.percentile(bootstrapped_aucs, [2.5, 97.5])
                    label_global = f'Global ROC (AUC = {global_auc:.2f} [95% CI: {lower_global:.2f}-{upper_global:.2f}])'

                print(label)
                print(label_global)

                fpr_global, tpr_global, _ = roc_curve(y_true_total, y_pred_proba_total)
                ax_roc.plot(fpr_global, tpr_global, color='b', label=label, lw=2, alpha=.8)
                ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"Cross-Validation ROC Curve for {pipeline_name}", xlabel="False Positive Rate", ylabel="True Positive Rate")
                ax_roc.legend(loc="lower right")

            else: # multiclass
                y_pred_proba_total = all_y_pred_proba_flat
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy'])

                for i, color in zip(range(len(class_names)), colors):
                    y_true_class_i = (y_true_total == i).astype(int)
                    if np.sum(y_true_class_i) == 0:
                        continue # Skip if no samples for this class
                    
                    y_pred_proba_class_i = y_pred_proba_total[:, i]
                    global_auc = roc_auc_score(y_true_class_i, y_pred_proba_class_i)
                    
                    bootstrapped_aucs = [roc_auc_score(y_true_class_i[indices], y_pred_proba_class_i[indices])
                                         for _ in range(1000)
                                         if len(np.unique(y_true_class_i[(indices := np.random.RandomState(config['RANDOM_STATE']+_).choice(len(y_true_class_i), len(y_true_class_i), replace=True))])) > 1]

                    label = f'Global ROC for {class_names[i]} (AUC = {global_auc:.2f})'
                    if bootstrapped_aucs:
                        lower, upper = np.percentile(bootstrapped_aucs, [2.5, 97.5])
                        label += f' [95% CI: {lower:.2f}-{upper:.2f}]'

                    fpr, tpr, _ = roc_curve(y_true_class_i, y_pred_proba_class_i)
                    ax_roc.plot(fpr, tpr, color=color, label=label, lw=2, alpha=.8)

                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"Global One-vs-Rest ROC Curves for {pipeline_name}", xlabel="False Positive Rate", ylabel="True Positive Rate")
                ax_roc.legend(loc="lower right")

            save_plot(fig_roc, f"{pipeline_name}_roc_curve_cross_validation", results_exploration)

            print(f"--- Step 4: Evaluating overall model performance for {pipeline_name} from CV ---")
            print("Overall Classification Report (from cross-validation):")
            print(classification_report(all_y_test_true, all_y_pred, target_names=class_names))
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            cm = confusion_matrix(all_y_test_true, all_y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            ax_cm.set_title(f"Overall Confusion Matrix for {pipeline_name} (from CV)")
            save_plot(fig_cm, f"{pipeline_name}_confusion_matrix_cv", results_exploration)
        

        # --- Step 3: Build Consensus Final Model ---
        print(f"\n--- Step 3: Building Consensus Final Model for {pipeline_name} on ALL data ---")

        # 1. Create consensus hyperparameters
        consensus_params = {}
        if fold_best_params:
            params_df = pd.DataFrame(fold_best_params)
            for col in params_df.columns:
                consensus_params[col] = int(np.round(params_df[col].mean())) if params_df[col].dtype == 'int64' else params_df[col].mean()
        print(f"  Consensus Hyperparameters: {consensus_params}")

        # 2. Create consensus feature list
        consensus_features = []
        if fold_feature_lists:
            feature_counts = Counter([item for sublist in fold_feature_lists for item in sublist])
            # consensus_threshold = int(np.floor(config['CROSS_VALIDATION_FOLDS'] / 2)) + 1
            consensus_threshold = 0
            consensus_features = sorted([f for f, c in feature_counts.items() if c >= consensus_threshold])
            print(f"  Found {len(consensus_features)} consensus features (selected in >= {consensus_threshold} of {config['CROSS_VALIDATION_FOLDS']} folds)")
        
        if not consensus_features:
            print("  Warning: No consensus features found. Using union of all features found across folds.")
            consensus_features = sorted(list(set([item for sublist in fold_feature_lists for item in sublist])))

        if config.get('EXPLORING_MODE', False):
            to_remove = set()
            for item in consensus_features:
                if item.endswith('_ismissing'):
                    to_remove.add(item)
                    # Remove the last 10 characters ('_ismissing') to get the base name
                    base_name = item[:-10] 
                    to_remove.add(base_name)

            # 2. Filter the list to exclude any item in the removal set
            consensus_features = [item for item in consensus_features if item not in to_remove]

        # 3. Train final model on ALL data with consensus settings
        if not consensus_features:
            print("  ERROR: No features available to train final model. Skipping.")
            final_model = None
        else:
            X_final = X[consensus_features]
            y_fit_all = y
            if analysis_type == 'survival':
                y_fit_all = np.where(y['OS'] == 1, y['OS_time'], -y['OS_time'])
            
            if feature_selection:
                master_final_feature_list = select_stable_features(
                    X_final[consensus_features], y_fit_all, ['ALL'], consensus_features, consensus_params,
                    len(class_names), analysis_type, config['RANDOM_STATE'],
                    pipeline_name=pipeline_name,
                    outer_fold=None)

            X_final = X[master_final_feature_list]

            num_class = len(class_names) if analysis_type == 'multiclass' and len(class_names) > 2 else 1
            model_params = {**consensus_params, 'random_state': config['RANDOM_STATE'], 'enable_categorical': True}
            final_model = xgb.XGBClassifier(objective='multi:softprob' if analysis_type == 'multiclass' else 'binary:logistic',
                                              eval_metric='mlogloss' if analysis_type == 'multiclass' else 'logloss',
                                              num_class=num_class if num_class > 1 else None, **model_params)
            if analysis_type == 'survival':
                final_model = xgb.XGBRegressor(objective='survival:cox', eval_metric='cox-nloglik', **model_params)

            sample_weights_all = None
            if analysis_type in ['binary', 'multiclass']:
                sample_weights_all = compute_sample_weight(class_weight='balanced', y=y_fit_all)

            final_model.fit(X_final, y_fit_all, sample_weight=sample_weights_all)
            # import re
            # import os
            # prospect = pd.read_csv(os.path.join(config['CLINICAL_DATA_DIR'], f"prospect.tsv"), sep='\t')
            # # prospect.columns = [re.sub(r'[\[\]<>()\s,]', '_', col) for col in prospect.columns]
            # external = pd.read_csv(os.path.join(config['CLINICAL_DATA_DIR'], f"external.tsv"), sep='\t')
            # # external.columns = [re.sub(r'[\[\]<>()\s,]', '_', col) for col in external.columns]

            # print(f"  Final consensus model trained on {len(master_final_feature_list)} features.")
            # print(f"ROC AUC on prospect: {roc_auc_score(prospect['Label'],final_model.predict_proba(prospect[final_model.feature_names_in_])[:,1])}")
            # print(f"ROC AUC on external: {roc_auc_score(external['Label'],final_model.predict_proba(external[final_model.feature_names_in_])[:,1])}")


        print("\n--- Step 4: Final Feature Set Verification (10-fold CV on entire dataset) ---")
        from sklearn.preprocessing import label_binarize # Required for multiclass ROC

        mean_v_auc = 0
        if final_model is not None and master_final_feature_list:
            X_verify = X[master_final_feature_list]
            cv_verify = StratifiedKFold(n_splits=10, shuffle=True, random_state=config['RANDOM_STATE'])

            verify_metrics = []
            tprs = []
            mean_fpr = np.linspace(0, 1, 100)

            y_verify_stratify = y['OS'] if analysis_type == 'survival' else (y.iloc[:,0] if isinstance(y, pd.DataFrame) else y)
            
            # Identify classes for multiclass handling
            unique_classes = np.unique(y_verify_stratify)
            n_classes = len(unique_classes)

            class_tprs = {i: [] for i in range(n_classes)} 
            tprs = [] # Still used for the macro-average
            verify_metrics = []
            mean_fpr = np.linspace(0, 1, 100)

            oof_predictions = pd.DataFrame(index=X_verify.index)
            oof_predictions['truth'] = y_verify_stratify
            # Note: 'probability' column logic below might need adjustment for multiclass if you want to save all class probs
            
            fig_roc, ax_roc = plt.subplots(figsize=(8, 8))

            for v_fold, (v_train_idx, v_test_idx) in enumerate(cv_verify.split(X_verify, y_verify_stratify)):
                X_v_train, X_v_test = X_verify.iloc[v_train_idx], X_verify.iloc[v_test_idx]
                y_v_train, y_v_test = y.iloc[v_train_idx], y.iloc[v_test_idx]
                
                if analysis_type == 'survival':
                    v_model = xgb.XGBRegressor(**final_model.get_params())
                    v_y_train_fit = np.where(y_v_train['OS'] == 1, y_v_train['OS_time'], -y_v_train['OS_time'])
                    v_model.fit(X_v_train, v_y_train_fit)
                    v_pred = v_model.predict(X_v_test)
                    verify_metrics.append(concordance_index(y_v_test['OS_time'], -v_pred, y_v_test['OS']))
                else:
                    v_model = xgb.XGBClassifier(**final_model.get_params())
                    v_sample_weights = compute_sample_weight(class_weight='balanced', y=y_v_train)
                    v_model.fit(X_v_train, y_v_train, sample_weight=v_sample_weights)
                    v_proba = v_model.predict_proba(X_v_test)
                    
                    if analysis_type == 'binary':
                        fpr, tpr, _ = roc_curve(y_v_test, v_proba[:, 1])
                        roc_auc = roc_auc_score(y_v_test, v_proba[:, 1])
                        verify_metrics.append(roc_auc)
                        
                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                        ax_roc.plot(fpr, tpr, lw=1, alpha=0.3, label='_nolegend_')

                    elif analysis_type == 'multiclass':
                        y_v_test_bin = label_binarize(y_v_test, classes=unique_classes)
                        roc_auc = roc_auc_score(y_v_test, v_proba, multi_class='ovr', average='macro')
                        verify_metrics.append(roc_auc)

                        fold_tprs = []
                        for i in range(n_classes):
                            fpr_c, tpr_c, _ = roc_curve(y_v_test_bin[:, i], v_proba[:, i])
                            interp_tpr = np.interp(mean_fpr, fpr_c, tpr_c)
                            interp_tpr[0] = 0.0
                            
                            # Store for individual class mean calculation
                            class_tprs[i].append(interp_tpr)
                            # Store for this fold's macro-average
                            fold_tprs.append(interp_tpr)
                        
                        tprs.append(np.mean(fold_tprs, axis=0))
            
            # --- Update OOF Predictions for CSV ---
            # We use the final_model trained on whole set for the "per-sample" view
            oof_predictions['predicted_label'] = final_model.predict(X_verify)
            # For simplicity, we save the probability of the predicted class or the first class
            oof_predictions['probability'] = np.max(final_model.predict_proba(X_verify), axis=1)

            # --- 1. SAVE PER-SAMPLE PREDICTIONS ---
            predictions_csv_path = results_pipeline / f"{pipeline_name}_sample_predictions.csv"
            oof_predictions.to_csv(predictions_csv_path)
            
            # --- 2. GENERATE AND SAVE CONFUSION MATRIX ---
            if analysis_type != 'survival':
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                valid_oof = oof_predictions.dropna(subset=['predicted_label'])
                cm = confusion_matrix(valid_oof['truth'], valid_oof['predicted_label'])
                fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=class_names, yticklabels=class_names)
                ax_cm.set_title(f"Confusion Matrix: {pipeline_name}\n(10-fold CV Aggregated)")
                save_plot(fig_cm, f"{pipeline_name}_confusion_matrix_cv", results_pipeline)

            # --- Finalize Statistics & Plotting ---
            mean_v_auc = np.mean(verify_metrics)
            m_name = "C-Index" if analysis_type == 'survival' else "AUC"
            confidence = 0.95
            n_folds = len(verify_metrics)
            se = stats.sem(verify_metrics)
            h = se * stats.t.ppf((1 + confidence) / 2., n_folds-1)
            lower_v, upper_v = mean_v_auc - h, mean_v_auc + h

            if analysis_type in ['binary']:
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                
                pipeline_curve_data = {'fpr': mean_fpr, 'tpr': mean_tpr, 'auc': mean_v_auc, 'lower': lower_v, 'upper': upper_v}
                
                label_prefix = "Mean" if analysis_type == 'binary' else "Macro-average"
                ax_roc.plot(mean_fpr, mean_tpr, color='b',
                            label=f'{label_prefix} ROC (AUC = {mean_v_auc:.2f} [{lower_v:.2f}- {upper_v:.2f}])', lw=2, alpha=.8)
                
                ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random classifier', alpha=.8)
                ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                            title=f"Verification {n_folds}-fold ROC - {analysis_type}",
                            xlabel="False Positive Rate", ylabel="True Positive Rate")
                ax_roc.legend(loc="lower right")
                save_plot(fig_roc, f"Final_{pipeline_name}_ROC", results_pipeline)

            # --- Finalize Statistics & Plotting ---
            if analysis_type == 'multiclass':
                # 1. Plot Individual Class Curves first (lighter lines)
                for i in range(n_classes):
                    mean_tpr_c = np.mean(class_tprs[i], axis=0)
                    mean_tpr_c[-1] = 1.0
                    class_auc = auc(mean_fpr, mean_tpr_c)
                    c_name = class_names[i] if 'class_names' in locals() else f"Class {i}"
                    
                    ax_roc.plot(mean_fpr, mean_tpr_c, lw=1.5, alpha=0.5,
                                label=f'ROC {c_name} (AUC = {class_auc:.2f})')

                # 2. Plot the Macro-Average Curve (thick line)
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                
                ax_roc.plot(mean_fpr, mean_tpr, color='navy', linestyle=':', lw=3,
                            label=f'Macro-average ROC (AUC = {mean_v_auc:.2f} [{lower_v:.2f}-{upper_v:.2f}])')

                # Standard plot setup
                ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
                ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                            title=f"Multiclass 10-fold ROC - {pipeline_name}",
                            xlabel="False Positive Rate", ylabel="True Positive Rate")
                ax_roc.legend(loc="lower right", fontsize='small')
                
                save_plot(fig_roc, f"Final_{pipeline_name}_Multiclass_ROC", results_pipeline)


        # # --- Step 4: Final Feature Set Verification (10-fold CV on entire dataset) ---
        # print("\n--- Step 4: Final Feature Set Verification (10-fold CV on entire dataset) ---")
        # mean_v_auc = 0
        # if final_model is not None and master_final_feature_list:
        #     X_verify = X[master_final_feature_list]
        #     cv_verify = StratifiedKFold(n_splits=10, shuffle=True, random_state=config['RANDOM_STATE'])

        #     verify_metrics = []
        #     tprs = []
        #     mean_fpr = np.linspace(0, 1, 100)

        #     y_verify_stratify = y['OS'] if analysis_type == 'survival' else (y.iloc[:,0] if isinstance(y, pd.DataFrame) else y)
            
        #     # --- NEW: Data Structures to capture per-sample predictions ---
        #     # We create a DataFrame to store the 'Truth' vs 'Predicted' for every sample
        #     oof_predictions = pd.DataFrame(index=X_verify.index)
        #     oof_predictions['truth'] = y_verify_stratify
        #     oof_predictions['predicted_label'] = final_model.predict(X_verify)
        #     oof_predictions['probability'] = final_model.predict_proba(X_verify)[:,1]

        #     fig_roc, ax_roc = plt.subplots(figsize=(8, 8))

        #     for v_fold, (v_train_idx, v_test_idx) in enumerate(cv_verify.split(X_verify, y_verify_stratify)):
        #         X_v_train, X_v_test = X_verify.iloc[v_train_idx], X_verify.iloc[v_test_idx]
        #         y_v_train, y_v_test = y.iloc[v_train_idx], y.iloc[v_test_idx]
                
        #         # Clone/Re-initialize model with consensus params
        #         if analysis_type == 'survival':
        #             v_model = xgb.XGBRegressor(**final_model.get_params())
        #             v_y_train_fit = np.where(y_v_train['OS'] == 1, y_v_train['OS_time'], -y_v_train['OS_time'])
        #             v_model.fit(X_v_train, v_y_train_fit)
        #             v_pred = v_model.predict(X_v_test)
        #             verify_metrics.append(concordance_index(y_v_test['OS_time'], -v_pred, y_v_test['OS']))
        #         else:
        #             v_model = xgb.XGBClassifier(**final_model.get_params())
        #             v_sample_weights = compute_sample_weight(class_weight='balanced', y=y_v_train)
        #             v_model.fit(X_v_train, y_v_train, sample_weight=v_sample_weights)
        #             v_proba = v_model.predict_proba(X_v_test)
        #             v_preds = v_model.predict(X_v_test)
                    
        #             # Binary ROC Tracking
        #             if analysis_type == 'binary':
        #                 fpr, tpr, _ = roc_curve(y_v_test, v_proba[:, 1])
        #                 roc_auc = roc_auc_score(y_v_test, v_proba[:, 1])
        #                 verify_metrics.append(roc_auc)
                        
        #                 # Interpolate TPR for mean curve
        #                 interp_tpr = np.interp(mean_fpr, fpr, tpr)
        #                 interp_tpr[0] = 0.0
        #                 tprs.append(interp_tpr)
                        
        #                 ax_roc.plot(fpr, tpr, lw=1, alpha=0.3, label='_nolegend_')
        #             else:
        #                 # Multiclass AUC
        #                 verify_metrics.append(roc_auc_score(y_v_test, v_proba, multi_class='ovr'))

        #     # --- 1. SAVE PER-SAMPLE PREDICTIONS ---
        #     # Save the CSV showing which specific samples were predicted as 0 or 1
        #     predictions_csv_path = results_pipeline / f"{pipeline_name}_sample_predictions.csv"
        #     oof_predictions.to_csv(predictions_csv_path)
        #     print(f"✅ Per-sample predictions saved to: {predictions_csv_path}")
            
        #     # --- 2. GENERATE AND SAVE CONFUSION MATRIX ---
        #     if analysis_type != 'survival':
        #         from sklearn.metrics import confusion_matrix
        #         import seaborn as sns

        #         # Filter out any NaNs if CV didn't cover all rows (it should)
        #         valid_oof = oof_predictions.dropna(subset=['predicted_label'])
        #         cm = confusion_matrix(valid_oof['truth'], valid_oof['predicted_label'])
                
        #         fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
        #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
        #                     xticklabels=class_names, yticklabels=class_names)
        #         ax_cm.set_title(f"Confusion Matrix: {pipeline_name}\n(10-fold CV Aggregated)")
        #         ax_cm.set_xlabel("Predicted")
        #         ax_cm.set_ylabel("Actual")
                
        #         save_plot(fig_cm, f"{pipeline_name}_confusion_matrix_cv", results_pipeline)

        #     # --- Finalize Statistics & Plotting ---
        #     mean_v_auc = np.mean(verify_metrics)
        #     m_name = "C-Index" if analysis_type == 'survival' else "AUC"
        #     confidence = 0.95
        #     n_folds = len(verify_metrics)
        #     se = stats.sem(verify_metrics) # Standard Error of the Mean
        #     h = se * stats.t.ppf((1 + confidence) / 2., n_folds-1)

        #     lower_v = mean_v_auc - h
        #     upper_v = mean_v_auc + h

        #     if analysis_type == 'binary':
        #         # Calculate Mean TPR and Std
        #         mean_tpr = np.mean(tprs, axis=0)
        #         mean_tpr[-1] = 1.0
                
        #         # Store the data needed for the final comparison plot
        #         pipeline_curve_data = {
        #             'fpr': mean_fpr,
        #             'tpr': mean_tpr,
        #             'auc': mean_v_auc,
        #             'lower': lower_v,
        #             'upper': upper_v
        #         }
        #         # Plot Mean Curve
        #         ax_roc.plot(mean_fpr, mean_tpr, color='b',
        #                     label=f'Mean ROC (AUC = {mean_v_auc:.2f} [{lower_v:.2f}- {upper_v:.2f}])', lw=2, alpha=.8)
                
        #         # Plot Chance Line
        #         ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random classifier', alpha=.8)
                
        #         ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        #                     title=f"Verification {n_folds}-fold ROC - {analysis_type}",
        #                     xlabel="False Positive Rate", ylabel="True Positive Rate")
        #         ax_roc.legend(loc="lower right")
                
        #         # Use your save_plot function if available, or plt.show()
        #         save_plot(fig_roc, f"Final_{pipeline_name}_ROC", results_pipeline)

                # cm = confusion_matrix(all_y_test_true, all_y_pred)
                # fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
                # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                # ax_cm.set_title(f"Overall Confusion Matrix for {pipeline_name} (from CV)")
                # save_plot(fig_cm, f"{pipeline_name}_confusion_matrix_cv", results_exploration)
        

        all_results[pipeline_name] = {
            'pipeline': final_model, # Storing the actual model object
            'vmodel': v_model,
            'selected_features': master_final_feature_list,
            'best_params': consensus_params,
            'verification_auc': mean_v_auc,
            'curve_data': pipeline_curve_data
        }

        shap_values = explain_with_shap(all_results, X, config['ANALYSIS_TYPE'], class_names, results_pipeline, mappings)

        # if analysis_type != 'survival':
        #     print(f"  ROC AUC of final model (fitted on all): {roc_auc_score(y_fit_all,final_model.predict_proba(X_final)[:,1])}")

    # --- FINAL COMPARISON PLOT ---
    if analysis_type == 'binary':
        print("\n--- Generating Final Pipeline Comparison Plot ---")
        fig_comp, ax_comp = plt.subplots(figsize=(10, 8))
        
        # Define colors for different pipelines
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        
        for (pipeline_name, res), color in zip(all_results.items(), colors):
            curve = res.get('curve_data')
            if curve is not None:
                label = f"{pipeline_name} (AUC = {curve['auc']:.2f} [{curve['lower']:.2f}-{curve['upper']:.2f}])"
                ax_comp.plot(curve['fpr'], curve['tpr'], color=color, lw=2.5, label=label)
        
        # Standard Plot Formatting
        ax_comp.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.5, label='Chance')
        ax_comp.set_xlim([-0.02, 1.02])
        ax_comp.set_ylim([-0.02, 1.02])
        ax_comp.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax_comp.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax_comp.set_title(f'Pipeline Comparison: {analysis_type.capitalize()} ROC Curves', fontsize=14)
        ax_comp.legend(loc="lower right", frameon=True, fontsize=10)
        ax_comp.grid(alpha=0.2)
        
        # Save the comparison plot
        save_plot(fig_comp, "Pipeline_Comparison_ROC", results_pipeline)

    return all_results

