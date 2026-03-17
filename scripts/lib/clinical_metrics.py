import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
from scripts.lib.savePlots import save_plot

def hosmer_lemeshow_test(y_true, y_proba, g=10):
    """
    Calculates the Hosmer-Lemeshow Goodness of Fit Test.
    g is the number of subgroups (typically 10 for deciles).
    """
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba')
    
    # Try to cut into deciles. If there are too many identical probabilities, 
    # 'duplicates=drop' will reduce the number of bins automatically.
    try:
        df['decile'] = pd.qcut(df['y_proba'], q=g, duplicates='drop')
    except Exception:
        return None, None
        
    grouped = df.groupby('decile', observed=False)
    
    # Observed events
    O_1 = grouped['y_true'].sum()
    O_0 = grouped['y_true'].count() - O_1
    
    # Expected events
    E_1 = grouped['y_proba'].sum()
    E_0 = grouped['y_proba'].count() - E_1
    
    # Calculate HL Statistic (adding small epsilon to prevent division by zero)
    hl_stat = np.sum(((O_1 - E_1)**2 / np.maximum(E_1, 1e-8)) + ((O_0 - E_0)**2 / np.maximum(E_0, 1e-8)))
    
    # Degrees of freedom: Number of bins - 2
    df_stat = len(grouped) - 2 
    if df_stat <= 0:
        return None, None
        
    p_value = 1 - stats.chi2.cdf(hl_stat, df_stat)
    return hl_stat, p_value


def evaluate_clinical_utility(y_true, y_proba, pipeline_name, dataset_name, out_dir, precalculated_threshold=None):
    """
    Calculates and plots Calibration, Decision Curve Analysis (DCA), 
    metrics at the optimal threshold, and saves predictions to CSV.
    """
    print(f"\n--- Calculating Clinical Utility Metrics for {pipeline_name} on {dataset_name} ---")
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    N = len(y_true)
    
    # 1. OPTIMAL THRESHOLD (Youden's J Statistic)
    if precalculated_threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = precalculated_threshold
    
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"  Optimal Threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")
    print(f"  PPV: {ppv:.4f} | NPV: {npv:.4f}")

    # 2. SAVE PREDICTIONS TO CSV
    predictions_df = pd.DataFrame({
        'Truth': y_true,
        'Probability': y_proba,
        'Predicted_Class': y_pred_opt
    })
    csv_path = out_dir / f"{dataset_name}_{pipeline_name}_predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"  Saved predictions to {csv_path}")

    # 3. CALIBRATION CURVE (Using Quantiles) + HOSMER-LEMESHOW TEST
    fig_cal, ax_cal = plt.subplots(figsize=(8, 8))
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='quantile')
    
    # Calculate HL Test
    hl_stat, hl_p_value = hosmer_lemeshow_test(y_true, y_proba, g=10)
    
    label_text = pipeline_name
    if hl_p_value is not None:
        hl_text = f"HL p-value = {hl_p_value:.3f}"
        print(f"  Hosmer-Lemeshow Test -> Stat: {hl_stat:.2f}, p-value: {hl_p_value:.4f}")
        # Append HL stats to the legend label
        label_text = f"{pipeline_name} ({hl_text})"
    else:
        print("  Hosmer-Lemeshow Test -> Could not be calculated (too few unique probability bins).")

    ax_cal.plot(prob_pred, prob_true, marker='o', linewidth=2, label=label_text)
    ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    ax_cal.set_title(f"Calibration Curve - {dataset_name}")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.legend(loc="lower right")
    ax_cal.grid(alpha=0.3)
    
    # Optionally: Add a text box inside the plot with the HL details
    if hl_p_value is not None:
        ax_cal.text(0.05, 0.95, f"Hosmer-Lemeshow $\chi^2$: {hl_stat:.2f}\np-value: {hl_p_value:.3f}", 
                    transform=ax_cal.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    save_plot(fig_cal, f"{dataset_name}_{pipeline_name}_Calibration", out_dir)

    # 4. DECISION CURVE ANALYSIS (DCA)
    threshold_probs = np.linspace(0.01, 0.99, 99)
    net_benefits_model = []
    net_benefits_treat_all = []
    
    for pt in threshold_probs:
        y_pred = (y_proba >= pt).astype(int)
        tp_model = np.sum((y_pred == 1) & (y_true == 1))
        fp_model = np.sum((y_pred == 1) & (y_true == 0))
        nb_model = (tp_model / N) - (fp_model / N) * (pt / (1 - pt))
        net_benefits_model.append(nb_model)
        
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        nb_all = (tp_all / N) - (fp_all / N) * (pt / (1 - pt))
        net_benefits_treat_all.append(nb_all)

    fig_dca, ax_dca = plt.subplots(figsize=(10, 8))
    
    ax_dca.plot(threshold_probs, net_benefits_model, label=f'{pipeline_name} Model', color='blue', linewidth=2)
    ax_dca.plot(threshold_probs, net_benefits_treat_all, label='Treat All', color='gray', linestyle='--')
    ax_dca.axhline(0, color='black', label='Treat None', linestyle=':')
    
    ax_dca.set_xlim([0.0, 1.0])
    ax_dca.set_ylim([-0.05, max(max(net_benefits_model), max(net_benefits_treat_all)) * 1.2]) 
    
    ax_dca.set_title(f"Decision Curve Analysis - {dataset_name}")
    ax_dca.set_xlabel("Threshold Probability")
    ax_dca.set_ylabel("Net Benefit")
    ax_dca.legend(loc="upper right")
    ax_dca.grid(alpha=0.3)
    
    save_plot(fig_dca, f"{dataset_name}_{pipeline_name}_DCA", out_dir)
    
    return optimal_threshold

