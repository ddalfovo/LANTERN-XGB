import pandas as pd
import numpy as np
import os
from fpdf import FPDF
import datetime

# --- BIOLOGICAL KNOWLEDGE BASE & TITLES ---
FEATURE_TITLES = {
    # Clinical
    "CEA": "Carcinoembryonic Antigen",
    "Psize": "Primary Tumor Size",
    "SCC_Ag": "Squamous Cell Carcinoma Antigen",
    "NSE": "Neuron-Specific Enolase",
    "Age": "Patient Age",
    # CT Conventional
    "CONVENTIONAL_HUmax": "Maximum Tumor Density",
    "CONVENTIONAL_HUmin": "Minimum Tumor Density",
    "CONVENTIONAL_HUmean": "Mean Tumor Density",
    "CONVENTIONAL_HUQ1": "25th Percentile Tumor Density",
    "CONVENTIONAL_HUQ3": "75th Percentile Tumor Density",
    # PET Conventional
    "CONVENTIONAL_SUVbwmax": "Maximum Metabolic Activity (SUVmax)",
    "CONVENTIONAL_SUVbwmin": "Minimum Metabolic Activity (SUVmin)",
    "CONVENTIONAL_SUVbwmean": "Mean Metabolic Activity (SUVmean)",
    "CONVENTIONAL_SUVbwQ1": "25th Percentile Metabolic Activity",
    "CONVENTIONAL_SUVbwQ3": "75th Percentile Metabolic Activity",
    # GLZLM
    "GLZLM_LGZE": "Large Zone Emphasis",
    "GLZLM_LZLGE": "Large Zone Low Gray-Level Emphasis",
    "GLZLM_HGZE": "High Gray-Level Zone Emphasis",
    "GLZLM_SZE": "Small Zone Emphasis",
    "GLZLM_SZHGE": "Small Zone High Gray-Level Emphasis",
    "GLZLM_SZLGE": "Small Zone Low Gray-Level Emphasis",
    # GLCM
    "GLCM_Correlation": "Tumor Pixel Correlation",
    "GLCM_Contrast": "Tumor Local Contrast",
    "GLCM_Entropy": "Tumor Texture Entropy",
    "GLCM_Energy": "Tumor Texture Energy",
    # GLRLM
    "GLRLM_LRHGE": "Long Run High Gray-Level Emphasis",
    "GLRLM_SRHGE": "Short Run High Gray-Level Emphasis",
    "GLRLM_SRLGE": "Short Run Low Gray-Level Emphasis",
    "GLRLM_LRE": "Long Run Emphasis",
    "GLRLM_SRE": "Short Run Emphasis",
    # SHAPE
    "SHAPE_Sphericity": "Tumor Sphericity",
    "SHAPE_SurfaceVolumeRatio": "Surface to Volume Ratio",
    "SHAPE_Maximum3DDiameter": "Maximum 3D Tumor Diameter",
    "SHAPE_Volume": "Tumor Volume"
}

BIO_KNOWLEDGE_BASE = {
    "CEA": "Elevated levels are strongly associated with increased tumor burden and potential for micrometastasis.",
    "Psize": "Larger primary tumor size correlates with an increased likelihood of microscopic local invasion and lymphatic spread.",
    "SCC_Ag": "Serum levels reflect tumor proliferation, burden, and squamous cell differentiation.",
    "NSE": "Linked to neuroendocrine differentiation; elevated levels often indicate biologically aggressive phenotypes.",
    "Age": "Patient age influences immune surveillance, tissue microenvironment, and baseline metabolic rates.",
    "CONVENTIONAL_SUVbwmin": "Indicates the areas of lowest metabolic activity within the tumor, which may represent necrotic or poorly perfused regions.",
    "CONVENTIONAL_SUVbwmax": "Represents the peak glucose avidity (Warburg effect), strongly correlating with tumor aggressiveness and cellular proliferation.",
    "CONVENTIONAL_SUVbw": "Reflects the baseline metabolic activity and glucose consumption of the tumor tissue.",
    "CONVENTIONAL_HUmin": "Reflects the least dense areas within the tumor on CT, often corresponding to cavitation, necrosis, or cystic degeneration.",
    "CONVENTIONAL_HUmax": "Reflects the most dense solid components or calcifications within the primary tumor lesion.",
    "CONVENTIONAL_HU": "Reflects the overall physical density, solid components, and general attenuation characteristics of the tumor.",
    "GLZLM_LGZE": "Reflects the presence of large homogeneous areas; lower values suggest a chaotic, heterogeneous microenvironment.",
    "GLZLM_LZLGE": "Indicates the dominance of large regions with low attenuation or metabolism, pointing to extensive necrosis or ground-glass opacities.",
    "GLZLM_HGZE": "Highlights the distribution of high-density or highly metabolic active regions clustered within the tumor.",
    "GLZLM_SZE": "Reflects the proportion of small, fine texture zones, often correlating with a highly chaotic and erratic micro-architecture.",
    "GLZLM_SZHGE": "Indicates multiple small areas of intense density or metabolism, a marker of erratic and aggressive cellular proliferation.",
    "GLCM_Correlation": "Measures the linear dependency of gray levels; atypical values corroborate a complex, non-uniform spatial cellular arrangement.",
    "GLCM_Contrast": "Measures local variations and sharp transitions in density or metabolism, signifying an unstructured and variable tumor matrix.",
    "GLCM_Entropy": "Serves as a quantitative surrogate for intratumoral heterogeneity and unpredictability in tissue structure.",
    "GLRLM_LRHGE": "Points to continuous, elongated bands of high-density or highly active tumor tissue.",
    "GLRLM_SRHGE": "Reflects fragmented, short streaks of high density or metabolism, indicating a disjointed and aggressive tumor architecture.",
    "GLRLM_SRLGE": "Indicates fragmented areas of low density, often associated with diffuse micro-necrosis or loose tissue structure.",
    "SHAPE_Sphericity": "Quantifies how closely the tumor resembles a perfect sphere; lower values indicate irregular, spiculated, or highly invasive margins.",
    "SHAPE_SurfaceVolumeRatio": "Higher ratios suggest irregular, complex, or spiculated tumor surfaces that interface more aggressively with surrounding healthy tissue."
}

def get_human_title(feature_name):
    base_name = feature_name.split('.')[0]
    base_name_clean = base_name.split('(')[0]
    
    if base_name_clean in FEATURE_TITLES:
        title = FEATURE_TITLES[base_name_clean]
    elif base_name in FEATURE_TITLES:
        title = FEATURE_TITLES[base_name]
    else:
        if ".CT" in feature_name:
            title = "CT Radiomic Structural Feature"
        elif ".PET" in feature_name:
            title = "PET Radiomic Metabolic Feature"
        else:
            title = "Clinical Biomarker"
            
    if ".PET" in feature_name and "PET" not in title and "Metabolic" not in title:
        title += " (PET)"
    elif ".CT" in feature_name and "CT" not in title and "Density" not in title:
        title += " (CT)"
        
    return title

def get_dynamic_description(feature_name):
    base_name = feature_name.split('.')[0]
    base_name_clean = base_name.split('(')[0]
    
    if base_name_clean in BIO_KNOWLEDGE_BASE:
        return BIO_KNOWLEDGE_BASE[base_name_clean]
    
    for key in sorted(BIO_KNOWLEDGE_BASE.keys(), key=len, reverse=True):
        if key in feature_name:
            return BIO_KNOWLEDGE_BASE[key]
            
    if ".CT" in feature_name:
        return "Morphological or densitometric CT feature associated with spatial tumor arrangement."
    elif ".PET" in feature_name:
        return "Metabolic PET feature reflecting glucose avidity and cellular metabolic rate."
    
    return "Clinical or phenotypic marker contributing to the risk profile."

def generate_clinical_takeaway(report_data, pred):
    risk_class = "High-Risk" if pred == 1 else "Low-Risk"
    
    top_feature = None
    if pred == 1 and report_data.get("elevators"):
        top_feature = report_data["elevators"][0] 
    elif pred == 0 and report_data.get("protectors"):
        top_feature = report_data["protectors"][0]
    else:
        all_features = report_data.get("elevators", []) + report_data.get("protectors", [])
        if all_features:
            top_feature = sorted(all_features, key=lambda x: abs(x['impact_pct']), reverse=True)[0]
            
    if not top_feature:
        return f"The model classified this patient as {risk_class}, but specific feature drivers could not be extracted."

    if top_feature in report_data.get("elevators", []):
        direction_phrase = f"elevated the patient's absolute risk by +{abs(top_feature['impact_pct']):.1f}%"
    else:
        direction_phrase = f"reduced the patient's absolute risk by {abs(top_feature['impact_pct']):.1f}%"
        
    desc = top_feature['description']
    desc_formatted = desc[0].lower() + desc[1:] if desc else "contributed to the risk profile."
    
    takeaway = (
        f"The machine learning model classified this patient as {risk_class} for Occult Lymph Node (OLN) Metastasis. "
        f"The primary driver determining this prediction is the {top_feature['human_title']} ({top_feature['name']}), "
        f"which {direction_phrase}. Biologically, this feature {desc_formatted}"
    )
    
    return takeaway

def generate_biological_narrative_data(patient_features, patient_shap, X_train, patient_proba, base_proba, feature_impacts, pred, threshold, top_k=8, ice_narratives=None):
    sorted_indices = np.argsort(np.abs(patient_shap))[::-1]
    top_indices = sorted_indices[:top_k]
    
    risk_level = "HIGH-RISK" if pred == 1 else "LOW-RISK"
    risk_ratio = (patient_proba / threshold) if threshold > 0 else 0.0
    
    margin = abs(patient_proba - threshold)
    if margin < 0.05:
        confidence = "Borderline / Low Confidence"
    elif margin < 0.15:
        confidence = "Moderate Confidence"
    else:
        confidence = "High Confidence"
        
    report_data = {
        "risk_status": risk_level,
        "patient_proba": patient_proba,
        "base_proba": base_proba,
        "risk_ratio": risk_ratio,
        "pred": pred,
        "confidence": confidence,
        "threshold": threshold,
        "elevators": [],
        "protectors": []
    }
        
    for idx in top_indices:
        feature = patient_features.index[idx]
        val = patient_features.iloc[idx]
        shap_val = patient_shap[idx]
        impact_pct = feature_impacts.get(feature, 0.0)
        
        # Calculate cohort percentile AND determine two-sided extremity (Top / Bottom tail)
        if X_train is not None and feature in X_train.columns:
            feature_array = X_train[feature].dropna()
            if len(feature_array) > 0:
                percentile = (feature_array < val).mean() * 100
                
                # Classify the extremity for the clinical reader
                if percentile >= 75:
                    top_val = 100 - percentile
                    p_label = f"Top {top_val:.0f}%" if top_val >= 1 else "Top <1%"
                elif percentile <= 25:
                    p_label = f"Bottom {percentile:.0f}%" if percentile >= 1 else "Bottom <1%"
                else:
                    p_label = f"Avg ({percentile:.0f}th %ile)"
                    
                percentile_str = p_label
            else:
                percentile_str = "N/A"
        else:
            percentile_str = "N/A"
        
        feature_dict = {
            "name": feature,
            "human_title": get_human_title(feature),
            "value": round(val, 2) if isinstance(val, float) else val,
            "percentile": percentile_str,
            "impact_pct": impact_pct,
            "description": get_dynamic_description(feature)
        }
        
        if shap_val > 0:
            report_data["elevators"].append(feature_dict)
        else:
            report_data["protectors"].append(feature_dict)
            
    report_data["elevators"] = sorted(report_data["elevators"], key=lambda x: abs(x['impact_pct']), reverse=True)
    report_data["protectors"] = sorted(report_data["protectors"], key=lambda x: abs(x['impact_pct']), reverse=True)
    report_data["clinical_takeaway"] = generate_clinical_takeaway(report_data, pred)
    
    report_data["ice_narratives"] = ice_narratives or {}

    return report_data

def print_feature_row(pdf, item):
    start_x = pdf.get_x()
    
    # 1. Bold Title
    pdf.set_font("Arial", 'B', 11)
    title_text = item['human_title']
    title_width = pdf.get_string_width(title_text)
    pdf.cell(title_width + 1, 6, txt=title_text)
    
    # 2. Regular Name in Parentheses
    pdf.set_font("Arial", '', 11)
    name_text = f" ({item['name']})"
    name_width = pdf.get_string_width(name_text)
    pdf.cell(name_width + 1, 6, txt=name_text)
    
    # 3. Right-Aligned Value, Extremity Percentile & Probability
    val_text = f"Value: {item['value']} [{item['percentile']}] ({item['impact_pct']:+.1f}%)"
    remaining_width = 190 - (title_width + 1 + name_width + 1)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(remaining_width, 6, txt=val_text, align='R')
    pdf.ln(6)
    
    # 4. Description on a new line
    pdf.set_font("Arial", 'I', 10)
    pdf.set_x(start_x + 5) 
    pdf.multi_cell(185, 5, txt=item['description'])
    pdf.ln(3) 

def create_patient_pdf_report(patient_id, report_data, plot_image_path, output_dir):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- 1. Header & Summary ---
    pdf.set_font("Arial", 'B', 15)
    pdf.cell(0, 8, txt=f"Clinical Risk Assessment Report: Patient {patient_id}", ln=True, align='L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="1. Primary Outcome & Risk Summary", ln=True)
    pdf.set_font("Arial", '', 11)
    
    pdf.cell(0, 6, txt=f"- Target: Occult Lymph Node (OLN) Metastasis", ln=True)
    pdf.cell(0, 6, txt=f"- Model Prediction: {'Positive for Metastasis' if report_data['pred'] == 1 else 'Negative for Metastasis'}", ln=True)
    pdf.cell(0, 6, txt=f"- Prediction Confidence: {report_data['confidence']}", ln=True)
    pdf.cell(0, 6, txt=f"- Absolute Predicted Risk: {report_data['patient_proba']*100:.1f}% (Diagnostic Threshold: {report_data['threshold']*100:.1f}%)", ln=True)
    pdf.cell(0, 6, txt=f"- Relative Risk Multiplier: {report_data['risk_ratio']:.2f}x the diagnostic threshold", ln=True)
    
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, txt="Clinical Takeaway:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, txt=report_data['clinical_takeaway'])
    pdf.ln(6)

    # --- BORDERLINE ANALYSIS ---
    margin = abs(report_data['patient_proba'] - report_data['threshold'])
    
    # FIX: Tie this strictly to the confidence string instead of a hardcoded margin
    if report_data['confidence'] == "Borderline / Low Confidence" and report_data.get('ice_narratives'):
        pdf.ln(4)
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(200, 50, 50) # Dark red for emphasis
        pdf.cell(0, 6, txt="Clinical Sensitivity Analysis (Borderline Patient):", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 10)
        
        intro_text = "PREDICTION INSTABILITY WARNING: This patient's risk profile sits precisely on the borderline. Consequently, the model's prediction is highly sensitive. Minor biological variations or measurement errors in the following features would flip the final clinical prediction:"
        pdf.multi_cell(0, 5, txt=intro_text)
        pdf.ln(2)
        
        for feat_name, ice_data in report_data['ice_narratives'].items():
            human_name = get_human_title(feat_name)
            current = ice_data['current_raw']
            tipping = ice_data['tipping_point_raw']
            direction = ice_data['direction']
            pct_change = ice_data.get('pct_change', 0.0) 
            
            if abs(current) >= 10000 or (abs(current) < 0.01 and current != 0):
                current_str = f"{current:.2e}"
                tipping_str = f"{tipping:.2e}"
            else:
                current_str = f"{current:.2f}"
                tipping_str = f"{tipping:.2f}"

            # narrative = (
            #     f"- {human_name}: Currently at {current_str}. "
            #     f"A {pct_change:.1f}% {direction} (to {tipping_str}) "
            #     f"would alter the prediction outcome."
            # )
            narrative = (
                f"- {human_name}: "
                f"{direction} "
                f"would alter the prediction outcome."
            )
            pdf.multi_cell(0, 5, txt=narrative)
            
        pdf.ln(4)

    # --- 2. Elevators ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="2. Key Pathological & Imaging Drivers (Elevating Risk)", ln=True)
    
    if not report_data["elevators"]:
        pdf.set_font("Arial", 'I', 11)
        pdf.cell(0, 6, txt="No major risk-elevating features observed in the top drivers.", ln=True)
        pdf.ln(2)
    else:
        for item in report_data["elevators"]:
            print_feature_row(pdf, item)
            
    # --- 3. Protectors ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="3. Protective / Mitigating Factors (Reducing Risk)", ln=True)
    
    if not report_data["protectors"]:
        pdf.set_font("Arial", 'I', 11)
        pdf.cell(0, 6, txt="No major risk-reducing features observed in the top drivers.", ln=True)
    else:
        for item in report_data["protectors"]:
            print_feature_row(pdf, item)

    # --- Clinical Disclaimer Footer on Page 1 ---
    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    disclaimer = "DISCLAIMER: This report is generated by an investigational machine learning model. It is intended to provide adjunctive biological insights and must not replace standard clinical, radiological, or pathological evaluation."
    pdf.multi_cell(0, 4, txt=disclaimer, align='C')
    pdf.set_text_color(0, 0, 0) # reset color

    # --- PAGE 2: SHAP Plot ---
    if os.path.exists(plot_image_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="Machine Learning Feature Attribution (SHAP Decision Path)", ln=True, align='C')
        pdf.ln(5)
        pdf.image(plot_image_path, x=15, w=180)
    
    # --- PAGE 3: WHAT-IF SENSITIVITY PLOTS ---
    # FIX: Adding the ICE plots if they exist
    if report_data.get('ice_narratives'):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="Sensitivity Analysis: Feature Tipping Points", ln=True, align='C')
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 5, txt="The plots below demonstrate how variations in specific clinical or radiomic features affect the model's predicted risk. The red marker indicates the patient's current clinical state. Crossing the red dashed line changes the diagnosis.", align='C')
        pdf.ln(5)

        ice_dir = os.path.join(output_dir, "ice")
        y_offset = pdf.get_y()
        
        # Plot up to 4 What-If charts in a 2x2 grid
        for i, (feat_name, _) in enumerate(list(report_data['ice_narratives'].items())[:4]):
            clean_name = str(feat_name).replace("/", "_").replace("\\", "_")
            img_path = os.path.join(ice_dir, f"ICE_{patient_id}_{clean_name}.png")
            
            if os.path.exists(img_path):
                x_pos = 10 if i % 2 == 0 else 105
                if i > 1 and i % 2 == 0: y_offset += 75
                pdf.image(img_path, x=x_pos, y=y_offset, w=95)


    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"Clinical_Report_Patient_{patient_id}.pdf")
    pdf.output(pdf_path)
    print(f" - Biological report saved for Patient {patient_id}: {pdf_path}")