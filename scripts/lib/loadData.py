"""
Data Preparation & Integration Module
-------------------------------------
This module handles the complex workflow of loading clinical and genomic data,
performing outer joins across multiple modalities, and cleaning features for
ML tasks (binary, multiclass, or survival analysis).
"""

import pandas as pd
from pathlib import Path
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
import yaml

# --- CONFIGURATION LOADING ---
# Load settings from YAML to control paths, feature selection, and analysis parameters.
config = yaml.safe_load(open("./scripts/config.yml"))

# Flatten FEATURES_TO_DROP list in case it contains nested lists from the YAML
config['FEATURES_TO_DROP'] = [
    element for sublist in config['FEATURES_TO_DROP'] 
    for element in (sublist if isinstance(sublist, list) else [sublist])
]

def load_and_prepare_data(mutation_path, target_column, positive_class, analysis_type):
    """
    Loads, merges, and cleans clinical and genomic data for machine learning.

    Workflow:
    1. Iteratively loads clinical TSVs and merges them on patient IDs.
    2. Identifies and merges the 'Target' column if missing from initial files.
    3. Merges mutation data (genomic matrix) with the clinical master table.
    4. Handles analysis-specific target encoding (Survival, Binary, or Multiclass).
    5. Processes 'binned' variables from a data dictionary into categorical columns.
    6. Standardizes data types (Numeric vs. Categorical).

    Args:
        mutation_path (str): Path to the TSV file containing mutation/genomic data.
        target_column (str): The name of the label or outcome variable.
        positive_class (str): The value to treat as '1' in binary classification.
        analysis_type (str): 'binary', 'multiclass', or 'survival'.

    Returns:
        tuple: (X, X_display, y, class_names, mappings, modality_features)
    """
    print("Step 1: Loading and preparing data...")
    
    clinical_dir = Path(config['CLINICAL_DATA_DIR'])
    clinical_files = config['CLINICAL_FILES']
    
    # Resolve ID mapping: clinical_id (e.g., Patient ID) vs genomic_id (e.g., Sample ID)
    clinical_id = config.get('CLINICAL_ID', 'pt_record_id')
    genomic_id = config.get('GENOMIC_ID', clinical_id) 
    print(f"Using Clinical ID: {clinical_id}, Genomic ID: {genomic_id}")
    
    clinical_df = None
    modality_features = {} # Tracks which features came from which clinical file
    
    # --- 1. CLINICAL DATA LOADING & MERGING ---
    for file_name in clinical_files:
        file_path = clinical_dir / f"{file_name}_data.tsv"
        print(f"Loading clinical file: {file_path}")
        try:
            df = pd.read_csv(file_path, sep='\t')
            # Sanitize column names: replace special characters with underscores
            # df.columns = [re.sub(r'[\[\]<>()\s,]', '_', col) for col in df.columns]

            # Track features associated with this specific modality
            new_features = [c for c in df.columns if c not in [clinical_id, genomic_id, target_column]]
            if file_name not in modality_features:
                modality_features[file_name] = []
            modality_features[file_name].extend(new_features)

            # Standardize IDs as strings to prevent merge mismatches
            if clinical_id in df.columns:
                df[clinical_id] = df[clinical_id].astype(str)
            if genomic_id != clinical_id and genomic_id in df.columns:
                df[genomic_id] = df[genomic_id].astype(str)
            
            # Merge logic: use 'outer' join to keep all patients from all files
            if clinical_df is None:
                clinical_df = df
            else:
                clinical_df = pd.merge(clinical_df, df, on=clinical_id, how='outer', suffixes=('', '_dup'))
                
                # Coalesce genomic_id if it appeared as a duplicate during the merge
                if genomic_id != clinical_id:
                     dup_col = f"{genomic_id}_dup"
                     if genomic_id in clinical_df.columns and dup_col in clinical_df.columns:
                         clinical_df[genomic_id] = clinical_df[genomic_id].fillna(clinical_df[dup_col])
                         clinical_df.drop(columns=[dup_col], inplace=True)
                     elif dup_col in clinical_df.columns and genomic_id not in clinical_df.columns:
                         clinical_df.rename(columns={dup_col: genomic_id}, inplace=True)

        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Skipping.")
    
    if clinical_df is None:
        raise ValueError("No clinical data loaded. Check config and file paths.")

    # --- 2. TARGET COLUMN RETRIEVAL ---
    # If target is not in merged clinical files, look for it in a dedicated file based on prefix
    # if target_column not in clinical_df.columns:
    #     prefix = target_column.split('.')[0]
    #     potential_file = f"{prefix}_Clinical.tsv"
    #     potential_path = clinical_dir / potential_file
        
    #     if potential_path.exists():
    #         target_df = pd.read_csv(potential_path, sep='\t')
    #         # Extract only the keys and target column
    #         cols_to_load = [clinical_id, target_column]
    #         if genomic_id != clinical_id and genomic_id in target_df.columns:
    #             cols_to_load.append(genomic_id)
            
    #         clinical_df = pd.merge(clinical_df, target_df[cols_to_load], on=clinical_id, how='left', suffixes=('', '_dup'))
    if target_column not in clinical_df.columns:
        found = False
        # 1. Iterate through all files ending in _Clinical.tsv
        for potential_path in clinical_dir.glob("*_data.tsv"):
            
            # 2. Peak at the header first (efficient for large files)
            try:
                header = pd.read_csv(potential_path, sep='\t', nrows=0).columns
            except Exception:
                continue # Skip files that can't be read
                
            if target_column in header:
                print(f"  Target '{target_column}' found in {potential_path.name}. Merging...")
                
                # 3. Load only the necessary columns
                cols_to_load = [clinical_id, target_column]
                # Check if genomic_id exists in this specific file before adding
                if genomic_id != clinical_id:
                    # We check the 'header' we just read to see if genomic_id is present
                    if genomic_id in header:
                        cols_to_load.append(genomic_id)
                
                target_df = pd.read_csv(potential_path, sep='\t', usecols=cols_to_load)
                
                # 4. Merge into the main dataframe
                # We drop duplicates on clinical_id to prevent row explosion during merge
                target_df = target_df.drop_duplicates(subset=[clinical_id])
                clinical_df = pd.merge(
                    clinical_df, 
                    target_df, 
                    on=clinical_id, 
                    how='left', 
                    suffixes=('', '_dup')
                )
                
                found = True
                break  # Exit loop once the target is found and merged

        if not found:
            print(f"  Warning: Column '{target_column}' was not found in any clinical files.")

    # --- 3. MUTATION (GENOMIC) DATA MERGING ---
    print(f"Loading mutation data from: {mutation_path}")
    mut_matrix_transposed = pd.DataFrame()
    try:
        mut_matrix_binary = pd.read_csv(mutation_path, sep='\t', index_col=0)
        mut_matrix_transposed = mut_matrix_binary.T # Patients as rows, genes as columns
        mut_matrix_transposed.columns = [col.replace('-', '_') for col in mut_matrix_transposed.columns]
        modality_features['genomics'] = mut_matrix_transposed.columns.to_list()
    except Exception as e:
        print(f"Mutation loading failed: {e}")

    # Final Master Merge
    if not mut_matrix_transposed.empty and genomic_id in clinical_df.columns:
        master_df = pd.merge(clinical_df, mut_matrix_transposed, left_on=genomic_id, right_index=True, how="left")
    else:
        master_df = clinical_df

    # --- 4. TARGET CLEANING & SUBSETTING ---
    if analysis_type == 'survival':
        master_df.dropna(subset=['OS_time', 'OS'], inplace=True)
    else:
        # Filter for specific classes if specified in config
        if config['CLASS2KEEP']:
            master_df = master_df[master_df[target_column].isin(config['CLASS2KEEP'])]
        master_df.dropna(subset=[target_column], inplace=True)

    # --- 5. ENCODING TARGET (y) ---
    if analysis_type == 'binary':
        if config['BIN_MERGE'] and master_df[target_column].dtype!='int':
            # Force target into 'positive' vs 'not positive'
            positive_mask = master_df[target_column] == positive_class
            master_df.loc[~positive_mask, target_column] = f'not_{positive_class}'
            class_names = [f'not_{positive_class}', positive_class]
        else:
            class_names = ['0', '1']
        master_df['target'] = (master_df[target_column] == positive_class).astype(int)
        y = master_df.set_index(clinical_id)['target']

    elif analysis_type == 'multiclass':
        le = LabelEncoder()
        master_df['target'] = le.fit_transform(master_df[target_column].astype(str))
        class_names = le.classes_
        y = master_df.set_index(clinical_id)['target']

    elif analysis_type == 'survival':
        y = master_df[[clinical_id, 'OS_time', 'OS']].set_index(clinical_id)
        class_names = ['Low Risk', 'High Risk']

    # --- 6. FEATURE ENGINEERING (BINNED MERGING) ---
    # Uses 'dict.csv' to find multiple binary columns that should be one categorical column
    try:
        data_dict = pd.read_csv(clinical_dir / "dict.csv")
        # --- 1. Identify and Group Binned Variables ---
        if not data_dict.empty:
            # Filter for binned entries and group by their prefix (base variable name)
            binned_entries = data_dict[data_dict['type'] == 'binned'].copy()
            binned_groups = {}
            valid_columns = master_df.columns.tolist()
            binned_entries = binned_entries[binned_entries['label'].isin(valid_columns)]

            for _, row in binned_entries.iterrows():
                full_col_name = f"{row['group']}.{row['label']}"
                if '___' in full_col_name:
                    prefix, suffix = full_col_name.split('___', 1)
                    if prefix not in binned_groups:
                        binned_groups[prefix] = []
                    binned_groups[prefix].append((full_col_name, suffix))

            # --- 2. Merge Binned Variables in master_df ---
            for new_col, components in binned_groups.items():
                print(f"  Merging binary flags into categorical: {new_col}...")
                
                # Initialize the new categorical column with NaN
                master_df[new_col] = np.nan
                cols_to_drop = []

                for old_col, suffix in components:
                    if old_col in master_df.columns:
                        # Identify rows where the flag is 1
                        is_present = pd.to_numeric(master_df[old_col], errors='coerce') == 1
                        master_df.loc[is_present, new_col] = suffix
                        cols_to_drop.append(old_col)
                
                # Clean up the original one-hot encoded columns
                master_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    except FileNotFoundError:
        print("Warning: data/dict.csv not found. Skipping dictionary-based conversion.")
        data_dict = pd.DataFrame()

    # --- 7. TYPE CASTING & MAPPING ---
    # Final cleanup: Ensures numeric columns are floats and categorical columns are factors
    X = master_df.drop(columns=['OS_time', 'OS', 'target', target_column], errors='ignore').set_index(clinical_id)
    
    # (Mappings logic stores means/stdevs or category levels for potential pipeline deployment)
    numerical_mappings = {}
    categorical_mappings = {}
    gene_mappings = {}
    
    # Map column names to types for high-speed lookup
    type_lookup = {}
    if not data_dict.empty:
        for _, row in data_dict.iterrows():
            name = f"{row['label']}"
            # If it was binned, the merged column name is the prefix (part before ___)
            if row['type'] == 'binned' and '___' in name:
                type_lookup[name.split('___')[0]] = 'factor'
            else:
                type_lookup[name] = row['type']

    print("  Applying data types and calculating statistics...")
    for col in X.columns:
        dtype = type_lookup.get(col)

        if dtype == 'numeric':
            X[col] = pd.to_numeric(X[col], errors='coerce')
            numerical_mappings[col] = {
                'mean': X[col].mean(), 
                'std': X[col].std()
            }
        elif dtype in ['factor', 'bin']:
            X[col] = X[col].astype('category')
            categorical_mappings[col] = X[col].cat.categories.tolist()
        elif dtype == 'txt':
            # Text fields are generally excluded from ML models unless used for NLP
            pass
        else:
            # Fallback: Heuristic typing for columns not found in dictionary
            if col in mut_matrix_transposed.columns:
                gene_mappings[col] = {'mut': X[col].sum()}
            elif X[col].dtype == 'object':
                print(f"    Notice: Auto-casting '{col}' to category.")
                X[col] = X[col].astype('category')
            else:
                # Store numeric stats for pre-typed numerical columns (e.g. genomic)
                numerical_mappings[col] = {'mean': X[col].mean(), 'std': X[col].std()}

    X_display = X.copy()
    mappings = [categorical_mappings, numerical_mappings, gene_mappings]

    print(f"Data prepared successfully. Predicting '{target_column}'")
    return X, X_display, y, class_names, mappings, modality_features
