"""
DSF (Differential Scanning Fluorimetry) Analysis Package
=======================================================

This module provides a comprehensive analysis pipeline for DSF data, including:
- Data loading and preprocessing
- Melting curve analysis
- Model fitting and comparison
- Statistical analysis
- Visualization of results

This is an optimized version of the original notebook with:
- Improved memory management
- Cached negative control analysis
- More efficient data structures
- Support for UniProt ID mapping
- Support for SMILES code visualization
- Better documentation and code organization

Required Dependencies:
--------------------
- pandas: Data manipulation and analysis
- matplotlib: Basic plotting
- numpy: Numerical computations
- scipy: Scientific computing and optimization
- plotly: Interactive visualizations
- rdkit: Chemical structure visualization (for SMILES)
"""

# Standard library imports
import os
import glob
import re
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter, find_peaks
from sklearn.metrics import r2_score

# Optional imports for SMILES visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import IPythonConsole
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Global cache for negative control fits
# Structure: {(protein, plate, rep, neg_control_well): best_fit_result}
NEG_CONTROL_CACHE = {}

# Global cache for model fits
# Structure: {(protein, plate, rep, well): {model_name: {window_size: fit_result}}}
MODEL_FIT_CACHE = {}

# Global cache for derivative data
# Structure: {(protein, plate, rep, well): derivative_df}
DERIVATIVE_CACHE = {}

# Global mapping dictionaries
PROTEIN_TO_UNIPROT = {}
COMPOUND_TO_SMILES = {}

# --- DATA LOADING FUNCTIONS ---

def unify_plate_id(raw_plate_id):
    """
    Convert 'PM2A' -> 'PM2', 'PM3B' -> 'PM3', etc. 
    If it doesn't match pattern 'PM\\d+', return as-is.
    
    Parameters:
    -----------
    raw_plate_id : str
        The raw plate identifier
        
    Returns:
    --------
    str
        Unified plate identifier
    """
    match = re.match(r'^(PM\d+)', raw_plate_id)
    if match:
        return match.group(1)
    return raw_plate_id

def parse_protein_plate_from_subfolder(subfolder_name):
    """
    Extract protein and plate ID from subfolder name.
    E.g. 'B5_PM2A' -> (protein='B5', plate_id='PM2')
    
    Parameters:
    -----------
    subfolder_name : str
        Name of the subfolder
        
    Returns:
    --------
    tuple
        (protein, plate_id)
    """
    parts = subfolder_name.split("_")
    if len(parts) >= 2:
        protein = parts[0]
        raw_plate_id = parts[1]
        plate_id = unify_plate_id(raw_plate_id)
    else:
        protein = subfolder_name
        plate_id = "UnknownPlate"
    return protein, plate_id

def parse_replicate_from_filename(file_name):
    """
    Look for a pattern like '_2.eds.csv' to treat as replicate #2 -> 'Rep02'.
    
    Parameters:
    -----------
    file_name : str
        Name of the file
        
    Returns:
    --------
    str
        Replicate identifier (e.g., 'Rep02')
    """
    match = re.search(r"_([0-9]+)\.eds\.csv$", file_name)
    match2 = re.search(r"_trial([0-9]+)_amb\.eds\.csv$", file_name)
    match3 = re.search(r"_trial([0-9]+)_amb\.eds\.txt$", file_name)
    
    if match:
        rep_number_str = match.group(1)
        rep_number = int(rep_number_str)
        return f"Rep{str(rep_number).zfill(2)}"
    elif match2: 
        rep_number_str = match2.group(1)
        rep_number = int(rep_number_str)
        return f"Rep{str(rep_number).zfill(2)}"
    elif match3:
        rep_number_str = match3.group(1)
        rep_number = int(rep_number_str)
        return f"Rep{str(rep_number).zfill(2)}"
    return "Rep01"

def split_raw_derivative_boltzmann_data(file_path):
    """
    Splits a multi-segmented data file into (raw_data, derivative_data, boltzmann_data).
    Assumes each segment is demarcated by lines like 'Derivative' or 'Boltzmann Temperature'.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
        
    Returns:
    --------
    tuple
        (raw_data_df, derivative_data_df, boltzmann_data_df)
    """
    # Read the file once to identify section boundaries
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()

    boltzmann_start = None
    derivative_start = None
    
    for i, line in enumerate(lines):
        if "Boltzmann Temperature" in line and "Boltzmann Fluorescence" in line:
            boltzmann_start = i
        if "Derivative" in line:
            derivative_start = i
            
    if boltzmann_start is None:
        raise ValueError(f"No Boltzmann data found in the file: {file_path}")
    if derivative_start is None:
        raise ValueError(f"No Derivative data found in the file: {file_path}")
    
    # Determine file format (txt or csv)
    delimiter = ',' if file_path.endswith('.csv') else '\t'
    
    # Read the raw data up to the derivative section
    raw_data = pd.read_csv(
        file_path, 
        sep=delimiter, 
        encoding='ISO-8859-1', 
        nrows=derivative_start-2   # exclude header lines for derivative section
    )

    # Derivative data from derivative_start to right before boltzmann
    derivative_data = pd.read_csv(
        file_path, 
        sep=delimiter, 
        encoding='ISO-8859-1', 
        skiprows=derivative_start, 
        nrows=boltzmann_start - derivative_start - 2
    )
    
    # Boltzmann data from boltzmann_start onward
    boltzmann_data = pd.read_csv(
        file_path, 
        sep=delimiter, 
        encoding='ISO-8859-1', 
        skiprows=boltzmann_start - 1
    )
    
    return raw_data, derivative_data, boltzmann_data

def load_plate_map_csvs(plate_map_folder):
    """
    Finds all CSVs in plate_map_folder, e.g. 'PM2A_PlateMap_CSV.csv',
    unifies the plate_id, and loads into plate_map_dict[plate_id] = DataFrame
    
    Parameters:
    -----------
    plate_map_folder : str
        Path to the folder containing plate map CSV files
        
    Returns:
    --------
    dict
        Dictionary mapping plate IDs to plate map DataFrames
    """
    plate_map_dict = {}
    csv_files = glob.glob(os.path.join(plate_map_folder, "*.csv"))
    
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)
        raw_plate = base_name.split("_")[0]  # e.g. 'PM2A'
        plate_id = unify_plate_id(raw_plate)  # unify it
        df_map = pd.read_csv(csv_file)
        plate_map_dict[plate_id] = df_map
    
    return plate_map_dict

def build_well_to_ligand_map(plate_map_df, well_col="Well", ligand_col="Ligand", smiles_col=None):
    """
    From a plate map DataFrame, build { well: ligand_name } and optionally { well: smiles }
    
    Parameters:
    -----------
    plate_map_df : DataFrame
        DataFrame containing plate map information
    well_col : str, optional
        Column name for well positions
    ligand_col : str, optional
        Column name for ligand names
    smiles_col : str, optional
        Column name for SMILES codes
        
    Returns:
    --------
    tuple
        (well_to_ligand, well_to_smiles) dictionaries
    """
    well_to_ligand = {}
    well_to_smiles = {}
    
    for idx, row in plate_map_df.iterrows():
        w = str(row[well_col]).strip()
        lig = str(row[ligand_col]).strip()
        well_to_ligand[w] = lig
        
        # If SMILES column is provided, also build well_to_smiles mapping
        if smiles_col is not None and smiles_col in plate_map_df.columns:
            if pd.notna(row[smiles_col]):
                smiles = str(row[smiles_col]).strip()
                well_to_smiles[w] = smiles
                # Also update global COMPOUND_TO_SMILES dictionary
                if lig != "DMSO" and lig != "Buffer":
                    COMPOUND_TO_SMILES[lig] = smiles
    
    return well_to_ligand, well_to_smiles

def load_data_with_derivative_boltzmann(parent_folder, plate_map_folder, 
                                        protein_filter=None, plate_filter=None, 
                                        well_filter=None, use_uniprot=False,
                                        use_smiles=False):
    """
    Load DSF data from files and organize into nested dictionaries.
    
    Parameters:
    -----------
    parent_folder : str
        Path to the parent folder containing DSF data
    plate_map_folder : str
        Path to the folder containing plate map CSV files
    protein_filter : list, optional
        List of proteins to include (if None, include all)
    plate_filter : list, optional
        List of plates to include (if None, include all)
    well_filter : list, optional
        List of wells to include (if None, include all)
    use_uniprot : bool, optional
        Whether to use UniProt IDs for protein names
    use_smiles : bool, optional
        Whether to load SMILES codes for compounds
        
    Returns:
    --------
    tuple
        (raw_data_dfs, derivative_data_dfs, boltzmann_data_dfs, plate_map_dict)
    """
    # Load plate maps first
    plate_map_dict = load_plate_map_csvs(plate_map_folder)
    
    # Initialize data dictionaries
    raw_data_dfs = {}
    derivative_data_dfs = {}
    boltzmann_data_dfs = {}
    
    # Build well-to-ligand mappings for each plate
    plate_to_well_ligand_map = {}
    plate_to_well_smiles_map = {}
    
    for plate_id, plate_map_df in plate_map_dict.items():
        smiles_col = "SMILES" if use_smiles and "SMILES" in plate_map_df.columns else None
        well_to_ligand, well_to_smiles = build_well_to_ligand_map(
            plate_map_df, smiles_col=smiles_col)
        plate_to_well_ligand_map[plate_id] = well_to_ligand
        if use_smiles:
            plate_to_well_smiles_map[plate_id] = well_to_smiles
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if (file.endswith('.csv') or file.endswith('.txt')) and ('RawData' in file):
                file_path = os.path.join(root, file)
                
                # Extract protein, plate_id, and rep_id from path and filename
                rel_path = os.path.relpath(root, parent_folder)
                subfolder_parts = rel_path.split(os.sep)
                
                if len(subfolder_parts) >= 1:
                    protein, plate_id = parse_protein_plate_from_subfolder(subfolder_parts[0])
                    
                    # Apply filters if provided
                    if protein_filter and protein not in protein_filter:
                        continue
                    if plate_filter and plate_id not in plate_filter:
                        continue
                    
                    # Map protein to UniProt ID if requested and available
                    if use_uniprot and protein in PROTEIN_TO_UNIPROT:
                        protein = PROTEIN_TO_UNIPROT[protein]
                    
                    rep_id = parse_replicate_from_filename(file)
                    
                    try:
                        # Split the file into raw, derivative, and boltzmann data
                        raw_df, derivative_df, boltzmann_df = split_raw_derivative_boltzmann_data(file_path)
                        
                        # Initialize nested dictionaries if needed
                        if protein not in raw_data_dfs:
                            raw_data_dfs[protein] = {}
                            derivative_data_dfs[protein] = {}
                            boltzmann_data_dfs[protein] = {}
                        
                        if plate_id not in raw_data_dfs[protein]:
                            raw_data_dfs[protein][plate_id] = {}
                            derivative_data_dfs[protein][plate_id] = {}
                            boltzmann_data_dfs[protein][plate_id] = {}
                        
                        if rep_id not in raw_data_dfs[protein][plate_id]:
                            raw_data_dfs[protein][plate_id][rep_id] = {}
                            derivative_data_dfs[protein][plate_id][rep_id] = {}
                            boltzmann_data_dfs[protein][plate_id][rep_id] = {}
                        
                        # Split by well position and store in the nested dictionaries
                        for well, well_df in split_by_well_position(raw_df).items():
                            # Apply well filter if provided
                            if well_filter and well not in well_filter:
                                continue
                                
                            # Add ligand information if available
                            if plate_id in plate_to_well_ligand_map and well in plate_to_well_ligand_map[plate_id]:
                                ligand = plate_to_well_ligand_map[plate_id][well]
                                well_df['Ligand'] = ligand
                            
                            raw_data_dfs[protein][plate_id][rep_id][well] = well_df
                        
                        for well, well_df in split_by_well_position(derivative_df).items():
                            if well_filter and well not in well_filter:
                                continue
                            derivative_data_dfs[protein][plate_id][rep_id][well] = well_df
                        
                        for well, well_df in split_by_well_position(boltzmann_df).items():
                            if well_filter and well not in well_filter:
                                continue
                            boltzmann_data_dfs[protein][plate_id][rep_id][well] = well_df
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    # Force garbage collection to free memory
    gc.collect()
    
    return raw_data_dfs, derivative_data_dfs, boltzmann_data_dfs, plate_map_dict

def split_by_well_position(df, well_column="Well Position"):
    """
    Splits the given DataFrame by 'Well Position' -> { well: df_well, ... }
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to split
    well_column : str, optional
        Column name for well positions
        
    Returns:
    --------
    dict
        Dictionary mapping well positions to DataFrames
    """
    well_dfs = {}
    if well_column in df.columns:
        for well in df[well_column].unique():
            subset = df[df[well_column] == well].copy()
            well_dfs[well] = subset
    return well_dfs

def get_ligand_for_plate_well(raw_data_dfs, plate, well):
    """
    Get the ligand name for a specific plate and well.
    
    Parameters:
    -----------
    raw_data_dfs : dict
        Nested dictionary of raw data
    plate : str
        Plate identifier
    well : str
        Well position
        
    Returns:
    --------
    str
        Ligand name or "Unknown"
    """
    for protein in raw_data_dfs:
        if plate in raw_data_dfs[protein]:
            for rep in raw_data_dfs[protein][plate]:
                if well in raw_data_dfs[protein][plate][rep]:
                    df = raw_data_dfs[protein][plate][rep][well]
                    if 'Ligand' in df.columns:
                        return df['Ligand'].iloc[0]
    return "Unknown"

# --- MODEL FUNCTIONS ---

def s1_model(x, Asym, xmid, scal, d):
    """
    Single sigmoidal model.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    Asym, xmid, scal, d : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return Asym / (1 + np.exp((xmid - x) / scal)) * np.exp(d * (x - xmid))

def s1_d_model(x, Asym, xmid, scal, d, id_d, id_b):
    """
    Single sigmoidal model with decay.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    Asym, xmid, scal, d, id_d, id_b : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return s1_model(x, Asym, xmid, scal, d) + id_d * np.exp(id_b * x)

def s2_model(x, Asym, xmid, scal, d, Asym2, xmid2, scal2, d2):
    """
    Double sigmoidal model.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    Asym, xmid, scal, d, Asym2, xmid2, scal2, d2 : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return s1_model(x, Asym, xmid, scal, d) + s1_model(x, Asym2, xmid2, scal2, d2)

def s2_d_model(x, Asym, xmid, scal, d, Asym2, xmid2, scal2, d2, id_d, id_b):
    """
    Double sigmoidal model with decay.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    Asym, xmid, scal, d, Asym2, xmid2, scal2, d2, id_d, id_b : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return s2_model(x, Asym, xmid, scal, d, Asym2, xmid2, scal2, d2) + id_d * np.exp(id_b * x)

def modified_boltzmann_model(x, A, B, C, D, xmid, E):
    """
    A two-slope-plus-sigmoid model ("modified Boltzmann"), adapted to use
    normalized temperature x in [0..1]. Interpreting 'xmid' as the
    normalized Tm, and E as the 'width' or slope factor.
    
    The form is:
      y = A*x + B + (C*x + D) / (1 + exp((xmid - x)/E))
      
    Parameters:
    -----------
    x : array-like
        Independent variable (normalized temperature)
    A, B, C, D, xmid, E : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return A*x + B + (C*x + D) / (1 + np.exp((xmid - x) / E))

# Dictionary to store model definitions and parameters
MODEL_PARAMS = {
    "s1": {
        "model_func": s1_model,
        "p0": [1, 0.6, 0.03, -1],
        "bounds": ([0.1, 0.1, 0.01, -10], [5, 0.95, 1, 10]),
        "tm_idx": [1]
    },
    "s1_d": {
        "model_func": s1_d_model,
        "p0": [1, 0.6, 0.03, -1, 0.01, -5],
        "bounds": ([0.1, 0.1, 0.01, -10, 0.001, -20], [5, 0.95, 1, 10, 5, 0]),
        "tm_idx": [1]
    },
    "s2": {
        "model_func": s2_model,
        "p0": [1, 0.6, 0.03, -1, 0.5, 0.65, 0.03, -2],
        "bounds": ([0.01, 0.1, 0.01, -10, 0.01, 0.1, 0.01, -10],
                   [5, 0.95, 1, 10, 5, 0.95, 1, 10]),
        "tm_idx": [1, 5]
    },
    "s2_d": {
        "model_func": s2_d_model,
        "p0": [1, 0.6, 0.03, -1, 0.5, 0.65, 0.03, -2, 0.01, -5],
        "bounds": ([0.01, 0.1, 0.01, -10, 0.01, 0.1, 0.01, -10, 0.001, -20],
                   [5, 0.95, 1, 10, 5, 0.95, 1, 10, 5, 0]),
        "tm_idx": [1, 5]
    }, 
    "mB": {
        "model_func": modified_boltzmann_model,
        # Initial guess for [A, B, C, D, xmid, E]
        "p0": [0, 0, 0, 0, 0.5, 0.01],
        # Bounds are just an example; tweak to fit your data scale
        "bounds": ([-10, -10, -10, -10, 0.0, 1e-5], [10, 10, 10, 10, 1.0, 0.1]),
        # Tm is the 5th parameter in the list (index=4) 
        "tm_idx": [4]
    }
}

# --- DATA PRE-PROCESSING & DERIVATIVE FUNCTIONS ---

def normalize_df(df):
    """
    Normalize Temperature and Fluorescence, and store original bounds as attributes.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing Temperature and Fluorescence columns
        
    Returns:
    --------
    DataFrame
        DataFrame with added normalized columns and bounds attributes
    """
    df = df.copy()
    t_min, t_max = df["Temperature"].min(), df["Temperature"].max()
    f_min, f_max = df["Fluorescence"].min(), df["Fluorescence"].max()
    df["Temperature_norm"] = (df["Temperature"] - t_min) / (t_max - t_min)
    df["value_norm"] = (df["Fluorescence"] - f_min) / (f_max - f_min)
    df.attrs["T_bounds"] = (t_min, t_max)
    df.attrs["F_bounds"] = (f_min, f_max)
    return df

def find_derivative_peaks(df, window_size=11, derivative_threshold=0.0002):
    """
    Apply Savitzky-Golay filter to compute the first and second derivatives,
    and detect peaks in the first derivative.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing temperature and fluorescence data
    window_size : int, optional
        Window size for Savitzky-Golay filter
    derivative_threshold : float, optional
        Threshold for peak detection
        
    Returns:
    --------
    tuple
        (DataFrame with derivatives, peak indices)
    """
    # Check if result is already in cache
    for key, cached_df in DERIVATIVE_CACHE.items():
        if id(df) == id(cached_df):
            # Return cached result
            peaks, _ = find_peaks(cached_df["sgd1"], height=derivative_threshold, distance=20)
            return cached_df, peaks
    
    df = df.copy()
    # Ensure an odd window size and a minimum value
    window_size = max(5, window_size if window_size % 2 == 1 else window_size+1)
    
    if "Temperature_norm" not in df.columns or "value_norm" not in df.columns:
        df = normalize_df(df)
    
    df["sgd1"] = savgol_filter(df["value_norm"], window_size, polyorder=3, deriv=1)
    df["sgd2"] = savgol_filter(df["value_norm"], window_size, polyorder=3, deriv=2)
    peaks, _ = find_peaks(df["sgd1"], height=derivative_threshold, distance=20)
    
    # Cache the result
    DERIVATIVE_CACHE[id(df)] = df
    
    return df, peaks

def get_deriv_peak_temp(df):
    """
    Return the Temperature corresponding to the highest peak in the first derivative.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing temperature and fluorescence data
        
    Returns:
    --------
    float or None
        Temperature at the highest derivative peak, or None if no peaks found
    """
    df_with_deriv, peaks = find_derivative_peaks(df)
    if len(peaks) == 0:
        return None
    best_peak_idx = peaks[np.argmax(df_with_deriv["sgd1"].iloc[peaks].values)]
    return df_with_deriv["Temperature"].iloc[best_peak_idx]

def compute_derivative(df, window=11, polyorder=2):
    """
    Compute and add the first derivative to the dataframe.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing temperature and fluorescence data
    window : int, optional
        Window size for Savitzky-Golay filter
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filter
        
    Returns:
    --------
    DataFrame
        DataFrame with added derivative column
    """
    df = df.copy().sort_values("Temperature").reset_index(drop=True)
    x = df["Temperature"].values
    y = df["value_norm"].values
    delta = np.mean(np.diff(x)) if len(x) > 1 else 1.0
    window = max(5, window if window % 2 == 1 else window+1)
    deriv = savgol_filter(y, window_length=window, polyorder=polyorder, deriv=1, delta=delta)
    df["Derivative"] = deriv
    return df

def estimate_minor_peak_temp(df, t_min, t_max, edge_margin=5, window_size=11, major_peak=None):
    """
    Estimate a candidate minor peak (for Tm2) from the second derivative.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with temperature and fluorescence data
    t_min, t_max : float
        Temperature bounds for the current window
    edge_margin : float, optional
        Margin to exclude from edges
    window_size : int, optional
        Window size for Savitzky-Golay filter
    major_peak : float, optional
        Temperature of the major peak to avoid
        
    Returns:
    --------
    float or None
        Normalized temperature of the minor peak, or None if no candidate qualifies
    """
    # Work on a copy
    df = df.copy()
    # Compute first and second derivatives on normalized fluorescence
    df["sgd1"] = savgol_filter(df["value_norm"], window_size, polyorder=3, deriv=1)
    df["sgd2"] = savgol_filter(df["value_norm"], window_size, polyorder=3, deriv=2)
    
    # Only consider points that are not near the edges
    valid_idxs = [i for i in range(len(df))
                  if (df["Temperature"].iloc[i] >= t_min + edge_margin and 
                      df["Temperature"].iloc[i] <= t_max - edge_margin)]
    if not valid_idxs:
        return None

    # Find valleys in sgd2 (i.e. peaks in the inverted signal)
    valleys, _ = find_peaks(-df["sgd2"])
    
    # Filter valleys: they must be in valid_idxs and have sgd1 > 0
    candidate_idxs = []
    for i in valleys:
        if i in valid_idxs and df["sgd1"].iloc[i] > 0:
            # Convert normalized temperature to real temperature
            T_real = df["Temperature_norm"].iloc[i] * (t_max - t_min) + t_min
            if major_peak is not None and abs(T_real - major_peak) < 4:
                continue  # Ignore candidate too close to major peak
            candidate_idxs.append(i)
    
    if not candidate_idxs:
        return None
    
    # Choose the candidate with the most negative sgd2 (deepest trough)
    chosen_idx = min(candidate_idxs, key=lambda i: df["sgd2"].iloc[i])
    return df["Temperature_norm"].iloc[chosen_idx]

# --- MODEL FITTING & ANALYSIS FUNCTIONS ---

def fit_model_generic(model_name, x, y, custom_p0=None):
    """
    Generic model fitting function that selects the model function, initial guess, and bounds
    based on the model name from MODEL_PARAMS.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to fit
    x : array-like
        Independent variable values
    y : array-like
        Dependent variable values
    custom_p0 : list, optional
        Custom initial parameter values
        
    Returns:
    --------
    array-like or None
        Optimized parameters or None if fitting failed
    """
    if model_name not in MODEL_PARAMS:
        raise ValueError("Unknown model name. Choose from: " + ", ".join(MODEL_PARAMS.keys()))
    
    params = MODEL_PARAMS[model_name]
    p0 = custom_p0 if custom_p0 is not None else params["p0"]
    bounds = params["bounds"]
    model_func = params["model_func"]
    
    try:
        popt, _ = curve_fit(model_func, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except Exception as e:
        print(f"[Fit failed] {model_name}: {e}")
        return None

def select_best_model_for_trace(df_raw, candidate_models=None, window_mode="optimized", 
                               window_sizes=None, fixed_window_size=20, verbose=False, 
                               show_plot=False, protein=None, plate_ID=None, rep=None, 
                               well=None, ligand=None):
    """
    For a given trace, try different models and window sizes to find the best fit.
    
    Parameters:
    -----------
    df_raw : DataFrame
        Raw data for a single trace
    candidate_models : list, optional
        List of model names to try
    window_mode : str, optional
        Mode for window selection ('optimized', 'fixed', or 'full')
    window_sizes : list, optional
        List of window sizes to try
    fixed_window_size : int, optional
        Fixed window size to use if window_mode='fixed'
    verbose : bool, optional
        Whether to print verbose output
    show_plot : bool, optional
        Whether to show a plot of the fit
    protein, plate_ID, rep, well, ligand : str, optional
        Identifiers for the current trace
        
    Returns:
    --------
    tuple
        (best_fit_result, summary_df)
    """
    # Check if result is already in cache
    cache_key = (protein, plate_ID, rep, well)
    if cache_key in MODEL_FIT_CACHE:
        model_cache = MODEL_FIT_CACHE[cache_key]
        
        # Check if we have results for the requested models and window sizes
        for model_name in candidate_models or list(MODEL_PARAMS.keys()):
            if model_name in model_cache:
                if window_mode == "fixed" and fixed_window_size in model_cache[model_name]:
                    # Return cached result for fixed window
                    best_fit = model_cache[model_name][fixed_window_size]
                    return best_fit, pd.DataFrame([best_fit])
                elif window_mode == "optimized" and all(ws in model_cache[model_name] for ws in window_sizes or [20]):
                    # Return best cached result for optimized window
                    all_fits = [model_cache[model_name][ws] for ws in window_sizes or [20]]
                    best_fit = max(all_fits, key=lambda x: x["R2"])
                    return best_fit, pd.DataFrame(all_fits)
    
    if candidate_models is None:
        candidate_models = list(MODEL_PARAMS.keys())
    if window_sizes is None:
        window_sizes = [10, 15, 20, 30, 40]
    
    # Initialize cache for this trace if needed
    if cache_key not in MODEL_FIT_CACHE:
        MODEL_FIT_CACHE[cache_key] = {}
    
    # Normalize the raw data
    df_norm_all = normalize_df(df_raw)
    t_min_all, t_max_all = df_norm_all.attrs["T_bounds"]
    
    all_fit_results = []
    
    for model_name in candidate_models:
        # Initialize model cache if needed
        if model_name not in MODEL_FIT_CACHE[cache_key]:
            MODEL_FIT_CACHE[cache_key][model_name] = {}
            
        if window_mode == "full":
            window_sizes_to_try = [None]  # Just one "window" = the full range
        elif window_mode == "fixed":
            window_sizes_to_try = [fixed_window_size]
        else:  # "optimized"
            window_sizes_to_try = window_sizes
        
        for window_size in window_sizes_to_try:
            if window_size in MODEL_FIT_CACHE[cache_key][model_name]:
                # Use cached result
                fit_result = MODEL_FIT_CACHE[cache_key][model_name][window_size]
                all_fit_results.append(fit_result)
                continue
                
            if window_size is None:
                # Use the full temperature range
                df_window = df_norm_all
                t_start, t_end = t_min_all, t_max_all
            else:
                # Find the derivative peak as a starting point
                deriv_peak_temp = get_deriv_peak_temp(df_raw)
                if deriv_peak_temp is None:
                    # If no peak found, use the midpoint of the temperature range
                    deriv_peak_temp = (t_min_all + t_max_all) / 2
                
                # Define the window around the peak
                half_window = window_size / 2
                t_start = max(t_min_all, deriv_peak_temp - half_window)
                t_end = min(t_max_all, deriv_peak_temp + half_window)
                
                # Extract data within the window
                df_window = df_norm_all[(df_norm_all["Temperature"] >= t_start) & 
                                       (df_norm_all["Temperature"] <= t_end)].copy()
            
            # Skip if window is too small
            if len(df_window) < 5:
                continue
                
            # Normalize within the window
            df_window_norm = normalize_df(df_window)
            
            # Fit the model
            x = df_window_norm["Temperature_norm"].values
            y = df_window_norm["value_norm"].values
            
            popt = fit_model_generic(model_name, x, y)
            if popt is None:
                continue
                
            # Calculate R² for the fit
            model_func = MODEL_PARAMS[model_name]["model_func"]
            y_pred = model_func(x, *popt)
            r2 = r2_score(y, y_pred)
            
            # Calculate Tm values (convert from normalized to real temperature)
            t_min_window, t_max_window = df_window_norm.attrs["T_bounds"]
            tm_idxs = MODEL_PARAMS[model_name]["tm_idx"]
            tm_values = []
            
            for tm_idx in tm_idxs:
                tm_norm = popt[tm_idx]
                tm_real = tm_norm * (t_max_window - t_min_window) + t_min_window
                tm_values.append(tm_real)
            
            # Store the fit result
            fit_result = {
                "model": model_name,
                "window_size": window_size,
                "t_start": t_start,
                "t_end": t_end,
                "popt": popt,
                "R2": r2,
                "Tm1": tm_values[0],
                "protein": protein,
                "plate_ID": plate_ID,
                "rep": rep,
                "well": well,
                "ligand": ligand
            }
            
            if len(tm_values) > 1:
                fit_result["Tm2"] = tm_values[1]
            
            # Cache the result
            MODEL_FIT_CACHE[cache_key][model_name][window_size] = fit_result
            all_fit_results.append(fit_result)
    
    if not all_fit_results:
        return None, pd.DataFrame()
    
    # Find the best fit based on R²
    best_fit = max(all_fit_results, key=lambda x: x["R2"])
    summary_df = pd.DataFrame(all_fit_results)
    
    # Plot if requested
    if show_plot:
        plot_model_fit(df_raw, best_fit, title=f"{protein}, {ligand} ({plate_ID}, {well}, {rep})")
    
    return best_fit, summary_df

def get_negative_control_fit(protein, plate, rep, neg_control_well="A01", 
                            raw_data_dfs=None, candidate_models=None,
                            window_mode="optimized", window_sizes=None,
                            fixed_window_size=20, verbose=False):
    """
    Get the model fit for a negative control well, using caching to avoid redundant computations.
    
    Parameters:
    -----------
    protein, plate, rep : str
        Identifiers for the protein, plate, and replicate
    neg_control_well : str, optional
        Well position for the negative control
    raw_data_dfs : dict, optional
        Nested dictionary of raw data
    candidate_models, window_mode, window_sizes, fixed_window_size, verbose : 
        Parameters for model fitting
        
    Returns:
    --------
    dict or None
        Best fit result for the negative control
    """
    # Check if result is already in cache
    cache_key = (protein, plate, rep, neg_control_well)
    if cache_key in NEG_CONTROL_CACHE:
        return NEG_CONTROL_CACHE[cache_key]
    
    # Get the negative control data
    if raw_data_dfs is None or protein not in raw_data_dfs or plate not in raw_data_dfs[protein] or \
       rep not in raw_data_dfs[protein][plate] or neg_control_well not in raw_data_dfs[protein][plate][rep]:
        if verbose:
            print(f"Negative control data not found for {protein}, {plate}, {rep}, {neg_control_well}")
        return None
    
    df_neg = raw_data_dfs[protein][plate][rep][neg_control_well]
    ligand_neg = get_ligand_for_plate_well(raw_data_dfs, plate, neg_control_well)
    
    # Fit the model
    best_fit_neg, _ = select_best_model_for_trace(
        df_neg, candidate_models=candidate_models,
        window_mode=window_mode, window_sizes=window_sizes,
        fixed_window_size=fixed_window_size, verbose=verbose,
        protein=protein, plate_ID=plate, rep=rep,
        well=neg_control_well, ligand=ligand_neg
    )
    
    # Cache the result
    NEG_CONTROL_CACHE[cache_key] = best_fit_neg
    
    return best_fit_neg

# --- VISUALIZATION FUNCTIONS ---

def plot_model_fit(raw_df, best_fit, title=None):
    """
    Plot the raw data and the fitted model.
    
    Parameters:
    -----------
    raw_df : DataFrame
        Raw data for a single trace
    best_fit : dict
        Best fit result from select_best_model_for_trace
    title : str, optional
        Plot title
        
    Returns:
    --------
    None
    """
    if best_fit is None:
        print("No valid fit to plot.")
        return
    
    fig, (ax_main, ax_deriv) = plt.subplots(2, 1, figsize=(10, 8), 
                                           gridspec_kw={'height_ratios': [3, 1]},
                                           sharex=True)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Plot raw data
    ax_main.plot(raw_df["Temperature"], raw_df["Fluorescence"], 'o-', 
                color="black", alpha=0.5, label="Raw Data")
    
    # Get model parameters
    model_name = best_fit["model"]
    popt = best_fit["popt"]
    t_start = best_fit["t_start"]
    t_end = best_fit["t_end"]
    tm1 = best_fit["Tm1"]
    tm2 = best_fit.get("Tm2", None)
    
    # Highlight the window used for fitting
    ax_main.axvspan(t_start, t_end, alpha=0.2, color='gray')
    
    # Extract the window data
    df_window = raw_df[(raw_df["Temperature"] >= t_start) & 
                       (raw_df["Temperature"] <= t_end)].copy()
    df_window_norm = normalize_df(df_window)
    
    # Get bounds for denormalization
    t_min_fit, t_max_fit = df_window_norm.attrs["T_bounds"]
    f_min_raw, f_max_raw = raw_df["Fluorescence"].min(), raw_df["Fluorescence"].max()
    f_min_fit, f_max_fit = df_window_norm.attrs["F_bounds"]
    
    # Generate dense x values for smooth curve
    temp_dense = np.linspace(t_min_fit, t_max_fit, 300)
    temp_dense_norm = (temp_dense - t_min_fit) / (t_max_fit - t_min_fit)
    
    # Calculate model predictions
    model_func = MODEL_PARAMS[model_name]["model_func"]
    y_model_fit_norm = model_func(temp_dense_norm, *popt)
    y_model_fit_real = y_model_fit_norm * (f_max_fit - f_min_fit) + f_min_fit
    
    # Create label with model info
    label = f"{model_name} (R²={best_fit['R2']:.3f}, Tm={tm1:.1f}°C"
    if tm2 is not None:
        label += f", Tm2={tm2:.1f}°C"
    label += ")"
    
    # Plot Tm2 if available
    if tm2 is not None:
        ax_main.axvline(x=tm2, linestyle=":", color="blue", zorder=9)
    
    # Plot model curve with high zorder
    ax_main.plot(temp_dense, y_model_fit_real, label=label, linestyle="--", zorder=10)
    # Draw vertical dashed line at Tm1
    ax_main.axvline(x=tm1, linestyle="--", color="gray", zorder=9)
    
    ax_main.set_ylabel("Fluorescence (RFU)")
    ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_main.legend(fontsize=9)
    
    # For derivative plotting, use the full raw data
    df_deriv = find_derivative_peaks(raw_df)[0]
    # Convert normalized derivative to unnormalized
    norm_factor = (f_max_raw - f_min_raw)
    df_deriv["sgd1"] = df_deriv["sgd1"] * norm_factor
    
    ax_deriv.plot(df_deriv["Temperature"], df_deriv["sgd1"], color="black", label="1st Derivative", zorder=3)
    # Filter peaks with find_peaks
    peak_idxs = find_peaks(df_deriv["sgd1"])[0]
    if len(peak_idxs) > 0:
        peak_vals = df_deriv["sgd1"].iloc[peak_idxs].values
        global_idx = peak_idxs[np.argmax(peak_vals)]
        for peak in peak_idxs:
            temp = df_deriv["Temperature"].iloc[peak]
            dval = df_deriv["sgd1"].iloc[peak]
            if dval < 0.0002 * norm_factor:
                continue
            if peak == global_idx:
                ax_deriv.plot(temp, dval, 'o', color='red', markersize=10,
                              markeredgecolor='black', zorder=4)
                ax_deriv.text(temp, dval + (0.01 * norm_factor), f"{temp:.1f}", fontsize=9,
                              ha='center', color='black', fontweight='bold', zorder=5)
            else:
                ax_deriv.plot(temp, dval, 'o', color='red', markersize=6,
                              markeredgecolor='black', zorder=4)
                ax_deriv.text(temp, dval + (0.01 * norm_factor), f"{temp:.1f}", fontsize=8, ha='center',
                              color='black', zorder=5)
    
    ax_deriv.set_xlabel("Temperature (°C)")
    ax_deriv.set_ylabel("dF/dT (RFU/°C)")
    y_vals = df_deriv["sgd1"].values
    y_min_val, y_max_val = np.min(y_vals), np.max(y_vals)
    y_range = y_max_val - y_min_val
    ax_deriv.set_ylim(y_min_val - 0.1 * y_range, y_max_val + 0.3 * y_range)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_reps_and_neg_control(protein, plate, treatment_well, neg_control_well="A01",
                             raw_data_dfs=None, candidate_models=None,
                             window_mode="optimized", window_sizes=None,
                             fixed_window_size=20, verbose=False,
                             include_tmchange_subplot=True,
                             include_norm_subplot=False,
                             save_png=False, png_filename=None, dTm_threshold=2,
                             use_uniprot=False, use_smiles=False):
    """
    Plot replicates for a given protein, plate, and treatment well with corresponding negative control.
    
    Parameters:
    -----------
    protein, plate, treatment_well : str
        Identifiers for the protein, plate, and treatment well
    neg_control_well : str, optional
        Well position for the negative control
    raw_data_dfs : dict, optional
        Nested dictionary of raw data
    candidate_models, window_mode, window_sizes, fixed_window_size, verbose : 
        Parameters for model fitting
    include_tmchange_subplot : bool, optional
        Whether to include a subplot showing Tm changes
    include_norm_subplot : bool, optional
        Whether to include subplots with normalized data
    save_png : bool, optional
        Whether to save the plot as a PNG
    png_filename : str, optional
        Filename for saving the plot
    dTm_threshold : float, optional
        Threshold for highlighting significant Tm changes
    use_uniprot : bool, optional
        Whether to use UniProt IDs for protein names
    use_smiles : bool, optional
        Whether to include SMILES visualization
        
    Returns:
    --------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks
    
    # Use a color-blind friendly palette
    treatment_colors = ["#0072B2", "#E69F00", "#009E73", "#F0E442", "#56B4E9", "#D55E00", "#CC79A7"]
    if candidate_models is None:
        candidate_models = list(MODEL_PARAMS.keys())
    if window_sizes is None:
        window_sizes = [10, 15, 20, 30, 40]
    
    # Map protein to UniProt ID if requested and available
    display_protein = protein
    if use_uniprot and protein in PROTEIN_TO_UNIPROT:
        display_protein = PROTEIN_TO_UNIPROT[protein]
    
    # Get replicates for the given protein and plate
    reps = list(raw_data_dfs[protein][plate].keys())
    if verbose:
        print(f"Found replicates for {protein} {plate}: {reps}")
    
    first_ligand = None

    # Decide on subplot layout
    if include_tmchange_subplot and include_norm_subplot:
        nrows = 5
        height_ratios = [1, 3, 3, 1, 1]
        fig, axs = plt.subplots(nrows, 1, figsize=(10, 14), sharex=True,
                                gridspec_kw={'height_ratios': height_ratios})
        ax_tmchange, ax_main, ax_norm, ax_deriv, ax_deriv_norm = axs
    elif include_tmchange_subplot and not include_norm_subplot:
        nrows = 3
        height_ratios = [1, 3, 1]
        fig, axs = plt.subplots(nrows, 1, figsize=(10, 10), sharex=True,
                                gridspec_kw={'height_ratios': height_ratios})
        ax_tmchange, ax_main, ax_deriv = axs
    elif not include_tmchange_subplot and include_norm_subplot:
        nrows = 4
        height_ratios = [3, 3, 1, 1]
        fig, axs = plt.subplots(nrows, 1, figsize=(10, 12), sharex=True,
                                gridspec_kw={'height_ratios': height_ratios})
        ax_main, ax_norm, ax_deriv, ax_deriv_norm = axs
    else:
        nrows = 2
        height_ratios = [3, 1]
        fig, axs = plt.subplots(nrows, 1, figsize=(10, 8), sharex=True,
                                gridspec_kw={'height_ratios': height_ratios})
        ax_main, ax_deriv = axs

    # If using the Tm-change subplot, prepare to assign a unique y-coordinate per replicate
    if include_tmchange_subplot:
        tmchange_ax = ax_tmchange
        tmchange_ax.set_ylabel("Replicate")
        count = 0  # y-coordinate counter

    # Process each replicate
    for i, rep in enumerate(reps):
        rep_data = raw_data_dfs[protein][plate][rep]
        if treatment_well not in rep_data or neg_control_well not in rep_data:
            if verbose:
                print(f"{rep}: Missing required wells. Skipping.")
            continue

        df_treat = rep_data[treatment_well]
        df_neg = rep_data[neg_control_well]

        ligand_treat = get_ligand_for_plate_well(raw_data_dfs, plate, treatment_well)
        if first_ligand is None:
            first_ligand = ligand_treat
        ligand_neg = get_ligand_for_plate_well(raw_data_dfs, plate, neg_control_well)

        # Get model fits using cached negative control
        best_fit_neg = get_negative_control_fit(
            protein, plate, rep, neg_control_well,
            raw_data_dfs, candidate_models,
            window_mode, window_sizes,
            fixed_window_size, verbose
        )
        
        best_fit_t, _ = select_best_model_for_trace(
            df_treat, candidate_models=candidate_models,
            window_mode=window_mode, window_sizes=window_sizes,
            fixed_window_size=fixed_window_size, verbose=verbose,
            protein=protein, plate_ID=plate, rep=rep,
            well=treatment_well, ligand=ligand_treat
        )

        # Skip if either fit failed
        if best_fit_t is None or best_fit_neg is None:
            if verbose:
                print(f"{rep}: Model fitting failed. Skipping.")
            continue

        # Plot raw data
        rep_color = treatment_colors[i % len(treatment_colors)]
        ax_main.plot(df_treat["Temperature"], df_treat["Fluorescence"],
                    color=rep_color, alpha=0.7, linestyle="-", linewidth=2,
                    label=f"{rep} {ligand_treat}")
        ax_main.plot(df_neg["Temperature"], df_neg["Fluorescence"],
                    color=rep_color, alpha=0.3, linestyle="--", linewidth=1)

        # Compute Tm change
        dTm_model = best_fit_t["Tm1"] - best_fit_neg["Tm1"]
        
        # Compute derivative peaks
        df_treat_deriv, _ = find_derivative_peaks(df_treat)
        df_neg_deriv, _ = find_derivative_peaks(df_neg)
        
        # Get derivative peak temperatures
        deriv_peak_t = get_deriv_peak_temp(df_treat)
        deriv_peak_neg = get_deriv_peak_temp(df_neg)
        
        if deriv_peak_t is not None and deriv_peak_neg is not None:
            dTm_deriv = deriv_peak_t - deriv_peak_neg
        else:
            dTm_deriv = np.nan

        # Plot derivative data
        f_min_t, f_max_t = df_treat["Fluorescence"].min(), df_treat["Fluorescence"].max()
        norm_factor_t = f_max_t - f_min_t
        df_treat_deriv["sgd1_real"] = df_treat_deriv["sgd1"] * norm_factor_t
        
        f_min_neg, f_max_neg = df_neg["Fluorescence"].min(), df_neg["Fluorescence"].max()
        norm_factor_neg = f_max_neg - f_min_neg
        df_neg_deriv["sgd1_real"] = df_neg_deriv["sgd1"] * norm_factor_neg
        
        ax_deriv.plot(df_treat_deriv["Temperature"], df_treat_deriv["sgd1_real"],
                     color=rep_color, alpha=0.7, linestyle="-", linewidth=2)
        ax_deriv.plot(df_neg_deriv["Temperature"], df_neg_deriv["sgd1_real"],
                     color=rep_color, alpha=0.3, linestyle="--", linewidth=1)
        
        # Add Tm markers to derivative plot
        if deriv_peak_t is not None:
            peak_val_t = df_treat_deriv.loc[df_treat_deriv["Temperature"] == deriv_peak_t, "sgd1_real"].values[0]
            ax_deriv.plot(deriv_peak_t, peak_val_t, 'o', color=rep_color, markersize=8, markeredgecolor='black')
            ax_deriv.text(deriv_peak_t, peak_val_t + 0.05 * norm_factor_t, f"{deriv_peak_t:.1f}°C",
                         fontsize=8, ha='center', va='bottom')
        
        # Add normalized plots if requested
        if include_norm_subplot:
            df_treat_norm = normalize_df(df_treat)
            df_neg_norm = normalize_df(df_neg)
            
            ax_norm.plot(df_treat_norm["Temperature"], df_treat_norm["value_norm"],
                        color=rep_color, alpha=0.7, linestyle="-", linewidth=2)
            ax_norm.plot(df_neg_norm["Temperature"], df_neg_norm["value_norm"],
                        color=rep_color, alpha=0.3, linestyle="--", linewidth=1)
            
            ax_deriv_norm.plot(df_treat_deriv["Temperature"], df_treat_deriv["sgd1"],
                              color=rep_color, alpha=0.7, linestyle="-", linewidth=2)
            ax_deriv_norm.plot(df_neg_deriv["Temperature"], df_neg_deriv["sgd1"],
                              color=rep_color, alpha=0.3, linestyle="--", linewidth=1)
        
        # Add Tm change subplot if requested
        if include_tmchange_subplot:
            y_pos = count + 0.5
            
            # Plot model-based Tm change
            tmchange_ax.plot([best_fit_neg["Tm1"], best_fit_t["Tm1"]], [y_pos, y_pos],
                            '-o', color=rep_color, linewidth=2, markersize=6)
            
            # Add dTm label
            if not np.isnan(dTm_model):
                label_text = f"ΔTm = {dTm_model:.1f}°C"
                weight = 'bold' if dTm_model >= dTm_threshold else 'normal'
                tmchange_ax.text(best_fit_t["Tm1"] + 1, y_pos, label_text,
                                va='center', fontweight=weight, fontsize=9)
            
            # Plot derivative-based Tm change
            if deriv_peak_t is not None and deriv_peak_neg is not None:
                tmchange_ax.plot([deriv_peak_neg, deriv_peak_t], [y_pos - 0.2, y_pos - 0.2],
                                '--o', color=rep_color, linewidth=1, markersize=4, alpha=0.7)
                
                # Add derivative dTm label
                if not np.isnan(dTm_deriv):
                    label_text = f"ΔTm(deriv) = {dTm_deriv:.1f}°C"
                    weight = 'normal'
                    tmchange_ax.text(deriv_peak_t + 1, y_pos - 0.2, label_text,
                                    va='center', fontweight=weight, fontsize=8, alpha=0.7)
            
            # Add replicate label
            tmchange_ax.text(min(df_treat["Temperature"].min(), df_neg["Temperature"].min()) - 2,
                            y_pos, rep, va='center', ha='right', fontsize=9)
            
            count += 1
    
    # Set up axes and labels
    if include_tmchange_subplot:
        tmchange_ax.set_title(f"{display_protein}, {first_ligand} ({plate}, {treatment_well})")
        tmchange_ax.set_ylim(-0.5, count + 0.5)
        tmchange_ax.set_yticks([])
        tmchange_ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    else:
        ax_main.set_title(f"{display_protein}, {first_ligand} ({plate}, {treatment_well})")
    
    ax_main.set_ylabel("Fluorescence (RFU)")
    ax_main.legend(fontsize=9, loc='upper right')
    ax_main.grid(True, linestyle='--', alpha=0.3)
    
    if include_norm_subplot:
        ax_norm.set_ylabel("Normalized Fluorescence")
        ax_norm.grid(True, linestyle='--', alpha=0.3)
        ax_deriv_norm.set_ylabel("Normalized dF/dT")
        ax_deriv_norm.set_xlabel("Temperature (°C)")
        ax_deriv_norm.grid(True, linestyle='--', alpha=0.3)
    else:
        ax_deriv.set_xlabel("Temperature (°C)")
    
    ax_deriv.set_ylabel("dF/dT (RFU/°C)")
    ax_deriv.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Add SMILES visualization if requested and available
    if use_smiles and first_ligand in COMPOUND_TO_SMILES and RDKIT_AVAILABLE:
        smiles = COMPOUND_TO_SMILES[first_ligand]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Create a new figure for the chemical structure
            fig_chem = plt.figure(figsize=(4, 4))
            ax_chem = fig_chem.add_subplot(111)
            img = Draw.MolToImage(mol, size=(300, 300))
            ax_chem.imshow(img)
            ax_chem.set_title(f"{first_ligand}\n{smiles}")
            ax_chem.axis('off')
            plt.tight_layout()
    
    # Save figure if requested
    if save_png:
        if png_filename is None:
            png_filename = f"{protein}_{plate}_{treatment_well}.png"
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_interactive_heatmap(df, x_col, y_col, value_col, title=None, colorscale='RdBu_r',
                              hover_template=None, sig_figs=2):
    """
    Create an interactive heatmap using Plotly.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the data
    x_col, y_col, value_col : str
        Column names for x-axis, y-axis, and values
    title : str, optional
        Plot title
    colorscale : str, optional
        Colorscale for the heatmap
    hover_template : str, optional
        Custom hover template
    sig_figs : int, optional
        Number of significant figures for hover text
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive heatmap figure
    """
    # Round values to specified significant figures for display
    df = df.copy()
    if sig_figs > 0:
        # Format the value column to specified significant figures
        df[f'{value_col}_formatted'] = df[value_col].apply(
            lambda x: f"{{:.{sig_figs}g}}".format(x) if pd.notnull(x) else "N/A"
        )
    
    # Create the heatmap
    if hover_template is None:
        hover_template = (
            f"<b>{x_col}</b>: %{{x}}<br>" +
            f"<b>{y_col}</b>: %{{y}}<br>" +
            f"<b>{value_col}</b>: %{{customdata}}"
        )
    
    fig = go.Figure(data=go.Heatmap(
        z=df[value_col].values,
        x=df[x_col].unique(),
        y=df[y_col].unique(),
        colorscale=colorscale,
        customdata=df[f'{value_col}_formatted'] if sig_figs > 0 else df[value_col],
        hovertemplate=hover_template
    ))
    
    if title:
        fig.update_layout(title=title)
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=600,
        width=800
    )
    
    return fig

# --- UTILITY FUNCTIONS ---

def load_uniprot_mapping(mapping_file):
    """
    Load protein to UniProt ID mapping from a file.
    
    Parameters:
    -----------
    mapping_file : str
        Path to the mapping file (CSV or JSON)
        
    Returns:
    --------
    dict
        Dictionary mapping protein names to UniProt IDs
    """
    global PROTEIN_TO_UNIPROT
    
    if mapping_file.endswith('.csv'):
        df = pd.read_csv(mapping_file)
        if 'Protein' in df.columns and 'UniProt' in df.columns:
            PROTEIN_TO_UNIPROT = dict(zip(df['Protein'], df['UniProt']))
        else:
            print("Warning: CSV file must contain 'Protein' and 'UniProt' columns")
    elif mapping_file.endswith('.json'):
        with open(mapping_file, 'r') as f:
            PROTEIN_TO_UNIPROT = json.load(f)
    else:
        print("Warning: Unsupported file format. Use CSV or JSON.")
    
    return PROTEIN_TO_UNIPROT

def load_smiles_mapping(mapping_file):
    """
    Load compound to SMILES mapping from a file.
    
    Parameters:
    -----------
    mapping_file : str
        Path to the mapping file (CSV or JSON)
        
    Returns:
    --------
    dict
        Dictionary mapping compound names to SMILES codes
    """
    global COMPOUND_TO_SMILES
    
    if mapping_file.endswith('.csv'):
        df = pd.read_csv(mapping_file)
        if 'Compound' in df.columns and 'SMILES' in df.columns:
            COMPOUND_TO_SMILES = dict(zip(df['Compound'], df['SMILES']))
        else:
            print("Warning: CSV file must contain 'Compound' and 'SMILES' columns")
    elif mapping_file.endswith('.json'):
        with open(mapping_file, 'r') as f:
            COMPOUND_TO_SMILES = json.load(f)
    else:
        print("Warning: Unsupported file format. Use CSV or JSON.")
    
    return COMPOUND_TO_SMILES

def create_output_directory(base_dir, protein=None, plate=None, create=True):
    """
    Create a unique output directory to prevent overriding existing files.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for outputs
    protein : str, optional
        Protein identifier
    plate : str, optional
        Plate identifier
    create : bool, optional
        Whether to create the directory if it doesn't exist
        
    Returns:
    --------
    str
        Path to the output directory
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir) and create:
        os.makedirs(base_dir)
    
    # Create subdirectory for protein and plate if provided
    if protein is not None and plate is not None:
        subdir = os.path.join(base_dir, f"{protein}_{plate}")
        
        # Check if directory already exists
        if os.path.exists(subdir):
            # Find a unique name by appending a number
            i = 1
            while os.path.exists(f"{subdir}_{i}"):
                i += 1
            subdir = f"{subdir}_{i}"
        
        if create:
            os.makedirs(subdir)
        
        return subdir
    
    return base_dir

def clear_caches():
    """
    Clear all global caches to free memory.
    
    Returns:
    --------
    None
    """
    global NEG_CONTROL_CACHE, MODEL_FIT_CACHE, DERIVATIVE_CACHE
    NEG_CONTROL_CACHE.clear()
    MODEL_FIT_CACHE.clear()
    DERIVATIVE_CACHE.clear()
    gc.collect()

def reanalyze_conditions(raw_data_dfs, master_df_path, conditions, 
                        output_path=None, candidate_models=None,
                        window_mode="optimized", window_sizes=None):
    """
    Reanalyze specific conditions and update the master dataframe.
    
    Parameters:
    -----------
    raw_data_dfs : dict
        Nested dictionary of raw data
    master_df_path : str
        Path to the master dataframe CSV
    conditions : list of dict
        List of conditions to reanalyze, each with 'protein', 'plate', 'rep', 'well'
    output_path : str, optional
        Path to save the updated master dataframe
    candidate_models, window_mode, window_sizes : 
        Parameters for model fitting
        
    Returns:
    --------
    DataFrame
        Updated master dataframe
    """
    # Load the master dataframe
    master_df = pd.read_csv(master_df_path)
    
    # Reanalyze each condition
    for condition in conditions:
        protein = condition['protein']
        plate = condition['plate']
        rep = condition['rep']
        well = condition['well']
        
        # Skip if data not available
        if (protein not in raw_data_dfs or plate not in raw_data_dfs[protein] or
            rep not in raw_data_dfs[protein][plate] or well not in raw_data_dfs[protein][plate][rep]):
            print(f"Data not found for {protein}, {plate}, {rep}, {well}. Skipping.")
            continue
        
        # Get the raw data
        df_raw = raw_data_dfs[protein][plate][rep][well]
        ligand = get_ligand_for_plate_well(raw_data_dfs, plate, well)
        
        # Fit the model
        best_fit, _ = select_best_model_for_trace(
            df_raw, candidate_models=candidate_models,
            window_mode=window_mode, window_sizes=window_sizes,
            protein=protein, plate_ID=plate, rep=rep,
            well=well, ligand=ligand
        )
        
        if best_fit is None:
            print(f"Model fitting failed for {protein}, {plate}, {rep}, {well}. Skipping.")
            continue
        
        # Update the master dataframe
        mask = ((master_df['protein'] == protein) & 
                (master_df['plate_ID'] == plate) & 
                (master_df['rep'] == rep) & 
                (master_df['well'] == well))
        
        if mask.any():
            # Update existing row
            for key, value in best_fit.items():
                if key in master_df.columns:
                    master_df.loc[mask, key] = value
        else:
            # Add new row
            new_row = pd.DataFrame([best_fit])
            master_df = pd.concat([master_df, new_row], ignore_index=True)
    
    # Save the updated dataframe if output path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        master_df.to_csv(output_path, index=False)
    
    return master_df

# --- MAIN ANALYSIS FUNCTION ---

def analyze_dsf_data(parent_folder, plate_map_folder, output_dir="./output",
                   protein_filter=None, plate_filter=None, well_filter=None,
                   candidate_models=None, window_mode="optimized", window_sizes=None,
                   neg_control_well="A01", use_uniprot=False, uniprot_mapping=None,
                   use_smiles=False, smiles_mapping=None, sig_figs=2):
    """
    Main function to analyze DSF data and generate results.
    
    Parameters:
    -----------
    parent_folder : str
        Path to the parent folder containing DSF data
    plate_map_folder : str
        Path to the folder containing plate map CSV files
    output_dir : str, optional
        Directory for output files
    protein_filter, plate_filter, well_filter : list, optional
        Lists of proteins, plates, and wells to include
    candidate_models : list, optional
        List of model names to try
    window_mode : str, optional
        Mode for window selection ('optimized', 'fixed', or 'full')
    window_sizes : list, optional
        List of window sizes to try
    neg_control_well : str, optional
        Well position for the negative control
    use_uniprot : bool, optional
        Whether to use UniProt IDs for protein names
    uniprot_mapping : str, optional
        Path to UniProt mapping file
    use_smiles : bool, optional
        Whether to include SMILES visualization
    smiles_mapping : str, optional
        Path to SMILES mapping file
    sig_figs : int, optional
        Number of significant figures for plotly hover text
        
    Returns:
    --------
    tuple
        (raw_data_dfs, master_df, output_dir)
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load mappings if provided
    if use_uniprot and uniprot_mapping is not None:
        load_uniprot_mapping(uniprot_mapping)
    
    if use_smiles and smiles_mapping is not None:
        load_smiles_mapping(smiles_mapping)
    
    # Load data
    print(f"Loading data from {parent_folder}...")
    raw_data_dfs, derivative_data_dfs, boltzmann_data_dfs, plate_map_dict = load_data_with_derivative_boltzmann(
        parent_folder, plate_map_folder, protein_filter, plate_filter, well_filter,
        use_uniprot, use_smiles
    )
    
    # Initialize master dataframe
    master_df_columns = [
        'protein', 'plate_ID', 'rep', 'well', 'ligand',
        'model', 'window_size', 't_start', 't_end',
        'R2', 'Tm1', 'Tm2'
    ]
    master_df = pd.DataFrame(columns=master_df_columns)
    
    # Process each protein, plate, rep, well
    for protein in raw_data_dfs:
        for plate in raw_data_dfs[protein]:
            # Create protein/plate specific output directory
            plate_output_dir = create_output_directory(output_dir, protein, plate)
            
            for rep in raw_data_dfs[protein][plate]:
                for well in raw_data_dfs[protein][plate][rep]:
                    # Skip negative control well
                    if well == neg_control_well:
                        continue
                    
                    # Get raw data
                    df_raw = raw_data_dfs[protein][plate][rep][well]
                    ligand = get_ligand_for_plate_well(raw_data_dfs, plate, well)
                    
                    # Fit model
                    best_fit, _ = select_best_model_for_trace(
                        df_raw, candidate_models=candidate_models,
                        window_mode=window_mode, window_sizes=window_sizes,
                        protein=protein, plate_ID=plate, rep=rep,
                        well=well, ligand=ligand
                    )
                    
                    if best_fit is not None:
                        # Add to master dataframe
                        master_df = pd.concat([master_df, pd.DataFrame([best_fit])], ignore_index=True)
            
            # Generate plots for each treatment well
            treatment_wells = set()
            for rep in raw_data_dfs[protein][plate]:
                for well in raw_data_dfs[protein][plate][rep]:
                    if well != neg_control_well:
                        treatment_wells.add(well)
            
            for well in treatment_wells:
                try:
                    # Create plot
                    plot_reps_and_neg_control(
                        protein, plate, well, neg_control_well,
                        raw_data_dfs, candidate_models,
                        window_mode, window_sizes,
                        save_png=True,
                        png_filename=os.path.join(plate_output_dir, f"{protein}_{plate}_{well}.png"),
                        use_uniprot=use_uniprot,
                        use_smiles=use_smiles
                    )
                except Exception as e:
                    print(f"Error plotting {protein}, {plate}, {well}: {e}")
            
            # Create heatmap of Tm values
            try:
                plate_df = master_df[(master_df['protein'] == protein) & (master_df['plate_ID'] == plate)]
                if not plate_df.empty:
                    # Pivot the data for heatmap
                    heatmap_df = plate_df.pivot_table(
                        index='well', columns='rep', values='Tm1', aggfunc='mean'
                    ).reset_index()
                    
                    # Create interactive heatmap
                    fig = create_interactive_heatmap(
                        heatmap_df.melt(id_vars='well', var_name='rep', value_name='Tm1'),
                        'rep', 'well', 'Tm1',
                        title=f"{protein} {plate} - Tm Values",
                        sig_figs=sig_figs
                    )
                    
                    # Save as HTML
                    fig.write_html(os.path.join(plate_output_dir, f"{protein}_{plate}_heatmap.html"))
            except Exception as e:
                print(f"Error creating heatmap for {protein}, {plate}: {e}")
    
    # Save master dataframe
    master_df.to_csv(os.path.join(output_dir, "master_results.csv"), index=False)
    
    # Clear caches to free memory
    clear_caches()
    
    return raw_data_dfs, master_df, output_dir

# Example usage template
if __name__ == "__main__":
    # Set paths
    parent_folder = "./data"
    plate_map_folder = "./plate_maps"
    output_dir = "./output"
    
    # Optional: Load UniProt and SMILES mappings
    uniprot_mapping = "./mappings/uniprot_mapping.csv"
    smiles_mapping = "./mappings/smiles_mapping.csv"
    
    # Run analysis
    raw_data_dfs, master_df, output_dir = analyze_dsf_data(
        parent_folder, plate_map_folder, output_dir,
        use_uniprot=True, uniprot_mapping=uniprot_mapping,
        use_smiles=True, smiles_mapping=smiles_mapping,
        sig_figs=2
    )
    
    print(f"Analysis complete. Results saved to {output_dir}")
