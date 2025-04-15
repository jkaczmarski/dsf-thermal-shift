"""
Core data loading functionality for DSF analysis.

This module provides functions for loading and preprocessing DSF data files,
including parsing file structures, extracting well information, and organizing
data into appropriate structures for analysis.
"""

import os
import glob
import re
import gc
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np

# Global mapping dictionaries
PROTEIN_TO_UNIPROT = {}
COMPOUND_TO_SMILES = {}

def unify_plate_id(raw_plate_id: str) -> str:
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

def parse_protein_plate_from_subfolder(subfolder_name: str) -> Tuple[str, str]:
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

def parse_replicate_from_filename(file_name: str) -> str:
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

def split_raw_derivative_boltzmann_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

def load_plate_map_csvs(plate_map_folder: str) -> Dict[str, pd.DataFrame]:
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

def build_well_to_ligand_map(plate_map_df: pd.DataFrame, well_col: str = "Well", 
                           ligand_col: str = "Ligand", smiles_col: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
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

def load_data_with_derivative_boltzmann(parent_folder: str, plate_map_folder: str, 
                                      protein_filter: Optional[List[str]] = None, 
                                      plate_filter: Optional[List[str]] = None, 
                                      well_filter: Optional[List[str]] = None, 
                                      use_uniprot: bool = False,
                                      use_smiles: bool = False) -> Tuple[Dict, Dict, Dict, Dict]:
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

def split_by_well_position(df: pd.DataFrame, well_column: str = "Well Position") -> Dict[str, pd.DataFrame]:
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

def get_ligand_for_plate_well(raw_data_dfs: Dict, plate: str, well: str) -> str:
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
