"""
Utility functions for mapping in DSF analysis.

This module provides functions for mapping protein names to UniProt IDs
and compound names to SMILES codes.
"""

import os
import json
import warnings
import pandas as pd
from typing import Dict, Optional

def load_uniprot_mapping(mapping_file: str) -> Dict[str, str]:
    """
    Load protein to UniProt ID mapping from a file (CSV or JSON).
    
    Parameters:
    -----------
    mapping_file : str
        Path to the mapping file (CSV or JSON)
        
    Returns:
    --------
    dict
        Dictionary mapping protein names to UniProt IDs
    """
    # Import to avoid circular imports
    from dsf_analysis.core.data_loading import PROTEIN_TO_UNIPROT
    
    if not os.path.exists(mapping_file):
        warnings.warn(f"UniProt mapping file not found: {mapping_file}")
        return {}
    
    if mapping_file.endswith('.csv'):
        df = pd.read_csv(mapping_file)
        if 'Protein' in df.columns and 'UniProt' in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row['UniProt']) and row['UniProt'].strip():
                    PROTEIN_TO_UNIPROT[row['Protein']] = row['UniProt'].strip()
        else:
            warnings.warn("CSV file must contain 'Protein' and 'UniProt' columns")
    elif mapping_file.endswith('.json'):
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            
        # Handle both flat and nested JSON structures
        for protein, value in data.items():
            if isinstance(value, str):
                # Flat structure: {"B2": "P12345"}
                PROTEIN_TO_UNIPROT[protein] = value
            elif isinstance(value, dict) and 'uniprot' in value:
                # Nested structure: {"B2": {"uniprot": "P12345"}}
                PROTEIN_TO_UNIPROT[protein] = value['uniprot']
    else:
        warnings.warn("Unsupported file format. Use CSV or JSON.")
    
    print(f"Loaded {len(PROTEIN_TO_UNIPROT)} UniProt mappings")
    return PROTEIN_TO_UNIPROT

def load_smiles_mapping(mapping_file: str) -> Dict[str, str]:
    """
    Load compound to SMILES mapping from a file (CSV or JSON).
    
    Parameters:
    -----------
    mapping_file : str
        Path to the mapping file (CSV or JSON)
        
    Returns:
    --------
    dict
        Dictionary mapping compound names to SMILES codes
    """
    # Import to avoid circular imports
    from dsf_analysis.core.data_loading import COMPOUND_TO_SMILES
    
    if not os.path.exists(mapping_file):
        warnings.warn(f"SMILES mapping file not found: {mapping_file}")
        return {}
    
    if mapping_file.endswith('.csv'):
        df = pd.read_csv(mapping_file)
        if 'Compound' in df.columns and 'SMILES' in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row['SMILES']) and row['SMILES'].strip():
                    COMPOUND_TO_SMILES[row['Compound']] = row['SMILES'].strip()
        else:
            warnings.warn("CSV file must contain 'Compound' and 'SMILES' columns")
    elif mapping_file.endswith('.json'):
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            
        # Handle both flat and nested JSON structures
        for compound, value in data.items():
            if isinstance(value, str):
                # Flat structure: {"Compound1": "SMILES1"}
                COMPOUND_TO_SMILES[compound] = value
            elif isinstance(value, dict) and 'smiles' in value:
                # Nested structure: {"Compound1": {"smiles": "SMILES1"}}
                COMPOUND_TO_SMILES[compound] = value['smiles']
    else:
        warnings.warn("Unsupported file format. Use CSV or JSON.")
    
    print(f"Loaded {len(COMPOUND_TO_SMILES)} SMILES mappings")
    return COMPOUND_TO_SMILES

def extract_smiles_from_plate_maps(plate_map_dict: Dict[str, pd.DataFrame], 
                                  smiles_col: str = "SMILES") -> Dict[str, str]:
    """
    Extract SMILES codes from plate map DataFrames and update the global mapping.
    
    Parameters:
    -----------
    plate_map_dict : dict
        Dictionary mapping plate IDs to plate map DataFrames
    smiles_col : str, optional
        Column name for SMILES codes
        
    Returns:
    --------
    dict
        Updated dictionary mapping compound names to SMILES codes
    """
    # Import to avoid circular imports
    from dsf_analysis.core.data_loading import COMPOUND_TO_SMILES
    
    for plate_id, df in plate_map_dict.items():
        if smiles_col in df.columns:
            ligand_col = "Ligand" if "Ligand" in df.columns else "Compound"
            
            if ligand_col in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row[smiles_col]) and pd.notna(row[ligand_col]):
                        ligand = str(row[ligand_col]).strip()
                        smiles = str(row[smiles_col]).strip()
                        if ligand and smiles and ligand not in ["DMSO", "Buffer", "Control"]:
                            COMPOUND_TO_SMILES[ligand] = smiles
    
    return COMPOUND_TO_SMILES

def get_uniprot_info(uniprot_id: str) -> Optional[Dict]:
    """
    Get information about a protein from UniProt.
    
    Parameters:
    -----------
    uniprot_id : str
        UniProt ID
        
    Returns:
    --------
    dict or None
        Dictionary with protein information, or None if retrieval failed
    """
    try:
        import requests
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.json"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            warnings.warn(f"Failed to retrieve UniProt information for {uniprot_id}: {response.status_code}")
            return None
    except Exception as e:
        warnings.warn(f"Error retrieving UniProt information: {e}")
        return None

def get_compound_info(compound_name: str) -> Optional[Dict]:
    """
    Get information about a compound from PubChem.
    
    Parameters:
    -----------
    compound_name : str
        Compound name
        
    Returns:
    --------
    dict or None
        Dictionary with compound information, or None if retrieval failed
    """
    try:
        import requests
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/JSON"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            warnings.warn(f"Failed to retrieve PubChem information for {compound_name}: {response.status_code}")
            return None
    except Exception as e:
        warnings.warn(f"Error retrieving PubChem information: {e}")
        return None
