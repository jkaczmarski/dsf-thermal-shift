"""
Utility functions for formatting in DSF analysis.

This module provides functions for formatting values, strings, and other data
for consistent display in the DSF analysis package.
"""

import numpy as np
from typing import Union, Any, Optional

def format_to_sig_figs(value: Union[float, int], sig_figs: int = 2) -> str:
    """
    Format a number to a specified number of significant figures.
    
    Parameters:
    -----------
    value : float or int
        Number to format
    sig_figs : int, optional
        Number of significant figures
        
    Returns:
    --------
    str
        Formatted string
    """
    if not isinstance(value, (int, float)) or np.isnan(value):
        return str(value)
    
    return f"{value:.{sig_figs}g}"

def format_temperature(value: Union[float, int], decimal_places: int = 1) -> str:
    """
    Format a temperature value with a specified number of decimal places.
    
    Parameters:
    -----------
    value : float or int
        Temperature value
    decimal_places : int, optional
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted temperature string with °C
    """
    if not isinstance(value, (int, float)) or np.isnan(value):
        return str(value)
    
    return f"{value:.{decimal_places}f}°C"

def format_delta_tm(value: Union[float, int], decimal_places: int = 1) -> str:
    """
    Format a delta Tm value with a specified number of decimal places.
    
    Parameters:
    -----------
    value : float or int
        Delta Tm value
    decimal_places : int, optional
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted delta Tm string with sign and °C
    """
    if not isinstance(value, (int, float)) or np.isnan(value):
        return str(value)
    
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimal_places}f}°C"

def format_r_squared(value: Union[float, int], decimal_places: int = 3) -> str:
    """
    Format an R² value with a specified number of decimal places.
    
    Parameters:
    -----------
    value : float or int
        R² value
    decimal_places : int, optional
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted R² string
    """
    if not isinstance(value, (int, float)) or np.isnan(value):
        return str(value)
    
    return f"R² = {value:.{decimal_places}f}"

def format_concentration(value: Union[float, int], unit: str = "μM") -> str:
    """
    Format a concentration value with appropriate units.
    
    Parameters:
    -----------
    value : float or int
        Concentration value
    unit : str, optional
        Concentration unit
        
    Returns:
    --------
    str
        Formatted concentration string with unit
    """
    if not isinstance(value, (int, float)) or np.isnan(value):
        return str(value)
    
    # Format based on magnitude
    if value >= 1000:
        return f"{value/1000:.1f} m{unit}"
    elif value >= 1:
        return f"{value:.1f} {unit}"
    elif value >= 0.001:
        return f"{value*1000:.1f} n{unit}"
    else:
        return f"{value*1000000:.1f} p{unit}"

def format_plate_well(plate: str, well: str) -> str:
    """
    Format a plate and well identifier.
    
    Parameters:
    -----------
    plate : str
        Plate identifier
    well : str
        Well identifier
        
    Returns:
    --------
    str
        Formatted plate and well string
    """
    return f"{plate}:{well}"

def format_protein_name(protein: str, uniprot_id: Optional[str] = None) -> str:
    """
    Format a protein name with optional UniProt ID.
    
    Parameters:
    -----------
    protein : str
        Protein identifier
    uniprot_id : str, optional
        UniProt ID
        
    Returns:
    --------
    str
        Formatted protein name
    """
    if uniprot_id:
        return f"{protein} ({uniprot_id})"
    return protein
