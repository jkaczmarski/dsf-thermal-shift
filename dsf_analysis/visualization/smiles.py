"""
SMILES visualization functionality for DSF analysis.

This module provides functions for visualizing chemical structures from SMILES codes
in the context of DSF (Differential Scanning Fluorimetry) data analysis.
"""

import os
import io
import base64
import warnings
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import RDKit, but provide graceful fallback if not available
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Chemical structure visualization will be disabled.")

def smiles_to_mol(smiles: str) -> Optional[object]:
    """
    Convert a SMILES string to an RDKit molecule object.
    
    Parameters:
    -----------
    smiles : str
        SMILES code for a chemical compound
        
    Returns:
    --------
    rdkit.Chem.rdchem.Mol or None
        RDKit molecule object, or None if conversion failed
    """
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Add hydrogen atoms and compute 2D coordinates
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
            mol = Chem.RemoveHs(mol)  # Remove hydrogens for cleaner visualization
        return mol
    except Exception as e:
        warnings.warn(f"Error converting SMILES to molecule: {e}")
        return None

def mol_to_image(mol, size: Tuple[int, int] = (300, 300), 
                highlight_atoms: List[int] = None) -> Optional[object]:
    """
    Convert an RDKit molecule to an image.
    
    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    size : tuple, optional
        Image size (width, height)
    highlight_atoms : list, optional
        List of atom indices to highlight
        
    Returns:
    --------
    PIL.Image or None
        PIL Image object, or None if conversion failed
    """
    if not RDKIT_AVAILABLE or mol is None:
        return None
    
    try:
        if highlight_atoms:
            # Create a molecule drawer with atom highlighting
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
            drawer.FinishDrawing()
            png = drawer.GetDrawingText()
            from PIL import Image
            import io
            return Image.open(io.BytesIO(png))
        else:
            # Use the simpler Draw.MolToImage function
            return Draw.MolToImage(mol, size=size)
    except Exception as e:
        warnings.warn(f"Error converting molecule to image: {e}")
        return None

def get_mol_for_compound(compound: str) -> Optional[object]:
    """
    Get an RDKit molecule object for a compound using the global SMILES mapping.
    
    Parameters:
    -----------
    compound : str
        Compound name or identifier
        
    Returns:
    --------
    rdkit.Chem.rdchem.Mol or None
        RDKit molecule object, or None if not found or conversion failed
    """
    if not RDKIT_AVAILABLE:
        return None
    
    # Import to avoid circular imports
    from dsf_analysis.core.data_loading import COMPOUND_TO_SMILES
    
    if compound in COMPOUND_TO_SMILES:
        return smiles_to_mol(COMPOUND_TO_SMILES[compound])
    return None

def get_image_for_compound(compound: str, size: Tuple[int, int] = (300, 300)) -> Optional[object]:
    """
    Get an image of the chemical structure for a compound.
    
    Parameters:
    -----------
    compound : str
        Compound name or identifier
    size : tuple, optional
        Image size (width, height)
        
    Returns:
    --------
    PIL.Image or None
        PIL Image object, or None if not found or conversion failed
    """
    mol = get_mol_for_compound(compound)
    if mol is not None:
        return mol_to_image(mol, size)
    return None

def add_structure_to_matplotlib_figure(fig, compound: str, 
                                      position: Tuple[float, float, float, float],
                                      title: str = None) -> None:
    """
    Add a chemical structure to a matplotlib figure at the specified position.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    compound : str
        Compound name or identifier
    position : tuple
        Position (left, bottom, width, height) in figure coordinates
    title : str, optional
        Title to display above the structure
        
    Returns:
    --------
    None
    """
    if not RDKIT_AVAILABLE:
        return
    
    img = get_image_for_compound(compound)
    if img is not None:
        ax = fig.add_axes(position)
        ax.imshow(img)
        ax.axis('off')
        if title:
            ax.set_title(title)

def mol_to_svg(mol, width: int = 300, height: int = 300) -> Optional[str]:
    """
    Convert an RDKit molecule to an SVG string.
    
    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    width, height : int, optional
        Dimensions of the SVG image
        
    Returns:
    --------
    str or None
        SVG string, or None if conversion failed
    """
    if not RDKIT_AVAILABLE or mol is None:
        return None
    
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        warnings.warn(f"Error converting molecule to SVG: {e}")
        return None

def mol_to_plotly_image(mol, format: str = 'svg') -> Optional[str]:
    """
    Convert an RDKit molecule to a format suitable for Plotly.
    
    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    format : str, optional
        Output format ('svg' or 'png')
        
    Returns:
    --------
    str or None
        Image data in the specified format, or None if conversion failed
    """
    if not RDKIT_AVAILABLE or mol is None:
        return None
    
    try:
        if format == 'svg':
            return mol_to_svg(mol)
        elif format == 'png':
            img = mol_to_image(mol)
            if img is not None:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        return None
    except Exception as e:
        warnings.warn(f"Error converting molecule to Plotly image: {e}")
        return None

def plot_dsf_curve_with_structure(df_raw: pd.DataFrame, 
                                 best_fit: Dict, 
                                 title: str = None,
                                 show_structure: bool = True) -> plt.Figure:
    """
    Plot DSF curve with chemical structure.
    
    Parameters:
    -----------
    df_raw : DataFrame
        Raw DSF data
    best_fit : dict
        Best fit result from model fitting
    title : str, optional
        Plot title
    show_structure : bool, optional
        Whether to show the chemical structure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with DSF curve and chemical structure
    """
    if best_fit is None:
        print("No valid fit to plot.")
        return None
    
    # Create figure with extra space for the chemical structure
    if show_structure and RDKIT_AVAILABLE and 'ligand' in best_fit:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        ax_main = fig.add_subplot(gs[0, 0])
        ax_deriv = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_chem = fig.add_subplot(gs[0, 1])
    else:
        fig, (ax_main, ax_deriv) = plt.subplots(2, 1, figsize=(10, 8), 
                                               gridspec_kw={'height_ratios': [3, 1]},
                                               sharex=True)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Plot raw data
    ax_main.plot(df_raw["Temperature"], df_raw["Fluorescence"], 'o-', 
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
    df_window = df_raw[(df_raw["Temperature"] >= t_start) & 
                       (df_raw["Temperature"] <= t_end)].copy()
    
    # Get bounds for denormalization
    t_min_fit, t_max_fit = t_start, t_end
    f_min_raw, f_max_raw = df_raw["Fluorescence"].min(), df_raw["Fluorescence"].max()
    
    # Generate dense x values for smooth curve
    temp_dense = np.linspace(t_min_fit, t_max_fit, 300)
    
    # Import model functions
    from dsf_analysis.core.models import MODEL_PARAMS
    
    # Get model function
    model_func = MODEL_PARAMS[model_name]["model_func"]
    
    # Normalize temperature for model input
    temp_norm = (temp_dense - t_min_fit) / (t_max_fit - t_min_fit)
    
    # Calculate model predictions
    y_model_norm = model_func(temp_norm, *popt)
    
    # Denormalize model predictions
    f_min_window, f_max_window = df_window["Fluorescence"].min(), df_window["Fluorescence"].max()
    y_model_fit_real = y_model_norm * (f_max_window - f_min_window) + f_min_window
    
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
    
    # For derivative plotting
    # Calculate derivative
    temp = df_raw["Temperature"].values
    fluor = df_raw["Fluorescence"].values
    deriv = np.gradient(fluor, temp)
    
    # Plot derivative
    ax_deriv.plot(temp, deriv, color="black", label="1st Derivative", zorder=3)
    
    ax_deriv.set_xlabel("Temperature (°C)")
    ax_deriv.set_ylabel("dF/dT (RFU/°C)")
    
    # Add chemical structure if available
    if show_structure and RDKIT_AVAILABLE and 'ligand' in best_fit:
        ligand = best_fit['ligand']
        mol = get_mol_for_compound(ligand)
        if mol is not None:
            img = mol_to_image(mol)
            if img is not None:
                ax_chem.imshow(img)
                ax_chem.set_title(f"{ligand}")
                ax_chem.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def create_structure_grid(compounds: List[str], 
                         ncols: int = 3, 
                         size: Tuple[int, int] = (200, 200)) -> plt.Figure:
    """
    Create a grid of chemical structures.
    
    Parameters:
    -----------
    compounds : list
        List of compound names
    ncols : int, optional
        Number of columns in the grid
    size : tuple, optional
        Size of each structure image
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with grid of chemical structures
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available. Cannot create structure grid.")
        return None
    
    # Import to avoid circular imports
    from dsf_analysis.core.data_loading import COMPOUND_TO_SMILES
    
    # Filter compounds to those with valid SMILES
    valid_compounds = [c for c in compounds if c in COMPOUND_TO_SMILES]
    if not valid_compounds:
        print("No valid compounds with SMILES codes found.")
        return None
    
    # Calculate grid dimensions
    n = len(valid_compounds)
    nrows = (n + ncols - 1) // ncols
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Add structures to grid
    for i, compound in enumerate(valid_compounds):
        if i < len(axes):
            img = get_image_for_compound(compound, size)
            if img is not None:
                axes[i].imshow(img)
                axes[i].set_title(compound)
                axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(valid_compounds), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
