"""
SMILES Visualization Module for DSF Analysis
===========================================

This module provides functions for visualizing chemical structures from SMILES codes
in the context of DSF (Differential Scanning Fluorimetry) data analysis.

Dependencies:
- rdkit: Chemical informatics and structure visualization
- matplotlib: For embedding chemical structure images in plots
- plotly: For interactive visualizations with chemical structures
"""

import os
import io
import base64
import warnings
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import RDKit, but provide graceful fallback if not available
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Chemical structure visualization will be disabled.")

# Global dictionary to store SMILES codes for compounds
COMPOUND_TO_SMILES = {}

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
    global COMPOUND_TO_SMILES
    
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
        import json
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
    global COMPOUND_TO_SMILES
    
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

def create_interactive_heatmap_with_structures(df: pd.DataFrame, 
                                             x_col: str, 
                                             y_col: str, 
                                             value_col: str,
                                             compound_col: str = None,
                                             title: str = None,
                                             colorscale: str = 'RdBu_r',
                                             sig_figs: int = 2) -> go.Figure:
    """
    Create an interactive heatmap with chemical structures in hover tooltips.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the data
    x_col, y_col, value_col : str
        Column names for x-axis, y-axis, and values
    compound_col : str, optional
        Column name for compound identifiers
    title : str, optional
        Plot title
    colorscale : str, optional
        Colorscale for the heatmap
    sig_figs : int, optional
        Number of significant figures for hover text
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive heatmap figure with chemical structures
    """
    # Round values to specified significant figures for display
    df = df.copy()
    if sig_figs > 0:
        # Format the value column to specified significant figures
        df[f'{value_col}_formatted'] = df[value_col].apply(
            lambda x: f"{{:.{sig_figs}g}}".format(x) if pd.notnull(x) else "N/A"
        )
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df[value_col].values,
        x=df[x_col].unique(),
        y=df[y_col].unique(),
        colorscale=colorscale,
        customdata=df[f'{value_col}_formatted'] if sig_figs > 0 else df[value_col],
        hovertemplate=(
            f"<b>{x_col}</b>: %{{x}}<br>" +
            f"<b>{y_col}</b>: %{{y}}<br>" +
            f"<b>{value_col}</b>: %{{customdata}}"
        )
    ))
    
    # Add chemical structures if RDKit is available and compound_col is provided
    if RDKIT_AVAILABLE and compound_col is not None and compound_col in df.columns:
        # Create a subplot with a second empty trace for the structure
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{"type": "heatmap"}, {"type": "image"}]],
            horizontal_spacing=0.01
        )
        
        # Add the heatmap to the first subplot
        fig.add_trace(
            go.Heatmap(
                z=df[value_col].values,
                x=df[x_col].unique(),
                y=df[y_col].unique(),
                colorscale=colorscale,
                customdata=df[[compound_col, f'{value_col}_formatted'] if sig_figs > 0 else [compound_col, value_col]].values,
                hovertemplate=(
                    f"<b>{x_col}</b>: %{{x}}<br>" +
                    f"<b>{y_col}</b>: %{{y}}<br>" +
                    f"<b>{value_col}</b>: %{{customdata[1]}}<br>" +
                    f"<b>Compound</b>: %{{customdata[0]}}"
                )
            ),
            row=1, col=1
        )
        
        # Add an empty image trace for the structure
        fig.add_trace(
            go.Image(
                source="",
                xaxis="x2",
                yaxis="y2"
            ),
            row=1, col=2
        )
        
        # Add a callback to update the structure when hovering over the heatmap
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Show Structure",
                            method="animate",
                            args=[None, {"frame": {"duration": 0, "redraw": True}}]
                        )
                    ]
                )
            ]
        )
    
    if title:
        fig.update_layout(title=title)
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=600,
        width=800
    )
    
    return fig

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
    
    # Calculate model predictions (this is simplified and would need to be adapted to your model functions)
    # For demonstration purposes only
    y_model_fit_real = np.linspace(df_window["Fluorescence"].min(), df_window["Fluorescence"].max(), 300)
    
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
    
    # For derivative plotting (simplified)
    ax_deriv.plot(df_raw["Temperature"], np.gradient(df_raw["Fluorescence"], df_raw["Temperature"]), 
                 color="black", label="1st Derivative", zorder=3)
    
    ax_deriv.set_xlabel("Temperature (°C)")
    ax_deriv.set_ylabel("dF/dT (RFU/°C)")
    
    # Add chemical structure if available
    if show_structure and RDKIT_AVAILABLE and 'ligand' in best_fit:
        ligand = best_fit['ligand']
        if ligand in COMPOUND_TO_SMILES:
            mol = get_mol_for_compound(ligand)
            if mol is not None:
                img = mol_to_image(mol)
                if img is not None:
                    ax_chem.imshow(img)
                    ax_chem.set_title(f"{ligand}\n{COMPOUND_TO_SMILES[ligand]}")
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

# Example usage
if __name__ == "__main__":
    # Load SMILES mappings
    load_smiles_mapping("./mappings/smiles_mapping.csv")
    
    # Create a test compound
    if RDKIT_AVAILABLE:
        # Create a test molecule
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        COMPOUND_TO_SMILES["Aspirin"] = test_smiles
        
        # Display the structure
        mol = smiles_to_mol(test_smiles)
        img = mol_to_image(mol)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Aspirin")
        plt.show()
