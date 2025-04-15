"""
Utility functions for output management in DSF analysis.

This module provides functions for managing output directories and files,
including creating unique directories, saving files with conflict prevention,
and organizing output files.
"""

import os
import shutil
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

def create_output_directory(base_dir: str, protein: Optional[str] = None, 
                           plate: Optional[str] = None, create: bool = True,
                           timestamp: bool = False) -> str:
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
    timestamp : bool, optional
        Whether to add a timestamp to the directory name
        
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
        # Add timestamp if requested
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            subdir = os.path.join(base_dir, f"{protein}_{plate}_{timestamp_str}")
        else:
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

def get_unique_filename(directory: str, base_filename: str, extension: str = "") -> str:
    """
    Generate a unique filename to prevent overwriting existing files.
    
    Parameters:
    -----------
    directory : str
        Directory where the file will be saved
    base_filename : str
        Base name for the file (without extension)
    extension : str, optional
        File extension (with or without leading dot)
        
    Returns:
    --------
    str
        Unique filename
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Normalize extension
    if extension and not extension.startswith('.'):
        extension = f".{extension}"
    
    # Check if file already exists
    filename = os.path.join(directory, f"{base_filename}{extension}")
    if not os.path.exists(filename):
        return filename
    
    # Find a unique name by appending a number
    i = 1
    while os.path.exists(os.path.join(directory, f"{base_filename}_{i}{extension}")):
        i += 1
    
    return os.path.join(directory, f"{base_filename}_{i}{extension}")

def save_dataframe(df, directory: str, base_filename: str, 
                  extension: str = "csv", index: bool = False,
                  overwrite: bool = False) -> str:
    """
    Save a DataFrame to a file with a unique filename.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    directory : str
        Directory where the file will be saved
    base_filename : str
        Base name for the file (without extension)
    extension : str, optional
        File extension (without leading dot)
    index : bool, optional
        Whether to include the index in the output
    overwrite : bool, optional
        Whether to overwrite existing files
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Determine filename
    if overwrite:
        filename = os.path.join(directory, f"{base_filename}.{extension}")
    else:
        filename = get_unique_filename(directory, base_filename, extension)
    
    # Save DataFrame based on extension
    if extension.lower() == "csv":
        df.to_csv(filename, index=index)
    elif extension.lower() == "excel" or extension.lower() == "xlsx":
        df.to_excel(filename, index=index)
    elif extension.lower() == "json":
        df.to_json(filename, orient="records")
    elif extension.lower() == "html":
        df.to_html(filename, index=index)
    elif extension.lower() == "pickle" or extension.lower() == "pkl":
        df.to_pickle(filename)
    else:
        # Default to CSV
        filename = f"{os.path.splitext(filename)[0]}.csv"
        df.to_csv(filename, index=index)
    
    return filename

def save_figure(fig, directory: str, base_filename: str, 
               formats: List[str] = ["png"], dpi: int = 300,
               overwrite: bool = False) -> List[str]:
    """
    Save a matplotlib figure to files with unique filenames.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    directory : str
        Directory where the file will be saved
    base_filename : str
        Base name for the file (without extension)
    formats : list, optional
        List of formats to save (e.g., ["png", "pdf", "svg"])
    dpi : int, optional
        Resolution for raster formats
    overwrite : bool, optional
        Whether to overwrite existing files
        
    Returns:
    --------
    list
        Paths to the saved files
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    saved_files = []
    
    for fmt in formats:
        # Determine filename
        if overwrite:
            filename = os.path.join(directory, f"{base_filename}.{fmt}")
        else:
            filename = get_unique_filename(directory, base_filename, fmt)
        
        # Save figure
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        saved_files.append(filename)
    
    return saved_files

def save_plotly_figure(fig, directory: str, base_filename: str,
                      formats: List[str] = ["html", "png", "json"],
                      overwrite: bool = False) -> List[str]:
    """
    Save a plotly figure to files with unique filenames.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to save
    directory : str
        Directory where the file will be saved
    base_filename : str
        Base name for the file (without extension)
    formats : list, optional
        List of formats to save (e.g., ["html", "png", "json"])
    overwrite : bool, optional
        Whether to overwrite existing files
        
    Returns:
    --------
    list
        Paths to the saved files
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    saved_files = []
    
    for fmt in formats:
        # Determine filename
        if overwrite:
            filename = os.path.join(directory, f"{base_filename}.{fmt}")
        else:
            filename = get_unique_filename(directory, base_filename, fmt)
        
        # Save figure based on format
        if fmt.lower() == "html":
            fig.write_html(filename)
        elif fmt.lower() == "png":
            fig.write_image(filename)
        elif fmt.lower() == "jpg" or fmt.lower() == "jpeg":
            fig.write_image(filename)
        elif fmt.lower() == "svg":
            fig.write_image(filename)
        elif fmt.lower() == "pdf":
            fig.write_image(filename)
        elif fmt.lower() == "json":
            with open(filename, "w") as f:
                f.write(fig.to_json())
        
        saved_files.append(filename)
    
    return saved_files

def save_analysis_metadata(directory: str, metadata: Dict[str, Any],
                          base_filename: str = "analysis_metadata",
                          overwrite: bool = False) -> str:
    """
    Save analysis metadata to a JSON file.
    
    Parameters:
    -----------
    directory : str
        Directory where the file will be saved
    metadata : dict
        Metadata to save
    base_filename : str, optional
        Base name for the file (without extension)
    overwrite : bool, optional
        Whether to overwrite existing files
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Add timestamp to metadata
    metadata["timestamp"] = datetime.datetime.now().isoformat()
    
    # Determine filename
    if overwrite:
        filename = os.path.join(directory, f"{base_filename}.json")
    else:
        filename = get_unique_filename(directory, base_filename, "json")
    
    # Save metadata
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return filename
