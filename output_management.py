"""
Output Management Module for DSF Analysis
========================================

This module provides functions for managing output directories and files
in the context of DSF (Differential Scanning Fluorimetry) data analysis.
It ensures proper organization of results and prevents overwriting existing files.
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
    import pandas as pd
    
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
    import matplotlib.pyplot as plt
    
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

def create_analysis_report(directory: str, results: Dict[str, Any],
                          base_filename: str = "analysis_report",
                          format: str = "html",
                          overwrite: bool = False) -> str:
    """
    Create an analysis report in the specified format.
    
    Parameters:
    -----------
    directory : str
        Directory where the file will be saved
    results : dict
        Analysis results to include in the report
    base_filename : str, optional
        Base name for the file (without extension)
    format : str, optional
        Report format ("html", "md", or "txt")
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
        filename = os.path.join(directory, f"{base_filename}.{format}")
    else:
        filename = get_unique_filename(directory, base_filename, format)
    
    # Generate report content
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if format.lower() == "html":
        content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DSF Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>DSF Analysis Report</h1>
            <p>Generated: {timestamp}</p>
        """
        
        # Add results sections
        for section, data in results.items():
            content += f"<h2>{section}</h2>"
            
            if isinstance(data, dict):
                content += "<table><tr><th>Parameter</th><th>Value</th></tr>"
                for key, value in data.items():
                    content += f"<tr><td>{key}</td><td>{value}</td></tr>"
                content += "</table>"
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                if data:
                    headers = data[0].keys()
                    content += "<table><tr>"
                    for header in headers:
                        content += f"<th>{header}</th>"
                    content += "</tr>"
                    
                    for item in data:
                        content += "<tr>"
                        for header in headers:
                            content += f"<td>{item.get(header, '')}</td>"
                        content += "</tr>"
                    
                    content += "</table>"
            else:
                content += f"<p>{data}</p>"
        
        content += """
        </body>
        </html>
        """
    
    elif format.lower() == "md":
        content = f"# DSF Analysis Report\n\nGenerated: {timestamp}\n\n"
        
        # Add results sections
        for section, data in results.items():
            content += f"## {section}\n\n"
            
            if isinstance(data, dict):
                content += "| Parameter | Value |\n|-----------|-------|\n"
                for key, value in data.items():
                    content += f"| {key} | {value} |\n"
                content += "\n"
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                if data:
                    headers = data[0].keys()
                    content += "| " + " | ".join(headers) + " |\n"
                    content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    
                    for item in data:
                        content += "| " + " | ".join(str(item.get(header, '')) for header in headers) + " |\n"
                    
                    content += "\n"
            else:
                content += f"{data}\n\n"
    
    else:  # Plain text
        content = f"DSF Analysis Report\n{'=' * 20}\n\nGenerated: {timestamp}\n\n"
        
        # Add results sections
        for section, data in results.items():
            content += f"{section}\n{'-' * len(section)}\n\n"
            
            if isinstance(data, dict):
                for key, value in data.items():
                    content += f"{key}: {value}\n"
                content += "\n"
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                if data:
                    headers = data[0].keys()
                    header_str = "  ".join(str(header).ljust(15) for header in headers)
                    content += header_str + "\n"
                    content += "  ".join("-" * 15 for _ in headers) + "\n"
                    
                    for item in data:
                        content += "  ".join(str(item.get(header, '')).ljust(15) for header in headers) + "\n"
                    
                    content += "\n"
            else:
                content += f"{data}\n\n"
    
    # Save report
    with open(filename, "w") as f:
        f.write(content)
    
    return filename

def check_output_conflicts(output_dir: str, required_files: List[str]) -> List[str]:
    """
    Check for potential file conflicts in the output directory.
    
    Parameters:
    -----------
    output_dir : str
        Output directory to check
    required_files : list
        List of filenames that will be created
        
    Returns:
    --------
    list
        List of conflicting files
    """
    conflicts = []
    
    if os.path.exists(output_dir):
        for filename in required_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                conflicts.append(filepath)
    
    return conflicts

def backup_directory(directory: str, backup_suffix: str = "_backup") -> str:
    """
    Create a backup of a directory.
    
    Parameters:
    -----------
    directory : str
        Directory to backup
    backup_suffix : str, optional
        Suffix to add to the backup directory name
        
    Returns:
    --------
    str
        Path to the backup directory
    """
    if not os.path.exists(directory):
        return None
    
    # Create backup directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{directory}{backup_suffix}_{timestamp}"
    
    # Copy directory
    shutil.copytree(directory, backup_dir)
    
    return backup_dir

def organize_output_files(output_dir: str, file_patterns: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Organize output files into subdirectories based on patterns.
    
    Parameters:
    -----------
    output_dir : str
        Output directory containing files
    file_patterns : dict
        Dictionary mapping subdirectory names to file patterns
        
    Returns:
    --------
    dict
        Dictionary mapping subdirectory names to lists of moved files
    """
    import glob
    
    moved_files = {}
    
    for subdir, pattern in file_patterns.items():
        # Create subdirectory
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        
        # Find matching files
        full_pattern = os.path.join(output_dir, pattern)
        matching_files = glob.glob(full_pattern)
        
        # Move files to subdirectory
        moved = []
        for file_path in matching_files:
            if os.path.isfile(file_path) and os.path.dirname(file_path) == output_dir:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(subdir_path, filename)
                
                # Ensure destination filename is unique
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    i = 1
                    while os.path.exists(os.path.join(subdir_path, f"{base}_{i}{ext}")):
                        i += 1
                    dest_path = os.path.join(subdir_path, f"{base}_{i}{ext}")
                
                # Move file
                shutil.move(file_path, dest_path)
                moved.append(dest_path)
        
        moved_files[subdir] = moved
    
    return moved_files

# Example usage
if __name__ == "__main__":
    # Create a unique output directory
    output_dir = create_output_directory("./output", "B2", "PM1", timestamp=True)
    print(f"Created output directory: {output_dir}")
    
    # Generate a unique filename
    filename = get_unique_filename(output_dir, "results", "csv")
    print(f"Unique filename: {filename}")
    
    # Create some test files
    with open(os.path.join(output_dir, "test1.txt"), "w") as f:
        f.write("Test file 1")
    
    with open(os.path.join(output_dir, "test2.txt"), "w") as f:
        f.write("Test file 2")
    
    with open(os.path.join(output_dir, "data.csv"), "w") as f:
        f.write("col1,col2\n1,2\n3,4")
    
    # Organize files
    file_patterns = {
        "text_files": "*.txt",
        "data_files": "*.csv"
    }
    
    moved_files = organize_output_files(output_dir, file_patterns)
    print("Organized files:")
    for subdir, files in moved_files.items():
        print(f"  {subdir}: {len(files)} files")
