"""
Additional Features Module for DSF Analysis
=========================================

This module provides additional features for the DSF analysis package:
- Plotly figure formatting with customizable significant figures
- Reanalysis of specific conditions with master dataframe updates
"""

import os
import re
import json
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def format_plotly_hover_template(fig, sig_figs=2):
    """
    Format all hover templates in a Plotly figure to use the specified number of significant figures.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Plotly figure to modify
    sig_figs : int, optional
        Number of significant figures to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Modified figure with updated hover templates
    """
    # Create a formatter function for the specified number of significant figures
    def format_value(val):
        if isinstance(val, (int, float)) and not np.isnan(val):
            return f"{{:.{sig_figs}g}}".format(val)
        return val
    
    # Process each trace in the figure
    for i, trace in enumerate(fig.data):
        # Skip if no hovertemplate
        if not hasattr(trace, 'hovertemplate') or trace.hovertemplate is None:
            continue
        
        # Get the current hovertemplate
        template = trace.hovertemplate
        
        # Replace numeric format specifiers with our custom format
        # This regex matches format specifiers like %{y:.3f} or %{customdata[0]:.2f}
        pattern = r'%\{([^}]+?)(:\.[\d]+[efgd])\}'
        
        def replace_format(match):
            var_name = match.group(1)
            return f'%{{{var_name}}}'
        
        # Remove existing format specifiers
        template = re.sub(pattern, replace_format, template)
        
        # Update the hovertemplate
        fig.data[i].hovertemplate = template
        
        # If the trace has customdata, format it to the specified significant figures
        if hasattr(trace, 'customdata') and trace.customdata is not None:
            # For 1D arrays of customdata
            if isinstance(trace.customdata, np.ndarray) and trace.customdata.ndim == 1:
                formatted_data = np.array([format_value(val) for val in trace.customdata])
                fig.data[i].customdata = formatted_data
            
            # For 2D arrays of customdata
            elif isinstance(trace.customdata, np.ndarray) and trace.customdata.ndim == 2:
                formatted_data = np.array([[format_value(val) for val in row] for row in trace.customdata])
                fig.data[i].customdata = formatted_data
    
    return fig

def format_plotly_legend(fig, sig_figs=2):
    """
    Format all legend entries in a Plotly figure to use the specified number of significant figures.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Plotly figure to modify
    sig_figs : int, optional
        Number of significant figures to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Modified figure with updated legend entries
    """
    # Create a formatter function for the specified number of significant figures
    def format_value(val):
        if isinstance(val, (int, float)) and not np.isnan(val):
            return f"{{:.{sig_figs}g}}".format(val)
        return val
    
    # Process each trace in the figure
    for i, trace in enumerate(fig.data):
        # Skip if no name
        if not hasattr(trace, 'name') or trace.name is None:
            continue
        
        # Get the current name
        name = trace.name
        
        # Replace numeric values with formatted values
        # This regex matches numeric values in the legend name
        pattern = r'(\d+\.\d+)'
        
        def replace_number(match):
            try:
                val = float(match.group(1))
                return format_value(val)
            except:
                return match.group(1)
        
        # Format numeric values in the name
        name = re.sub(pattern, replace_number, name)
        
        # Update the trace name
        fig.data[i].name = name
    
    return fig

def format_plotly_figure(fig, sig_figs=2):
    """
    Format a Plotly figure to use the specified number of significant figures
    for both hover templates and legend entries.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Plotly figure to modify
    sig_figs : int, optional
        Number of significant figures to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Modified figure with updated formatting
    """
    fig = format_plotly_hover_template(fig, sig_figs)
    fig = format_plotly_legend(fig, sig_figs)
    return fig

def create_interactive_heatmap(df, x_col, y_col, value_col, title=None, colorscale='RdBu_r',
                              hover_template=None, sig_figs=2):
    """
    Create an interactive heatmap using Plotly with customizable significant figures.
    
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
    
    # Create the hovertemplate
    if hover_template is None:
        hover_template = (
            f"<b>{x_col}</b>: %{{x}}<br>" +
            f"<b>{y_col}</b>: %{{y}}<br>" +
            f"<b>{value_col}</b>: %{{customdata}}"
        )
    
    # Create the heatmap
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

def reanalyze_conditions(raw_data_dfs, master_df_path, conditions, 
                        output_path=None, candidate_models=None,
                        window_mode="optimized", window_sizes=None,
                        verbose=False):
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
    verbose : bool, optional
        Whether to print verbose output
        
    Returns:
    --------
    DataFrame
        Updated master dataframe
    """
    # Import here to avoid circular imports
    from dsf_optimized import select_best_model_for_trace, get_ligand_for_plate_well
    
    # Load the master dataframe
    if not os.path.exists(master_df_path):
        raise FileNotFoundError(f"Master dataframe not found: {master_df_path}")
    
    master_df = pd.read_csv(master_df_path)
    
    if verbose:
        print(f"Loaded master dataframe with {len(master_df)} rows")
        print(f"Reanalyzing {len(conditions)} conditions")
    
    # Reanalyze each condition
    for condition in conditions:
        protein = condition['protein']
        plate = condition['plate']
        rep = condition['rep']
        well = condition['well']
        
        if verbose:
            print(f"Reanalyzing {protein}, {plate}, {rep}, {well}")
        
        # Skip if data not available
        if (protein not in raw_data_dfs or plate not in raw_data_dfs[protein] or
            rep not in raw_data_dfs[protein][plate] or well not in raw_data_dfs[protein][plate][rep]):
            warnings.warn(f"Data not found for {protein}, {plate}, {rep}, {well}. Skipping.")
            continue
        
        # Get the raw data
        df_raw = raw_data_dfs[protein][plate][rep][well]
        ligand = get_ligand_for_plate_well(raw_data_dfs, plate, well)
        
        # Fit the model
        best_fit, _ = select_best_model_for_trace(
            df_raw, candidate_models=candidate_models,
            window_mode=window_mode, window_sizes=window_sizes,
            protein=protein, plate_ID=plate, rep=rep,
            well=well, ligand=ligand,
            verbose=verbose
        )
        
        if best_fit is None:
            warnings.warn(f"Model fitting failed for {protein}, {plate}, {rep}, {well}. Skipping.")
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
            
            if verbose:
                print(f"Updated existing row in master dataframe")
        else:
            # Add new row
            new_row = pd.DataFrame([best_fit])
            master_df = pd.concat([master_df, new_row], ignore_index=True)
            
            if verbose:
                print(f"Added new row to master dataframe")
    
    # Save the updated dataframe if output path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        master_df.to_csv(output_path, index=False)
        
        if verbose:
            print(f"Saved updated master dataframe to {output_path}")
    
    return master_df

def batch_reanalyze_by_criteria(raw_data_dfs, master_df_path, criteria, 
                               output_path=None, candidate_models=None,
                               window_mode="optimized", window_sizes=None,
                               verbose=False):
    """
    Reanalyze conditions that match specified criteria and update the master dataframe.
    
    Parameters:
    -----------
    raw_data_dfs : dict
        Nested dictionary of raw data
    master_df_path : str
        Path to the master dataframe CSV
    criteria : dict
        Dictionary of column-value pairs to filter conditions
    output_path : str, optional
        Path to save the updated master dataframe
    candidate_models, window_mode, window_sizes : 
        Parameters for model fitting
    verbose : bool, optional
        Whether to print verbose output
        
    Returns:
    --------
    DataFrame
        Updated master dataframe
    """
    # Load the master dataframe
    if not os.path.exists(master_df_path):
        raise FileNotFoundError(f"Master dataframe not found: {master_df_path}")
    
    master_df = pd.read_csv(master_df_path)
    
    # Filter rows that match the criteria
    query = ' & '.join([f"{col} == '{val}'" if isinstance(val, str) else f"{col} == {val}" 
                        for col, val in criteria.items()])
    
    filtered_df = master_df.query(query) if query else master_df
    
    if verbose:
        print(f"Found {len(filtered_df)} conditions matching criteria: {criteria}")
    
    # Convert filtered rows to conditions list
    conditions = []
    for _, row in filtered_df.iterrows():
        condition = {
            'protein': row['protein'],
            'plate': row['plate_ID'],
            'rep': row['rep'],
            'well': row['well']
        }
        conditions.append(condition)
    
    # Reanalyze the conditions
    return reanalyze_conditions(
        raw_data_dfs=raw_data_dfs,
        master_df_path=master_df_path,
        conditions=conditions,
        output_path=output_path,
        candidate_models=candidate_models,
        window_mode=window_mode,
        window_sizes=window_sizes,
        verbose=verbose
    )

def compare_analysis_results(original_df_path, updated_df_path, output_path=None):
    """
    Compare original and updated analysis results to identify changes.
    
    Parameters:
    -----------
    original_df_path : str
        Path to the original master dataframe CSV
    updated_df_path : str
        Path to the updated master dataframe CSV
    output_path : str, optional
        Path to save the comparison results
        
    Returns:
    --------
    DataFrame
        DataFrame with comparison results
    """
    # Load the dataframes
    original_df = pd.read_csv(original_df_path)
    updated_df = pd.read_csv(updated_df_path)
    
    # Create a unique identifier for each condition
    def create_id(row):
        return f"{row['protein']}_{row['plate_ID']}_{row['rep']}_{row['well']}"
    
    original_df['condition_id'] = original_df.apply(create_id, axis=1)
    updated_df['condition_id'] = updated_df.apply(create_id, axis=1)
    
    # Find common conditions
    common_ids = set(original_df['condition_id']).intersection(set(updated_df['condition_id']))
    
    # Initialize comparison dataframe
    comparison_columns = ['condition_id', 'protein', 'plate_ID', 'rep', 'well', 'ligand',
                         'original_model', 'updated_model', 'original_Tm1', 'updated_Tm1',
                         'Tm1_change', 'original_R2', 'updated_R2', 'R2_change']
    
    comparison_df = pd.DataFrame(columns=comparison_columns)
    
    # Compare each common condition
    for condition_id in common_ids:
        original_row = original_df[original_df['condition_id'] == condition_id].iloc[0]
        updated_row = updated_df[updated_df['condition_id'] == condition_id].iloc[0]
        
        # Calculate changes
        tm1_change = updated_row['Tm1'] - original_row['Tm1']
        r2_change = updated_row['R2'] - original_row['R2']
        
        # Create comparison row
        comparison_row = {
            'condition_id': condition_id,
            'protein': original_row['protein'],
            'plate_ID': original_row['plate_ID'],
            'rep': original_row['rep'],
            'well': original_row['well'],
            'ligand': original_row['ligand'],
            'original_model': original_row['model'],
            'updated_model': updated_row['model'],
            'original_Tm1': original_row['Tm1'],
            'updated_Tm1': updated_row['Tm1'],
            'Tm1_change': tm1_change,
            'original_R2': original_row['R2'],
            'updated_R2': updated_row['R2'],
            'R2_change': r2_change
        }
        
        # Add to comparison dataframe
        comparison_df = pd.concat([comparison_df, pd.DataFrame([comparison_row])], ignore_index=True)
    
    # Find new conditions in updated dataframe
    new_ids = set(updated_df['condition_id']) - set(original_df['condition_id'])
    for condition_id in new_ids:
        updated_row = updated_df[updated_df['condition_id'] == condition_id].iloc[0]
        
        # Create comparison row 
(Content truncated due to size limit. Use line ranges to read in chunks)