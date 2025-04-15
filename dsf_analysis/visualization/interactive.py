"""
Interactive visualization functionality for DSF analysis.

This module provides functions for creating interactive visualizations of DSF data,
including heatmaps, interactive plots, and chemical structure visualizations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any

def create_interactive_heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str,
                              title: Optional[str] = None, colorscale: str = 'RdBu_r',
                              hover_template: Optional[str] = None, sig_figs: int = 2) -> go.Figure:
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
    # Import formatting function
    from dsf_analysis.utils.formatting import format_to_sig_figs
    
    # Round values to specified significant figures for display
    df = df.copy()
    if sig_figs > 0:
        # Format the value column to specified significant figures
        df[f'{value_col}_formatted'] = df[value_col].apply(
            lambda x: format_to_sig_figs(x, sig_figs) if pd.notnull(x) else "N/A"
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

def create_interactive_heatmap_with_structures(df: pd.DataFrame, x_col: str, y_col: str, 
                                             value_col: str, compound_col: str,
                                             title: Optional[str] = None, 
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
    compound_col : str
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
    # Try to import RDKit-related functions
    try:
        from dsf_analysis.visualization.smiles import (
            get_mol_for_compound,
            mol_to_plotly_image
        )
        RDKIT_AVAILABLE = True
    except ImportError:
        RDKIT_AVAILABLE = False
    
    # Import formatting function
    from dsf_analysis.utils.formatting import format_to_sig_figs
    
    # Round values to specified significant figures for display
    df = df.copy()
    if sig_figs > 0:
        # Format the value column to specified significant figures
        df[f'{value_col}_formatted'] = df[value_col].apply(
            lambda x: format_to_sig_figs(x, sig_figs) if pd.notnull(x) else "N/A"
        )
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
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
    ))
    
    # Add chemical structures if RDKit is available
    if RDKIT_AVAILABLE:
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

def create_interactive_curve_plot(df_raw: pd.DataFrame, best_fit: Dict, 
                                 title: Optional[str] = None) -> go.Figure:
    """
    Create an interactive plot of raw data and model fit using Plotly.
    
    Parameters:
    -----------
    df_raw : DataFrame
        Raw data for a single trace
    best_fit : dict
        Best fit result from select_best_model_for_trace
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure
    """
    if best_fit is None:
        print("No valid fit to plot.")
        return None
    
    # Import formatting function
    from dsf_analysis.utils.formatting import format_to_sig_figs
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Fluorescence vs Temperature", "First Derivative"),
        row_heights=[0.7, 0.3]
    )
    
    # Get model parameters
    model_name = best_fit["model"]
    popt = best_fit["popt"]
    t_start = best_fit["t_start"]
    t_end = best_fit["t_end"]
    tm1 = best_fit["Tm1"]
    tm2 = best_fit.get("Tm2", None)
    
    # Add raw data
    fig.add_trace(
        go.Scatter(
            x=df_raw["Temperature"],
            y=df_raw["Fluorescence"],
            mode="markers+lines",
            name="Raw Data",
            marker=dict(color="black", size=5, opacity=0.5),
            line=dict(color="black", width=1, dash="dot"),
            hovertemplate=(
                "<b>Temperature</b>: %{x:.2f}°C<br>" +
                "<b>Fluorescence</b>: %{y:.2f}"
            )
        ),
        row=1, col=1
    )
    
    # Extract the window data
    df_window = df_raw[(df_raw["Temperature"] >= t_start) & 
                       (df_raw["Temperature"] <= t_end)].copy()
    
    # Highlight the window used for fitting
    fig.add_vrect(
        x0=t_start, x1=t_end,
        fillcolor="gray", opacity=0.2,
        layer="below", line_width=0,
        row=1, col=1
    )
    
    # Generate dense x values for smooth curve
    temp_dense = np.linspace(t_start, t_end, 300)
    
    # Import model functions
    from dsf_analysis.core.models import MODEL_PARAMS
    
    # Get model function
    model_func = MODEL_PARAMS[model_name]["model_func"]
    
    # Normalize temperature for model input
    temp_norm = (temp_dense - t_start) / (t_end - t_start)
    
    # Calculate model predictions
    y_model_norm = model_func(temp_norm, *popt)
    
    # Denormalize model predictions
    f_min_window, f_max_window = df_window["Fluorescence"].min(), df_window["Fluorescence"].max()
    y_model_fit_real = y_model_norm * (f_max_window - f_min_window) + f_min_window
    
    # Create label with model info
    label = f"{model_name} (R²={format_to_sig_figs(best_fit['R2'], 3)}, Tm={format_to_sig_figs(tm1, 1)}°C"
    if tm2 is not None:
        label += f", Tm2={format_to_sig_figs(tm2, 1)}°C"
    label += ")"
    
    # Add model curve
    fig.add_trace(
        go.Scatter(
            x=temp_dense,
            y=y_model_fit_real,
            mode="lines",
            name=label,
            line=dict(color="red", width=2),
            hovertemplate=(
                "<b>Temperature</b>: %{x:.2f}°C<br>" +
                "<b>Fitted Value</b>: %{y:.2f}"
            )
        ),
        row=1, col=1
    )
    
    # Add vertical line at Tm1
    fig.add_vline(
        x=tm1,
        line=dict(color="gray", width=1, dash="dash"),
        row="all", col=1
    )
    
    # Add vertical line at Tm2 if available
    if tm2 is not None:
        fig.add_vline(
            x=tm2,
            line=dict(color="blue", width=1, dash="dot"),
            row="all", col=1
        )
    
    # Calculate derivative
    temp = df_raw["Temperature"].values
    fluor = df_raw["Fluorescence"].values
    deriv = np.gradient(fluor, temp)
    
    # Add derivative
    fig.add_trace(
        go.Scatter(
            x=temp,
            y=deriv,
            mode="lines",
            name="1st Derivative",
            line=dict(color="black", width=1),
            hovertemplate=(
                "<b>Temperature</b>: %{x:.2f}°C<br>" +
                "<b>dF/dT</b>: %{y:.4f}"
            )
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis2_title="Temperature (°C)",
        yaxis_title="Fluorescence (RFU)",
        yaxis2_title="dF/dT (RFU/°C)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=700,
        width=800,
        hovermode="closest"
    )
    
    return fig

def create_interactive_comparison_plot(df_list: List[pd.DataFrame], 
                                      labels: List[str],
                                      title: Optional[str] = None,
                                      colors: Optional[List[str]] = None) -> go.Figure:
    """
    Create an interactive comparison plot of multiple DSF curves.
    
    Parameters:
    -----------
    df_list : list
        List of DataFrames containing raw data
    labels : list
        List of labels for each curve
    title : str, optional
        Plot title
    colors : list, optional
        List of colors for each curve
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure
    """
    if not df_list or not labels or len(df_list) != len(labels):
        print("Invalid input: df_list and labels must have the same length.")
        return None
    
    # Default colors if not provided
    if colors is None:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Fluorescence vs Temperature", "First Derivative"),
        row_heights=[0.7, 0.3]
    )
    
    # Add each curve
    for i, (df, label) in enumerate(zip(df_list, labels)):
        color = colors[i % len(colors)]
        
        # Add raw data
        fig.add_trace(
            go.Scatter(
                x=df["Temperature"],
                y=df["Fluorescence"],
                mode="markers+lines",
                name=f"{label} (Raw)",
                marker=dict(color=color, size=5, opacity=0.5),
                line=dict(color=color, width=1, dash="dot"),
                hovertemplate=(
                    f"<b>{label}</b><br>" +
                    "<b>Temperature</b>: %{x:.2f}°C<br>" +
                    "<b>Fluorescence</b>: %{y:.2f}"
                )
            ),
            row=1, col=1
        )
        
        # Calculate derivative
        temp = df["Temperature"].values
        fluor = df["Fluorescence"].values
        deriv = np.gradient(fluor, temp)
        
        # Add derivative
        fig.add_trace(
            go.Scatter(
                x=temp,
                y=deriv,
                mode="lines",
                name=f"{label} (Derivative)",
                line=dict(color=color, width=1),
                hovertemplate=(
                    f"<b>{label}</b><br>" +
                    "<b>Temperature</b>: %{x:.2f}°C<br>" +
                    "<b>dF/dT</b>: %{y:.4f}"
                )
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis2_title="Temperature (°C)",
        yaxis_title="Fluorescence (RFU)",
        yaxis2_title="dF/dT (RFU/°C)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=700,
        width=800,
        hovermode="closest"
    )
    
    return fig
