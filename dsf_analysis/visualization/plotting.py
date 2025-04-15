"""
Visualization plotting functionality for DSF analysis.

This module provides functions for creating static plots of DSF data,
including model fits, replicate comparisons, and derivative plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

def plot_model_fit(df_raw: pd.DataFrame, best_fit: Dict, title: Optional[str] = None) -> plt.Figure:
    """
    Plot raw data and model fit.
    
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
    matplotlib.figure.Figure
        Figure object
    """
    if best_fit is None:
        print("No valid fit to plot.")
        return None
    
    # Create figure with two subplots
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
    
    # Mark Tm1 on derivative plot
    ax_deriv.axvline(x=tm1, linestyle="--", color="gray", zorder=9)
    
    # Mark Tm2 on derivative plot if available
    if tm2 is not None:
        ax_deriv.axvline(x=tm2, linestyle=":", color="blue", zorder=9)
    
    ax_deriv.set_xlabel("Temperature (°C)")
    ax_deriv.set_ylabel("dF/dT (RFU/°C)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def plot_reps_and_neg_control(protein: str, plate: str, treatment_well: str, 
                             neg_control_well: str = "A01", 
                             raw_data_dfs: Optional[Dict] = None, 
                             candidate_models: Optional[List[str]] = None,
                             window_mode: str = "optimized", 
                             window_sizes: Optional[List[int]] = None,
                             include_tmchange_subplot: bool = True) -> plt.Figure:
    """
    Plot replicates for a treatment well with negative control comparison.
    
    Parameters:
    -----------
    protein, plate : str
        Identifiers for the protein and plate
    treatment_well : str
        Well position for the treatment
    neg_control_well : str, optional
        Well position for the negative control
    raw_data_dfs : dict, optional
        Nested dictionary of raw data
    candidate_models, window_mode, window_sizes : 
        Parameters for model fitting
    include_tmchange_subplot : bool, optional
        Whether to include a subplot showing Tm changes
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Import necessary functions
    from dsf_analysis.core.models import get_negative_control_fit, select_best_model_for_trace
    from dsf_analysis.core.data_loading import get_ligand_for_plate_well
    
    # Check if raw_data_dfs is provided
    if raw_data_dfs is None:
        print("No raw data provided.")
        return None
    
    # Check if protein and plate exist in raw_data_dfs
    if protein not in raw_data_dfs or plate not in raw_data_dfs[protein]:
        print(f"Data not found for {protein}, {plate}")
        return None
    
    # Get all replicates for this protein/plate
    reps = list(raw_data_dfs[protein][plate].keys())
    
    # Check if treatment_well exists in any replicate
    treatment_exists = False
    for rep in reps:
        if treatment_well in raw_data_dfs[protein][plate][rep]:
            treatment_exists = True
            break
    
    if not treatment_exists:
        print(f"Treatment well {treatment_well} not found for {protein}, {plate}")
        return None
    
    # Get ligand names
    ligand_treatment = get_ligand_for_plate_well(raw_data_dfs, plate, treatment_well)
    ligand_neg = get_ligand_for_plate_well(raw_data_dfs, plate, neg_control_well)
    
    # Determine number of subplots
    n_subplots = 3 if include_tmchange_subplot else 2
    
    # Create figure
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 10), 
                            gridspec_kw={'height_ratios': [3, 2, 1] if include_tmchange_subplot else [3, 2]})
    
    # Set title
    fig.suptitle(f"{protein}, {plate}, {treatment_well} ({ligand_treatment}) vs {neg_control_well} ({ligand_neg})", 
                fontsize=14)
    
    # Plot raw data for each replicate
    ax_raw = axes[0]
    ax_deriv = axes[1]
    
    # Lists to store Tm values
    tm1_values_treatment = []
    tm1_values_neg = []
    
    # Process each replicate
    for rep in reps:
        # Skip if treatment_well or neg_control_well not in this replicate
        if treatment_well not in raw_data_dfs[protein][plate][rep] or \
           neg_control_well not in raw_data_dfs[protein][plate][rep]:
            continue
        
        # Get raw data
        df_treatment = raw_data_dfs[protein][plate][rep][treatment_well]
        df_neg = raw_data_dfs[protein][plate][rep][neg_control_well]
        
        # Plot raw data
        ax_raw.plot(df_treatment["Temperature"], df_treatment["Fluorescence"], 'o-', 
                   alpha=0.7, label=f"{treatment_well} ({rep})")
        ax_raw.plot(df_neg["Temperature"], df_neg["Fluorescence"], 'o-', 
                   alpha=0.3, label=f"{neg_control_well} ({rep})")
        
        # Calculate derivatives
        temp_treatment = df_treatment["Temperature"].values
        fluor_treatment = df_treatment["Fluorescence"].values
        deriv_treatment = np.gradient(fluor_treatment, temp_treatment)
        
        temp_neg = df_neg["Temperature"].values
        fluor_neg = df_neg["Fluorescence"].values
        deriv_neg = np.gradient(fluor_neg, temp_neg)
        
        # Plot derivatives
        ax_deriv.plot(temp_treatment, deriv_treatment, '-', 
                     alpha=0.7, label=f"{treatment_well} ({rep})")
        ax_deriv.plot(temp_neg, deriv_neg, '-', 
                     alpha=0.3, label=f"{neg_control_well} ({rep})")
        
        # Fit models
        best_fit_treatment, _ = select_best_model_for_trace(
            df_treatment, candidate_models=candidate_models,
            window_mode=window_mode, window_sizes=window_sizes,
            protein=protein, plate_ID=plate, rep=rep,
            well=treatment_well, ligand=ligand_treatment
        )
        
        best_fit_neg = get_negative_control_fit(
            protein=protein, plate=plate, rep=rep,
            neg_control_well=neg_control_well,
            raw_data_dfs=raw_data_dfs,
            candidate_models=candidate_models,
            window_mode=window_mode,
            window_sizes=window_sizes
        )
        
        # Store Tm values if fits were successful
        if best_fit_treatment is not None and "Tm1" in best_fit_treatment:
            tm1_values_treatment.append(best_fit_treatment["Tm1"])
            
            # Draw vertical line at Tm1
            ax_raw.axvline(x=best_fit_treatment["Tm1"], linestyle="--", 
                          color="gray", alpha=0.7, zorder=9)
            ax_deriv.axvline(x=best_fit_treatment["Tm1"], linestyle="--", 
                            color="gray", alpha=0.7, zorder=9)
        
        if best_fit_neg is not None and "Tm1" in best_fit_neg:
            tm1_values_neg.append(best_fit_neg["Tm1"])
            
            # Draw vertical line at Tm1
            ax_raw.axvline(x=best_fit_neg["Tm1"], linestyle=":", 
                          color="gray", alpha=0.3, zorder=8)
            ax_deriv.axvline(x=best_fit_neg["Tm1"], linestyle=":", 
                            color="gray", alpha=0.3, zorder=8)
    
    # Set labels
    ax_raw.set_ylabel("Fluorescence (RFU)")
    ax_raw.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_raw.legend(fontsize=9, loc='upper left')
    
    ax_deriv.set_ylabel("dF/dT (RFU/°C)")
    if not include_tmchange_subplot:
        ax_deriv.set_xlabel("Temperature (°C)")
    else:
        ax_deriv.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_deriv.legend(fontsize=9, loc='upper left')
    
    # Add Tm change subplot if requested
    if include_tmchange_subplot and tm1_values_treatment and tm1_values_neg:
        ax_tm = axes[2]
        
        # Calculate average Tm values
        avg_tm1_treatment = np.mean(tm1_values_treatment)
        avg_tm1_neg = np.mean(tm1_values_neg)
        
        # Calculate delta Tm
        delta_tm1 = avg_tm1_treatment - avg_tm1_neg
        
        # Create bar plot
        bars = ax_tm.bar([0, 1, 2], [avg_tm1_neg, avg_tm1_treatment, delta_tm1], 
                        color=['lightgray', 'darkgray', 'red' if delta_tm1 > 0 else 'blue'])
        
        # Add error bars for the first two bars
        if len(tm1_values_neg) > 1:
            std_tm1_neg = np.std(tm1_values_neg)
            ax_tm.errorbar(0, avg_tm1_neg, yerr=std_tm1_neg, fmt='none', color='black')
        
        if len(tm1_values_treatment) > 1:
            std_tm1_treatment = np.std(tm1_values_treatment)
            ax_tm.errorbar(1, avg_tm1_treatment, yerr=std_tm1_treatment, fmt='none', color='black')
        
        # Add labels
        ax_tm.set_xticks([0, 1, 2])
        ax_tm.set_xticklabels([f"{neg_control_well}\n({ligand_neg})", 
                              f"{treatment_well}\n({ligand_treatment})", 
                              "ΔTm"])
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            value = avg_tm1_neg if i == 0 else avg_tm1_treatment if i == 1 else delta_tm1
            ax_tm.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                      f"{value:.1f}°C", ha='center', va='bottom', fontsize=10)
        
        ax_tm.set_ylabel("Temperature (°C)")
        ax_tm.set_xlabel("")
        
        # Set y-limits to focus on the interesting region
        y_min = min(avg_tm1_neg, avg_tm1_treatment) - 5
        y_max = max(avg_tm1_neg, avg_tm1_treatment) + 5
        ax_tm.set_ylim(y_min, y_max)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def plot_derivative_peaks(df_raw: pd.DataFrame, window_size: int = 11, 
                         derivative_threshold: float = 0.0002, 
                         title: Optional[str] = None) -> plt.Figure:
    """
    Plot raw data with derivative and detected peaks.
    
    Parameters:
    -----------
    df_raw : DataFrame
        Raw data for a single trace
    window_size : int, optional
        Window size for Savitzky-Golay filter
    derivative_threshold : float, optional
        Threshold for peak detection
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Import necessary functions
    from dsf_analysis.core.models import find_derivative_peaks
    
    # Calculate derivatives and find peaks
    df_with_deriv, peaks = find_derivative_peaks(
        df_raw, window_size=window_size, derivative_threshold=derivative_threshold)
    
    # Create figure with two subplots
    fig, (ax_main, ax_deriv) = plt.subplots(2, 1, figsize=(10, 8), 
                                           gridspec_kw={'height_ratios': [3, 1]},
                                           sharex=True)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Plot raw data
    ax_main.plot(df_raw["Temperature"], df_raw["Fluorescence"], 'o-', 
                color="black", alpha=0.5, label="Raw Data")
    
    # Plot normalized data
    ax_main.plot(df_with_deriv["Temperature"], df_with_deriv["value_norm"], 'o-', 
                color="blue", alpha=0.5, label="Normalized Data")
    
    ax_main.set_ylabel("Fluorescence (RFU)")
    ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_main.legend(fontsize=9)
    
    # Plot derivative
    ax_deriv.plot(df_with_deriv["Temperature"], df_with_deriv["sgd1"], 
                 color="black", label="1st Derivative", zorder=3)
    
    # Mark peaks
    for peak in peaks:
        temp = df_with_deriv["Temperature"].iloc[peak]
        deriv = df_with_deriv["sgd1"].iloc[peak]
        ax_deriv.plot(temp, deriv, 'ro', zorder=10)
        ax_deriv.text(temp, deriv, f"{temp:.1f}°C", 
                     ha='center', va='bottom', fontsize=9)
        
        # Mark peak on main plot
        ax_main.axvline(x=temp, linestyle="--", color="red", zorder=9)
    
    ax_deriv.set_xlabel("Temperature (°C)")
    ax_deriv.set_ylabel("dF/dT (normalized)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
