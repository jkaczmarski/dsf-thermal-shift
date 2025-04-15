"""
Core models and fitting functionality for DSF analysis.

This module provides model definitions, fitting functions, and data processing
utilities for analyzing DSF melting curves.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Global cache for negative control fits
# Structure: {(protein, plate, rep, neg_control_well): best_fit_result}
NEG_CONTROL_CACHE = {}

# Global cache for model fits
# Structure: {(protein, plate, rep, well): {model_name: {window_size: fit_result}}}
MODEL_FIT_CACHE = {}

# Global cache for derivative data
# Structure: {(protein, plate, rep, well): derivative_df}
DERIVATIVE_CACHE = {}

# --- MODEL FUNCTIONS ---

def s1_model(x: np.ndarray, Asym: float, xmid: float, scal: float, d: float) -> np.ndarray:
    """
    Single sigmoidal model.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    Asym, xmid, scal, d : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return Asym / (1 + np.exp((xmid - x) / scal)) * np.exp(d * (x - xmid))

def s1_d_model(x: np.ndarray, Asym: float, xmid: float, scal: float, d: float, 
              id_d: float, id_b: float) -> np.ndarray:
    """
    Single sigmoidal model with decay.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    Asym, xmid, scal, d, id_d, id_b : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return s1_model(x, Asym, xmid, scal, d) + id_d * np.exp(id_b * x)

def s2_model(x: np.ndarray, Asym: float, xmid: float, scal: float, d: float, 
            Asym2: float, xmid2: float, scal2: float, d2: float) -> np.ndarray:
    """
    Double sigmoidal model.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    Asym, xmid, scal, d, Asym2, xmid2, scal2, d2 : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return s1_model(x, Asym, xmid, scal, d) + s1_model(x, Asym2, xmid2, scal2, d2)

def s2_d_model(x: np.ndarray, Asym: float, xmid: float, scal: float, d: float, 
              Asym2: float, xmid2: float, scal2: float, d2: float, 
              id_d: float, id_b: float) -> np.ndarray:
    """
    Double sigmoidal model with decay.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    Asym, xmid, scal, d, Asym2, xmid2, scal2, d2, id_d, id_b : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return s2_model(x, Asym, xmid, scal, d, Asym2, xmid2, scal2, d2) + id_d * np.exp(id_b * x)

def modified_boltzmann_model(x: np.ndarray, A: float, B: float, C: float, 
                           D: float, xmid: float, E: float) -> np.ndarray:
    """
    A two-slope-plus-sigmoid model ("modified Boltzmann"), adapted to use
    normalized temperature x in [0..1]. Interpreting 'xmid' as the
    normalized Tm, and E as the 'width' or slope factor.
    
    The form is:
      y = A*x + B + (C*x + D) / (1 + exp((xmid - x)/E))
      
    Parameters:
    -----------
    x : array-like
        Independent variable (normalized temperature)
    A, B, C, D, xmid, E : float
        Model parameters
        
    Returns:
    --------
    array-like
        Model predictions
    """
    return A*x + B + (C*x + D) / (1 + np.exp((xmid - x) / E))

# Dictionary to store model definitions and parameters
MODEL_PARAMS = {
    "s1": {
        "model_func": s1_model,
        "p0": [1, 0.6, 0.03, -1],
        "bounds": ([0.1, 0.1, 0.01, -10], [5, 0.95, 1, 10]),
        "tm_idx": [1]
    },
    "s1_d": {
        "model_func": s1_d_model,
        "p0": [1, 0.6, 0.03, -1, 0.01, -5],
        "bounds": ([0.1, 0.1, 0.01, -10, 0.001, -20], [5, 0.95, 1, 10, 5, 0]),
        "tm_idx": [1]
    },
    "s2": {
        "model_func": s2_model,
        "p0": [1, 0.6, 0.03, -1, 0.5, 0.65, 0.03, -2],
        "bounds": ([0.01, 0.1, 0.01, -10, 0.01, 0.1, 0.01, -10],
                   [5, 0.95, 1, 10, 5, 0.95, 1, 10]),
        "tm_idx": [1, 5]
    },
    "s2_d": {
        "model_func": s2_d_model,
        "p0": [1, 0.6, 0.03, -1, 0.5, 0.65, 0.03, -2, 0.01, -5],
        "bounds": ([0.01, 0.1, 0.01, -10, 0.01, 0.1, 0.01, -10, 0.001, -20],
                   [5, 0.95, 1, 10, 5, 0.95, 1, 10, 5, 0]),
        "tm_idx": [1, 5]
    }, 
    "mB": {
        "model_func": modified_boltzmann_model,
        # Initial guess for [A, B, C, D, xmid, E]
        "p0": [0, 0, 0, 0, 0.5, 0.01],
        # Bounds are just an example; tweak to fit your data scale
        "bounds": ([-10, -10, -10, -10, 0.0, 1e-5], [10, 10, 10, 10, 1.0, 0.1]),
        # Tm is the 5th parameter in the list (index=4) 
        "tm_idx": [4]
    }
}

# --- DATA PRE-PROCESSING & DERIVATIVE FUNCTIONS ---

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Temperature and Fluorescence, and store original bounds as attributes.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing Temperature and Fluorescence columns
        
    Returns:
    --------
    DataFrame
        DataFrame with added normalized columns and bounds attributes
    """
    df = df.copy()
    t_min, t_max = df["Temperature"].min(), df["Temperature"].max()
    f_min, f_max = df["Fluorescence"].min(), df["Fluorescence"].max()
    df["Temperature_norm"] = (df["Temperature"] - t_min) / (t_max - t_min)
    df["value_norm"] = (df["Fluorescence"] - f_min) / (f_max - f_min)
    df.attrs["T_bounds"] = (t_min, t_max)
    df.attrs["F_bounds"] = (f_min, f_max)
    return df

def find_derivative_peaks(df: pd.DataFrame, window_size: int = 11, 
                         derivative_threshold: float = 0.0002) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply Savitzky-Golay filter to compute the first and second derivatives,
    and detect peaks in the first derivative.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing temperature and fluorescence data
    window_size : int, optional
        Window size for Savitzky-Golay filter
    derivative_threshold : float, optional
        Threshold for peak detection
        
    Returns:
    --------
    tuple
        (DataFrame with derivatives, peak indices)
    """
    # Check if result is already in cache
    for key, cached_df in DERIVATIVE_CACHE.items():
        if id(df) == id(cached_df):
            # Return cached result
            peaks, _ = find_peaks(cached_df["sgd1"], height=derivative_threshold, distance=20)
            return cached_df, peaks
    
    df = df.copy()
    # Ensure an odd window size and a minimum value
    window_size = max(5, window_size if window_size % 2 == 1 else window_size+1)
    
    if "Temperature_norm" not in df.columns or "value_norm" not in df.columns:
        df = normalize_df(df)
    
    df["sgd1"] = savgol_filter(df["value_norm"], window_size, polyorder=3, deriv=1)
    df["sgd2"] = savgol_filter(df["value_norm"], window_size, polyorder=3, deriv=2)
    peaks, _ = find_peaks(df["sgd1"], height=derivative_threshold, distance=20)
    
    # Cache the result
    DERIVATIVE_CACHE[id(df)] = df
    
    return df, peaks

def get_deriv_peak_temp(df: pd.DataFrame) -> Optional[float]:
    """
    Return the Temperature corresponding to the highest peak in the first derivative.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing temperature and fluorescence data
        
    Returns:
    --------
    float or None
        Temperature at the highest derivative peak, or None if no peaks found
    """
    df_with_deriv, peaks = find_derivative_peaks(df)
    if len(peaks) == 0:
        return None
    best_peak_idx = peaks[np.argmax(df_with_deriv["sgd1"].iloc[peaks].values)]
    return df_with_deriv["Temperature"].iloc[best_peak_idx]

def compute_derivative(df: pd.DataFrame, window: int = 11, polyorder: int = 2) -> pd.DataFrame:
    """
    Compute and add the first derivative to the dataframe.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing temperature and fluorescence data
    window : int, optional
        Window size for Savitzky-Golay filter
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filter
        
    Returns:
    --------
    DataFrame
        DataFrame with added derivative column
    """
    df = df.copy().sort_values("Temperature").reset_index(drop=True)
    x = df["Temperature"].values
    y = df["value_norm"].values
    delta = np.mean(np.diff(x)) if len(x) > 1 else 1.0
    window = max(5, window if window % 2 == 1 else window+1)
    deriv = savgol_filter(y, window_length=window, polyorder=polyorder, deriv=1, delta=delta)
    df["Derivative"] = deriv
    return df

def estimate_minor_peak_temp(df: pd.DataFrame, t_min: float, t_max: float, 
                           edge_margin: float = 5, window_size: int = 11, 
                           major_peak: Optional[float] = None) -> Optional[float]:
    """
    Estimate a candidate minor peak (for Tm2) from the second derivative.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with temperature and fluorescence data
    t_min, t_max : float
        Temperature bounds for the current window
    edge_margin : float, optional
        Margin to exclude from edges
    window_size : int, optional
        Window size for Savitzky-Golay filter
    major_peak : float, optional
        Temperature of the major peak to avoid
        
    Returns:
    --------
    float or None
        Normalized temperature of the minor peak, or None if no candidate qualifies
    """
    # Work on a copy
    df = df.copy()
    # Compute first and second derivatives on normalized fluorescence
    df["sgd1"] = savgol_filter(df["value_norm"], window_size, polyorder=3, deriv=1)
    df["sgd2"] = savgol_filter(df["value_norm"], window_size, polyorder=3, deriv=2)
    
    # Only consider points that are not near the edges
    valid_idxs = [i for i in range(len(df))
                  if (df["Temperature"].iloc[i] >= t_min + edge_margin and 
                      df["Temperature"].iloc[i] <= t_max - edge_margin)]
    if not valid_idxs:
        return None

    # Find valleys in sgd2 (i.e. peaks in the inverted signal)
    valleys, _ = find_peaks(-df["sgd2"])
    
    # Filter valleys: they must be in valid_idxs and have sgd1 > 0
    candidate_idxs = []
    for i in valleys:
        if i in valid_idxs and df["sgd1"].iloc[i] > 0:
            # Convert normalized temperature to real temperature
            T_real = df["Temperature_norm"].iloc[i] * (t_max - t_min) + t_min
            if major_peak is not None and abs(T_real - major_peak) < 4:
                continue  # Ignore candidate too close to major peak
            candidate_idxs.append(i)
    
    if not candidate_idxs:
        return None
    
    # Choose the candidate with the most negative sgd2 (deepest trough)
    chosen_idx = min(candidate_idxs, key=lambda i: df["sgd2"].iloc[i])
    return df["Temperature_norm"].iloc[chosen_idx]

# --- MODEL FITTING & ANALYSIS FUNCTIONS ---

def fit_model_generic(model_name: str, x: np.ndarray, y: np.ndarray, 
                     custom_p0: Optional[List[float]] = None) -> Optional[np.ndarray]:
    """
    Generic model fitting function that selects the model function, initial guess, and bounds
    based on the model name from MODEL_PARAMS.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to fit
    x : array-like
        Independent variable values
    y : array-like
        Dependent variable values
    custom_p0 : list, optional
        Custom initial parameter values
        
    Returns:
    --------
    array-like or None
        Optimized parameters or None if fitting failed
    """
    if model_name not in MODEL_PARAMS:
        raise ValueError("Unknown model name. Choose from: " + ", ".join(MODEL_PARAMS.keys()))
    
    params = MODEL_PARAMS[model_name]
    p0 = custom_p0 if custom_p0 is not None else params["p0"]
    bounds = params["bounds"]
    model_func = params["model_func"]
    
    try:
        popt, _ = curve_fit(model_func, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except Exception as e:
        print(f"[Fit failed] {model_name}: {e}")
        return None

def select_best_model_for_trace(df_raw: pd.DataFrame, 
                              candidate_models: Optional[List[str]] = None, 
                              window_mode: str = "optimized", 
                              window_sizes: Optional[List[int]] = None, 
                              fixed_window_size: int = 20, 
                              verbose: bool = False, 
                              show_plot: bool = False, 
                              protein: Optional[str] = None, 
                              plate_ID: Optional[str] = None, 
                              rep: Optional[str] = None, 
                              well: Optional[str] = None, 
                              ligand: Optional[str] = None) -> Tuple[Optional[Dict], pd.DataFrame]:
    """
    For a given trace, try different models and window sizes to find the best fit.
    
    Parameters:
    -----------
    df_raw : DataFrame
        Raw data for a single trace
    candidate_models : list, optional
        List of model names to try
    window_mode : str, optional
        Mode for window selection ('optimized', 'fixed', or 'full')
    window_sizes : list, optional
        List of window sizes to try
    fixed_window_size : int, optional
        Fixed window size to use if window_mode='fixed'
    verbose : bool, optional
        Whether to print verbose output
    show_plot : bool, optional
        Whether to show a plot of the fit
    protein, plate_ID, rep, well, ligand : str, optional
        Identifiers for the current trace
        
    Returns:
    --------
    tuple
        (best_fit_result, summary_df)
    """
    # Check if result is already in cache
    cache_key = (protein, plate_ID, rep, well)
    if cache_key in MODEL_FIT_CACHE:
        model_cache = MODEL_FIT_CACHE[cache_key]
        
        # Check if we have results for the requested models and window sizes
        for model_name in candidate_models or list(MODEL_PARAMS.keys()):
            if model_name in model_cache:
                if window_mode == "fixed" and fixed_window_size in model_cache[model_name]:
                    # Return cached result for fixed window
                    best_fit = model_cache[model_name][fixed_window_size]
                    return best_fit, pd.DataFrame([best_fit])
                elif window_mode == "optimized" and all(ws in model_cache[model_name] for ws in window_sizes or [20]):
                    # Return best cached result for optimized window
                    all_fits = [model_cache[model_name][ws] for ws in window_sizes or [20]]
                    best_fit = max(all_fits, key=lambda x: x["R2"])
                    return best_fit, pd.DataFrame(all_fits)
    
    if candidate_models is None:
        candidate_models = list(MODEL_PARAMS.keys())
    if window_sizes is None:
        window_sizes = [10, 15, 20, 30, 40]
    
    # Initialize cache for this trace if needed
    if cache_key not in MODEL_FIT_CACHE:
        MODEL_FIT_CACHE[cache_key] = {}
    
    # Normalize the raw data
    df_norm_all = normalize_df(df_raw)
    t_min_all, t_max_all = df_norm_all.attrs["T_bounds"]
    
    all_fit_results = []
    
    for model_name in candidate_models:
        # Initialize model cache if needed
        if model_name not in MODEL_FIT_CACHE[cache_key]:
            MODEL_FIT_CACHE[cache_key][model_name] = {}
            
        if window_mode == "full":
            window_sizes_to_try = [None]  # Just one "window" = the full range
        elif window_mode == "fixed":
            window_sizes_to_try = [fixed_window_size]
        else:  # "optimized"
            window_sizes_to_try = window_sizes
        
        for window_size in window_sizes_to_try:
            if window_size in MODEL_FIT_CACHE[cache_key][model_name]:
                # Use cached result
                fit_result = MODEL_FIT_CACHE[cache_key][model_name][window_size]
                all_fit_results.append(fit_result)
                continue
                
            if window_size is None:
                # Use the full temperature range
                df_window = df_norm_all
                t_start, t_end = t_min_all, t_max_all
            else:
                # Find the derivative peak as a starting point
                deriv_peak_temp = get_deriv_peak_temp(df_raw)
                if deriv_peak_temp is None:
                    # If no peak found, use the midpoint of the temperature range
                    deriv_peak_temp = (t_min_all + t_max_all) / 2
                
                # Define the window around the peak
                half_window = window_size / 2
                t_start = max(t_min_all, deriv_peak_temp - half_window)
                t_end = min(t_max_all, deriv_peak_temp + half_window)
                
                # Extract data within the window
                df_window = df_norm_all[(df_norm_all["Temperature"] >= t_start) & 
                                       (df_norm_all["Temperature"] <= t_end)].copy()
            
            # Skip if window is too small
            if len(df_window) < 5:
                continue
                
            # Normalize within the window
            df_window_norm = normalize_df(df_window)
            
            # Fit the model
            x = df_window_norm["Temperature_norm"].values
            y = df_window_norm["value_norm"].values
            
            popt = fit_model_generic(model_name, x, y)
            if popt is None:
                continue
                
            # Calculate R² for the fit
            model_func = MODEL_PARAMS[model_name]["model_func"]
            y_pred = model_func(x, *popt)
            r2 = r2_score(y, y_pred)
            
            # Calculate Tm values (convert from normalized to real temperature)
            t_min_window, t_max_window = df_window_norm.attrs["T_bounds"]
            tm_idxs = MODEL_PARAMS[model_name]["tm_idx"]
            tm_values = []
            
            for tm_idx in tm_idxs:
                tm_norm = popt[tm_idx]
                tm_real = tm_norm * (t_max_window - t_min_window) + t_min_window
                tm_values.append(tm_real)
            
            # Store the fit result
            fit_result = {
                "model": model_name,
                "window_size": window_size,
                "t_start": t_start,
                "t_end": t_end,
                "popt": popt,
                "R2": r2,
                "Tm1": tm_values[0],
                "protein": protein,
                "plate_ID": plate_ID,
                "rep": rep,
                "well": well,
                "ligand": ligand
            }
            
            if len(tm_values) > 1:
                fit_result["Tm2"] = tm_values[1]
            
            # Cache the result
            MODEL_FIT_CACHE[cache_key][model_name][window_size] = fit_result
            all_fit_results.append(fit_result)
    
    if not all_fit_results:
        return None, pd.DataFrame()
    
    # Find the best fit based on R²
    best_fit = max(all_fit_results, key=lambda x: x["R2"])
    summary_df = pd.DataFrame(all_fit_results)
    
    # Plot if requested
    if show_plot:
        from dsf_analysis.visualization.plotting import plot_model_fit
        plot_model_fit(df_raw, best_fit, title=f"{protein}, {ligand} ({plate_ID}, {well}, {rep})")
    
    return best_fit, summary_df

def get_negative_control_fit(protein: str, plate: str, rep: str, 
                           neg_control_well: str = "A01", 
                           raw_data_dfs: Optional[Dict] = None, 
                           candidate_models: Optional[List[str]] = None,
                           window_mode: str = "optimized", 
                           window_sizes: Optional[List[int]] = None,
                           fixed_window_size: int = 20, 
                           verbose: bool = False) -> Optional[Dict]:
    """
    Get the model fit for a negative control well, using caching to avoid redundant computations.
    
    Parameters:
    -----------
    protein, plate, rep : str
        Identifiers for the protein, plate, and replicate
    neg_control_well : str, optional
        Well position for the negative control
    raw_data_dfs : dict, optional
        Nested dictionary of raw data
    candidate_models, window_mode, window_sizes, fixed_window_size, verbose : 
        Parameters for model fitting
        
    Returns:
    --------
    dict or None
        Best fit result for the negative control
    """
    # Check if result is already in cache
    cache_key = (protein, plate, rep, neg_control_well)
    if cache_key in NEG_CONTROL_CACHE:
        return NEG_CONTROL_CACHE[cache_key]
    
    # Get the negative control data
    if raw_data_dfs is None or protein not in raw_data_dfs or plate not in raw_data_dfs[protein] or \
       rep not in raw_data_dfs[protein][plate] or neg_control_well not in raw_data_dfs[protein][plate][rep]:
        if verbose:
            print(f"Negative control data not found for {protein}, {plate}, {rep}, {neg_control_well}")
        return None
    
    df_neg = raw_data_dfs[protein][plate][rep][neg_control_well]
    
    # Get ligand name if available
    ligand_neg = None
    if 'Ligand' in df_neg.columns:
        ligand_neg = df_neg['Ligand'].iloc[0]
    else:
        from dsf_analysis.core.data_loading import get_ligand_for_plate_well
        ligand_neg = get_ligand_for_plate_well(raw_data_dfs, plate, neg_control_well)
    
    # Fit the model
    best_fit_neg, _ = select_best_model_for_trace(
        df_neg, candidate_models=candidate_models,
        window_mode=window_mode, window_sizes=window_sizes,
        fixed_window_size=fixed_window_size, verbose=verbose,
        protein=protein, plate_ID=plate, rep=rep,
        well=neg_control_well, ligand=ligand_neg
    )
    
    # Cache the result
    NEG_CONTROL_CACHE[cache_key] = best_fit_neg
    
    return best_fit_neg

def clear_caches() -> None:
    """
    Clear all global caches to free memory.
    
    Returns:
    --------
    None
    """
    global NEG_CONTROL_CACHE, MODEL_FIT_CACHE, DERIVATIVE_CACHE
    NEG_CONTROL_CACHE.clear()
    MODEL_FIT_CACHE.clear()
    DERIVATIVE_CACHE.clear()
    import gc
    gc.collect()
