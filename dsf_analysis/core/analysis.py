"""
Core analysis functionality for DSF analysis.

This module provides the main analysis functions for processing DSF data,
including the primary analyze_dsf_data function that orchestrates the entire workflow.
"""

import os
import gc
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from dsf_analysis.core.data_loading import (
    load_data_with_derivative_boltzmann,
    get_ligand_for_plate_well
)

from dsf_analysis.core.models import (
    select_best_model_for_trace,
    get_negative_control_fit,
    clear_caches
)

def analyze_dsf_data(parent_folder: str, plate_map_folder: str, output_dir: str,
                    candidate_models: Optional[List[str]] = None,
                    window_mode: str = "optimized",
                    window_sizes: Optional[List[int]] = None,
                    neg_control_well: str = "A01",
                    protein_filter: Optional[List[str]] = None,
                    plate_filter: Optional[List[str]] = None,
                    well_filter: Optional[List[str]] = None,
                    use_uniprot: bool = False,
                    uniprot_mapping: Optional[str] = None,
                    use_smiles: bool = False,
                    smiles_mapping: Optional[str] = None,
                    create_plots: bool = True,
                    create_heatmaps: bool = True,
                    verbose: bool = True) -> Tuple[Dict, pd.DataFrame, str]:
    """
    Main function to analyze DSF data.
    
    Parameters:
    -----------
    parent_folder : str
        Path to the parent folder containing DSF data
    plate_map_folder : str
        Path to the folder containing plate map CSV files
    output_dir : str
        Path to the output directory
    candidate_models : list, optional
        List of model names to try
    window_mode : str, optional
        Mode for window selection ('optimized', 'fixed', or 'full')
    window_sizes : list, optional
        List of window sizes to try
    neg_control_well : str, optional
        Well position for the negative control
    protein_filter, plate_filter, well_filter : list, optional
        Lists of proteins, plates, and wells to include
    use_uniprot : bool, optional
        Whether to use UniProt IDs for protein names
    uniprot_mapping : str, optional
        Path to UniProt mapping file
    use_smiles : bool, optional
        Whether to load SMILES codes for compounds
    smiles_mapping : str, optional
        Path to SMILES mapping file
    create_plots : bool, optional
        Whether to create plots for each well
    create_heatmaps : bool, optional
        Whether to create heatmaps for each protein/plate
    verbose : bool, optional
        Whether to print verbose output
        
    Returns:
    --------
    tuple
        (raw_data_dfs, master_df, output_dir)
    """
    # Import here to avoid circular imports
    from dsf_analysis.utils.mapping import load_uniprot_mapping, load_smiles_mapping
    from dsf_analysis.utils.output import create_output_directory, save_dataframe
    
    # Load mappings if provided
    if use_uniprot and uniprot_mapping:
        load_uniprot_mapping(uniprot_mapping)
    
    if use_smiles and smiles_mapping:
        load_smiles_mapping(smiles_mapping)
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    if verbose:
        print(f"Loading data from {parent_folder}...")
    
    raw_data_dfs, derivative_data_dfs, boltzmann_data_dfs, plate_map_dict = load_data_with_derivative_boltzmann(
        parent_folder=parent_folder,
        plate_map_folder=plate_map_folder,
        protein_filter=protein_filter,
        plate_filter=plate_filter,
        well_filter=well_filter,
        use_uniprot=use_uniprot,
        use_smiles=use_smiles
    )
    
    if verbose:
        print(f"Loaded data for {len(raw_data_dfs)} proteins")
    
    # Initialize results list
    all_results = []
    
    # Process each protein, plate, rep, well
    for protein in raw_data_dfs:
        for plate in raw_data_dfs[protein]:
            # Create protein/plate specific output directory
            plate_output_dir = create_output_directory(output_dir, protein, plate)
            
            if verbose:
                print(f"Processing {protein}, {plate}...")



         
            # Get model fit for negative control (use first rep that has it)
            neg_fit = None
            for rep in raw_data_dfs[protein][plate]:
                if neg_control_well in raw_data_dfs[protein][plate][rep]:
                    neg_fit = get_negative_control_fit(
                        protein=protein,
                        plate=plate,
                        rep=rep,
                        neg_control_well=neg_control_well,
                        raw_data_dfs=raw_data_dfs,
                        candidate_models=candidate_models,
                        window_mode=window_mode,
                        window_sizes=window_sizes,
                        verbose=verbose
                    )
                    break

            # Get Tm_D for negative control
            tm_d_neg = None
            for rep in raw_data_dfs[protein][plate]:
                if neg_control_well in raw_data_dfs[protein][plate][rep]:
                    df_neg = raw_data_dfs[protein][plate][rep][neg_control_well]
                    temp_neg = df_neg["Temperature"].values
                    fluor_neg = df_neg["Fluorescence"].values
                    deriv_neg = np.gradient(fluor_neg, temp_neg)
                    tm_d_neg = temp_neg[np.argmax(deriv_neg)]
                    break

            for rep in raw_data_dfs[protein][plate]: 
                # Process each well
                for well in raw_data_dfs[protein][plate][rep]:
                    # Skip negative control well
                    if well == neg_control_well:
                        continue
                    
                    # Get raw data for this well
                    df_raw = raw_data_dfs[protein][plate][rep][well]
                    
                    # Get ligand name
                    ligand = get_ligand_for_plate_well(raw_data_dfs, plate, well)
                    print(f"Processing {protein}, {plate}, {rep}, {well}")
                    # Fit model
                    best_fit, _ = select_best_model_for_trace(
                        df_raw=df_raw,
                        candidate_models=candidate_models,
                        window_mode=window_mode,
                        window_sizes=window_sizes,
                        verbose=verbose,
                        protein=protein,
                        plate_ID=plate,
                        rep=rep,
                        well=well,
                        ligand=ligand
                    )
                    
                    if best_fit is None:
                        warnings.warn(f"Failed to fit model for {protein}, {plate}, {rep}, {well}")
                        continue

                    # Calculate Tm D (derivative peak) for this well
                    temp = df_raw["Temperature"].values
                    fluor = df_raw["Fluorescence"].values
                    deriv = np.gradient(fluor, temp)
                    tm_d = temp[np.argmax(deriv)]
                    best_fit["Tm_D"] = tm_d

                    # Add control comparison values if available
                    if neg_fit is not None and "Tm1" in neg_fit:
                        best_fit["Tm1_neg"] = neg_fit["Tm1"]
                        best_fit["dTm1"] = best_fit["Tm1"] - neg_fit["Tm1"]

                        if "Tm2" in best_fit and "Tm2" in neg_fit:
                            best_fit["Tm2_neg"] = neg_fit["Tm2"]
                            best_fit["dTm2"] = best_fit["Tm2"] - neg_fit["Tm2"]

                    if tm_d_neg is not None:
                        best_fit["Tm_D_neg"] = tm_d_neg
                        best_fit["dTm_D"] = tm_d - tm_d_neg


                    
                    # Add to results list
                    all_results.append(best_fit)
                    
                    # Create plot if requested
                    if create_plots:
                        from dsf_analysis.visualization.plotting import plot_model_fit, plot_reps_and_neg_control
                        from dsf_analysis.utils.output import save_figure
                        import matplotlib.pyplot as plt
                        

                        # # Plot model fit
                        # fig = None
                        # try:
                        #     fig = plot_model_fit(
                        #         df_raw=df_raw,
                        #         best_fit=best_fit,
                        #         title=f"{protein}, {plate}, {rep}, {well}, {ligand}"
                        #     )
                        #     if fig:
                        #         save_figure(
                        #             fig=fig,
                        #             directory=plate_output_dir,
                        #             base_filename=f"{protein}_{plate}_{rep}_{well}",
                        #             formats=["png"],
                        #             overwrite=True
                        #         )
                        # finally:
                        #     if fig:
                        #         plt.close(fig)

                        # Plot with negative control comparison
                        fig2 = None
                        try:
                            fig2 = plot_reps_and_neg_control(
                                protein=protein,
                                plate=plate,
                                treatment_well=well,
                                neg_control_well=neg_control_well,
                                raw_data_dfs=raw_data_dfs,
                                candidate_models=candidate_models,
                                window_mode=window_mode,
                                window_sizes=window_sizes,
                                include_tmchange_subplot=True
                            )
                            if fig2:
                                save_figure(
                                    fig=fig2,
                                    directory=plate_output_dir,
                                    base_filename=f"{protein}_{plate}_{rep}_{well}_with_neg",
                                    formats=["png"],
                                    overwrite=True
                                )
                        finally:
                            if fig2:
                                plt.close(fig2)


                        
                # Free memory after processing each replicate
                gc.collect()
    
    # Create master dataframe
    master_df = pd.DataFrame(all_results)
    
    # Save master dataframe
    if len(master_df) > 0:
        save_dataframe(
            df=master_df,
            directory=output_dir,
            base_filename="master_results",
            extension="csv",
            index=False,
            overwrite=True
        )
    
    # Create heatmaps if requested
    if create_heatmaps and len(master_df) > 0:
        try:
            from dsf_analysis.visualization.interactive import create_interactive_heatmap
            from dsf_analysis.utils.output import save_plotly_figure
            
            # Create heatmap for each protein/plate combination
            for protein in master_df["protein"].unique():
                for plate in master_df[master_df["protein"] == protein]["plate_ID"].unique():
                    # Filter data for this protein/plate
                    df_subset = master_df[(master_df["protein"] == protein) & 
                                         (master_df["plate_ID"] == plate)]
                    
                    # Create heatmap for Tm1
                    fig = create_interactive_heatmap(
                        df=df_subset,
                        x_col="ligand",
                        y_col="well",
                        value_col="Tm1",
                        title=f"{protein}, {plate} - Tm1",
                        sig_figs=2
                    )
                    
                    # Save heatmap
                    save_plotly_figure(
                        fig=fig,
                        directory=create_output_directory(output_dir, protein, plate),
                        base_filename=f"{protein}_{plate}_Tm1_heatmap",
                        formats=["html", "png"],
                        overwrite=True
                    )
                    
                    # Create heatmap for dTm1 if available
                    if "dTm1" in df_subset.columns:
                        fig = create_interactive_heatmap(
                            df=df_subset,
                            x_col="ligand",
                            y_col="well",
                            value_col="dTm1",
                            title=f"{protein}, {plate} - ΔTm1",
                            sig_figs=2
                        )
                        
                        # Save heatmap
                        save_plotly_figure(
                            fig=fig,
                            directory=create_output_directory(output_dir, protein, plate),
                            base_filename=f"{protein}_{plate}_dTm1_heatmap",
                            formats=["html", "png"],
                            overwrite=True
                        )
        except Exception as e:
            warnings.warn(f"Failed to create heatmaps: {e}")
    
    # Clear caches to free memory
    clear_caches()
    
    return raw_data_dfs, master_df, output_dir
