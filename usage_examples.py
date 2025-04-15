"""
DSF Analysis Package - Usage Examples
====================================

This file provides comprehensive examples of how to use the DSF analysis package
for various analysis scenarios.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dsf_optimized import (
    # Core functions
    load_data_with_derivative_boltzmann,
    analyze_dsf_data,
    
    # Model fitting functions
    select_best_model_for_trace,
    get_negative_control_fit,
    
    # Visualization functions
    plot_model_fit,
    plot_reps_and_neg_control,
    create_interactive_heatmap,
    
    # Utility functions
    load_uniprot_mapping,
    load_smiles_mapping,
    create_output_directory,
    clear_caches,
    reanalyze_conditions
)

# Optional imports for SMILES visualization
try:
    from smiles_visualization import (
        get_image_for_compound,
        create_structure_grid,
        plot_dsf_curve_with_structure
    )
    SMILES_AVAILABLE = True
except ImportError:
    SMILES_AVAILABLE = False

#######################
# Basic Data Loading #
#######################

def example_data_loading():
    """
    Example of loading DSF data and plate maps.
    """
    print("Loading DSF data...")
    
    # Set paths
    parent_folder = "./data"
    plate_map_folder = "./plate_maps"
    
    # Load all data
    raw_data_dfs, derivative_data_dfs, boltzmann_data_dfs, plate_map_dict = load_data_with_derivative_boltzmann(
        parent_folder=parent_folder,
        plate_map_folder=plate_map_folder
    )
    
    # Print summary
    proteins = list(raw_data_dfs.keys())
    print(f"Loaded data for {len(proteins)} proteins: {proteins}")
    
    for protein in proteins:
        plates = list(raw_data_dfs[protein].keys())
        print(f"  Protein {protein} has {len(plates)} plates: {plates}")
        
        for plate in plates:
            reps = list(raw_data_dfs[protein][plate].keys())
            print(f"    Plate {plate} has {len(reps)} replicates: {reps}")
            
            for rep in reps:
                wells = list(raw_data_dfs[protein][plate][rep].keys())
                print(f"      Replicate {rep} has {len(wells)} wells")
    
    # Print plate map summary
    print("\nPlate maps:")
    for plate_id, plate_df in plate_map_dict.items():
        print(f"  Plate {plate_id}: {len(plate_df)} wells")
    
    return raw_data_dfs, derivative_data_dfs, boltzmann_data_dfs, plate_map_dict

#######################
# Selective Loading  #
#######################

def example_selective_loading():
    """
    Example of selectively loading specific proteins, plates, and wells.
    """
    print("Selectively loading DSF data...")
    
    # Set paths
    parent_folder = "./data"
    plate_map_folder = "./plate_maps"
    
    # Load specific proteins and plates
    raw_data_dfs, _, _, _ = load_data_with_derivative_boltzmann(
        parent_folder=parent_folder,
        plate_map_folder=plate_map_folder,
        protein_filter=["B2", "C2"],  # Only load these proteins
        plate_filter=["PM1"],         # Only load this plate
        well_filter=["A01", "A02"]    # Only load these wells
    )
    
    # Print summary
    proteins = list(raw_data_dfs.keys())
    print(f"Loaded filtered data for {len(proteins)} proteins: {proteins}")
    
    return raw_data_dfs

#######################
# Model Fitting      #
#######################

def example_model_fitting(raw_data_dfs):
    """
    Example of fitting models to DSF data.
    
    Parameters:
    -----------
    raw_data_dfs : dict
        Nested dictionary of raw data from load_data_with_derivative_boltzmann
    """
    print("Fitting models to DSF data...")
    
    # Get a sample trace
    protein = list(raw_data_dfs.keys())[0]
    plate = list(raw_data_dfs[protein].keys())[0]
    rep = list(raw_data_dfs[protein][plate].keys())[0]
    well = list(raw_data_dfs[protein][plate][rep].keys())[0]
    
    df_raw = raw_data_dfs[protein][plate][rep][well]
    
    print(f"Fitting models for {protein}, {plate}, {rep}, {well}")
    
    # Try different models
    candidate_models = ["mB", "s1", "s2"]
    window_sizes = [20, 30, 40, 50]
    
    best_fit, summary_df = select_best_model_for_trace(
        df_raw=df_raw,
        candidate_models=candidate_models,
        window_mode="optimized",
        window_sizes=window_sizes,
        verbose=True,
        protein=protein,
        plate_ID=plate,
        rep=rep,
        well=well
    )
    
    print("\nBest fit result:")
    for key, value in best_fit.items():
        if key != "popt":  # Skip printing the parameters array
            print(f"  {key}: {value}")
    
    print("\nSummary of all fits:")
    print(summary_df[["model", "window_size", "R2", "Tm1"]].to_string(index=False))
    
    # Plot the best fit
    plt.figure(figsize=(10, 6))
    plot_model_fit(df_raw, best_fit, title=f"{protein}, {plate}, {rep}, {well}")
    plt.tight_layout()
    plt.show()
    
    return best_fit, summary_df

#######################
# Negative Controls  #
#######################

def example_negative_control_analysis(raw_data_dfs):
    """
    Example of analyzing negative controls.
    
    Parameters:
    -----------
    raw_data_dfs : dict
        Nested dictionary of raw data from load_data_with_derivative_boltzmann
    """
    print("Analyzing negative controls...")
    
    # Get a sample protein and plate
    protein = list(raw_data_dfs.keys())[0]
    plate = list(raw_data_dfs[protein].keys())[0]
    rep = list(raw_data_dfs[protein][plate].keys())[0]
    
    # Assume A01 is the negative control
    neg_control_well = "A01"
    
    # Get negative control fit
    neg_fit = get_negative_control_fit(
        protein=protein,
        plate=plate,
        rep=rep,
        neg_control_well=neg_control_well,
        raw_data_dfs=raw_data_dfs,
        candidate_models=["mB", "s1"],
        window_mode="optimized",
        window_sizes=[20, 30, 40]
    )
    
    print(f"Negative control fit for {protein}, {plate}, {rep}, {neg_control_well}:")
    for key, value in neg_fit.items():
        if key != "popt":  # Skip printing the parameters array
            print(f"  {key}: {value}")
    
    return neg_fit

#######################
# Replicate Analysis #
#######################

def example_replicate_analysis(raw_data_dfs):
    """
    Example of analyzing replicates for a treatment well.
    
    Parameters:
    -----------
    raw_data_dfs : dict
        Nested dictionary of raw data from load_data_with_derivative_boltzmann
    """
    print("Analyzing replicates...")
    
    # Get a sample protein and plate
    protein = list(raw_data_dfs.keys())[0]
    plate = list(raw_data_dfs[protein].keys())[0]
    
    # Find a treatment well (not A01)
    treatment_well = None
    for rep in raw_data_dfs[protein][plate]:
        for well in raw_data_dfs[protein][plate][rep]:
            if well != "A01":
                treatment_well = well
                break
        if treatment_well:
            break
    
    if not treatment_well:
        print("No treatment wells found.")
        return
    
    print(f"Analyzing replicates for {protein}, {plate}, {treatment_well}")
    
    # Plot replicates with negative control
    plot_reps_and_neg_control(
        protein=protein,
        plate=plate,
        treatment_well=treatment_well,
        neg_control_well="A01",
        raw_data_dfs=raw_data_dfs,
        candidate_models=["mB", "s1"],
        window_mode="optimized",
        window_sizes=[20, 30, 40],
        include_tmchange_subplot=True
    )
    
    plt.tight_layout()
    plt.show()

#######################
# Full Analysis      #
#######################

def example_full_analysis():
    """
    Example of running a full analysis on all data.
    """
    print("Running full analysis...")
    
    # Set paths
    parent_folder = "./data"
    plate_map_folder = "./plate_maps"
    output_dir = "./output"
    
    # Run analysis
    raw_data_dfs, master_df, output_dir = analyze_dsf_data(
        parent_folder=parent_folder,
        plate_map_folder=plate_map_folder,
        output_dir=output_dir,
        candidate_models=["mB", "s1"],
        window_mode="optimized",
        window_sizes=[20, 30, 40, 50],
        neg_control_well="A01"
    )
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print(f"Master dataframe has {len(master_df)} rows")
    
    return raw_data_dfs, master_df, output_dir

#######################
# UniProt Integration #
#######################

def example_uniprot_integration():
    """
    Example of using UniProt ID mapping.
    """
    print("Using UniProt ID mapping...")
    
    # Create a sample mapping file
    os.makedirs("./mappings", exist_ok=True)
    
    # Sample mapping data
    mapping_data = pd.DataFrame({
        "Protein": ["B2", "C2", "D5"],
        "UniProt": ["P00918", "P07451", "P00915"],
        "Description": [
            "Carbonic anhydrase 2",
            "Carbonic anhydrase 3",
            "Carbonic anhydrase 1"
        ]
    })
    
    # Save to CSV
    mapping_file = "./mappings/uniprot_mapping.csv"
    mapping_data.to_csv(mapping_file, index=False)
    print(f"Created sample mapping file: {mapping_file}")
    
    # Load the mapping
    mapping = load_uniprot_mapping(mapping_file)
    print(f"Loaded {len(mapping)} UniProt mappings:")
    for protein, uniprot in mapping.items():
        print(f"  {protein} -> {uniprot}")
    
    # Run analysis with UniProt mapping
    parent_folder = "./data"
    plate_map_folder = "./plate_maps"
    output_dir = "./output_uniprot"
    
    # Run analysis (this would use the UniProt IDs in the output)
    print("\nRunning analysis with UniProt mapping...")
    print("(This is a demonstration - actual analysis would use the UniProt IDs)")
    
    return mapping

#######################
# SMILES Integration #
#######################

def example_smiles_integration():
    """
    Example of using SMILES code visualization.
    """
    if not SMILES_AVAILABLE:
        print("SMILES visualization not available. Install RDKit to use this feature.")
        return
    
    print("Using SMILES code visualization...")
    
    # Create a sample mapping file
    os.makedirs("./mappings", exist_ok=True)
    
    # Sample mapping data
    mapping_data = pd.DataFrame({
        "Compound": ["Caffeine", "Aspirin", "Ibuprofen"],
        "SMILES": [
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        ],
        "Description": [
            "1,3,7-Trimethylpurine-2,6-dione",
            "Acetylsalicylic acid",
            "2-(4-Isobutylphenyl)propanoic acid"
        ]
    })
    
    # Save to CSV
    mapping_file = "./mappings/smiles_mapping.csv"
    mapping_data.to_csv(mapping_file, index=False)
    print(f"Created sample mapping file: {mapping_file}")
    
    # Load the mapping
    mapping = load_smiles_mapping(mapping_file)
    print(f"Loaded {len(mapping)} SMILES mappings:")
    for compound, smiles in mapping.items():
        print(f"  {compound} -> {smiles}")
    
    # Create a grid of structures
    print("\nCreating structure grid...")
    fig = create_structure_grid(["Caffeine", "Aspirin", "Ibuprofen"])
    plt.tight_layout()
    plt.show()
    
    return mapping

#######################
# Reanalysis         #
#######################

def example_reanalysis(raw_data_dfs):
    """
    Example of reanalyzing specific conditions.
    
    Parameters:
    -----------
    raw_data_dfs : dict
        Nested dictionary of raw data from load_data_with_derivative_boltzmann
    """
    print("Reanalyzing specific conditions...")
    
    # Create a sample master dataframe
    master_df = pd.DataFrame({
        "protein": ["B2", "B2", "C2"],
        "plate_ID": ["PM1", "PM1", "PM1"],
        "rep": ["Rep01", "Rep01", "Rep01"],
        "well": ["A02", "A03", "A02"],
        "ligand": ["Compound1", "Compound2", "Compound1"],
        "model": ["s1", "s1", "s1"],
        "R2": [0.95, 0.92, 0.94],
        "Tm1": [55.2, 58.7, 62.1]
    })
    
    # Save to CSV
    os.makedirs("./output", exist_ok=True)
    master_file = "./output/master_results.csv"
    master_df.to_csv(master_file, index=False)
    print(f"Created sample master file: {master_file}")
    
    # Define conditions to reanalyze
    conditions = [
        {"protein": "B2", "plate": "PM1", "rep": "Rep01", "well": "A02"},
        {"protein": "C2", "plate": "PM1", "rep": "Rep01", "well": "A02"}
    ]
    
    # Reanalyze (this is a demonstration - actual reanalysis would use the raw data)
    print("\nReanalyzing conditions:")
    for condition in conditions:
        print(f"  {condition}")
    
    print("\nThis is a demonstration - actual reanalysis would update the master dataframe")
    
    return conditions

#######################
# Output Management  #
#######################

def example_output_management():
    """
    Example of managing output directories.
    """
    print("Managing output directories...")
    
    # Create output directories
    base_dir = "./output"
    
    # Create a directory for a specific protein and plate
    protein = "B2"
    plate = "PM1"
    
    output_dir = create_output_directory(base_dir, protein, plate)
    print(f"Created output directory: {output_dir}")
    
    # Create another directory for the same protein and plate (should get a unique name)
    output_dir2 = create_output_directory(base_dir, protein, plate)
    print(f"Created second output directory: {output_dir2}")
    
    return output_dir, output_dir2

#######################
# Memory Management  #
#######################

def example_memory_management():
    """
    Example of managing memory usage.
    """
    print("Managing memory usage...")
    
    # Clear caches to free memory
    clear_caches()
    print("Cleared all caches")
    
    # Create some large arrays to simulate memory usage
    print("Creating large arrays...")
    arrays = []
    for i in range(5):
        arrays.append(np.random.rand(1000, 1000))
    
    print(f"Created {len(arrays)} large arrays")
    
    # Delete arrays and clear memory
    print("Deleting arrays and clearing memory...")
    del arrays
    clear_caches()
    
    print("Memory cleared")

#######################
# Main Example       #
#######################

def main():
    """
    Run all examples.
    """
    print("DSF Analysis Package - Usage Examples")
    print("====================================")
    
    # Note: In a real scenario, you would run these examples with actual data
    # For demonstration purposes, we'll create minimal examples
    
    # Create sample data directory structure
    os.makedirs("./data/B2_PM1", exist_ok=True)
    os.makedirs("./data/C2_PM1", exist_ok=True)
    os.makedirs("./plate_maps", exist_ok=True)
    
    print("\nThis is a demonstration script. In a real scenario, you would run these")
    print("examples with actual DSF data files in the expected directory structure.")
    
    # Run examples that don't require actual data
    example_uniprot_integration()
    print("\n" + "-"*50 + "\n")
    
    example_smiles_integration()
    print("\n" + "-"*50 + "\n")
    
    example_output_management()
    print("\n" + "-"*50 + "\n")
    
    example_memory_management()
    print("\n" + "-"*50 + "\n")
    
    print("For examples that require actual DSF data, please see the function")
    print("definitions in this file and run them with your own data.")

if __name__ == "__main__":
    main()
