"""
DSF Analysis Package
===================

A comprehensive package for analyzing Differential Scanning Fluorimetry (DSF) data
with optimized performance and advanced visualization features.

This package provides tools for:
- Data loading and preprocessing
- Melting curve analysis with multiple models
- Statistical analysis and visualization
- UniProt ID mapping
- Chemical structure visualization using SMILES
- Output management and organization

For more information, see the documentation at:
https://github.com/yourusername/dsf-analysis
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import core functionality
from dsf_analysis.core.data_loading import (
    load_data_with_derivative_boltzmann,
    split_by_well_position,
    get_ligand_for_plate_well
)

from dsf_analysis.core.models import (
    fit_model_generic,
    select_best_model_for_trace,
    get_negative_control_fit,
    normalize_df,
    find_derivative_peaks,
    get_deriv_peak_temp
)

from dsf_analysis.core.analysis import (
    analyze_dsf_data,
    clear_caches
)

# Import visualization functions
from dsf_analysis.visualization.plotting import (
    plot_model_fit,
    plot_reps_and_neg_control
)

from dsf_analysis.visualization.interactive import (
    create_interactive_heatmap
)

# Import utility functions
from dsf_analysis.utils.mapping import (
    load_uniprot_mapping,
    load_smiles_mapping
)

from dsf_analysis.utils.output import (
    create_output_directory,
    save_dataframe,
    save_figure
)

# # Import additional features
# from dsf_analysis.utils.reanalysis import (
#     reanalyze_conditions,
#     batch_reanalyze_by_criteria,
#     compare_analysis_results
# )

# Define what's available when using "from dsf_analysis import *"
__all__ = [
    # Core functionality
    'load_data_with_derivative_boltzmann',
    'split_by_well_position',
    'get_ligand_for_plate_well',
    'fit_model_generic',
    'select_best_model_for_trace',
    'get_negative_control_fit',
    'normalize_df',
    'find_derivative_peaks',
    'get_deriv_peak_temp',
    'analyze_dsf_data',
    'clear_caches',
    
    # Visualization
    'plot_model_fit',
    'plot_reps_and_neg_control',
    'create_interactive_heatmap',
    'plot_replicates_with_comparison',
    
    # Utilities
    'load_uniprot_mapping',
    'load_smiles_mapping',
    'create_output_directory',
    'save_dataframe',
    'save_figure',
    
    # Additional features
    'reanalyze_conditions',
    'batch_reanalyze_by_criteria',
    'compare_analysis_results'
]
