# DSF Analysis Package Structure

This directory contains the restructured DSF analysis code as a proper Python package.

## Package Structure

```
dsf_analysis/
├── __init__.py             # Package initialization
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── data_loading.py     # Data loading functions
│   ├── models.py           # Model definitions and fitting
│   └── analysis.py         # Main analysis functions
├── visualization/          # Visualization components
│   ├── __init__.py
│   ├── plotting.py         # Basic plotting functions
│   ├── interactive.py      # Interactive visualizations
│   └── smiles.py           # Chemical structure visualization
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── output.py           # Output management
│   ├── mapping.py          # UniProt and SMILES mapping
│   └── helpers.py          # Helper functions
├── cli/                    # Command-line interface
│   ├── __init__.py
│   └── commands.py         # CLI commands
└── examples/               # Example scripts
    ├── basic_analysis.py
    ├── advanced_features.py
    └── batch_processing.py
```

## Implementation Plan

1. Create the package directory structure
2. Move existing code into appropriate modules
3. Create proper imports and exports
4. Add package metadata and setup.py
5. Implement command-line interface
6. Create example scripts
7. Add tests

## Dependencies

- pandas
- numpy
- matplotlib
- scipy
- plotly
- rdkit (optional, for chemical structure visualization)
