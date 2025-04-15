# SMILES Code Integration Template

This file provides a template for integrating SMILES codes for chemical compounds in the DSF analysis package.

## What are SMILES Codes?

SMILES (Simplified Molecular Input Line Entry System) is a notation that allows you to represent chemical structures as text strings. For example, the SMILES code for caffeine is `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`.

## CSV Format

The recommended format for providing SMILES codes is a CSV file with the following columns:

```
Compound,SMILES,Description
Compound1,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,Caffeine
Compound2,CC(=O)OC1=CC=CC=C1C(=O)O,Aspirin
```

The minimum required columns are:
- `Compound`: The compound identifier used in your plate maps (e.g., "Compound1")
- `SMILES`: The SMILES code for the compound

Optional columns:
- `Description`: A human-readable description or common name of the compound
- `MolecularWeight`: Molecular weight of the compound
- `Formula`: Molecular formula

## JSON Format

Alternatively, you can provide the mapping as a JSON file:

```json
{
  "Compound1": {
    "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "description": "Caffeine",
    "molecular_weight": 194.19,
    "formula": "C8H10N4O2"
  },
  "Compound2": {
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "description": "Aspirin",
    "molecular_weight": 180.16,
    "formula": "C9H8O4"
  }
}
```

For simple mappings, you can use a flat JSON structure:

```json
{
  "Compound1": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  "Compound2": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

## Plate Map Integration

You can also add SMILES codes directly to your plate map CSV files by adding a `SMILES` column:

```
Well,Ligand,Concentration,SMILES
A01,DMSO,100,
A02,Compound1,100,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
A03,Compound2,100,CC(=O)OC1=CC=CC=C1C(=O)O
```

## How to Use

1. Create your SMILES mapping file using one of the formats above
2. Install RDKit for chemical structure visualization:
   ```
   conda install -c conda-forge rdkit
   ```
3. Load the mapping when running the analysis:

```python
from dsf_optimized import analyze_dsf_data

analyze_dsf_data(
    parent_folder="./data",
    plate_map_folder="./plate_maps",
    output_dir="./output",
    use_smiles=True,
    smiles_mapping="./mappings/smiles_mapping.csv"
)
```

Or load the mapping separately:

```python
from dsf_optimized import load_smiles_mapping, analyze_dsf_data

# Load the mapping
mapping = load_smiles_mapping("./mappings/smiles_mapping.csv")
print(f"Loaded {len(mapping)} SMILES mappings")

# Run analysis with the loaded mapping
analyze_dsf_data(
    parent_folder="./data",
    plate_map_folder="./plate_maps",
    output_dir="./output",
    use_smiles=True
)
```

## Finding SMILES Codes

You can find SMILES codes for compounds using various chemical databases:
1. PubChem: https://pubchem.ncbi.nlm.nih.gov/
2. ChemSpider: http://www.chemspider.com/
3. ChEMBL: https://www.ebi.ac.uk/chembl/

## Example

Here's an example of a complete SMILES mapping CSV file:

```
Compound,SMILES,Description,Formula,MolecularWeight
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,1,3,7-Trimethylpurine-2,6-dione,C8H10N4O2,194.19
Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,Acetylsalicylic acid,C9H8O4,180.16
Ibuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,2-(4-Isobutylphenyl)propanoic acid,C13H18O2,206.28
```
