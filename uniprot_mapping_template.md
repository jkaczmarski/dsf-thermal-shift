# UniProt ID Mapping Template

This file provides a template for mapping protein names to UniProt IDs in the DSF analysis package.

## CSV Format

The recommended format for providing UniProt ID mappings is a CSV file with the following columns:

```
Protein,UniProt,Description
B2,P12345,Protein name or description (optional)
C2,P23456,Protein name or description (optional)
```

The minimum required columns are:
- `Protein`: The protein identifier used in your DSF data (e.g., "B2", "C2")
- `UniProt`: The corresponding UniProt ID (e.g., "P12345")

Optional columns:
- `Description`: A human-readable description or full name of the protein
- `Gene`: Gene name
- `Organism`: Source organism

## JSON Format

Alternatively, you can provide the mapping as a JSON file:

```json
{
  "B2": {
    "uniprot": "P12345",
    "description": "Protein name or description",
    "gene": "GENE1",
    "organism": "Homo sapiens"
  },
  "C2": {
    "uniprot": "P23456",
    "description": "Another protein description",
    "gene": "GENE2",
    "organism": "Homo sapiens"
  }
}
```

For simple mappings, you can use a flat JSON structure:

```json
{
  "B2": "P12345",
  "C2": "P23456"
}
```

## How to Use

1. Create your mapping file using one of the formats above
2. Load the mapping when running the analysis:

```python
from dsf_optimized import analyze_dsf_data

analyze_dsf_data(
    parent_folder="./data",
    plate_map_folder="./plate_maps",
    output_dir="./output",
    use_uniprot=True,
    uniprot_mapping="./mappings/uniprot_mapping.csv"
)
```

Or load the mapping separately:

```python
from dsf_optimized import load_uniprot_mapping, analyze_dsf_data

# Load the mapping
mapping = load_uniprot_mapping("./mappings/uniprot_mapping.csv")
print(f"Loaded {len(mapping)} UniProt mappings")

# Run analysis with the loaded mapping
analyze_dsf_data(
    parent_folder="./data",
    plate_map_folder="./plate_maps",
    output_dir="./output",
    use_uniprot=True
)
```

## Finding UniProt IDs

You can find UniProt IDs for your proteins using the UniProt website:
1. Go to https://www.uniprot.org/
2. Search for your protein by name, gene, or organism
3. Select the appropriate entry
4. The UniProt ID is displayed at the top of the entry (e.g., P12345)

## Example

Here's an example of a complete UniProt mapping CSV file:

```
Protein,UniProt,Description,Gene,Organism
B2,P00918,Carbonic anhydrase 2,CA2,Homo sapiens
C2,P07451,Carbonic anhydrase 3,CA3,Homo sapiens
D5,P00915,Carbonic anhydrase 1,CA1,Homo sapiens
```
