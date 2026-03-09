# Analysis Scripts

This directory contains Python scripts for analysing the STARS accretion model grid.

## Scripts

| Script                          | Purpose                                              |
|---------------------------------|------------------------------------------------------|
| `read_model.py`                 | Load a single model file into a pandas DataFrame     |
| `extract_surface_params.py`     | Extract stellar surface parameters across the grid   |
| `extract_surface_abundances.py` | Extract surface chemical abundances across the grid  |
| `plot_hr_diagram.py`            | Plot Hertzsprung–Russell diagrams for selected models|

## Quick Start

```bash
# Install dependencies (from the repository root)
pip install -r requirements.txt

# Extract surface parameters for all models and save to CSV
python scripts/extract_surface_params.py --data-dir data/models --output output/surface_params.csv

# Extract surface abundances
python scripts/extract_surface_abundances.py --data-dir data/models --output output/surface_abundances.csv

# Plot an HR diagram for a subset of models
python scripts/plot_hr_diagram.py --data-dir data/models --metallicity 0.014 --output plots/hr_diagram.png
```

Run any script with `--help` to see all available options.
