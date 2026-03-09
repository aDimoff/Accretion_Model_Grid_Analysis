# Accretion Model Grid Analysis

Codes and tools to analyse the large grid of STARS stellar evolution models
with accretion (Dimoff et al. 2025).

---

## Overview

This repository provides Python scripts and Jupyter notebooks for working with
a grid of stellar evolution models computed with the
[STARS code](https://www.ast.cam.ac.uk/~stars/).  The grid covers a wide range
of initial masses, metallicities, and accretion rates, and is described in:

> **Dimoff et al. (2025)** — *Title of Paper*, Journal, Volume, Pages.
> DOI: [10.XXXX/XXXXXX](https://doi.org/10.XXXX/XXXXXX)

The analysis tools let you:

- **Load** individual model output files into pandas DataFrames
- **Extract** stellar surface parameters (T_eff, log g, luminosity, radius, …)
- **Extract** surface chemical abundances (H, He, C, N, O, …) across the grid
- **Plot** Hertzsprung–Russell diagrams for selected subsets of models

---

## Downloading the Model Grid

The full model grid is **not stored in this repository** due to its large size.
It can be downloaded from Zenodo:

> **Dimoff et al. (2025)** — *Grid of STARS stellar evolution models with accretion*
> [https://zenodo.org/records/XXXXXXX](https://zenodo.org/records/XXXXXXX)

After downloading, extract the archive into the `data/` directory.
See [`data/README.md`](data/README.md) for the expected directory layout and
a description of the model file format.

---

## Repository Structure

```
Accretion_Model_Grid_Analysis/
├── README.md                        ← this file
├── LICENSE                          ← CC0 1.0 — public domain
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── data/
│   └── README.md                    ← how to download the model grid
│
├── scripts/
│   ├── README.md                    ← script usage summary
│   ├── read_model.py                ← load one model file → DataFrame
│   ├── extract_surface_params.py    ← surface parameters for all models
│   ├── extract_surface_abundances.py← surface abundances for all models
│   └── plot_hr_diagram.py           ← HR diagram plots
│
└── notebooks/
    ├── README.md
    └── example_analysis.ipynb       ← end-to-end worked example
```

---

## Installation

Python 3.8 or later is required.

```bash
git clone https://github.com/aDimoff/Accretion_Model_Grid_Analysis.git
cd Accretion_Model_Grid_Analysis
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Extract final-timestep surface parameters for every model → CSV
python scripts/extract_surface_params.py \
    --data-dir data/models \
    --output output/surface_params.csv

# Extract surface abundances
python scripts/extract_surface_abundances.py \
    --data-dir data/models \
    --output output/surface_abundances.csv

# Plot an HR diagram for solar-metallicity models
python scripts/plot_hr_diagram.py \
    --data-dir data/models \
    --metallicity 0.014 \
    --output plots/hr_Z0014.png
```

Run any script with `--help` to see all available options.

For an interactive walkthrough, open the example notebook:

```bash
jupyter notebook notebooks/example_analysis.ipynb
```

---

## License

This repository is released under the
[CC0 1.0 Universal](LICENSE) licence — it is placed in the public domain.
