# Model Grid Data

The full grid of STARS stellar evolution models with accretion (Dimoff et al. 2025)
is **not stored in this repository** due to its large size.

## Downloading the Model Grid

The complete model grid can be downloaded from Zenodo:

> **Dimoff et al. (2025)** — *Grid of STARS stellar evolution models with accretion*
> [https://zenodo.org/records/XXXXXXX](https://zenodo.org/records/XXXXXXX)
> *(replace XXXXXXX with the actual Zenodo record ID)*

Once downloaded, extract the archive and place the model files in this `data/` directory,
preserving the folder structure described below.

## Expected Directory Layout

After downloading and extracting, this directory should look like:

```
data/
├── README.md             ← this file
└── models/
    ├── Z0.014/           ← metallicity folders (e.g. solar metallicity)
    │   ├── M01.0/        ← initial mass folders (e.g. 1.0 Msun)
    │   │   ├── mdot1e-7/ ← accretion-rate folders
    │   │   │   └── *.dat ← individual model output files
    │   │   └── ...
    │   └── ...
    └── ...
```

## Model File Format

Each model output file is a plain-text table with one row per time-step and columns
for the physical quantities listed in the header.  Key columns include:

| Column          | Description                              |
|-----------------|------------------------------------------|
| `age`           | Stellar age (yr)                         |
| `mass`          | Total stellar mass (Msun)                |
| `log_L`         | Log surface luminosity (Lsun)            |
| `log_Teff`      | Log effective temperature (K)            |
| `log_g`         | Log surface gravity (cm/s²)              |
| `radius`        | Stellar radius (Rsun)                    |
| `X_surf`        | Surface hydrogen mass fraction           |
| `Y_surf`        | Surface helium mass fraction             |
| `Z_surf`        | Surface total metal mass fraction        |
| `C_surf`        | Surface carbon mass fraction             |
| `N_surf`        | Surface nitrogen mass fraction           |
| `O_surf`        | Surface oxygen mass fraction             |

See the paper for a complete description of all output columns and the physics
implemented in the models.
