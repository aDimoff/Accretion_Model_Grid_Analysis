"""
extract_surface_params.py
--------------------------
Walk the model grid directory tree and extract the final (or optionally
all) time-step surface parameters for every model into a single CSV file.

Surface parameters extracted
-----------------------------
- ``log_L``   – log surface luminosity (log L/Lsun)
- ``log_Teff`` – log effective temperature (log Teff/K)
- ``log_g``   – log surface gravity (cm/s²)
- ``radius``  – stellar radius (Rsun)
- ``mass``    – total stellar mass at that time-step (Msun)
- ``age``     – stellar age (yr)

Grid parameter columns added automatically
------------------------------------------
The metallicity (``Z``), initial mass (``M_init``), and accretion rate
(``mdot``) are parsed from the directory names and appended as columns.

Usage
-----
::

    python scripts/extract_surface_params.py \\
        --data-dir data/models \\
        --output output/surface_params.csv

Run with ``--all-timesteps`` to keep every time-step rather than just the
final one.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Columns that constitute "surface parameters"
SURFACE_PARAM_COLS = ["age", "mass", "log_L", "log_Teff", "log_g", "radius"]


def _parse_grid_params(model_file):
    """Extract Z, M_init, and mdot from a model file's directory path.

    Expects a path of the form ``…/Z<value>/M<value>/<mdot_label>/file.dat``.

    Returns a dict with keys ``Z``, ``M_init``, ``mdot``.  Missing values
    are returned as ``None``.
    """
    parts = model_file.parts
    params = {"Z": None, "M_init": None, "mdot": None}

    for part in parts:
        if m := re.fullmatch(r"Z([\d.eE+\-]+)", part):
            params["Z"] = float(m.group(1))
        elif m := re.fullmatch(r"M([\d.eE+\-]+)", part):
            params["M_init"] = float(m.group(1))
        elif "mdot" in part.lower():
            # e.g. "mdot1e-7" → 1e-7
            if m := re.search(r"mdot([\d.eE+\-]+)", part, re.IGNORECASE):
                params["mdot"] = float(m.group(1))
    return params


def extract_surface_params(data_dir, all_timesteps=False):
    """Extract surface parameters from every model in *data_dir*.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the model grid (contains metallicity sub-folders).
    all_timesteps : bool, optional
        If ``True``, include every time-step.  If ``False`` (default), keep
        only the final time-step for each model.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with one row per (model, time-step) and columns
        for the grid parameters plus the surface parameters.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    records = []
    model_files = sorted(data_dir.rglob("*.dat"))

    if not model_files:
        print(f"Warning: no .dat files found under '{data_dir}'.", file=sys.stderr)
        return pd.DataFrame()

    for model_file in model_files:
        try:
            df = pd.read_csv(model_file, sep=r"\s+", comment="#")
        except Exception as exc:
            print(f"Warning: skipping '{model_file}': {exc}", file=sys.stderr)
            continue

        # Keep only the columns we care about (skip silently if missing)
        available = [c for c in SURFACE_PARAM_COLS if c in df.columns]
        if not available:
            print(
                f"Warning: no surface-parameter columns found in '{model_file}'.",
                file=sys.stderr,
            )
            continue

        subset = df[available].copy()
        if not all_timesteps:
            subset = subset.tail(1)

        grid_params = _parse_grid_params(model_file)
        for key, val in grid_params.items():
            subset.insert(0, key, val)

        records.append(subset)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract stellar surface parameters from the accretion model grid."
    )
    parser.add_argument(
        "--data-dir",
        default="data/models",
        help="Root directory of the model grid (default: data/models).",
    )
    parser.add_argument(
        "--output",
        default="output/surface_params.csv",
        help="Output CSV file path (default: output/surface_params.csv).",
    )
    parser.add_argument(
        "--all-timesteps",
        action="store_true",
        help="Include all time-steps instead of only the final one.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    print(f"Extracting surface parameters from '{args.data_dir}' …")
    df = extract_surface_params(args.data_dir, all_timesteps=args.all_timesteps)

    if df.empty:
        print("No data extracted. Check that the data directory is correct.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to '{output_path}'.")


if __name__ == "__main__":
    main()
