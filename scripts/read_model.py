"""
read_model.py
-------------
Utility for loading a single STARS accretion model output file into a
pandas DataFrame.

Usage
-----
As a module::

    from scripts.read_model import load_model
    df = load_model("data/models/Z0.014/M01.0/mdot1e-7/model.dat")

As a command-line tool::

    python scripts/read_model.py data/models/Z0.014/M01.0/mdot1e-7/model.dat
"""

import argparse
from pathlib import Path

import pandas as pd


def load_model(filepath):
    """Load a STARS model output file.

    Parameters
    ----------
    filepath : str or Path
        Path to the model output file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per time-step and one column per physical
        quantity.  Column names are taken from the file header.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be parsed (e.g. unexpected format).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # The STARS output format uses a single comment/header line starting with '#'
    # followed by whitespace-separated columns.
    try:
        df = pd.read_csv(filepath, sep=r"\s+", comment="#")
    except Exception as exc:
        raise ValueError(f"Could not parse model file '{filepath}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Model file '{filepath}' contains no data rows.")

    return df


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Load and display a STARS accretion model file."
    )
    parser.add_argument("filepath", help="Path to the model output file.")
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        metavar="N",
        help="Print the first N rows (default: 10).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    df = load_model(args.filepath)
    print(f"Loaded {len(df)} time-steps from '{args.filepath}'.")
    print(f"Columns: {list(df.columns)}")
    print()
    print(df.head(args.head).to_string(index=False))


if __name__ == "__main__":
    main()
