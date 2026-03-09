"""
plot_hr_diagram.py
-------------------
Plot a Hertzsprung–Russell (HR) diagram for a selection of models from the
accretion model grid.

Each evolutionary track is drawn as a line from the ZAMS to the end of the
computed evolution.  Models are colour-coded by initial mass; different
metallicities and accretion rates can be selected with command-line flags.

Usage
-----
::

    # All solar-metallicity models
    python scripts/plot_hr_diagram.py \\
        --data-dir data/models \\
        --metallicity 0.014 \\
        --output plots/hr_Z0014.png

    # Subset by accretion rate and initial masses
    python scripts/plot_hr_diagram.py \\
        --data-dir data/models \\
        --metallicity 0.014 \\
        --mdot 1e-7 \\
        --masses 1.0 2.0 5.0 \\
        --output plots/hr_subset.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def _parse_grid_params(model_file):
    """Extract Z, M_init, mdot from directory names."""
    parts = model_file.parts
    params = {"Z": None, "M_init": None, "mdot": None}
    for part in parts:
        if m := re.fullmatch(r"Z([\d.eE+\-]+)", part):
            params["Z"] = float(m.group(1))
        elif m := re.fullmatch(r"M([\d.eE+\-]+)", part):
            params["M_init"] = float(m.group(1))
        elif "mdot" in part.lower():
            if m := re.search(r"mdot([\d.eE+\-]+)", part, re.IGNORECASE):
                params["mdot"] = float(m.group(1))
    return params


def plot_hr_diagram(
    data_dir,
    metallicity=None,
    mdot=None,
    masses=None,
    output=None,
    show=False,
):
    """Create an HR diagram from the model grid.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the model grid.
    metallicity : float or None
        Select only models with this metallicity (``Z`` value).  ``None``
        includes all metallicities.
    mdot : float or None
        Select only models with this accretion rate.  ``None`` includes all.
    masses : list of float or None
        Select only models whose initial mass is in *masses*.  ``None``
        includes all masses.
    output : str or Path or None
        Save the figure to this path.  If ``None``, the figure is not saved.
    show : bool
        If ``True``, display the figure interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    model_files = sorted(data_dir.rglob("*.dat"))
    if not model_files:
        raise FileNotFoundError(f"No model files found under '{data_dir}'.")

    # Collect tracks
    tracks = []
    for model_file in model_files:
        params = _parse_grid_params(model_file)

        if metallicity is not None and params["Z"] != metallicity:
            continue
        if mdot is not None and params["mdot"] != mdot:
            continue
        if masses is not None and params["M_init"] not in masses:
            continue

        try:
            df = pd.read_csv(model_file, sep=r"\s+", comment="#")
        except Exception as exc:
            print(f"Warning: skipping '{model_file}': {exc}", file=sys.stderr)
            continue

        if "log_Teff" not in df.columns or "log_L" not in df.columns:
            print(
                f"Warning: '{model_file}' missing log_Teff or log_L columns.",
                file=sys.stderr,
            )
            continue

        tracks.append((params, df))

    if not tracks:
        print("No tracks to plot after applying filters.", file=sys.stderr)
        sys.exit(1)

    # Colour-code by initial mass
    all_masses = sorted({t[0]["M_init"] for t in tracks if t[0]["M_init"] is not None})
    cmap = cm.viridis
    mass_to_color = {
        m: cmap(i / max(len(all_masses) - 1, 1)) for i, m in enumerate(all_masses)
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    plotted_masses = set()
    for params, df in tracks:
        m_init = params["M_init"]
        color = mass_to_color.get(m_init, "grey")
        label = f"{m_init} M☉" if m_init not in plotted_masses else None
        ax.plot(df["log_Teff"], df["log_L"], color=color, lw=0.8, label=label)
        plotted_masses.add(m_init)

    ax.invert_xaxis()
    ax.set_xlabel(r"$\log\,T_{\rm eff}$ [K]", fontsize=13)
    ax.set_ylabel(r"$\log\,(L/L_\odot)$", fontsize=13)

    title_parts = ["HR diagram"]
    if metallicity is not None:
        title_parts.append(f"Z={metallicity}")
    if mdot is not None:
        title_parts.append(rf"$\dot{{M}}={mdot:.0e}$")
    ax.set_title("  —  ".join(title_parts), fontsize=13)

    # Legend: only show if not too many masses
    if len(all_masses) <= 20:
        ax.legend(title="Initial mass", fontsize=8, loc="upper left")

    plt.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Figure saved to '{output}'.")

    if show:
        plt.show()

    return fig


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot an HR diagram from the accretion model grid."
    )
    parser.add_argument(
        "--data-dir",
        default="data/models",
        help="Root directory of the model grid (default: data/models).",
    )
    parser.add_argument(
        "--metallicity",
        type=float,
        default=None,
        help="Select models with this metallicity (e.g. 0.014).",
    )
    parser.add_argument(
        "--mdot",
        type=float,
        default=None,
        help="Select models with this accretion rate (e.g. 1e-7).",
    )
    parser.add_argument(
        "--masses",
        type=float,
        nargs="+",
        default=None,
        metavar="M",
        help="Select models with these initial masses (e.g. 1.0 2.0 5.0).",
    )
    parser.add_argument(
        "--output",
        default="plots/hr_diagram.png",
        help="Output figure path (default: plots/hr_diagram.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    plot_hr_diagram(
        data_dir=args.data_dir,
        metallicity=args.metallicity,
        mdot=args.mdot,
        masses=args.masses,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
