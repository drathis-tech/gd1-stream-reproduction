"""Scatter plot of phi1 vs phi2 for the v2 final candidates, restricted to |phi2| < 1 deg."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"

INPUT_PATH = DATA_DIR / "gd1_final_v2.parquet"
PLOT_PATH = PLOT_DIR / "gd1_final_narrow.png"

PHI2_MAX_ABS = 1.0


def main():
    df = pd.read_parquet(INPUT_PATH, columns=["phi1", "phi2"])
    narrow = df[df["phi2"].abs() < PHI2_MAX_ABS]

    print(f"Total candidates: {len(df)}")
    print(f"Candidates with |phi2| < {PHI2_MAX_ABS}: {len(narrow)}")

    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(narrow["phi1"], narrow["phi2"], s=8, alpha=0.7, edgecolor="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_ylim(-PHI2_MAX_ABS, PHI2_MAX_ABS)
    ax.set_title(f"GD-1 final candidates (v2), |phi2| < {PHI2_MAX_ABS} deg")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


if __name__ == "__main__":
    main()
