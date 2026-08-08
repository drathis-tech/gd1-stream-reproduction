"""Plot proper motions of GD-1-region stars to locate the stream's PM locus.

GD-1 stars share a tight, cold proper motion consistent with their common
orbit, so they show up as a compact overdensity against the broad field-star
background in pm_phi1_cosphi2 vs pm_phi2 space. This plot is a visual aid for
picking that region before applying a proper-motion cut.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"

INPUT_PATH = DATA_DIR / "gaia_gd1_transformed.parquet"
PLOT_PATH = PLOT_DIR / "gd1_pm_locus.png"


def main():
    df = pd.read_parquet(INPUT_PATH, columns=["pm_phi1_cosphi2", "pm_phi2"])

    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["pm_phi1_cosphi2"], df["pm_phi2"], s=0.5, alpha=0.1, edgecolor="none", rasterized=True)
    ax.set_xlim(-20, 5)
    ax.set_ylim(-10, 10)
    ax.set_xlabel(r"$\mu_{\phi_1}\cos\phi_2$ [mas/yr]")
    ax.set_ylabel(r"$\mu_{\phi_2}$ [mas/yr]")
    ax.set_title("GD-1 region proper motions")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)

    print(f"Row count: {len(df)}")


if __name__ == "__main__":
    main()
