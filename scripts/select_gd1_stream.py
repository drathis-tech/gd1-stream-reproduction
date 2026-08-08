"""Select GD-1 stream candidates by a phi1-dependent proper-motion cut.

Rather than a single fixed PM box, stars are kept if they fall within
PM_TOLERANCE of the Price-Whelan & Bonaca (2018) PM track (see
gd1_pm_track.py) evaluated at each star's own phi1. This follows the track's
slope in pm_phi1_cosphi2/pm_phi2 vs phi1, so it isolates the stream's cold,
coherent motion more tightly than a box that has to be wide enough to cover
the track's full swing across phi1.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from gd1_pm_track import PM_PHI1_TRACK, PM_PHI2_TRACK, TRACK_PHI1_RANGE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"

INPUT_PATH = DATA_DIR / "gaia_gd1_transformed.parquet"
OUTPUT_PATH = DATA_DIR / "gd1_candidates.parquet"
PLOT_PATH = PLOT_DIR / "gd1_stream_detection.png"

PM_TOLERANCE = 2.5  # mas/yr, applied independently to each PM component

BIN_SIZE = 0.2


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Row count before cut: {len(df)}")

    in_track_range = df["phi1"].between(*TRACK_PHI1_RANGE)
    pm_phi1_track_at_star = PM_PHI1_TRACK(df["phi1"])
    pm_phi2_track_at_star = PM_PHI2_TRACK(df["phi1"])
    mask = (
        in_track_range
        & ((df["pm_phi1_cosphi2"] - pm_phi1_track_at_star).abs() < PM_TOLERANCE)
        & ((df["pm_phi2"] - pm_phi2_track_at_star).abs() < PM_TOLERANCE)
    )
    candidates = df[mask]
    print(f"Row count after cut: {len(candidates)}")

    candidates.to_parquet(OUTPUT_PATH, index=False)

    phi1_bins = np.arange(candidates["phi1"].min(), candidates["phi1"].max() + BIN_SIZE, BIN_SIZE)
    phi2_bins = np.arange(candidates["phi2"].min(), candidates["phi2"].max() + BIN_SIZE, BIN_SIZE)

    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    _, _, _, im = ax.hist2d(
        candidates["phi1"], candidates["phi2"], bins=[phi1_bins, phi2_bins], norm=LogNorm(), cmap="viridis"
    )
    fig.colorbar(im, ax=ax, label="stars per bin")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("GD-1 stream candidates (proper-motion selected)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


if __name__ == "__main__":
    main()
