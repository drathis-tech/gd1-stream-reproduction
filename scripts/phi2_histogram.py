"""1D histogram of phi2 for the final GD-1 candidates, marginalized over phi1.

If the proper-motion + isochrone selected sample contains real GD-1 stream
members, they should cluster near phi2=0 (the great-circle track by
construction) and show up as a density excess over the smoother
contamination from remaining field stars.

The raw histogram shape is a mix of that real excess and the survey
footprint itself (the RA/Dec box is not uniform in phi2), so it's normalized
by the phi2 distribution of the full unfiltered star count (computed from
gaia_gd1_raw.parquet via the same GD1 frame used elsewhere) as a footprint
density proxy. The bins around phi2=0 are then checked against a Poisson
background estimated from nearby flanking bins.
"""

from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

from transform_gd1 import GD1_FRAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"

CANDIDATES_PATH = DATA_DIR / "gd1_final_candidates.parquet"
RAW_PATH = DATA_DIR / "gaia_gd1_raw.parquet"
PLOT_PATH = PLOT_DIR / "gd1_phi2_histogram_normalized.png"

BIN_SIZE = 0.25
CENTRAL_RANGE = (-1.0, 1.0)
BACKGROUND_RANGE = (-5.0, 5.0)  # flanking region used to estimate the local background rate


def raw_phi2():
    raw = pd.read_parquet(RAW_PATH, columns=["ra", "dec"])
    icrs = coord.SkyCoord(ra=raw["ra"].to_numpy() * u.deg, dec=raw["dec"].to_numpy() * u.deg, frame="icrs")
    return icrs.transform_to(GD1_FRAME).lat.deg


def main():
    candidates = pd.read_parquet(CANDIDATES_PATH, columns=["phi2"])["phi2"]
    print(f"Candidate row count: {len(candidates)}")

    raw = raw_phi2()

    bins = np.arange(candidates.min(), candidates.max() + BIN_SIZE, BIN_SIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    counts_candidates, _ = np.histogram(candidates, bins=bins)
    counts_raw, _ = np.histogram(raw, bins=bins)

    normalized_pct = 100 * counts_candidates / counts_raw

    PLOT_DIR.mkdir(exist_ok=True)
    fig, (ax_raw, ax_norm) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_raw.stairs(counts_raw, bins, fill=True)
    ax_raw.axvline(0, color="red", linestyle="--", linewidth=1)
    ax_raw.set_ylabel(f"raw stars per {BIN_SIZE} deg bin")
    ax_raw.set_title("Unfiltered footprint density (proxy for sky area vs phi2)")

    ax_norm.stairs(normalized_pct, bins, fill=True)
    ax_norm.axvline(0, color="red", linestyle="--", linewidth=1, label=r"$\phi_2=0$ (stream track)")
    ax_norm.set_xlabel(r"$\phi_2$ [deg]")
    ax_norm.set_ylabel("candidates / raw stars [%]")
    ax_norm.set_title("Footprint-normalized candidate density")
    ax_norm.legend()

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)

    # Poisson check on raw candidate counts (not the normalized ratio -- Poisson
    # statistics apply to counts) around phi2=0, using flanking bins as background.
    central_mask = (bin_centers >= CENTRAL_RANGE[0]) & (bin_centers <= CENTRAL_RANGE[1])
    background_mask = (
        (bin_centers >= BACKGROUND_RANGE[0]) & (bin_centers < CENTRAL_RANGE[0])
    ) | (
        (bin_centers > CENTRAL_RANGE[1]) & (bin_centers <= BACKGROUND_RANGE[1])
    )

    background_rate = counts_candidates[background_mask].mean()
    print(f"\nLocal background rate (flanking {BACKGROUND_RANGE} deg, excluding {CENTRAL_RANGE}): "
          f"{background_rate:.2f} stars/bin, from {background_mask.sum()} bins")

    print(f"\nBins in phi2 = {CENTRAL_RANGE}:")
    print(f"{'phi2 center':>12} {'observed':>9} {'expected':>9} {'z':>6} {'p(>=obs)':>10} elevated?")
    n_elevated = 0
    for center, observed in zip(bin_centers[central_mask], counts_candidates[central_mask]):
        z = (observed - background_rate) / np.sqrt(background_rate)
        p = poisson.sf(observed - 1, background_rate)
        elevated = p < 0.05
        n_elevated += elevated
        print(f"{center:12.3f} {observed:9d} {background_rate:9.2f} {z:6.2f} {p:10.4f} {'yes' if elevated else ''}")

    total_observed = counts_candidates[central_mask].sum()
    total_expected = background_rate * central_mask.sum()
    p_agg = poisson.sf(total_observed - 1, total_expected)
    print(f"\n{n_elevated} of {central_mask.sum()} central bins individually elevated above background at p<0.05")
    print(f"Aggregate: {total_observed} observed vs {total_expected:.2f} expected in [{CENTRAL_RANGE[0]}, {CENTRAL_RANGE[1]}] deg, "
          f"p(>=observed | background) = {p_agg:.2e}")


if __name__ == "__main__":
    main()
