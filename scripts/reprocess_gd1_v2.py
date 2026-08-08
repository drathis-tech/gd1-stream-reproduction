"""Reprocess raw Gaia data with phi1-dependent PM, distance, and footprint cuts.

Builds on the official Price-Whelan & Bonaca (2018) GD-1 track (pm_phi1_cosphi2,
pm_phi2, and distance as functions of phi1 -- see gd1_pm_track.py) rather than
the single fixed-box cuts used previously. galstreams itself couldn't be
installed (its hard dependency `gala` has no Windows wheels), so the same
underlying official track table is read directly instead.
"""

from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from gd1_pm_track import PM_PHI1_TRACK, PM_PHI2_TRACK, TRACK_PHI1_RANGE, distance_kpc
from isochrone_cut import COLOR_CUT_WIDTH, DISTANCE_MODULUS, ISOCHRONE_CACHE_PATH, fetch_mist_isochrone, load_isochrone
from transform_gd1 import GD1_FRAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"

RAW_PATH = DATA_DIR / "gaia_gd1_raw.parquet"
OUTPUT_PATH = DATA_DIR / "gd1_final_v2.parquet"
PLOT_PATH = PLOT_DIR / "gd1_stream_v2.png"

PM_TOLERANCE = 2.0  # mas/yr, applied independently to each PM component
PARALLAX_N_SIGMA = 3.0  # multiples of each star's own Gaia parallax_error
PHI2_MAX_ABS = 3.0  # deg
PHI1_MIN, PHI1_MAX = -85.0, 20.0
BIN_SIZE = 0.2


def main():
    df = pd.read_parquet(RAW_PATH)
    print(f"Row count loaded (raw): {len(df)}")

    icrs = coord.SkyCoord(
        ra=df["ra"].to_numpy() * u.deg,
        dec=df["dec"].to_numpy() * u.deg,
        pm_ra_cosdec=df["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=df["pmdec"].to_numpy() * u.mas / u.yr,
        frame="icrs",
    )
    gd1 = icrs.transform_to(GD1_FRAME)
    df["phi1"] = gd1.lon.wrap_at(180 * u.deg).deg
    df["phi2"] = gd1.lat.deg
    df["pm_phi1_cosphi2"] = gd1.pm_lon_coslat.to(u.mas / u.yr).value
    df["pm_phi2"] = gd1.pm_lat.to(u.mas / u.yr).value

    # phi1-dependent PM selection (galstreams/PW18 track)
    in_track_range = df["phi1"].between(*TRACK_PHI1_RANGE)
    pm_phi1_track_at_star = PM_PHI1_TRACK(df["phi1"])
    pm_phi2_track_at_star = PM_PHI2_TRACK(df["phi1"])
    pm_mask = (
        in_track_range
        & ((df["pm_phi1_cosphi2"] - pm_phi1_track_at_star).abs() < PM_TOLERANCE)
        & ((df["pm_phi2"] - pm_phi2_track_at_star).abs() < PM_TOLERANCE)
    )
    df = df[pm_mask]
    print(f"Row count after phi1-dependent PM cut (+/-{PM_TOLERANCE} mas/yr of PW18 track): {len(df)}")

    # phi1-dependent distance/parallax selection (galstreams/PW18 track)
    parallax_expected = 1.0 / distance_kpc(df["phi1"])  # mas, for distance in kpc
    parallax_mask = (df["parallax"] - parallax_expected).abs() < PARALLAX_N_SIGMA * df["parallax_error"]
    df = df[parallax_mask]
    print(f"Row count after phi1-dependent parallax cut (+/-{PARALLAX_N_SIGMA} sigma of PW18 distance track): {len(df)}")

    # tighter sky footprint, matching published GD-1 analyses
    footprint_mask = (df["phi2"].abs() < PHI2_MAX_ABS) & df["phi1"].between(PHI1_MIN, PHI1_MAX)
    df = df[footprint_mask]
    print(f"Row count after footprint cut (|phi2|<{PHI2_MAX_ABS}, {PHI1_MIN}<phi1<{PHI1_MAX}): {len(df)}")

    # isochrone color cut, same MIST track/tolerance as isochrone_cut.py
    bp_rp = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    iso_path = Path(ISOCHRONE_CACHE_PATH)
    fetch_mist_isochrone(iso_path)
    iso = load_isochrone(iso_path)

    iso_g = iso["Gaia_G_EDR3"] + DISTANCE_MODULUS
    iso_bp_rp = iso["Gaia_BP_EDR3"] - iso["Gaia_RP_EDR3"]
    order = np.argsort(iso_g.to_numpy())
    iso_g_sorted = iso_g.to_numpy()[order]
    iso_bp_rp_sorted = iso_bp_rp.to_numpy()[order]

    iso_color_at_g = np.interp(df["phot_g_mean_mag"], iso_g_sorted, iso_bp_rp_sorted)
    in_g_range = df["phot_g_mean_mag"].between(iso_g_sorted.min(), iso_g_sorted.max())
    color_mask = (bp_rp - iso_color_at_g).abs() < COLOR_CUT_WIDTH
    final = df[in_g_range & color_mask]
    print(f"Row count after isochrone color cut (+/-{COLOR_CUT_WIDTH} mag): {len(final)}")

    DATA_DIR.mkdir(exist_ok=True)
    final.to_parquet(OUTPUT_PATH, index=False)

    phi1_bins = np.arange(final["phi1"].min(), final["phi1"].max() + BIN_SIZE, BIN_SIZE)
    phi2_bins = np.arange(final["phi2"].min(), final["phi2"].max() + BIN_SIZE, BIN_SIZE)

    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    _, _, _, im = ax.hist2d(
        final["phi1"], final["phi2"], bins=[phi1_bins, phi2_bins], norm=LogNorm(), cmap="viridis"
    )
    fig.colorbar(im, ax=ax, label="stars per bin")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("GD-1 stream candidates (PW18 phi1-dependent PM + distance + isochrone selected)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


if __name__ == "__main__":
    main()
