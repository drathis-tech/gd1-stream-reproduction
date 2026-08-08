"""Summary diagnostic plots for the final GD-1 stream candidate sample (v2).

Four plots, each showing a different line of evidence that gd1_final_v2.parquet
(688 stars) is a real detection of the GD-1 stream rather than residual field
contamination: where the candidates sit in proper-motion space relative to
the field, whether they trace the expected isochrone in color-magnitude
space, how much the full selection pipeline concentrated the footprint, and
whether the along-stream density shows the literature-reported gaps.
"""

from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isochrone_cut import DISTANCE_MODULUS, ISOCHRONE_CACHE_PATH, fetch_mist_isochrone, load_isochrone
from transform_gd1 import GD1_FRAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"

CANDIDATES_PATH = DATA_DIR / "gd1_final_v2.parquet"
RAW_PATH = DATA_DIR / "gaia_gd1_raw.parquet"
TRANSFORMED_PATH = DATA_DIR / "gaia_gd1_transformed.parquet"

RAW_SAMPLE_SIZE = 50_000
RANDOM_STATE = 42


def plot_pm_diagnostic(candidates):
    raw = pd.read_parquet(RAW_PATH, columns=["ra", "dec", "pmra", "pmdec"])
    raw_sample = raw.sample(n=RAW_SAMPLE_SIZE, random_state=RANDOM_STATE)

    icrs = coord.SkyCoord(
        ra=raw_sample["ra"].to_numpy() * u.deg,
        dec=raw_sample["dec"].to_numpy() * u.deg,
        pm_ra_cosdec=raw_sample["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=raw_sample["pmdec"].to_numpy() * u.mas / u.yr,
        frame="icrs",
    )
    gd1 = icrs.transform_to(GD1_FRAME)
    raw_pm_phi1 = gd1.pm_lon_coslat.to(u.mas / u.yr).value
    raw_pm_phi2 = gd1.pm_lat.to(u.mas / u.yr).value

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(raw_pm_phi1, raw_pm_phi2, s=2, alpha=0.15, color="lightgray", edgecolor="none", label=f"{RAW_SAMPLE_SIZE:,} random raw stars")
    ax.scatter(candidates["pm_phi1_cosphi2"], candidates["pm_phi2"], s=6, alpha=0.8, color="red", edgecolor="none", label="688 final candidates")
    ax.set_xlim(-15, 5)
    ax.set_ylim(-8, 8)
    ax.set_xlabel(r"$\mu_{\phi_1}\cos\phi_2$ [mas/yr]")
    ax.set_ylabel(r"$\mu_{\phi_2}$ [mas/yr]")
    ax.set_title("Final candidates in proper-motion space vs. field population")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pm_diagnostic.png", dpi=150)
    plt.close(fig)

    # Caption: the 688 final candidates (red) sit in a tight, coherent clump in
    # proper-motion space, offset from the bulk of the field population (light
    # gray, a random 50,000-star sample of the whole raw field), which is
    # exactly the kind of localized overdensity a real, kinematically cold
    # stream should produce against disk/halo field stars.


def plot_cmd_final_candidates(candidates):
    bp_rp = candidates["phot_bp_mean_mag"] - candidates["phot_rp_mean_mag"]

    iso_path = Path(ISOCHRONE_CACHE_PATH)
    fetch_mist_isochrone(iso_path)
    iso = load_isochrone(iso_path)
    iso_g = iso["Gaia_G_EDR3"] + DISTANCE_MODULUS
    iso_bp_rp = iso["Gaia_BP_EDR3"] - iso["Gaia_RP_EDR3"]
    order = np.argsort(iso_g.to_numpy())
    iso_g_sorted = iso_g.to_numpy()[order]
    iso_bp_rp_sorted = iso_bp_rp.to_numpy()[order]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(bp_rp, candidates["phot_g_mean_mag"], s=10, alpha=0.7, color="red", edgecolor="none", label="688 final candidates")
    ax.plot(iso_bp_rp_sorted, iso_g_sorted, color="black", lw=1.5, label="MIST isochrone (12 Gyr, [Fe/H]=-2.2)")
    ax.invert_yaxis()
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$G$")
    ax.set_title("Final candidates: color-magnitude diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cmd_final_candidates.png", dpi=150)
    plt.close(fig)

    # Caption: the 688 final candidates trace the 12 Gyr, [Fe/H]=-2.2 MIST
    # isochrone (shifted to GD-1's distance) closely along the main sequence
    # and subgiant branch, confirming the isochrone cut selected a single old,
    # metal-poor stellar population consistent with GD-1 rather than a mix of
    # unrelated field stars of different ages/metallicities.


def plot_before_after(candidates):
    transformed = pd.read_parquet(TRANSFORMED_PATH, columns=["phi1", "phi2"])
    narrow = candidates[candidates["phi2"].abs() < 1.0]

    phi1_min, phi1_max = transformed["phi1"].min(), transformed["phi1"].max()

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

    ax_before.scatter(transformed["phi1"], transformed["phi2"], s=0.3, alpha=0.05, color="steelblue", edgecolor="none", rasterized=True)
    ax_before.set_xlim(phi1_min, phi1_max)
    ax_before.set_xlabel(r"$\phi_1$ [deg]")
    ax_before.set_ylabel(r"$\phi_2$ [deg]")
    ax_before.set_title(f"Before: raw footprint ({len(transformed):,} stars)")

    ax_after.scatter(narrow["phi1"], narrow["phi2"], s=8, alpha=0.7, color="red", edgecolor="none")
    ax_after.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_after.set_xlim(phi1_min, phi1_max)
    ax_after.set_ylim(-1, 1)
    ax_after.set_xlabel(r"$\phi_1$ [deg]")
    ax_after.set_title(f"After: final candidates, |phi2|<1 deg ({len(narrow)} stars)")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "before_after_comparison.png", dpi=150)
    plt.close(fig)

    # Caption: same phi1 axis range on both panels. The left panel is the
    # entire unfiltered survey footprint (2.6M stars spanning the full RA/Dec
    # box); the right panel is the final PM + distance + footprint +
    # isochrone selected sample, restricted to |phi2|<1 deg. The selection
    # pipeline collapsed a diffuse, roughly uniform field of millions of stars
    # down to a comparatively tiny, spatially concentrated sample tracing a
    # narrow band, which is the visual signature of successfully isolating a
    # thin stream from the general field.


def plot_phi1_density(candidates):
    bin_size = 1.5
    bins = np.arange(candidates["phi1"].min(), candidates["phi1"].max() + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(candidates["phi1"], bins=bins, histtype="stepfilled", edgecolor="black", linewidth=0.5)
    ax.axvline(-20, color="red", linestyle="--", linewidth=1, label="literature gaps (phi1=-20, -40)")
    ax.axvline(-40, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(f"candidates per {bin_size} deg bin")
    ax.set_title("Final candidates: along-stream (phi1) density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "phi1_density.png", dpi=150)
    plt.close(fig)

    # Caption: along-stream density of the 688 final candidates, with dashed
    # lines marking phi1=-20 and phi1=-40, the approximate locations of gaps
    # reported by Price-Whelan & Bonaca (2018) and de Boer et al. (2018). At
    # this sample size (~1-2 stars/bin away from the densest region), Poisson
    # noise is large enough that any dip at these locations should be read as
    # suggestive at best, not a confirmed independent detection of the gaps.


def main():
    candidates = pd.read_parquet(CANDIDATES_PATH)
    PLOT_DIR.mkdir(exist_ok=True)

    plot_pm_diagnostic(candidates)
    plot_cmd_final_candidates(candidates)
    plot_before_after(candidates)
    plot_phi1_density(candidates)


if __name__ == "__main__":
    main()
