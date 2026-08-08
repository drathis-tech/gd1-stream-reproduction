"""Apply an isochrone color cut to GD-1 proper-motion candidates.

A MIST isochrone (age=12 Gyr, [Fe/H]=-2.2 -- GD-1's old, metal-poor stellar
population) is fetched from the MIST web interpolator (https://mist.science)
using Gaia EDR3 photometry (identical passbands to Gaia DR3), and cached
locally so repeat runs don't depend on the network. It's shifted to GD-1's
distance and overlaid on the candidates' color-magnitude diagram; stars far
in color from the track at their magnitude are field-star contamination and
are cut.
"""

import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LogNorm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"

INPUT_PATH = DATA_DIR / "gd1_candidates.parquet"
OUTPUT_PATH = DATA_DIR / "gd1_final_candidates.parquet"
CMD_PLOT_PATH = PLOT_DIR / "gd1_cmd_isochrone.png"
FOOTPRINT_PLOT_PATH = PLOT_DIR / "gd1_stream_final.png"
ISOCHRONE_CACHE_PATH = DATA_DIR / "mist_iso_12gyr_feh-2.2.txt"

MIST_AGE_YR = 12.0e9
MIST_FEH = -2.2
DISTANCE_MODULUS = 14.6  # ~8.3 kpc, within GD-1's literature range of 7.5-9 kpc
COLOR_CUT_WIDTH = 0.05  # mag, in bp_rp, applied on either side of the isochrone track
MAX_PHASE = 2  # MIST phase: 0=main sequence, 2=RGB (keep pre-horizontal-branch only)
BIN_SIZE = 0.2

RUWE_MAX = 1.2
PARALLAX_MIN, PARALLAX_MAX = -0.5, 0.5


def fetch_mist_isochrone(path):
    if path.exists():
        return
    path.parent.mkdir(exist_ok=True)
    form_data = {
        "version": "MIST1",
        "v_div_vcrit": "vvcrit0.4",
        "age_scale": "linear",
        "age_type": "single",
        "age_value": str(MIST_AGE_YR),
        "FeH_value": str(MIST_FEH),
        "alpha_value": "p0",
        "output_option": "photometry",
        "output": "UBVRIplus",
        "Av_value": "0",
    }
    resp = requests.post("https://mist.science/iso_form.php", data=form_data, timeout=60)
    resp.raise_for_status()
    link = resp.text.split('href="')[1].split('"')[0]
    zip_resp = requests.get(f"https://mist.science/{link}", timeout=60)
    zip_resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(zip_resp.content))
    iso_name = next(n for n in zf.namelist() if n.endswith(".UBVRIplus"))
    path.write_bytes(zf.read(iso_name))


def load_isochrone(path):
    with open(path) as f:
        for line in f:
            if line.startswith("#") and "EEP" in line and "isochrone_age_yr" in line:
                columns = line.lstrip("#").split()
                break
    iso = pd.read_csv(path, comment="#", sep=r"\s+", names=columns)
    return iso[iso["phase"] <= MAX_PHASE]


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Row count loaded (PM-selected candidates): {len(df)}")

    df = df[df["ruwe"] < RUWE_MAX]
    print(f"Row count after ruwe < {RUWE_MAX} cut: {len(df)}")

    df = df[df["parallax"].between(PARALLAX_MIN, PARALLAX_MAX)]
    print(f"Row count after parallax in [{PARALLAX_MIN}, {PARALLAX_MAX}] cut: {len(df)}")

    bp_rp = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]

    iso_path = Path(ISOCHRONE_CACHE_PATH)
    fetch_mist_isochrone(iso_path)
    iso = load_isochrone(iso_path)

    iso_g = iso["Gaia_G_EDR3"] + DISTANCE_MODULUS
    iso_bp_rp = iso["Gaia_BP_EDR3"] - iso["Gaia_RP_EDR3"]
    order = np.argsort(iso_g.to_numpy())
    iso_g_sorted = iso_g.to_numpy()[order]
    iso_bp_rp_sorted = iso_bp_rp.to_numpy()[order]

    PLOT_DIR.mkdir(exist_ok=True)

    # color-magnitude diagram with isochrone overlay, before the isochrone cut
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(bp_rp, df["phot_g_mean_mag"], s=1, alpha=0.3, edgecolor="none", rasterized=True, label="ruwe+parallax candidates")
    ax.plot(iso_bp_rp_sorted, iso_g_sorted, color="red", lw=1.5, label="MIST isochrone (12 Gyr, [Fe/H]=-2.2)")
    ax.invert_yaxis()
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$G$")
    ax.set_title("GD-1 candidates with isochrone overlay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CMD_PLOT_PATH, dpi=150)

    iso_color_at_g = np.interp(df["phot_g_mean_mag"], iso_g_sorted, iso_bp_rp_sorted)
    in_g_range = df["phot_g_mean_mag"].between(iso_g_sorted.min(), iso_g_sorted.max())
    color_mask = (bp_rp - iso_color_at_g).abs() < COLOR_CUT_WIDTH
    final = df[in_g_range & color_mask]

    print(f"Row count after isochrone color cut (+/-{COLOR_CUT_WIDTH} mag): {len(final)}")

    DATA_DIR.mkdir(exist_ok=True)
    final.to_parquet(OUTPUT_PATH, index=False)

    phi1_bins = np.arange(final["phi1"].min(), final["phi1"].max() + BIN_SIZE, BIN_SIZE)
    phi2_bins = np.arange(final["phi2"].min(), final["phi2"].max() + BIN_SIZE, BIN_SIZE)

    fig, ax = plt.subplots(figsize=(10, 4))
    _, _, _, im = ax.hist2d(
        final["phi1"], final["phi2"], bins=[phi1_bins, phi2_bins], norm=LogNorm(), cmap="viridis"
    )
    fig.colorbar(im, ax=ax, label="stars per bin")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("GD-1 stream candidates (proper motion + isochrone selected)")
    fig.tight_layout()
    fig.savefig(FOOTPRINT_PLOT_PATH, dpi=150)


if __name__ == "__main__":
    main()
