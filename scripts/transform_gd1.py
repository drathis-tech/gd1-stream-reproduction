"""Transform Gaia DR3 stars into the GD-1 stream coordinate frame.

The GD-1 great-circle frame is defined by the pole and origin from Koposov,
Rix & Hogg (2010, https://arxiv.org/abs/0907.1085), built as an
astropy SkyOffsetFrame centered on the origin and rotated so that its pole
coincides with the GD-1 pole. The pole/origin below are derived from the
rotation matrix given in the appendix of that paper (as also used by
gala.coordinates.GD1Koposov10Frame) and the resulting SkyOffsetFrame was
verified to reproduce that matrix transform exactly.
"""

from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"

INPUT_PATH = DATA_DIR / "gaia_gd1_raw.parquet"
OUTPUT_PATH = DATA_DIR / "gaia_gd1_transformed.parquet"
PLOT_PATH = PLOT_DIR / "gd1_footprint.png"

GD1_ORIGIN = coord.SkyCoord(ra=200.000000 * u.deg, dec=59.450434 * u.deg, frame="icrs")
GD1_POLE = coord.SkyCoord(ra=34.598700 * u.deg, dec=29.733100 * u.deg, frame="icrs")
GD1_ROTATION = GD1_ORIGIN.position_angle(GD1_POLE)
GD1_FRAME = GD1_ORIGIN.skyoffset_frame(rotation=GD1_ROTATION)


def main():
    df = pd.read_parquet(INPUT_PATH)

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

    df.to_parquet(OUTPUT_PATH, index=False)

    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df["phi1"], df["phi2"], s=0.5, alpha=0.1, edgecolor="none", rasterized=True)
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("GD-1 footprint")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)

    print(f"Row count: {len(df)}")


if __name__ == "__main__":
    main()
