"""GD-1's expected proper-motion track vs phi1, from Price-Whelan & Bonaca (2018).

Fetches the densely-sampled GD-1-PB18 stream track (fit by Price-Whelan &
Bonaca 2018 to their Gaia DR2 + PanSTARRS-1 selected members) from the
galstreams library (https://github.com/cmateu/galstreams), caches it
locally, transforms it into the GD-1 frame, and fits 5th-degree polynomials
to pm_phi1_cosphi2(phi1) and pm_phi2(phi1) -- the same polynomial degree
galstreams itself uses to build this track.
"""

from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import requests

from transform_gd1 import GD1_FRAME

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRACK_URL = "https://raw.githubusercontent.com/cmateu/galstreams/main/galstreams/tracks/track.st.GD-1.pricewhelan2018.ecsv"
TRACK_CACHE_PATH = DATA_DIR / "gd1_pw18_track.ecsv"

POLY_DEGREE = 5


def fetch_track(path=TRACK_CACHE_PATH):
    if not path.exists():
        resp = requests.get(TRACK_URL, timeout=60)
        resp.raise_for_status()
        DATA_DIR.mkdir(exist_ok=True)
        path.write_bytes(resp.content)
    return path


def load_track_phi1_pm(path=TRACK_CACHE_PATH):
    fetch_track(path)
    df = pd.read_csv(path, comment="#")

    icrs = coord.SkyCoord(
        ra=df["ra"].to_numpy() * u.deg,
        dec=df["dec"].to_numpy() * u.deg,
        pm_ra_cosdec=df["pm_ra_cosdec"].to_numpy() * u.mas / u.yr,
        pm_dec=df["pm_dec"].to_numpy() * u.mas / u.yr,
        frame="icrs",
    )
    gd1 = icrs.transform_to(GD1_FRAME)

    phi1 = gd1.lon.wrap_at(180 * u.deg).deg
    pm_phi1_cosphi2 = gd1.pm_lon_coslat.to(u.mas / u.yr).value
    pm_phi2 = gd1.pm_lat.to(u.mas / u.yr).value

    order = np.argsort(phi1)
    return phi1[order], pm_phi1_cosphi2[order], pm_phi2[order]


def fit_pm_polynomials(path=TRACK_CACHE_PATH):
    phi1, pm_phi1_cosphi2, pm_phi2 = load_track_phi1_pm(path)
    pm_phi1_poly = np.poly1d(np.polyfit(phi1, pm_phi1_cosphi2, POLY_DEGREE))
    pm_phi2_poly = np.poly1d(np.polyfit(phi1, pm_phi2, POLY_DEGREE))
    phi1_range = (phi1.min(), phi1.max())
    return pm_phi1_poly, pm_phi2_poly, phi1_range


PM_PHI1_TRACK, PM_PHI2_TRACK, TRACK_PHI1_RANGE = fit_pm_polynomials()


def distance_kpc(phi1):
    """GD-1 distance track, d(kpc) = 0.05*phi1 + 10 (Price-Whelan & Bonaca 2018).

    Verified against the distance column of the official PW18 track table
    (track.st.GD-1.pricewhelan2018.ecsv): matches to ~1e-9 kpc over the
    track's actual phi1 domain (about -90 to +10 deg).
    """
    return 0.05 * phi1 + 10
