"""Query Gaia DR3 for a sky region and save the results to a local parquet file.

Uses the AIP Gaia mirror (https://gaia.aip.de/tap) rather than the ESA archive:
the ESA async TAP endpoint was returning server errors on job submission at the
time this was written, while AIP mirrors the same gaiadr3.gaia_source table.

The full region (RA 130-200, Dec 20-60) matches ~2.6M rows. Querying it in one
shot hits the AIP server's statement timeout, so the RA range is split into
5-degree chunks that are queried and concatenated.
"""

from pathlib import Path

import pandas as pd
import pyvo

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TAP_URL = "https://gaia.aip.de/tap"

RA_MIN, RA_MAX = 130, 200
DEC_MIN, DEC_MAX = 20, 60
RA_CHUNK_SIZE = 5

QUERY_TEMPLATE = """
SELECT source_id, ra, dec, pmra, pmra_error, pmdec, pmdec_error,
       parallax, parallax_error, phot_g_mean_mag, phot_bp_mean_mag,
       phot_rp_mean_mag, ruwe
FROM gaiadr3.gaia_source
WHERE ra BETWEEN {ra_lo} AND {ra_hi}
  AND dec BETWEEN {dec_lo} AND {dec_hi}
  AND parallax < 1
  AND phot_g_mean_mag < 20.5
"""

OUTPUT_PATH = DATA_DIR / "gaia_gd1_raw.parquet"


def fetch_chunk(service, ra_lo, ra_hi):
    query = QUERY_TEMPLATE.format(ra_lo=ra_lo, ra_hi=ra_hi, dec_lo=DEC_MIN, dec_hi=DEC_MAX)
    job = service.submit_job(query, maxrec=3_000_000)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
    job.raise_if_error()
    return job.fetch_result().to_table().to_pandas()


def main():
    service = pyvo.dal.TAPService(TAP_URL)
    chunks = []
    ra_lo = RA_MIN
    while ra_lo < RA_MAX:
        ra_hi = min(ra_lo + RA_CHUNK_SIZE, RA_MAX)
        print(f"Fetching RA {ra_lo}-{ra_hi}...")
        chunk = fetch_chunk(service, ra_lo, ra_hi)
        print(f"  got {len(chunk)} rows")
        chunks.append(chunk)
        ra_lo = ra_hi

    df = pd.concat(chunks, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Row count: {len(df)}")


if __name__ == "__main__":
    main()
