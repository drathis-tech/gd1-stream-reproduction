# GD-1 Stream Detection Pipeline

Detects the GD-1 stellar stream in Gaia DR3 by successively cutting a raw
field sample on proper motion, parallax/distance, sky position, and
color-magnitude (isochrone) selections, following the general approach of
Price-Whelan & Bonaca (2018).

## Setup

```
python -m venv venv
venv\Scripts\activate        # venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Scripts read/write relative to the repo root (`data/`, `plots/`) regardless
of the working directory they're invoked from, so run them as e.g.:

```
python scripts/fetch_gaia.py
```

## Pipeline

Run in this order (each step reads the previous step's parquet output from
`data/`):

1. **`scripts/fetch_gaia.py`** — queries the Gaia DR3 archive (via the AIP
   mirror, `gaia.aip.de/tap`) for RA 130-200, Dec 20-60, parallax < 1,
   G < 20.5. Saves `data/gaia_gd1_raw.parquet` (~2.6M rows).
2. **`scripts/transform_gd1.py`** — transforms `ra`/`dec`/`pmra`/`pmdec` into
   the GD-1 stream frame (`phi1`, `phi2`, `pm_phi1_cosphi2`, `pm_phi2`),
   defined per Koposov, Rix & Hogg (2010). Saves
   `data/gaia_gd1_transformed.parquet`.
3. **`scripts/select_gd1.py`** — plots the raw proper-motion distribution
   (`plots/gd1_pm_locus.png`) to visualize the stream's PM locus.
4. **`scripts/gd1_pm_track.py`** — not run directly; provides the
   phi1-dependent PM and distance tracks (from Price-Whelan & Bonaca 2018,
   via the `galstreams` project's published track data) used by the next two
   steps.
5. **`scripts/select_gd1_stream.py`** — selects stars within 2.5 mas/yr of
   the PW18 PM track at each star's own `phi1`. Saves `data/gd1_candidates.parquet`.
6. **`scripts/isochrone_cut.py`** — cuts on `ruwe`, `parallax`, and a MIST
   isochrone (age=12 Gyr, [Fe/H]=-2.2, fetched from mist.science) color
   match. Saves `data/gd1_final_candidates.parquet`.
7. **`scripts/reprocess_gd1_v2.py`** — the sharper end-to-end version: starts
   from raw data and applies a phi1-dependent PM cut, a phi1-dependent
   parallax cut (from the PW18 distance track, d(kpc)=0.05·phi1+10), a tight
   footprint cut (|phi2|<3°, -85°<phi1<20°), then the isochrone cut. Saves
   `data/gd1_final_v2.parquet` (688 candidates) -- this is the pipeline's
   main result.
8. **`scripts/phi2_histogram.py`** / **`phi2_histogram_v2.py`** — 1D `phi2`
   histograms normalized by the raw star count per bin (a footprint-area
   proxy), with a Poisson significance check for an excess at phi2=0.
9. **`scripts/plot_final_narrow.py`**, **`scripts/final_plots.py`** —
   additional diagnostic plots on `gd1_final_v2.parquet` (proper-motion
   diagram, color-magnitude diagram, before/after footprint comparison,
   along-stream density).

## Result

The final sample (`data/gd1_final_v2.parquet`, 688 stars) shows a clear
excess at phi2=0 relative to the surrounding background -- 406 of 688
candidates fall within |phi2|<1°, versus ~141 expected if the central bins
had the same density as the flanking region (phi2 ∈ [-3°,-1°) ∪ (1°,3°]). A
simple Poisson comparison against that background gives p ≈ 1.2e-73, though
this should be read as a local significance check rather than a rigorous
global detection significance: the ±1° window was chosen after earlier
plots already showed the excess there, so no look-elsewhere correction has
been applied. The excess is visible directly as a narrow band in
`plots/gd1_stream_v2.png` and `plots/before_after_comparison.png`.

## Regenerating the data

`*.parquet` files are gitignored (large and fully reproducible), so a fresh
clone starts with `data/` empty of them. Regenerate from scratch by running
the pipeline scripts in order, starting with:

```
python scripts/fetch_gaia.py
```

This queries the Gaia DR3 archive and writes `data/gaia_gd1_raw.parquet`
(~2.6M rows, ~200MB) -- takes a few minutes since it's fetched in RA chunks
to stay under the archive's server-side query timeout. Every later script in
the Pipeline section above reads its input from a parquet file produced by
an earlier step, so rerun them in the listed order after this to rebuild the
rest of `data/`.

## Notes

- `galstreams` itself couldn't be installed (its dependency `gala` has no
  Windows wheels); `gd1_pm_track.py` reads the same underlying track data
  file directly from that project's GitHub repo instead.
