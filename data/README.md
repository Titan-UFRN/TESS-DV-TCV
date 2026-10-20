# Data and provenance

The repository does not track generated or archive-derived data. The exact
processed inputs used for the paper total approximately 11.6 MB and are pinned
in [`SHA256SUMS`](SHA256SUMS). They should be distributed as a versioned asset
with the corresponding GitHub/archival release and restored without changing
their relative paths.

Before publishing a paper release:

1. Package the six files listed in `SHA256SUMS` without renaming them.
2. Attach the bundle to the matching GitHub release and archival deposit.
3. Record the release URL and archival DOI in `CITATION.cff` and this file.
4. Verify the unpacked bundle with `python3 scripts/verify_inputs.py`.

Until an archival DOI is assigned, the repository's release page is:
<https://github.com/Titan-UFRN/TESS-DV-TCV/releases>.

## Pinned inputs

| Path | Bytes | Purpose |
|---|---:|---|
| `data/processed/candidates.csv` | 2,435,538 | Fixed 2026-07-18 archive/TIC candidate table |
| `data/processed/full_dv_xml_merged.csv` | 6,333,714 | Retrospective candidate-level tabular fusion table |
| `reports/candidate_triage_ranking.csv` | 470,410 | Frozen retrospective ranking input used by policy audits |
| `data/processed/temporal_cohort_20210617.csv` | 989,927 | Cutoff development and future-resolved cohort |
| `data/processed/temporal_dv_20210617.csv` | 378,204 | Cutoff-eligible DV header records |
| `data/processed/temporal_xml_20210617.csv` | 994,412 | Cutoff-eligible parsed DVR XML records |

## Upstream provenance

- Historical TOI catalog:
  `https://archive.stsci.edu/missions/tess/catalogs/toi/previous/tess20210617_toi-catalog.csv`
- Outcome catalog: MAST TOI catalog file created on 2026-03-14. This dated copy
  must come from the versioned data bundle because the live `tois.csv` changes.
- DV and DVR products: public TESS-SPOC/MAST products, limited to sectors ending
  no later than Sector 36 and product timestamps no later than 2021-06-17.
- Retrospective candidate snapshot: NASA Exoplanet Archive and TIC data queried
  on 2026-07-18, with candidate-level DV/DVR associations.

`make temporal-fetch` reconstructs the historical catalog and retrieves eligible
products from public services. It also downloads the current live outcome
catalog, so a later run is a new temporal cohort rather than an exact
reproduction of the 2026-03-14 endpoint. The versioned processed bundle remains
the authoritative source for the published result.

The source archives retain ownership and terms for their products. This
repository's BSD license applies to code, not to third-party data.
