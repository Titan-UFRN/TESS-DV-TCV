# Split Leakage Audit

Run date: 2026-07-18

Input: `../data/processed/full_dv_xml_merged.csv`

Seed: `11`

Leakage status: **PASS**

## Split Counts

| Split | Rows | Hosts | Missing TIC rows | Duplicate host rows | Known/confirmed | False positive/false alarm |
|---|---:|---:|---:|---:|---:|---:|
| train | 1486 | 1395 | 0 | 91 | 731 | 755 |
| calibration | 528 | 507 | 0 | 21 | 263 | 265 |
| test | 666 | 633 | 0 | 33 | 339 | 327 |

## Host Overlaps

| Pair | Overlap count |
|---|---:|
| train_calibration | 0 |
| train_test | 0 |
| calibration_test | 0 |

No TIC-level host leakage is present if all overlap counts are zero.
