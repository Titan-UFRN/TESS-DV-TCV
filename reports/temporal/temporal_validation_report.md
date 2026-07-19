# True Temporal Validation

Cutoff: 2021-06-17 archived MAST TOI catalog, Sectors 1-36.

Development: 396 rows on 390 hosts. 
Future-resolved test: 828 rows on 780 hosts.

## Repeated development-split results

The fixed future cohort is unchanged; only the historical train/calibration host partition varies.

| Model | Brier mean [split interval] | AP mean [split interval] | P@100 mean [split interval] |
|---|---:|---:|---:|
| `snapshot_catalog` | 0.2922 [0.2587, 0.3184] | 0.5863 [0.5590, 0.6146] | 0.602 [0.505, 0.675] |
| `snapshot_availability` | 0.2851 [0.2549, 0.3149] | 0.5920 [0.5640, 0.6213] | 0.606 [0.484, 0.676] |
| `snapshot_dv_header` | 0.2755 [0.2492, 0.3019] | 0.6265 [0.5968, 0.6576] | 0.662 [0.600, 0.720] |
| `snapshot_dvr_xml` | 0.2641 [0.2446, 0.2957] | 0.5931 [0.5601, 0.6360] | 0.627 [0.559, 0.701] |
| `all_cutoff_evidence` | 0.2454 [0.2256, 0.2765] | 0.6484 [0.6117, 0.6814] | 0.719 [0.645, 0.795] |

## Paired TIC-cluster bootstrap

Positive values favor all cutoff-eligible evidence over the snapshot catalog baseline.

| Metric | Mean improvement | 95% interval | Positive fraction |
|---|---:|---:|---:|
| `brier` | 0.0325 | [0.0203, 0.0441] | 1.000 |
| `average_precision` | 0.0521 | [0.0151, 0.0910] | 0.996 |
| `roc_auc` | 0.0302 | [0.0053, 0.0559] | 0.989 |
| `precision_at_100` | 0.1321 | [0.0300, 0.2200] | 0.997 |

## Diagnostic content beyond product availability

This comparison controls for the indicator that a cutoff-eligible DV/XML product exists.

| Metric | Mean improvement | 95% interval | Positive fraction |
|---|---:|---:|---:|
| `brier` | 0.0246 | [0.0131, 0.0357] | 1.000 |
| `average_precision` | 0.0452 | [0.0099, 0.0815] | 0.993 |
| `roc_auc` | 0.0212 | [-0.0021, 0.0440] | 0.961 |
| `precision_at_100` | 0.1289 | [0.0400, 0.2200] | 0.997 |

## Integrity and coverage

- Development-test TIC overlap: 0.
- Products dated after the cutoff: 0.
- Latest accepted product timestamp: 2021-06-07T21:13.
- DV matches by role: {'development_resolved_at_cutoff': 269, 'future_resolved_test': 370}.
- XML matches by role: {'development_resolved_at_cutoff': 269, 'future_resolved_test': 370}.
- Development positive rate: 0.747.
- Future-test positive rate: 0.378.
