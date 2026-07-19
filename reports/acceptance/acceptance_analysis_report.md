# Acceptance analyses

The endpoint is later TOI disposition, not independent planet truth or realized RV yield.

## Event association

- Rows: 1224
- Period only dv matches: 639
- Strict dv matches: 545
- Xml records found: 639
- Strict xml matches: 510
- Ambiguous rows: 129
- Duplicate strict xml products: 0

## Two-stage host bootstrap: all strict evidence minus catalog

| Metric | Mean improvement [95% interval] |
|---|---:|
| average_precision | 0.0612 [-0.0101, 0.1472] |
| brier | 0.0324 [-0.0102, 0.0813] |
| ece | 0.0325 [-0.0278, 0.1007] |
| precision_at_10 | -0.1250 [-0.6000, 0.3000] |
| precision_at_100 | 0.1368 [-0.0300, 0.3352] |
| precision_at_50 | 0.1409 [-0.1000, 0.3800] |
| roc_auc | 0.0354 [-0.0156, 0.0952] |

Full tables include repeated splits, complete-case and availability strata, label-category sensitivity, regularization sensitivity, and score-margin abstention.
