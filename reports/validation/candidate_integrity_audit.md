# Candidate Integrity Audit

Status: **PASS**

| Check | Value |
|---|---:|
| Candidate rows | 8064 |
| Merged rows | 8064 |
| Candidate duplicate rows | 0 |
| Merged duplicate rows | 0 |
| Missing candidate keys | 0 |
| Extra merged keys | 0 |
| Identity fields with mismatches | 0 |
| Candidate-matched DV products | 5555 |
| Candidate-matched XML products | 5555 |
| TICs matched to more than one TOI | 0 |

The audit fails if candidate identity changes during the merge, if a candidate key is duplicated or lost, or if one selected host product is assigned to more than one candidate.
