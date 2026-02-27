# Data Audit Report

## Scope
- Datasets: `PC_SAFT_NA_ML_Dataset.json`, `PC_SAFT_SA_ML_Dataset.json`.
- Checks: nulls, impossible ranges, extreme q01/q99 quantiles.
- Deterministic null policy for `hhb_cosm_kj_kmol`: class+family median (n>=3), else class median, else global median.
- Heavy-tail policy (applied to all scenarios): winsorize to [0.01, 0.99] then robust scale by median/IQR.

## Heavy-Tail Policy Parameters
| Feature | Winsor Low (q01) | Winsor High (q99) | Median | IQR |
|---|---:|---:|---:|---:|
| pc_bar | 9.393250 | 82.154000 | 33.500000 | 15.731900 |
| molcecular_weight | 34.010064 | 432.690770 | 127.012320 | 66.814870 |
| epsilonk_k | 151.331900 | 375.900730 | 259.971500 | 49.824500 |

## Audit Findings Summary
| Issue Type | Reason Code | Count |
|---|---|---:|
| EXTREME_QUANTILE | QUANTILE_HIGH_OUTLIER | 377 |
| EXTREME_QUANTILE | QUANTILE_LOW_OUTLIER | 304 |
| NULL | HHB_IMPUTE_CLASS_FAMILY_MEDIAN | 7 |
| NULL | HHB_IMPUTE_CLASS_MEDIAN | 2 |

## Impossible Range Findings
- Count: 0

## `hhb_cosm_kj_kmol` Null Handling (Affected Compounds)
| Dataset | Compound | CASRN | Action | Reason Code |
|---|---|---|---|---|
| NA | AMMONIUM CHLORIDE | "12125-02-9" | IMPUTED_TO_9.7871415 | HHB_IMPUTE_CLASS_MEDIAN |
| NA | MONOPALMITIN | "542-44-9" | IMPUTED_TO_9.7871415 | HHB_IMPUTE_CLASS_MEDIAN |
| NA | 1,4-BIS(3-AMINOPROPYL)PIPERAZINE | "7209-38-3" | IMPUTED_TO_4.219936 | HHB_IMPUTE_CLASS_FAMILY_MEDIAN |
| NA | 2-AMINODIPHENYL | "90-41-5" | IMPUTED_TO_5.580618 | HHB_IMPUTE_CLASS_FAMILY_MEDIAN |
| NA | TETRAETHYLENE GLYCOL MONOETHYL ETHER | "5650-20-4" | IMPUTED_TO_5.2321045 | HHB_IMPUTE_CLASS_FAMILY_MEDIAN |
| NA | 3-HYDROXYBUTYROLACTONE | "7331-52-4" | IMPUTED_TO_0.863232 | HHB_IMPUTE_CLASS_FAMILY_MEDIAN |
| NA | 4-((2-HYDROXYETHOXY)CARBONYL) BENZOIC ACID | "1137-99-1" | IMPUTED_TO_18.583367 | HHB_IMPUTE_CLASS_FAMILY_MEDIAN |
| NA | 1,4-BENZENEDICARBOXYLIC ACID, 2-HYDROXYETHYL METHYL ESTER | "3645-00-9" | IMPUTED_TO_0.863232 | HHB_IMPUTE_CLASS_FAMILY_MEDIAN |
| NA | METHYL RICINOLEATE | "141-24-2" | IMPUTED_TO_0.863232 | HHB_IMPUTE_CLASS_FAMILY_MEDIAN |

## Version-Controlled Outputs
- `audit/data_audit_report.csv`
- `audit/data_audit_report.md`
- `audit/PC_SAFT_NA_ML_Dataset_preprocessed.json`
- `audit/PC_SAFT_SA_ML_Dataset_preprocessed.json`
