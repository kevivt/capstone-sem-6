# Nutrient Dataset Calibration Outputs

Generated disease-specific nutrient suitability datasets and runtime calibration rules from the unified food KB.

## Summary

| disease      |   total_candidates |   top500_count |   score_mean |   score_p90 |   score_max |   high_tier_count |   moderate_tier_count |   low_tier_count |
|:-------------|-------------------:|---------------:|-------------:|------------:|------------:|------------------:|----------------------:|-----------------:|
| ckd          |               6313 |            500 |     0.499937 |    0.649968 |    0.675756 |              2149 |                  2082 |             2082 |
| diabetes     |               6313 |            500 |     0.499984 |    0.714881 |    0.844092 |              2147 |                  2083 |             2083 |
| hypertension |               6313 |            500 |     0.499984 |    0.711904 |    0.857556 |              2147 |                  2083 |             2083 |

## Outputs

- `{disease}_nutrient_candidates.csv` (all candidates with suitability score)
- `{disease}_top500_candidates.csv` (top-ranked foods for recommendation pipeline)
- `nutrient_dataset_summary.csv`
- `nutrient_suitability_distributions.png`
- `artifacts/nutrient_calibration_rules.json`