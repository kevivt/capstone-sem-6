# Threshold Tuning Applied

- Source diagnostics: `reports\current\model_diagnostics\threshold_diagnostics_summary.csv`
- Updated config: `artifacts\risk_thresholds_and_factors.json`
- Backup created: `artifacts\risk_thresholds_and_factors.backup_20260420_110939.json`

## Changes

| dataset      |   old_moderate |   new_moderate |   old_high |   new_high |   default_f1 |   best_f1 |   default_accuracy |   best_accuracy |
|:-------------|---------------:|---------------:|-----------:|-----------:|-------------:|----------:|-------------------:|----------------:|
| ckd          |       0.490566 |           0.49 |   0.640566 |   0.640566 |     0.138399 |  0.138719 |           0.809105 |        0.806482 |
| diabetes     |       0.35     |           0.1  |   0.9      |   0.9      |     0.803912 |  0.966346 |           0.68884  |        0.93676  |
| hypertension |       0.35     |           0.1  |   0.9      |   0.9      |     0.913327 |  0.985698 |           0.84299  |        0.97204  |
