# Large Medical Dataset Ingestion Guide

This guide adds a large-data pipeline using BRFSS and NHANES so you can defend dataset scale during reviews.

## Why this helps

Current internal dataset sizes are small:
- CKD: 400
- Diabetes: 768
- Hypertension: 4240

The new pipeline can use public datasets with much larger volume:
- BRFSS: 400k+ respondents per year
- NHANES: 10k+ participants per cycle (multiple cycles can be combined)

## New script

Use:

```powershell
python build_large_medical_datasets.py --brfss <path_to_brfss_file> --nhanes <path_to_nhanes_file>
```

Script location:
- `Sem_6_Capstone/build_large_medical_datasets.py`

## Supported input formats

For both BRFSS and NHANES files:
- `.csv`
- `.xpt`
- `.sas7bdat`

## Recommended official sources

- BRFSS annual data: https://www.cdc.gov/brfss/annual_data/annual_data.htm
- NHANES data page: https://www.cdc.gov/nchs/nhanes/

## Example run commands

From project root:

```powershell
Set-Location "d:/sem 6/capstone/Sem_6_Capstone"
& "../.venv/Scripts/Activate.ps1"
python build_large_medical_datasets.py --brfss "medical datasets/raw/BRFSS_2023.XPT" --nhanes "medical datasets/raw/NHANES_2017_2018_MERGED.csv"
```

If only one source is available, you can run with one input:

```powershell
python build_large_medical_datasets.py --brfss "medical datasets/raw/BRFSS_2023.XPT"
python build_large_medical_datasets.py --nhanes "medical datasets/raw/NHANES_2017_2018_MERGED.csv"
```

## Outputs created

Default output directory:
- `Sem_6_Capstone/medical datasets/large/`

Files:
- `unified_large_health.csv`
- `hypertension_large.csv`
- `diabetes_large.csv`
- `ckd_large.csv`
- `large_dataset_summary.json`

## What fields are produced

Unified table columns:
- source
- age
- male
- BMI
- sysBP
- diaBP
- glucose
- diabetes
- prevalentHyp
- ckd_label
- currentSmoker
- cigsPerDay
- education
- prevalentStroke

Disease labels:
- Hypertension target: `prevalentHyp`
- Diabetes target: `diabetes`
- CKD target: `ckd_label`

## Notes for teacher presentation

Use `large_dataset_summary.json` to show:
- total unified row count
- row count by source (BRFSS vs NHANES)
- row count per disease-specific table

This is the fastest way to prove your data scale increased materially.

## Next integration step (optional)

Your current model training script (`train_models.py`) uses legacy feature schemas from smaller datasets. To train directly on new large datasets, add a second training entry point with updated feature lists for:
- `hypertension_large.csv`
- `diabetes_large.csv`
- `ckd_large.csv`

Keep the existing training flow unchanged for demo stability, and use the large-data flow for evaluation and thesis evidence.
