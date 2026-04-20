# Terminal Demo Samples

This guide reflects the current full-feature serving models.

## 1. Open the Project

```powershell
Set-Location "d:/sem 6/capstone/Sem_6_Capstone"
& "../.venv/Scripts/Activate.ps1"
```

## 2. Show the Required Columns First

```powershell
python scripts/predict_from_terminal.py --disease ckd --show-template
python scripts/predict_from_terminal.py --disease hypertension --show-template
python scripts/predict_from_terminal.py --disease diabetes --show-template
```

## 3. Interactive Entry

Use this when you want to type every feature manually:

```powershell
python scripts/predict_from_terminal.py --disease ckd --interactive
python scripts/predict_from_terminal.py --disease hypertension --interactive
python scripts/predict_from_terminal.py --disease diabetes --interactive
```

## 4. JSON Demo Files

These sample files still match the current trained artifacts:

```powershell
python scripts/predict_from_terminal.py --input demo_inputs/terminal_samples/ckd_sample_1.json
python scripts/predict_from_terminal.py --input demo_inputs/terminal_samples/hypertension_sample_1.json
python scripts/predict_from_terminal.py --input demo_inputs/terminal_samples/diabetes_sample_1.json
```

To run all demo files:

```powershell
python scripts/run_demo_samples.py
```

## 5. Important Note

- Inputs are still model-space values, not patient-friendly raw values.
- This is intentional for now because the active models once again use full disease-specific column sets.
- The next implementation step is a proper raw-user-input layer that maps normal patient measurements into these feature spaces.
