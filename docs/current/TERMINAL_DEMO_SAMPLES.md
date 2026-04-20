# Terminal Demo Samples

This guide reflects the current serving models and both terminal modes:

- model-ready feature input mode
- patient-friendly raw-input mode

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

## 5. Raw Input Mode (New)

Show raw templates:

```powershell
python scripts/predict_from_terminal.py --disease ckd --show-raw-template
python scripts/predict_from_terminal.py --disease hypertension --show-raw-template
python scripts/predict_from_terminal.py --disease diabetes --show-raw-template
```

Run raw sample files:

```powershell
python scripts/predict_from_terminal.py --raw-input demo_inputs/raw_samples/ckd_raw_sample_1.json
python scripts/predict_from_terminal.py --raw-input demo_inputs/raw_samples/hypertension_raw_sample_1.json
python scripts/predict_from_terminal.py --raw-input demo_inputs/raw_samples/diabetes_raw_sample_1.json
```

Explanation-first raw demo:

```powershell
python scripts/predict_from_terminal.py --raw-input demo_inputs/raw_samples/ckd_explain_raw_sample_1.json --explain-raw
python scripts/predict_from_terminal.py --raw-input demo_inputs/raw_samples/hypertension_explain_raw_sample_1.json --explain-raw
python scripts/predict_from_terminal.py --raw-input demo_inputs/raw_samples/diabetes_explain_raw_sample_1.json --explain-raw
```

Guided raw entry:

```powershell
python scripts/predict_from_terminal.py --guided-raw
```

## 6. Important Note

- The raw-input flow maps patient-friendly values to deployed feature schema order.
- Some deployed artifacts still rely on reduced survey-style signals; warnings are returned when specific fields are forced/defaulted for compatibility.
- For backend + plan sensitivity validation in one command:

```powershell
python scripts/validate_raw_flow.py
```
