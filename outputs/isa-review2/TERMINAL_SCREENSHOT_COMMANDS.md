# Terminal Commands For Review Screenshots

Run these from project root:

```powershell
Set-Location "D:\sem 6\capstone\Sem_6_Capstone"
```

## 1. Environment Check

```powershell
& ".\.venv\Scripts\python.exe" -V
& ".\.venv\Scripts\python.exe" -c "import pandas, numpy, sklearn; print('pandas', pandas.__version__); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__)"
```

## 2. Full-Train Project Models

```powershell
& ".\.venv\Scripts\python.exe" scripts\train_models.py --data-source preprocessed
& ".\.venv\Scripts\python.exe" scripts\build_risk_profiles.py
```

## 3. Baseline Comparison Graph Inputs

```powershell
& ".\.venv\Scripts\python.exe" scripts\benchmark_baseline_models.py
& ".\.venv\Scripts\python.exe" scripts\generate_model_comparison_charts.py
```

## 4. Terminal Prediction Demo

```powershell
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --disease ckd --show-template
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --input demo_inputs\terminal_samples\ckd_sample_1.json
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --input demo_inputs\terminal_samples\hypertension_sample_1.json
& ".\.venv\Scripts\python.exe" scripts\predict_from_terminal.py --input demo_inputs\terminal_samples\diabetes_sample_1.json
```

## 5. Dataset Rows, Columns, And Train-Row Proof

```powershell
& ".\.venv\Scripts\python.exe" -c "from pathlib import Path; import pandas as pd; files=['ckd_large.csv','hypertension_large_500k.csv','diabetes_large_500k.csv']; base=Path('medical datasets/large'); train=Path('preprocessed_outputs'); train_files={'ckd_large.csv':'ckd_train_smote.csv','hypertension_large_500k.csv':'hypertension_train_smote.csv','diabetes_large_500k.csv':'diabetes_train_smote.csv'}; [print(f'\n{f}: {pd.read_csv(base/f).shape[0]} rows, {pd.read_csv(base/f).shape[1]} columns\nColumns: {', '.join(pd.read_csv(base/f, nrows=1).columns)}\nTrain rows used: {sum(1 for _ in open(train/train_files[f], encoding=\"utf-8\")) - 1}') for f in files]"
```

## 6. Backend Smoke Test

```powershell
& ".\.venv\Scripts\python.exe" scripts\backend_smoke_test.py
```

## Notes
- Project-final artifacts were trained on the full prepared train splits.
- The baseline comparison report still uses the built-in `120000` cap for runtime control.
