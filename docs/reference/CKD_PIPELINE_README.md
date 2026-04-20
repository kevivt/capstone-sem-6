# CKD Preprocessing and Modeling Pipeline

This project now includes a CKD-first preprocessing and baseline modeling script:

- File: `ckd_pipeline.py`
- Dependencies: `requirements-ckd.txt`

## What It Does

1. Loads CKD data from:
   - `--input` path if provided, or
   - auto-search in local CKD dataset folders, or
   - fallback to UCI via `ucimlrepo` (dataset id 336).
2. Cleans missing markers like `?` and `\t?`.
3. Drops `id` if present.
4. Encodes target class (`ckd` = 1, `notckd` = 0).
5. Infers numeric vs categorical features.
6. Applies preprocessing:
   - Numeric: `KNNImputer` + `MinMaxScaler`
   - Categorical: most-frequent imputation + one-hot encoding
7. Applies `SMOTE` on training data only.
8. Trains and evaluates baseline models with your requested CKD hyperparameters:
   - SVM: `C=0.241`, `kernel=linear`
   - KNN: `n_neighbors=1`, `weights=distance`, `algorithm=kd_tree`
   - XGBoost: `learning_rate=0.1`, `n_estimators=1000`, `max_depth=5`, `min_child_weight=6`, `reg_alpha=60.0`

## Run Steps

From `Sem_6_Capstone` folder:

```powershell
pip install -r requirements-ckd.txt
python ckd_pipeline.py
```

If you extract the local archive and want to point to a specific file:

```powershell
python ckd_pipeline.py --input "medical datasets/raw/chronic+kidney+disease/chronic_kidney_disease.csv"
```

## Notes

- Your CKD folder currently contains a `.rar` archive. If no local `.csv` or `.arff` is found, the script automatically downloads from UCI.
- If `xgboost` is unavailable, the script still runs SVM and KNN.
