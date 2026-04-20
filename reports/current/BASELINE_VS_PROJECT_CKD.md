# CKD Baseline vs Project Model

## Headline

- Project model: DecisionTreeClassifier
- Best baseline: DecisionTreeClassifier
- Delta Macro F1 (project - baseline): 0.000000
- Delta ROC AUC (project - baseline): 0.000000
- Delta Accuracy (project - baseline): 0.000000

## Direct Comparison

| dataset | candidate | model | accuracy | macro_f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| ckd | project_final_model | DecisionTreeClassifier | 0.753035 | 0.495961 | 0.585231 |
| ckd | best_baseline | DecisionTreeClassifier | 0.753035 | 0.495961 | 0.585231 |

## Baseline Leaderboard

| model | accuracy | macro_f1 | roc_auc |
| --- | --- | --- | --- |
| DecisionTreeClassifier | 0.753035 | 0.495961 | 0.585231 |
| RandomForestClassifier | 0.728341 | 0.487703 | 0.630633 |
| KNeighborsClassifier | 0.686440 | 0.473578 | 0.629385 |
| LogisticRegression | 0.587282 | 0.441963 | 0.713646 |
| GaussianNB | 0.478775 | 0.384597 | 0.705172 |
