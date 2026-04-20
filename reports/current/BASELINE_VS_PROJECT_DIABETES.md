# DIABETES Baseline vs Project Model

## Headline

- Project model: GaussianNB
- Best baseline: RandomForestClassifier
- Delta Macro F1 (project - baseline): -0.083373
- Delta ROC AUC (project - baseline): 0.045330
- Delta Accuracy (project - baseline): -0.147190

## Direct Comparison

| dataset | candidate | model | accuracy | macro_f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| diabetes | project_final_model | GaussianNB | 0.684320 | 0.522536 | 0.898276 |
| diabetes | best_baseline | RandomForestClassifier | 0.831510 | 0.605909 | 0.852946 |

## Baseline Leaderboard

| model | accuracy | macro_f1 | roc_auc |
| --- | --- | --- | --- |
| RandomForestClassifier | 0.831510 | 0.605909 | 0.852946 |
| DecisionTreeClassifier | 0.836360 | 0.603600 | 0.747341 |
| LogisticRegression | 0.795050 | 0.596841 | 0.901281 |
| KNeighborsClassifier | 0.799760 | 0.591255 | 0.851700 |
| GaussianNB | 0.684320 | 0.522536 | 0.898276 |
