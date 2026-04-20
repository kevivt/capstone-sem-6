# HYPERTENSION Baseline vs Project Model

## Headline

- Project model: LogisticRegression
- Best baseline: RandomForestClassifier
- Delta Macro F1 (project - baseline): -0.059659
- Delta ROC AUC (project - baseline): 0.049793
- Delta Accuracy (project - baseline): -0.087260

## Direct Comparison

| dataset | candidate | model | accuracy | macro_f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| hypertension | project_final_model | LogisticRegression | 0.843440 | 0.540416 | 0.926424 |
| hypertension | best_baseline | RandomForestClassifier | 0.930700 | 0.600075 | 0.876631 |

## Baseline Leaderboard

| model | accuracy | macro_f1 | roc_auc |
| --- | --- | --- | --- |
| RandomForestClassifier | 0.930700 | 0.600075 | 0.876631 |
| DecisionTreeClassifier | 0.925990 | 0.580417 | 0.718202 |
| KNeighborsClassifier | 0.869780 | 0.552409 | 0.856680 |
| LogisticRegression | 0.843440 | 0.540416 | 0.926424 |
| GaussianNB | 0.768030 | 0.494776 | 0.918117 |
