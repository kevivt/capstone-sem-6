# Baseline Model Comparison

This report compares baseline classifiers on the prepared disease datasets.

Training row cap per dataset: 120000

## Best Model Per Dataset (by Macro F1)

| dataset | model_key | model | train_rows | test_rows | accuracy | macro_f1 | roc_auc | is_project_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ckd | decision_tree | DecisionTreeClassifier | 120000 | 91118 | 0.753035 | 0.495961 | 0.585231 | False |
| diabetes | random_forest | RandomForestClassifier | 120000 | 100000 | 0.831510 | 0.605909 | 0.852946 | False |
| hypertension | random_forest | RandomForestClassifier | 120000 | 100000 | 0.930700 | 0.600075 | 0.876631 | False |

## Project Model vs Best Baseline (Per Disease)

| dataset | project_model | best_baseline | project_macro_f1 | baseline_macro_f1 | delta_macro_f1 | project_roc_auc | baseline_roc_auc | delta_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ckd | DecisionTreeClassifier | DecisionTreeClassifier | 0.495961 | 0.495961 | 0.000000 | 0.585231 | 0.585231 | 0.000000 |
| hypertension | LogisticRegression | RandomForestClassifier | 0.540416 | 0.600075 | -0.059659 | 0.926424 | 0.876631 | 0.049793 |
| diabetes | GaussianNB | RandomForestClassifier | 0.522536 | 0.605909 | -0.083373 | 0.898276 | 0.852946 | 0.045330 |

## Project Model Rows

| dataset | model | accuracy | macro_f1 | roc_auc |
| --- | --- | --- | --- | --- |
| ckd | DecisionTreeClassifier | 0.753035 | 0.495961 | 0.585231 |
| diabetes | GaussianNB | 0.684320 | 0.522536 | 0.898276 |
| hypertension | LogisticRegression | 0.843440 | 0.540416 | 0.926424 |

## Full Results

| dataset | model_key | model | train_rows | test_rows | accuracy | macro_f1 | roc_auc | is_project_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ckd | decision_tree | DecisionTreeClassifier | 120000 | 91118 | 0.753035 | 0.495961 | 0.585231 | False |
| ckd | project_final_model | DecisionTreeClassifier | 120000 | 91118 | 0.753035 | 0.495961 | 0.585231 | True |
| ckd | random_forest | RandomForestClassifier | 120000 | 91118 | 0.728341 | 0.487703 | 0.630633 | False |
| ckd | knn_k5 | KNeighborsClassifier | 120000 | 91118 | 0.686440 | 0.473578 | 0.629385 | False |
| ckd | logistic_regression | LogisticRegression | 120000 | 91118 | 0.587282 | 0.441963 | 0.713646 | False |
| ckd | gaussian_nb | GaussianNB | 120000 | 91118 | 0.478775 | 0.384597 | 0.705172 | False |
| diabetes | random_forest | RandomForestClassifier | 120000 | 100000 | 0.831510 | 0.605909 | 0.852946 | False |
| diabetes | decision_tree | DecisionTreeClassifier | 120000 | 100000 | 0.836360 | 0.603600 | 0.747341 | False |
| diabetes | logistic_regression | LogisticRegression | 120000 | 100000 | 0.795050 | 0.596841 | 0.901281 | False |
| diabetes | knn_k5 | KNeighborsClassifier | 120000 | 100000 | 0.799760 | 0.591255 | 0.851700 | False |
| diabetes | gaussian_nb | GaussianNB | 120000 | 100000 | 0.684320 | 0.522536 | 0.898276 | False |
| diabetes | project_final_model | GaussianNB | 120000 | 100000 | 0.684320 | 0.522536 | 0.898276 | True |
| hypertension | random_forest | RandomForestClassifier | 120000 | 100000 | 0.930700 | 0.600075 | 0.876631 | False |
| hypertension | decision_tree | DecisionTreeClassifier | 120000 | 100000 | 0.925990 | 0.580417 | 0.718202 | False |
| hypertension | knn_k5 | KNeighborsClassifier | 120000 | 100000 | 0.869780 | 0.552409 | 0.856680 | False |
| hypertension | logistic_regression | LogisticRegression | 120000 | 100000 | 0.843440 | 0.540416 | 0.926424 | False |
| hypertension | project_final_model | LogisticRegression | 120000 | 100000 | 0.843440 | 0.540416 | 0.926424 | True |
| hypertension | gaussian_nb | GaussianNB | 120000 | 100000 | 0.768030 | 0.494776 | 0.918117 | False |
