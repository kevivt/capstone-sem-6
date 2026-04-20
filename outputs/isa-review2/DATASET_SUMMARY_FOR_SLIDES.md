# Dataset Summary For Slides

Date: 2026-04-19

## Active Serving / Training Datasets

### CKD
- File: `medical datasets/large/ckd_large.csv`
- Rows: `455,590`
- Columns: `10`
- Final-model training rows used: `690,956`
- Target column: `ckd_label`
- Final model: `DecisionTreeClassifier`
- Column names:
`age, male, BMI, sysBP, diaBP, glucose, diabetes, prevalentHyp, source, ckd_label`

### Hypertension
- File: `medical datasets/large/hypertension_large_500k.csv`
- Rows: `500,000`
- Columns: `13`
- Final-model training rows used: `785,242`
- Target column: `prevalentHyp`
- Final model: `LogisticRegression`
- Column names:
`male, age, education, currentSmoker, cigsPerDay, prevalentStroke, BMI, sysBP, diaBP, glucose, diabetes, source, prevalentHyp`

### Diabetes
- File: `medical datasets/large/diabetes_large_500k.csv`
- Rows: `500,000`
- Columns: `10`
- Final-model training rows used: `754,970`
- Target column: `diabetes`
- Final model: `GaussianNB`
- Column names:
`age, male, BMI, sysBP, diaBP, glucose, prevalentHyp, currentSmoker, source, diabetes`

## Source Sites Used Across The Project Amalgamation
- UCI Machine Learning Repository
- Framingham Heart Study
- Pima Indians Diabetes Dataset
- CDC BRFSS Annual Data
- CDC NHANES 2017-2018
- ICMR-NIN IFCT 2017
- Mendeley Data
- Open Food Facts India
- Kaggle

## Benchmark Note
- Project-final artifacts were trained on the full available large train splits shown above.
- Baseline comparison for slides/report remains capped at `120,000` training rows per disease for runtime control.
