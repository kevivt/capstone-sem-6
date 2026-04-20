# Disease Threshold and Risk Factor Report

Generated: 2026-04-20T04:48:16.925089+00:00

## Chronic Kidney Disease

- Model: Decision Tree Classifier
- Moderate risk threshold: 0.490566
- High risk threshold: 0.640566
- Area Under the Receiver Operating Characteristic Curve: 0.591213
- Precision: 0.092369
- Recall: 0.300864
- F1 Score: 0.141344
- Top risk factors:
  - Body Mass Index (BMI) (importance=0.66679929, relative importance=0.66679929)
  - Age (age) (importance=0.30546209, relative importance=0.30546209)
  - Biological Sex (Male) (male) (importance=0.02534341, relative importance=0.02534341)
  - diabetes (importance=0.00239522, relative importance=0.00239522)
  - Systolic Blood Pressure (sysBP) (importance=0.0, relative importance=0.0)
  - Diastolic Blood Pressure (diaBP) (importance=0.0, relative importance=0.0)
  - Glucose (glucose) (importance=0.0, relative importance=0.0)
  - prevalentHyp (importance=0.0, relative importance=0.0)
  - source (importance=0.0, relative importance=0.0)

## Hypertension

- Model: Logistic Regression
- Moderate risk threshold: 0.35
- High risk threshold: 0.9
- Area Under the Receiver Operating Characteristic Curve: 0.926376
- Precision: 0.981843
- Recall: 0.999924
- F1 Score: 0.990801
- Top risk factors:
  - Age (age) (importance=5.51074737, relative importance=0.31684528)
  - Biological Sex (Male) (male) (importance=4.73704835, relative importance=0.27236077)
  - Body Mass Index (BMI) (importance=3.33005429, relative importance=0.19146441)
  - diabetes (importance=1.81869465, relative importance=0.10456745)
  - Education Level (education) (importance=1.05203261, relative importance=0.06048754)
  - prevalentStroke (importance=0.35107168, relative importance=0.02018518)
  - Cigarettes Smoked Per Day (cigsPerDay) (importance=0.33541028, relative importance=0.01928471)
  - currentSmoker (importance=0.2574909, relative importance=0.01480467)
  - Systolic Blood Pressure (sysBP) (importance=0.0, relative importance=0.0)
  - Diastolic Blood Pressure (diaBP) (importance=0.0, relative importance=0.0)

## Type Two Diabetes Mellitus

- Model: Gaussian Naive Bayes
- Moderate risk threshold: 0.35
- High risk threshold: 0.9
- Area Under the Receiver Operating Characteristic Curve: 0.899047
- Precision: 0.960558
- Recall: 0.984529
- F1 Score: 0.972396
- Top risk factors:
  - Biological Sex (Male) (male) (importance=0.01690931, relative importance=0.43190423)
  - Age (age) (importance=0.01492758, relative importance=0.38128605)
  - Body Mass Index (BMI) (importance=0.00575852, relative importance=0.14708636)
  - prevalentHyp (importance=0.00155519, relative importance=0.03972337)
  - Systolic Blood Pressure (sysBP) (importance=0.0, relative importance=0.0)
  - Diastolic Blood Pressure (diaBP) (importance=0.0, relative importance=0.0)
  - Glucose (glucose) (importance=0.0, relative importance=0.0)
  - currentSmoker (importance=0.0, relative importance=0.0)
  - source (importance=0.0, relative importance=0.0)
