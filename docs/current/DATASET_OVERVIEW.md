# Dataset Overview: Row Counts and Column Names

---

## CKD (Chronic Kidney Disease) Dataset

**File:** `preprocessed_outputs/ckd_preprocessed.csv`

**Number of Rows:** 455,590

**Number of Columns:** 10

**Column Names:**
1. age
2. male
3. BMI
4. sysBP
5. diaBP
6. glucose
7. diabetes
8. prevalentHyp
9. source
10. ckd_label

---

## Hypertension Dataset

**File:** `preprocessed_outputs/hypertension_preprocessed.csv`

**Number of Rows:** 500,000

**Number of Columns:** 13

**Column Names:**
1. male
2. age
3. education
4. currentSmoker
5. cigsPerDay
6. prevalentStroke
7. BMI
8. sysBP
9. diaBP
10. glucose
11. diabetes
12. source
13. prevalentHyp

---

## Diabetes Dataset

**File:** `preprocessed_outputs/diabetes_preprocessed.csv`

**Number of Rows:** 500,000

**Number of Columns:** 10

**Column Names:**
1. age
2. male
3. BMI
4. sysBP
5. diaBP
6. glucose
7. prevalentHyp
8. currentSmoker
9. source
10. diabetes

---

## Summary Table

| Dataset | File | Rows | Columns |
|---------|------|------|---------|
| CKD | ckd_preprocessed.csv | 455,590 | 10 |
| Hypertension | hypertension_preprocessed.csv | 500,000 | 13 |
| Diabetes | diabetes_preprocessed.csv | 500,000 | 10 |
| **Total** | **3 datasets** | **1,455,590** | **Varies** |

---

## Key Observations

- **CKD dataset** has the fewest rows (455,590) but sufficient for model training
- **Hypertension and Diabetes datasets** are larger (500,000 each) for better representation
- **Common features** across all datasets: age, male, BMI, glucose, diabetes, prevalentHyp
- **Disease-specific features:**
  - CKD: Includes clinical markers (systolic BP, diastolic BP)
  - Hypertension: Includes lifestyle factors (education, smoking, stroke history)
  - Diabetes: Includes cardiovascular risk factors (smoking, hypertension history)

---
