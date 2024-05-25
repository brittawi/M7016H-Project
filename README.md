# M7016H-Project

## Overview
With this project we aim to detect diabetes based on the following characteristics:
| Name            | Description |
| --------------- | ----------- |
| Pregnancies     | Number of times pregnant |
| Glucose         | Plasma glucose concentration is an oral glucose tolerance test |
| Blood Pressure  | Diastolic blood pressure (mm Hg) |
| Skin Thickness  | Triceps skin fold thickness (mm) |
| Insulin         | 2-Hour serum insulin (mu U/ml) |
| BMI             | Body mass index |
| Diabetes Pedigree Function | Diabetes pedigree function |
| Age             | Age (years) |
(Akmese, 2022)
We decided to compare the 5 models: KNN, Random Forest, SVM, Logistic Regression and MLP and choose the best one on the validation accuracy.

## Files
- Code related to preprocessing the data can be found in the file: `descriptove_statistics.py`
- The hyperparamter optimization and model comparison is done in: `hyperparameter_optimization.py`
- The testing of the final two models is done in `test.ipynb`
- Utility functions that have been used are in `utils.py`
