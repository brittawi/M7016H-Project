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

## TODOs
- which models should we use (go from easy to complex?):
    - KNN (Britta)
    - Decision Trees => use Random Forest instead? (Britta)
    - Logistic Regression (Emil)
    - SVM (Emil)
    - Neural Networks (Emil)
- data Normalization 
    - use Standardscaler!
    - Maybe try some that handles outliers better? (RobustScaler or QuantileTransformer)
    - https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section
    - how to handle outliers? => leave them in there for now, discuss later
- how to fill in missing values? (Britta)
    - split first
    - fill in missing values with mean in the beginning for both training and validation (keep it as one set so we can do Crossvalidation?!)
    - delete rows for test set
- when to do data split => before or after handling missing values
    - https://www.kaggle.com/discussions/questions-and-answers/471740 makes a lot of sense what they say there
    - when removing data do it before split
    - when filling in data do it after the split, as otherwise test set unintentionally influences the training process
