# M7016H-Project
Description...

## TODOs
- which models should we use (go from easy to complex?):
    - KNN (Britta)
    - Decision Trees => use Random Forest instead? (Britta)
    - Logistic Regression (Emil)
    - SVM (Emil)
    - Neural Networks (Emil)
- data Normalization 
    - use Standardscaler!
    - how to handle outliers? => leave them in there for now, discuss later
- how to fill in missing values? (Britta)
    - split first
    - fill in missing values with mean in the beginning for both training and validation (keep it as one set so we can do Crossvalidation?!)
    - delete rows for test set
- when to do data split => before or after handling missing values
    - https://www.kaggle.com/discussions/questions-and-answers/471740 makes a lot of sense what they say there
    - when removing data do it before split
    - when filling in data do it after the split, as otherwise test set unintentionally influences the training process