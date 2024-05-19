from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def grid_search(model, standard_scaler, params):
    pipe = Pipeline(steps=[
        ("scaler", standard_scaler), 
        ("classifier", model)
    ])
    cls = GridSearchCV(pipe, params)
    return cls

def halving_random_search(model, standard_scaler, params):
    pipe = Pipeline(steps=[
        ("scaler", standard_scaler), 
        ("classifier", model)
    ])
    cls = HalvingRandomSearchCV(pipe, params)
    return cls

def validate(cls, x_val, y_val):
    predictions = cls.predict(x_val)
    perfromance = classification_report(y_val, predictions)
    print(perfromance)
    cm = confusion_matrix(y_val, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[True,False])
    disp.plot()