from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def grid_search(model, standard_scaler, params):
    pipe = Pipeline(steps=[
        ("scaler", standard_scaler), 
        ("classifier", model)
    ])
    cls = GridSearchCV(pipe, params, cv=10)
    return cls

def halving_random_search(model, standard_scaler, params, random_state=42):
    pipe = Pipeline(steps=[
        ("scaler", standard_scaler), 
        ("classifier", model)
    ])
    # TODO
    cls = HalvingRandomSearchCV(pipe, params, random_state=random_state)
    return cls

def validate(cls, x_val, y_val):
    predictions = cls.predict(x_val)
    performance_str = classification_report(y_val, predictions, output_dict=False)
    print(performance_str)
    performance_dict = classification_report(y_val, predictions, output_dict=True)
    exclude_keys = ['support']
    macro_avg_wo_support = {k: performance_dict["macro avg"][k] for k in set(list(performance_dict["macro avg"].keys())) - set(exclude_keys)}
    accuracy = performance_dict["accuracy"]
    macro_avg_wo_support["accuracy"] = accuracy
    cm = confusion_matrix(y_val, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[True,False])
    disp.plot()
    return macro_avg_wo_support, cm