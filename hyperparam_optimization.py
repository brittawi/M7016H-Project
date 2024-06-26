from descriptive_statistics import DiabetesDataBase
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from utils import grid_search, halving_random_search, validate
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import numpy as np
from sklearn.svm import SVC
import torch
from collections import Counter

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Hyperparameters
SCALER = StandardScaler()
#SCALER = RobustScaler() # No difference to standard scaler for models I test except mlp
#SCALER = QuantileTransformer(n_quantiles=334) # Slightly better result for regression and SVM
RANDOM_STATES = [4,17,77,103,176]
Random_state_data = 17
csv_path = "diabetes.csv"

if __name__ == '__main__':
    
    # get data
    ddb = DiabetesDataBase(csv_path, random_state = Random_state_data, train_split=0.8, val_split=0.1, test_split=0.1, augment=True)
    X_train, X_val, X_test, y_train, y_val, y_test = ddb.get_splits()
        
    # tensorboard
    time = datetime.now().strftime("%Y%B%d_%H_%M")
    print(time)
    log_folder = "logs/"+time
    writer = SummaryWriter(log_dir=log_folder)
    
    results = {
        "KNN": {"Metrics": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
                "Params": []
            },
        "RandomForest":{"Metrics":  {
            "0": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
            "1": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
            "2": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
            "3": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
            "4": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),},
                "Params": {
                    "0": [],
                    "1": [],
                    "2": [],
                    "3": [],
                    "4": [],
                }
            },
        "LogReg":{"Metrics":  Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
                "Params": []
            },
        "SVM":{"Metrics":  Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
                "Params": []
            },
        "MLP":{"Metrics":  {
            "0": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
            "1": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
            "2": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
            "3": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
            "4": Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),},
                "Params": {
                    "0": [],
                    "1": [],
                    "2": [],
                    "3": [],
                    "4": [],
                }
            },
    }
    
    # KNN
    print("KNN")
    knn = KNeighborsClassifier()
    parameters = {
        "classifier__n_neighbors": list(range(1, 31)),
        "classifier__metric": ['euclidean','manhattan'],
        "classifier__weights":['uniform','distance']
    }
    knn_cls = grid_search(knn, SCALER, parameters)
    knn_cls.fit(X_train, y_train)
    knn_avg, knn_cm = validate(knn_cls, X_val, y_val)
    writer.add_scalars("knn", knn_avg)
    writer.flush()
    dict1 = Counter(knn_avg)
    results["KNN"]["Metrics"] += dict1
    results["KNN"]["Params"].append(knn_cls.best_estimator_.get_params()['classifier'])
    
    # Logistic Regression
    print("Log reg")
    log_reg = LogisticRegression(max_iter=200)
    parameters = {
        "classifier__penalty": [None, "l2"]
    }
    reg_cls = grid_search(log_reg, SCALER, parameters)
    reg_cls.fit(X_train, y_train)
    log_reg_avg, log_reg_cm = validate(reg_cls, X_val, y_val)
    writer.add_scalars("log_reg", log_reg_avg)
    writer.flush()
    dict1 = Counter(log_reg_avg)
    results["LogReg"]["Metrics"] += dict1
    results["LogReg"]["Params"].append(reg_cls.best_estimator_.get_params()['classifier'])
    
    # SVM
    print("SVM")
    svm = SVC()
    parameters = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__gamma": [0.01, 0.1, 1, 10, 100],
        "classifier__kernel": ["linear", "rbf", "sigmoid"]
    }
    svm_cls = grid_search(svm, SCALER, parameters)
    svm_cls.fit(X_train, y_train)
    svm_avg, svm_cm = validate(svm_cls, X_val, y_val)
    writer.add_scalars("svm", svm_avg)
    writer.flush()
    dict1 = Counter(svm_avg)
    results["SVM"]["Metrics"] += dict1
    results["SVM"]["Params"].append(svm_cls.best_estimator_.get_params()['classifier'])
    
    # for MLP and Random Forest get 5 models from hyperparameter optimization and run these 5 times
    print("MLP")
    for i in range(5):
        # MLP
        mlp = MLPClassifier(activation='relu', #relu
                    max_iter=30000, #300000
                    batch_size='auto',
                    # Early stopping kinda does CV too https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
                    early_stopping=True,
                    shuffle=True,
                    alpha=0.0001, # L2 loss strenght
                    epsilon=1e-8 # 1e-8 org Value for numerical stability in adam.
                    )

        parameters = {
            "classifier__solver": ["adam", "sgd"],
            "classifier__activation": ["relu", "tanh", "logistic"],
            "classifier__learning_rate_init": [0.0001, 0.001, 0.01, 0.005],
            "classifier__hidden_layer_sizes": [[10,10], [100,10], [50,100,50], [100], [10]],
            "classifier__beta_1": [round(i*0.001, 3) for i in range(90, 95)],
            "classifier__beta_2": [round(i*0.001, 4) for i in range(985, 999, 3)]
        }

        mlp_cls = halving_random_search(mlp, SCALER, parameters)
        mlp_cls.fit(X_train, y_train)

        for j, seed in enumerate(RANDOM_STATES):
            print("MLP:", i, j)
            # get params
            params = {}
            for key in mlp_cls.best_params_:
                newKey = key.replace("classifier__","")
                params[newKey] = mlp_cls.best_params_[key]
            mlp2 = MLPClassifier(activation='relu', #relu
                    max_iter=30000, #300000
                    batch_size='auto',
                    # Early stopping kinda does CV too https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
                    early_stopping=True,
                    shuffle=True,
                    alpha=0.0001, # L2 loss strenght
                    epsilon=1e-8, # 1e-8 org Value for numerical stability in adam.
                    random_state = seed
                    )
            mlp2.set_params(**params)
            pipe = Pipeline(steps=[
                ("scaler", SCALER), 
                ("classifier", mlp2)
            ])
            pipe.fit(X_train, y_train)
            mlp_avg, mlp_cm = validate(pipe, X_val, y_val)
            writer.add_scalars("mlp", mlp_avg)
            writer.flush()
            dict1 = Counter(mlp_avg)
            results["MLP"]["Metrics"][str(i)] += dict1
            if j == 0:
                results["MLP"]["Params"][str(i)].append(mlp_cls.best_estimator_.get_params()['classifier'])
    
    # Random Forest
    print("Random Forest")
    for i in range(5):
        random_forest = RandomForestClassifier()
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        parameters = {
            "classifier__n_estimators": [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)],
            "classifier__max_features": ['log2', 'sqrt'],
            "classifier__max_depth" : [int(x) for x in np.linspace(10, 110, num = 11)],
            "classifier__min_samples_split": [2,5,10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__bootstrap": [True, False],
        }
        random_forest_cls = halving_random_search(random_forest, SCALER, parameters)
        random_forest_cls.fit(X_train, y_train)
        
        for j, seed in enumerate(RANDOM_STATES):
            print("Random Forest", i, j)
            # get params
            params = {}
            for key in random_forest_cls.best_params_:
                newKey = key.replace("classifier__","")
                params[newKey] = random_forest_cls.best_params_[key]
            print(params)
            random_forest2 = RandomForestClassifier(random_state=seed)  
            random_forest2.set_params(**params)
            pipe = Pipeline(steps=[
                ("scaler", SCALER), 
                ("classifier", random_forest2)
            ])
            pipe.fit(X_train, y_train)
        
            random_forest_avg, random_forest_cm = validate(pipe, X_val, y_val)
            writer.add_scalars("random_forest", random_forest_avg)
            writer.flush()
            dict1 = Counter(random_forest_avg)
            results["RandomForest"]["Metrics"][str(i)]  += dict1
            if j == 0:
                results["RandomForest"]["Params"][str(i)] .append(random_forest_cls.best_estimator_.get_params()['classifier'])
        
    writer.close()
    # end for loop => calculate averages
    models = ["KNN", "LogReg", "SVM"]
    for model in models:
        print(f'{model}:')
        print(f'accuracy = {results[model]["Metrics"]["accuracy"]}')
        print(f'F1-Score = {results[model]["Metrics"]["f1-score"]}')
        print(f'Precision = {results[model]["Metrics"]["precision"]}')
        print(f'Recall = {results[model]["Metrics"]["recall"]}')
        print(f'Params = {results[model]["Params"]}')
        print()
        
    for model in ["RandomForest", "MLP"]:
        for i in range(5):
            print(f'{model}:')
            print(f'accuracy = {results[model]["Metrics"][str(i)]["accuracy"]/ len(RANDOM_STATES)}')
            print(f'F1-Score = {results[model]["Metrics"][str(i)]["f1-score"]/ len(RANDOM_STATES)}')
            print(f'Precision = {results[model]["Metrics"][str(i)]["precision"]/ len(RANDOM_STATES)}')
            print(f'Recall = {results[model]["Metrics"][str(i)]["recall"]/ len(RANDOM_STATES)}')
            print(f'Params = {results[model]["Params"][str(i)]}')
            print()
    
