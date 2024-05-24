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
SCALER2 = QuantileTransformer(n_quantiles=334) # Slightly better result for regression and SVM
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
        "RandomForest":{"Metrics":  Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
                "Params": []
            },
        "LogReg":{"Metrics":  Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
                "Params": []
            },
        "SVM":{"Metrics":  Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
                "Params": []
            },
        "MLP":{"Metrics":  Counter({'precision': 0, 'recall': 0, 'f1-score': 0, 'accuracy': 0}),
                "Params": []
            },
    }

    for seed in RANDOM_STATES:

        # KNN
        knn = KNeighborsClassifier()
        parameters = {
            "classifier__n_neighbors": list(range(1, 31)),
            "classifier__metric": ['euclidean','manhattan'],
            "classifier__weights":['uniform','distance']
        }
        knn_cls = grid_search(knn, SCALER, parameters)
        knn_cls.fit(X_train, y_train)
        knn_avg, knn_cm = validate(knn_cls, X_val, y_val)
        #writer.add_scalars("knn", knn_avg)
        #writer.flush()
        dict1 = Counter(knn_avg)
        results["KNN"]["Metrics"] += dict1
        results["KNN"]["Params"].append(knn_cls.best_estimator_.get_params()['classifier'])

        # Random Forest
        random_forest = RandomForestClassifier(random_state = seed)
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        parameters = {
            
            "classifier__n_estimators": [int(x) for x in np.linspace(start = 100, stop = 1000, num = 100)],
            "classifier__max_features": ['log2', 'sqrt'],
            "classifier__max_depth" : [int(x) for x in np.linspace(10, 110, num = 11)],
            "classifier__min_samples_split": [2,5,10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__bootstrap": [True, False],
        }
        random_forest_cls = halving_random_search(random_forest, SCALER, parameters, seed)
        random_forest_cls.fit(X_train, y_train)
        random_forest_avg, random_forest_cm = validate(random_forest_cls, X_val, y_val)
        #writer.add_scalars("random_forest", random_forest_avg)
        #writer.flush()
        dict1 = Counter(random_forest_avg)
        results["RandomForest"]["Metrics"] += dict1
        results["RandomForest"]["Params"].append(random_forest_cls.best_estimator_.get_params()['classifier'])
        
        # Logistic Regression
        log_reg = LogisticRegression(max_iter=200)
        parameters = {
            "classifier__penalty": [None, "l2"]
        }
        reg_cls = grid_search(log_reg, SCALER, parameters)
        reg_cls.fit(X_train, y_train)
        log_reg_avg, log_reg_cm = validate(reg_cls, X_val, y_val)
        #writer.add_scalars("log_reg", log_reg_avg)
        #writer.flush()
        dict1 = Counter(log_reg_avg)
        results["LogReg"]["Metrics"] += dict1
        results["LogReg"]["Params"].append(reg_cls.best_estimator_.get_params()['classifier'])
        
        # SVM
        svm = SVC()
        parameters = {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__gamma": [0.01, 0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf", "sigmoid"]
        }
        svm_cls = grid_search(svm, SCALER2, parameters)
        svm_cls.fit(X_train, y_train)
        svm_avg, svm_cm = validate(svm_cls, X_val, y_val)
        #writer.add_scalars("svm", svm_avg)
        #writer.flush()
        dict1 = Counter(svm_avg)
        results["SVM"]["Metrics"] += dict1
        results["SVM"]["Params"].append(svm_cls.best_estimator_.get_params()['classifier'])
        
        # MLP
        mlp = MLPClassifier(activation='relu', #relu
                    solver='adam', 
                    max_iter=30000, #300000
                    batch_size='auto',
                    learning_rate_init=0.001,
                    # Early stopping kinda does CV too https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
                    early_stopping=True,
                    shuffle=True,
                    random_state=seed,
                    alpha=0.0001, # L2 loss strenght
                    beta_1=0.9, # 0.9 org Exponential decay rate for estimates of first moment vector in adam
                    beta_2=0.999, # 0.999 org Exponential decay rate for estimates of second moment vector in adam
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

        mlp_cls = halving_random_search(mlp, SCALER2, parameters, seed)
        mlp_cls.fit(X_train, y_train)
        mlp_avg, mlp_cm = validate(mlp_cls, X_val, y_val)
        #writer.add_scalars("mlp", mlp_avg)
        #writer.flush()
        #writer.close()
        dict1 = Counter(mlp_avg)
        results["MLP"]["Metrics"] += dict1
        results["MLP"]["Params"].append(mlp_cls.best_estimator_.get_params()['classifier'])
        
    # end for loop => calculate averages
    models = ["KNN", "RandomForest", "LogReg", "SVM", "MLP"]
    for model in models:
        print(f'{model}:')
        print(f'accuracy = {results[model]["Metrics"]["accuracy"]/ len(RANDOM_STATES)}')
        print(f'F1-Score = {results[model]["Metrics"]["f1-score"]/ len(RANDOM_STATES)}')
        print(f'Precision = {results[model]["Metrics"]["precision"]/ len(RANDOM_STATES)}')
        print(f'Recall = {results[model]["Metrics"]["recall"]/ len(RANDOM_STATES)}')
        print(f'Params = {results[model]["Params"]}')
    