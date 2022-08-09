"""
This script runs RFC tunning for DS-PunoPeru

@author: Renato Quiliche
"""

import pandas as pd
import numpy as np

data_puno = pd.read_csv("Databases/peru_2019.csv")

# Set the random seed for reproductible results
np.random.seed(0)

Y = data_puno["disasters_risk"]
x = data_puno.iloc[:,6:]

# Feature MinMaxScaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x[["altura", "gpc"]] = scaler.fit_transform(x[["altura", "gpc"]])

# Init the grid search cross-validation on ENLR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, f1_score, roc_curve
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# I define here the NPV metric
def neg_pred_value(y_true,y_predicted):
    cm = confusion_matrix(y_true, y_predicted)
    if (cm[1][0]+cm[0][0])==0:
        neg_pred_value=0
    else:
        neg_pred_value = cm[0][0]/(cm[1][0]+cm[0][0])
    return neg_pred_value

def RFCexperiments(K_folds, Repeats):
    # Cross-validation method
    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=Repeats, random_state=0)
    
    # Hyperparameter grid for ENLR
    param_grid = {'criterion': ['gini'],
                  'max_depth': [2, 3, 4, 5, None],
                  'min_samples_leaf': np.arange(0,1.1,0.1),
                  'min_samples_split': np.arange(0,1.1,0.1),
                  'n_estimators': [10, 100, 200, 500],
                  'max_features': ['log2']}
    
    # I define the model here
    RFC = RandomForestClassifier(random_state=0, n_jobs=-1, bootstrap=True)
    scoring = {"AUC": "roc_auc"
               , "Accuracy": make_scorer(accuracy_score)
               , "F1-Score": "f1"
               , "MCC": make_scorer(matthews_corrcoef)
               , "NPV": make_scorer(neg_pred_value)}
    
    #Test CV
    search_ddnn = RandomizedSearchCV(random_state=0, estimator = RFC, param_distributions=param_grid
                                     , scoring=scoring, cv=cv, n_jobs=-1, refit="MCC", verbose=0, n_iter=200)
    results = search_ddnn.fit(x, Y)
    return results

# Loop guardar resultados de experimentos en CSV
Repeats = [15, 10, 8, 6]

import time

summary = {"param_criterion": [], "param_max_depth": [], "param_max_features": [], "param_min_samples_leaf": [],
           "param_min_samples_split": [], "param_n_estimators": [], "K_fold": []}

for i in range(2,6):
    start = time.time()
    print("Configuration")
    print("K_folds =", i, "------- Repeats =", Repeats[i-2]*i)
    exec(f'RFCResults_{i} = RFCexperiments({i}, Repeats[{(i-2)}])')
    #print("Elapsed Time ", (time.time() - start))
    exec(f'print("Optimal Parameters :", RFCResults_{i}.best_params_)')
    exec(f'pd.DataFrame(RFCResults_{i}.cv_results_).to_csv("Resultados/RFC/RFC_results_{i}.csv")')
    #Construimos resumen de resultados
    exec(f'summary["C"].append(RFCResults_{i}.best_params_["C"])')
    exec(f'summary["l1_ratio"].append(RFCResults_{i}.best_params_["l1_ratio"])')
    exec(f'summary["K_fold"].append({i})')
    Total_time = (time.time() - start)
    print("Iteration time: ", Total_time)

pd.DataFrame(summary).to_excel("Resultados/RFC/summary.xlsx")
