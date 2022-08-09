"""
This script runs ENLR tunning for DS-PunoPeru

@author: Renato Quiliche
"""

import pandas as pd
import numpy as np
import os
os.chdir('d:\\PaperCienciaDados-PunoPeru')

data_puno = pd.read_csv("Databases/peru_2019.csv")

# Set the random seed for reproductible results
np.random.seed(0)

Y = data_puno["disasters_risk"]
x = data_puno.iloc[:,6:]

# Feature MinMaxScaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x[["altura", "gpc"]] = scaler.fit_transform(x[["altura", "gpc"]])

# Get features format
x.info()

# Get balancing format
from collections import Counter

print(f"Original class counts: {Counter(Y)}")
print("Classes are balanced")

# Init the grid search cross-validation on ENLR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, f1_score, roc_curve
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# I define here the NPV metric
def neg_pred_value(y_true,y_predicted):
    cm = confusion_matrix(y_true, y_predicted)
    if (cm[1][0]+cm[0][0])==0:
        neg_pred_value=0
    else:
        neg_pred_value = cm[0][0]/(cm[1][0]+cm[0][0])
    return neg_pred_value

"""
Elastic net logistic regression
"""

def ENLRexperiments(K_folds, Repeats):
    # Cross-validation method
    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=Repeats, random_state=0)
    
    # Hyperparameter grid for ENLR
    param_grid = { 'C': [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
                  'l1_ratio': np.round_(np.arange(0.0,1.1,0.1), decimals=1)}
    
    # I define the model here
    ENLR = LogisticRegression(random_state=0, penalty='elasticnet', solver='saga', max_iter=4000, n_jobs=-1)
    scoring = {"AUC": "roc_auc"
               , "Accuracy": make_scorer(accuracy_score)
               , "F1-Score": "f1"
               , "MCC": make_scorer(matthews_corrcoef)
               , "NPV": make_scorer(neg_pred_value)}
    
    #Test CV
    search_ddnn= GridSearchCV(estimator = ENLR, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1, refit="MCC", verbose=0)
    results = search_ddnn.fit(x, Y)
    return results

# Loop guardar resultados de experimentos en CSV
Repeats = [15, 10, 8, 6]

import time

summary = {"C": [], "l1_ratio": [], "K_fold": []}

for i in range(2,6):
    start = time.time()
    print("Configuration")
    print("K_folds =", i, "------- Repeats =", Repeats[i-2]*i)
    exec(f'ENLRresults_{i} = ENLRexperiments({i}, Repeats[{(i-2)}])')
    #print("Elapsed Time ", (time.time() - start))
    exec(f'print("Optimal Parameters :", ENLRresults_{i}.best_params_)')
    exec(f'pd.DataFrame(ENLRresults_{i}.cv_results_).to_csv("Resultados/ENLR/ENLR_results_{i}.csv")')
    #Construimos resumen de resultados
    exec(f'summary["C"].append(ENLRresults_{i}.best_params_["C"])')
    exec(f'summary["l1_ratio"].append(ENLRresults_{i}.best_params_["l1_ratio"])')
    exec(f'summary["K_fold"].append({i})')
    Total_time = (time.time() - start)
    print("Iteration time: ", Total_time)

pd.DataFrame(summary).to_excel("Resultados/ENLR/summary.xlsx")

