"""
This script runs SVClassifier tunning for DS-PunoPeru

@author: Renato Quiliche
"""

# %%
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

# %%

# Importando los paquetes

# Method HPO
from sklearn.model_selection import RandomizedSearchCV

# Supervised learning algoritm
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, f1_score, roc_curve

# Cross-Validation method
from sklearn.model_selection import RepeatedStratifiedKFold

# I define here the NPV metric
def neg_pred_value(y_true,y_predicted):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_predicted)
    if (cm[1][0]+cm[0][0])==0:
        neg_pred_value=0
    else:
        neg_pred_value = cm[0][0]/(cm[1][0]+cm[0][0])
    return neg_pred_value

# %% HPO Function
from scipy.stats import uniform, randint

def SVCexperiments(K_folds, Repeats):
    # Cross-validation method
    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=Repeats, random_state=0)
    
    # Hyperparameter grid for ENLR
    param_grid = { 
        "C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
        "kernel" : ("poly", "rbf", "sigmoid"),
        'gamma': 10**np.arange(-5,5,1, dtype="float")
    }
    # randint.rvs(100,150, size=200)
    # Linear plantearlo como ElasticNet (no se puede, o es l1 o l2)
    # Hacer un RS para poly, tiene que variar el parametro degree
    # El poly acepta polinomios de tercer grado en las X
    
    # I define the model here
    SVClf = SVC()
    
    # Metrics
    scoring = {"AUC": "roc_auc"
               , "Accuracy": make_scorer(accuracy_score)
               , "F1-Score": "f1"
               , "MCC": make_scorer(matthews_corrcoef)
               , "NPV": make_scorer(neg_pred_value)}
    
    #Test CV
    search_ddnn = RandomizedSearchCV(random_state=0, estimator=SVClf, param_distributions=param_grid
                                     , scoring=scoring, cv=cv, n_jobs=-1, refit=False, verbose=4, n_iter=200)
    results = search_ddnn.fit(x, Y)
    return results

# %%

res = SVCexperiments(2, 2)
# %%
pd.DataFrame(res.cv_results_).to_csv("Resultados/SVC/first.csv")

