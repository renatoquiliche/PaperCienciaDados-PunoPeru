"""
This script runs RFC tunning for DS-PunoPeru

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

# %%
from scipy.stats import uniform, randint

def RFCexperiments(K_folds, Repeats):
    # Cross-validation method
    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=Repeats, random_state=0)
    
    # Hyperparameter grid for ENLR
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': randint(2,10),
                  'max_samples': [0.2, 0.5, 0.8],
                  'min_samples_split': uniform(0,0.6),
                  'min_samples_leaf': uniform(0, 0.4),
                  'n_estimators': randint(10, 300),
                  'max_features': ['log2', 'sqrt']}
    
    # I define the model here
    RFC = RandomForestClassifier(random_state=0, n_jobs=-1, bootstrap=True)
    scoring = {"AUC": "roc_auc"
               , "Accuracy": make_scorer(accuracy_score)
               , "F1-Score": "f1"
               , "MCC": make_scorer(matthews_corrcoef)
               , "NPV": make_scorer(neg_pred_value)}
    
    #Test CV
    search_ddnn = RandomizedSearchCV(random_state=0, estimator = RFC, param_distributions=param_grid
                                     , scoring=scoring, cv=cv, n_jobs=-1, refit="MCC", verbose=4, n_iter=400)
    results = search_ddnn.fit(x, Y)
    return results

# %%
# Loop guardar resultados de experimentos en CSV

#Repeats = [5, 4, 3, 2]
exec(f'RFCresults_{2} = RFCexperiments(5, 2)')
pd.DataFrame(RFCresults_2.cv_results_).to_csv("Resultados/RFC/scratch.csv")

# %%

import time

RFCsummary = {"param_criterion": [], "param_max_depth": [], "param_max_features": [], "param_min_samples_leaf": [],
           "param_min_samples_split": [], "param_n_estimators": [], "K_fold": []}

for i in range(2,6):
    start = time.time()
    print("Configuration")
    print("K_folds =", i, "------- Repeats =", Repeats[i-2]*i)
    exec(f'RFCresults_{i} = RFCexperiments({i}, Repeats[{(i-2)}])')
    #print("Elapsed Time ", (time.time() - start))
    exec(f'print("Optimal Parameters :", RFCresults_{i}.best_params_)')
    exec(f'pd.DataFrame(RFCresults_{i}.cv_results_).to_csv("Resultados/RFC/RFC_results_{i}.csv")')
    #Construimos resumen de resultados
    exec(f'RFCsummary["param_criterion"].append(RFCresults_{i}.best_params_["criterion"])')
    exec(f'RFCsummary["param_max_depth"].append(RFCresults_{i}.best_params_["max_depth"])')
    exec(f'RFCsummary["param_max_features"].append(RFCresults_{i}.best_params_["max_features"])')
    exec(f'RFCsummary["param_min_samples_leaf"].append(RFCresults_{i}.best_params_["min_samples_leaf"])')
    exec(f'RFCsummary["param_min_samples_split"].append(RFCresults_{i}.best_params_["min_samples_split"])')
    exec(f'RFCsummary["param_n_estimators"].append(RFCresults_{i}.best_params_["n_estimators"])')
    exec(f'RFCsummary["K_fold"].append({i})')
    Total_time = (time.time() - start)
    print("Iteration time: ", Total_time)

pd.DataFrame(RFCsummary).to_excel("Resultados/RFC/summary.xlsx")

# %%
for i in range(2,6):
    exec(f'RFC2_res = pd.read_csv("Resultados/RFC/RFC_results_{i}.csv")')
    #RFC2_res["Rank"] = RFC2_res["rank_test_MCC"]+RFC2_res["rank_test_NPV"]
    RFC2_res = RFC2_res.sort_values(by=["rank_test_MCC"]).iloc[0:30,:]
    RFC2_res = RFC2_res.loc[RFC2_res["rank_test_NPV"]==1]
    print(RFC2_res.iloc[0,5:10])

# %%
