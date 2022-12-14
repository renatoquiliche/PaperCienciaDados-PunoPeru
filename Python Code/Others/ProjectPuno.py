# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 19:58:47 2022

@author: Renato Quiliche
"""
import pandas as pd
import numpy as np
import os

os.chdir("G:\My Drive\Paper MachineLearning FB\Paper MachineLearning FB")
data_puno = pd.read_csv("Data/Output/peru_2019.csv")

data_puno.info()

#Set the random seed for reproductible results
np.random.seed(0)

Y = data_puno["disasters_risk"]
x = data_puno.iloc[:,6:]

#Binarization of Altitude and GPC
    #Making features lay in the same scale will improve ENLR SAGA solver performance

def Binarization_q(data, nquantiles, labels):
    categorized = pd.qcut(data, q=nquantiles, labels=labels)
    binarized = pd.get_dummies(categorized)
    binarized = pd.DataFrame(binarized, dtype="int64")
    return binarized

altitude_q = Binarization_q(x["altura"], 10, labels=["aq1", "aq2", "aq3" 
                                                          , "aq4", "aq5", "aq6", "aq7" 
                                                          , "aq8", "aq9", "aq10"])
gpc_q = Binarization_q(x["gpc"], 10, labels=["gq1", "gq2", "gq3" 
                                                          , "gq4", "gq5", "gq6", "gq7" 
                                                          , "gq8", "gq9", "gq10"])

# Concat to features matrix

x = x.drop(columns=["altura", "gpc"])
x = pd.concat([x, altitude_q, gpc_q], axis=1)

x.info()
print("Data type is int64")

from collections import Counter

print(f"Original class counts: {Counter(Y)}")
print("Classes are balanced")

# Init the grid search cross-validation on ENLR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
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


def ENLRexperiments(K_folds, mcc_weight):
    # Cross-validation method
    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=6, random_state=0)
    
    # Hyperparameter grid for ENLR
    param_grid = { 'C': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
                  'l1_ratio': np.round_(np.arange(0.0,1.1,0.1), decimals=1)}
    

    
    # The composite metric
    def composite_metric(y_true,y_predicted,??):
        composite = ??*matthews_corrcoef(y_true,y_predicted)+(1-??)*neg_pred_value(y_true,y_predicted)
        return composite
    
    # I define the model here
    ENLR = LogisticRegression(random_state=0, penalty='elasticnet', solver='saga', max_iter=4000, n_jobs=-1)
    composite_score = make_scorer(composite_metric, ??=mcc_weight)
    
    #Test CV
    search_ddnn= GridSearchCV(estimator = ENLR, param_grid=param_grid, scoring=composite_score, cv=cv, n_jobs=-1, refit=True)
    results = search_ddnn.fit(x, Y)
    return results

import time
start = time.time()
grid_length = 99

# Loop guardar resultados de experimentos en CSV
for i in range(3,6):
    for j in np.arange(0.0,1.1,0.1):
        print("Configuration")
        print("K_folds =", i, "------- npv_weight =", np.round_(j, decimals=1))
        exec(f'ENLRresults_{i}__{int(j)}_{int(j*10)} = ENLRexperiments({i}, {j})')
        print("Elapsed Time ", (time.time() - start))
        exec(f'print("Optimal Parameters :", ENLRresults_{i}__{int(j)}_{int(j*10)}.best_params_)')
        exec(f'convex = pd.Series(np.full_like(range(grid_length), {j}, dtype=np.double), name="lambda")')
        exec(f'pd.concat([pd.DataFrame(ENLRresults_{i}__{int(j)}_{int(j*10)}.cv_results_), convex], axis=1).to_csv("Data/Output/ENLR_results_{i}__{int(j)}_{int(j*10)}.csv")')        
        
Total_time = (time.time() - start)

#aux = ENLRexperiments(3, 0.4)

#pd.DataFrame(aux.cv_results_).to_excel("Data/Output/AuxExp/aux_results.xlsx")

# Resumen de experimentos

summary = {"C": [], "l1_ratio": [], "K_fold": [], "lambda": [], "composite": []}
for i in range(3,6):
    for j in np.arange(0.0,1.1,0.1):
        exec(f'summary["C"].append(ENLRresults_{i}__{int(j)}_{int(j*10)}.best_params_["C"])')
        exec(f'summary["l1_ratio"].append(ENLRresults_{i}__{int(j)}_{int(j*10)}.best_params_["l1_ratio"])')
        exec(f'summary["K_fold"].append({i})')
        exec(f'summary["lambda"].append(np.round_({j}, decimals=1))')
        exec(f'summary["composite"].append(ENLRresults_{i}__{int(j)}_{int(j*10)}.best_score_)')

pd.DataFrame(summary).to_excel("Data/Output/ENLRsummary.xlsx")
summary = pd.DataFrame(summary)

"""
Random forest classifier
"""

from sklearn.ensemble import RandomForestClassifier

def RFCexperiments(K_folds, mcc_weight):

    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=6, random_state=0)
    
    param_grid = {'criterion': ['gini'],
                  'max_depth': [2, 3, 4, None],
                  'min_samples_leaf': [0.1, 0.25, 0.5],
                  'min_samples_split': [0.1, 0.25, 0.5],
                  'n_estimators': [10, 100, 200],
                  'max_features': ['log2']}
    
    # The composite metric
    def composite_metric(y_true,y_predicted,??):
        composite = ??*matthews_corrcoef(y_true,y_predicted)+(1-??)*neg_pred_value(y_true,y_predicted)
        return composite
    
    # I define the model here
    RFC = RandomForestClassifier(random_state=0, n_jobs=-1, bootstrap=True)
    composite_score = make_scorer(composite_metric, ??=mcc_weight)
    
    #Test CV
    search_ddnn= GridSearchCV(estimator = RFC, param_grid=param_grid, scoring=composite_score, cv=cv, n_jobs=-1, refit=True)
    results = search_ddnn.fit(x, Y)
    return results

import time
start = time.time()

grid_length = 144

# Loop guardar resultados de experimentos en CSV
for i in range(3,6):
    for j in np.arange(0.0,1.1,0.1):
        print("Configuration")
        print("K_folds =", i, "------- npv_weight =", np.round_(j, decimals=1))
        exec(f'RFCresults_{i}__{int(j)}_{int(j*10)} = RFCexperiments({i}, {j})')
        print("Elapsed Time ", (time.time() - start))
        exec(f'print("Optimal Parameters :", RFCresults_{i}__{int(j)}_{int(j*10)}.best_params_)')
        exec(f'convex = pd.Series(np.full_like(range(grid_length), {j}, dtype=np.double), name="lambda")')
        exec(f'pd.concat([pd.DataFrame(RFCresults_{i}__{int(j)}_{int(j*10)}.cv_results_), convex], axis=1).to_csv("Data/Output/RFC_results_{i}__{int(j)}_{int(j*10)}.csv")')

print((time.time() - start))

RFCsummary = {"param_criterion": [], "param_max_depth": [], "param_max_features": [], "param_min_samples_leaf": [],
           "param_min_samples_split": [], "param_n_estimators": [], "K_fold": [], "lambda": [], "composite": []}			

for i in range(3,6):
    for j in np.arange(0.0,1.1,0.1):
        exec(f'RFCsummary["param_criterion"].append(RFCresults_{i}__{int(j)}_{int(j*10)}.best_params_["criterion"])')
        exec(f'RFCsummary["param_max_depth"].append(RFCresults_{i}__{int(j)}_{int(j*10)}.best_params_["max_depth"])')
        exec(f'RFCsummary["param_max_features"].append(RFCresults_{i}__{int(j)}_{int(j*10)}.best_params_["max_features"])')
        exec(f'RFCsummary["param_min_samples_leaf"].append(RFCresults_{i}__{int(j)}_{int(j*10)}.best_params_["min_samples_leaf"])')
        exec(f'RFCsummary["param_min_samples_split"].append(RFCresults_{i}__{int(j)}_{int(j*10)}.best_params_["min_samples_split"])')
        exec(f'RFCsummary["param_n_estimators"].append(RFCresults_{i}__{int(j)}_{int(j*10)}.best_params_["n_estimators"])')
        exec(f'RFCsummary["K_fold"].append({i})')
        exec(f'RFCsummary["lambda"].append(np.round_({j}, decimals=1))')
        exec(f'RFCsummary["composite"].append(RFCresults_{i}__{int(j)}_{int(j*10)}.best_score_)')
        
pd.DataFrame(RFCsummary).to_excel("Data/Output/RFC_summary.xlsx")

RFCresults_5__1_10.best_params_

#pd.DataFrame(res.cv_results_).to_excel("Data/Output/AuxExp/RFC_preliminar3.xlsx")

# Hyperparameter final tunning and model selection

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def ConfMatrix(results, test):
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=test, random_state=0, stratify=Y)
    
    res = results.best_estimator_.fit(X_train, y_train)
    predictions_LR = res.predict(X_test)
    cm = confusion_matrix(y_test, predictions_LR, labels=results.best_estimator_.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=results.best_estimator_.classes_)
    disp.plot()
    print("MCC", matthews_corrcoef(y_test, predictions_LR))
    print("NPV", neg_pred_value(y_test, predictions_LR))
    return cm

#Calculamos la matriz de confusion
ConfMatrix(results=res, test=1/3)

def ConfMatrix_eval(C, l1_ratio, test):
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=test, random_state=0, stratify=Y)
    ENLR = LogisticRegression(random_state=0, penalty='elasticnet', solver='saga',
                              max_iter=4000, n_jobs=-1, C=C, l1_ratio=l1_ratio)
    res = ENLR.fit(X_train, y_train)
    predictions_LR = res.predict(X_test)
    cm = confusion_matrix(y_test, predictions_LR, labels=ENLR.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=ENLR.classes_)
    disp.plot()
    print("MCC", matthews_corrcoef(y_test, predictions_LR))
    print("NPV", neg_pred_value(y_test, predictions_LR))
    return cm

a = {'C': 0.001, 'l1_ratio': 0.0}
cm1 = ConfMatrix_eval(C=a['C'], l1_ratio=a['l1_ratio'], test=0.5)

a = {'C': 0.0001, 'l1_ratio': 0.5}
cm2 = ConfMatrix_eval(C=a['C'], l1_ratio=a['l1_ratio'], test=0.5)

#results=results_3__1_10.best_params_

def Costs(cm):
    total_cost = -1*cm[0][0] + 1*cm[0][1] + 0.5*cm[1][0] + 1*cm[1][1]
    logistic_cost = 0.5*cm[1][0]+1*cm[1][1]
    return total_cost, logistic_cost

Costs(cm1)

Costs(cm2)