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

from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, f1_score, roc_curve
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

# I define here the NPV metric
def neg_pred_value(y_true,y_predicted):
    cm = confusion_matrix(y_true, y_predicted)
    if (cm[1][0]+cm[0][0])==0:
        neg_pred_value=0
    else:
        neg_pred_value = cm[0][0]/(cm[1][0]+cm[0][0])
    return neg_pred_value

# %%
XGB = xgb.XGBClassifier(objective="binary:logistic", tree_method='gpu_hist', gpu_id=0)
XGB.fit(x, Y)


# %%
from scipy.stats import uniform, randint

def XGBexperiments(K_folds, Repeats):
    # Cross-validation method
    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=Repeats, random_state=0)
    
    # Hyperparameter grid for ENLR
    param_grid = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3), # default 0.1 
        "max_depth": randint(2, 6), # default 3
        "n_estimators": randint(100, 200), # default 100
        "subsample": uniform(0.6, 0.4)
    }
    # randint.rvs(100,150, size=200)
    
    # I define the model here
    XGB = xgb.XGBClassifier(objective="binary:logistic")
    scoring = {"AUC": "roc_auc"
               , "Accuracy": make_scorer(accuracy_score)
               , "F1-Score": "f1"
               , "MCC": make_scorer(matthews_corrcoef)
               , "NPV": make_scorer(neg_pred_value)}
    
    #Test CV
    search_ddnn = RandomizedSearchCV(random_state=0, estimator=XGB, param_distributions=param_grid
                                     , scoring=scoring, cv=cv, n_jobs=-1, refit="MCC", verbose=4, n_iter=200)
    results = search_ddnn.fit(x, Y)
    return results
# %%

res = XGBexperiments(5, 2)
# %%
pd.DataFrame(res.cv_results_).to_csv("Resultados/XGB/first.csv")

