"""
This script runs RFC tunning for DS-PunoPeru

@author: Renato Quiliche
"""
# %%
import pandas as pd
import numpy as np
import os
#Read the data from GitHub directly
url = "https://github.com/renatoquiliche/PaperCienciaDados-PunoPeru/blob/main/Databases/peru_2019.csv?raw=true"

data_puno = pd.read_csv(url)

# %%
# Set the random seed for reproductible results
np.random.seed(0)

Y = data_puno["disasters_risk"]
# %%
from preprocessing import preprocessing
x = preprocessing(data_puno, umbral=0.02)

# %%

# Init the grid search cross-validation on RFC
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

def RFCexperiments(K_folds, Repeats, n_iter):
    # Cross-validation method
    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=Repeats, random_state=0)
    
    # Hyperparameter grid for RFC
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': randint(2,10),
                  'min_samples_split': uniform(0, 0.5),
                  'min_samples_leaf': uniform(0, 0.4),
                  'n_estimators': [10, 100, 200, 250, 300, 450],
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
                                     , scoring=scoring, cv=cv, n_jobs=-1, refit="MCC", verbose=4, n_iter=n_iter)
    results = search_ddnn.fit(x, Y)
    return results

# %%
import time
start = time.time()

K = 10
Repeats = 2
n_iter = 2000
RFC_results =  RFCexperiments(K, Repeats, n_iter)

Total_time = (time.time() - start)
print("Training time: ", Total_time, " seconds")

# %%

pd.DataFrame(RFC_results.cv_results_).to_csv("..\Resultados\RFC\RFC_results.csv")

# %%

import seaborn as sns
import pandas as pd

results_dataset = pd.read_csv("..\Resultados\RFC\RFC_results.csv")

#color_dict = dict({0.01:'brown',
#                  0.1:'green',
#                  1.0: 'orange',
#                  10.0: 'red',
#                   100.0: 'dodgerblue'})

sns.scatterplot(data=results_dataset, x="param_n_estimators", y="mean_test_MCC", hue="param_max_depth")
# %%
sorted = results_dataset.sort_values(by=["rank_test_MCC"])
sorted.iloc[:,5:12].head()
# %%
