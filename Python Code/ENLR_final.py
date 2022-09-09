"""
This script runs ENLR tunning for DS-PunoPeru

@author: Renato Quiliche
"""
# %%

import pandas as pd
import numpy as np
import os

# %%
#Read the data from GitHub directly
url = "https://github.com/renatoquiliche/PaperCienciaDados-PunoPeru/blob/main/Databases/peru_2019.csv?raw=true"

data_puno = pd.read_csv(url)

# %%

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

# %%

# Get balancing format
from collections import Counter

print(f"Original class counts: {Counter(Y)}")
print("Classes are balanced")

# Init the grid search cross-validation on ENLR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, f1_score, roc_curve
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint

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

# %%
def ENLRexperiments(K_folds, Repeats, n_iter):
    # Cross-validation method
    cv = RepeatedStratifiedKFold(n_splits=K_folds, n_repeats=Repeats, random_state=0)
    
    # Hyperparameter grid for ENLR
    param_grid = { 'C': [1e-2, 1e-1, 1.0, 10.0, 100.0],
                  'l1_ratio': uniform(0,1)}
    
    # I define the model here
    ENLR = LogisticRegression(random_state=0, penalty='elasticnet', solver='saga', max_iter=4000, n_jobs=-1)
    scoring = {"AUC": "roc_auc"
               , "Accuracy": make_scorer(accuracy_score)
               , "F1-Score": "f1"
               , "MCC": make_scorer(matthews_corrcoef)
               , "NPV": make_scorer(neg_pred_value)}
    
    #Test CV
    search_ddnn= RandomizedSearchCV(random_state=0, estimator=ENLR, param_distributions=param_grid,
                                    scoring=scoring, cv=cv, n_jobs=-1, refit="MCC", verbose=-1, n_iter=n_iter)
    results = search_ddnn.fit(x, Y)
    return results

# %%
import time
start = time.time()

K = 10
Repeats = 2
n_iter = 2000
ENLR_results =  ENLRexperiments(K, Repeats, n_iter)

Total_time = (time.time() - start)
print("Training time: ", Total_time, " seconds")

# %%

pd.DataFrame(ENLR_results.cv_results_).to_csv("..\Resultados\ENLR\ENLR_results.csv")

# %%
cutoff = round(n_iter*0.05)

results_dataset = pd.DataFrame(ENLR_results.cv_results_)
sorted_results = results_dataset.sort_values(by=["rank_test_MCC"])

# Select the 5% best MCC
step4 = sorted_results.iloc[0:cutoff,:]

# Select the best NPV over 5% best MCC
step5 = step4.sort_values(by=["rank_test_NPV"])
print(step4["mean_test_MCC"].iloc[0], step4["mean_test_NPV"].iloc[0])
print(step5["mean_test_MCC"].iloc[0], step5["mean_test_NPV"].iloc[0])

print("Change in MCC: ", 100*(step5["mean_test_MCC"].iloc[0]-step4["mean_test_MCC"].iloc[0])/(step4["mean_test_MCC"].iloc[0]))
print("Change in NPV: ", 100*(step5["mean_test_NPV"].iloc[0]-step4["mean_test_NPV"].iloc[0])/(step4["mean_test_NPV"].iloc[0]))
print("Percentil 5 cutoff: ", cutoff)

# %%
display(step4[["rank_test_MCC", "rank_test_NPV"]])
display(step5[["rank_test_MCC", "rank_test_NPV"]])

print("Parameters before NPV maximization: ")
display(step4["params"].iloc[0])

print("Parameters after NPV maximization: ")
display(step5["params"].iloc[0])

# %%
import seaborn as sns

NPV_repeats = {"Repeat1": [], "Repeat2": []}
for i in range(10):
    NPV_repeats["Repeat1"].append(f'split{i}_test_NPV')

for i in range(10,20):
    NPV_repeats["Repeat2"].append(f'split{i}_test_NPV')    

NPV = pd.DataFrame()

NPV_repeats1 = pd.concat([step5.iloc[0:1].T.loc[NPV_repeats["Repeat1"]].reset_index(), 
            pd.Series(np.ones(10).astype("int"))], axis=1, ignore_index=True)

NPV_repeats2 = pd.concat([step5.iloc[0:1].T.loc[NPV_repeats["Repeat2"]].reset_index(), 
            pd.Series(np.ones(10).astype("int"))+1], axis=1, ignore_index=True)            


NPV_repeats = pd.concat([NPV_repeats1, NPV_repeats2], axis=0)
NPV_repeats.columns = ["experiment", "NPV", "Repeat"]

sns.set_theme()
sns.boxplot(data=NPV_repeats, y="NPV", x="Repeat")

# %%
import seaborn as sns
import pandas as pd

results_dataset = pd.read_csv("..\Resultados\ENLR\ENLR_results.csv")

color_dict = dict({0.01:'brown',
                  0.1:'green',
                  1.0: 'orange',
                  10.0: 'red',
                   100.0: 'dodgerblue'})

sns.scatterplot(data=results_dataset, x="param_l1_ratio", y="mean_test_MCC", hue="param_C"
                ,palette=color_dict)
# %%
