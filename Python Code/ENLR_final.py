"""
This script runs ENLR tunning for DS-PunoPeru

@author: Renato Quiliche
"""
# %%

from functools import partial
from multiprocessing.connection import Pipe
import pandas as pd
import numpy as np

# %%
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

print("AUC", "\t\t Accuracy", "\t\t F1-Score")
print(step4["mean_test_AUC"].iloc[0], step4["mean_test_Accuracy"].iloc[0], step4["mean_test_F1-Score"].iloc[0])
print(step5["mean_test_AUC"].iloc[0], step5["mean_test_Accuracy"].iloc[0], step5["mean_test_F1-Score"].iloc[0])

# %%
display(step4[["rank_test_MCC", "rank_test_NPV"]].head(10))
display(step5[["rank_test_MCC", "rank_test_NPV"]].head(10))

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

MCC_repeats = {"Repeat1": [], "Repeat2": []}
for i in range(10):
    MCC_repeats["Repeat1"].append(f'split{i}_test_MCC')

for i in range(10,20):
    MCC_repeats["Repeat2"].append(f'split{i}_test_MCC')    

MCC = pd.DataFrame()

MCC_repeats1 = pd.concat([step5.iloc[0:1].T.loc[MCC_repeats["Repeat1"]].reset_index(), 
            pd.Series(np.ones(10).astype("int"))], axis=1, ignore_index=True)

MCC_repeats2 = pd.concat([step5.iloc[0:1].T.loc[MCC_repeats["Repeat2"]].reset_index(), 
            pd.Series(np.ones(10).astype("int"))+1], axis=1, ignore_index=True)            


MCC_repeats = pd.concat([MCC_repeats1, MCC_repeats2], axis=0)
MCC_repeats.columns = ["experiment", "MCC", "Repeat"]

sns.set_theme()
sns.boxplot(data=MCC_repeats, y="MCC", x="Repeat")
# %%
Coefficients = pd.DataFrame(columns=["Variable", "Coefficient"])

Coefficients["Coefficient"] = pd.DataFrame(ENLR_results.best_estimator_.coef_.T)

Coefficients["Variable"] = pd.DataFrame(ENLR_results.feature_names_in_)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=results_dataset, x="param_l1_ratio", 
              y="mean_test_MCC", hue="param_C"
                , palette="tab10")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# %%
sns.scatterplot(data=results_dataset, x="param_l1_ratio", 
              y="mean_test_NPV", hue="param_C"
                , palette="tab10")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# %%
params = step5["params"].iloc[0]

import ast
#params = ast.literal_eval(params)

winner_model = LogisticRegression(C=params["C"], l1_ratio=params["l1_ratio"], 
                                  max_iter=4000, n_jobs=-1, penalty="elasticnet",
                                  solver="saga", random_state=0)
# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

x_train, x_test, y_train, y_test = train_test_split(x, Y, random_state=0, test_size=0.2, stratify=Y)
winner_model.fit(x_train, y_train)
plt.rcParams.update(plt.rcParamsDefault)
predictions = winner_model.predict(x_test)
cm = confusion_matrix(y_test, predictions, labels=winner_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Otherwise", "Household at risk"])
disp.plot()


print('Negative predictive value: ', neg_pred_value(y_test, predictions))
print('Mathews Correlation Coefficient: ', matthews_corrcoef(y_test, predictions))


# %%

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, Y, random_state=0, test_size=0.2, stratify=Y)
winner_model.fit(x_train, y_train)
feature_importances = pd.DataFrame({'Features': winner_model.feature_names_in_,
                                    'Importances': np.abs(winner_model.coef_[0])})

sorted_results = feature_importances.sort_values(by=['Importances'], ascending=False)[0:15]

sorted_results["Features"] = ["Internet", "Household is totally owned",
                              "Household is located in a rural area", 
                              "Concrete roof", "Manure cooking", "The head is married",
                              "Outside tracks are paved", "Concrete floor", "Outside paths",
                              "Concrete walls", "The head is employed", "Electric iron",
                              "Motorcycle", "No public good", "Ligthing pole"]

sns.barplot(data=sorted_results, y='Features', x='Importances', palette='GnBu_r')

# %%

grid = np.arange(0.1,0.99,0.01)

res = {"test_size": [], "MCC": [], "NPV": []}

for i in grid:
    x_train, x_test, y_train, y_test = train_test_split(x, Y, random_state=0, test_size=i, stratify=Y)
    winner_model.fit(x_train, y_train)
    predictions = winner_model.predict(x_test)
    res["test_size"].append(i*100)
    res["MCC"].append(matthews_corrcoef(y_test, predictions)*100)
    res["NPV"].append(neg_pred_value(y_test, predictions)*100)
    
results = pd.DataFrame(res)

# %%
sns.set_theme()
g = sns.jointplot(data=results, x="test_size", y="MCC", kind="kde", fill=True, joint_kws={'alpha': 0.7})
g.plot_joint(sns.scatterplot, alpha=0.8)

# %%

g = sns.jointplot(data=results, x="test_size", y="NPV", kind="kde", fill=True, joint_kws={'alpha': 0.7})
g.plot_joint(sns.scatterplot, alpha=0.8)
# %%

