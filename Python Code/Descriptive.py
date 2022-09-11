# %%
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, jaccard_score, cohen_kappa_score
from locale import normalize

url = "https://github.com/renatoquiliche/PaperCienciaDados-PunoPeru/blob/main/Databases/peru_2019.csv?raw=true"

data_puno = pd.read_csv(url)

#data_puno = pd.read_csv("Databases\peru_2019.csv")

# %%

x = data_puno.iloc[:,6:]

corr_matrix = x.corr(method='pearson')

# %%

treshold = 0.7
get_index = corr_matrix[((corr_matrix > treshold) | (corr_matrix < -treshold)) & (corr_matrix != 1)].dropna(how='all').index

import seaborn as sns
import matplotlib.pyplot as plt

cmap_reversed = plt.cm.get_cmap('Spectral_r')
corr_subset = corr_matrix.loc[get_index, get_index]

plt.figure(figsize=(12,12))
sns.heatmap(corr_subset[corr_matrix != 1], cbar=False
, cmap='bwr', annot=True, robust=True, fmt="0.1%", center=0
, xticklabels=corr_subset.columns, yticklabels=corr_subset.columns)

plt.savefig("Resultados\Graphics\corr_matrix.png", dpi=600)
# %%

plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix[corr_matrix != 1], cbar=False, center=0
, cmap='bwr', robust=True, yticklabels=False, xticklabels=False)

plt.savefig("Resultados\Graphics\corr_matrix_full.png", dpi=600)

# %%
x.columns.values.tolist()

# %%


# %%
umbral = 0.02

frequencies = data_puno.iloc[:,7:-1].mean()
index_features = frequencies[frequencies > umbral].index
print("Features removed :", x.shape[1]-len(index_features))
print("Old data dimension", x.shape)
print("New data dimension", x.loc[:, index_features].shape)

# %%

final_data = x.loc[:,index_features]
descriptive_cat = pd.concat([final_data.mean(), final_data.sum()], axis=1)

descriptive_cat.to_excel("descriptive_cat.xlsx")

final_data[["gpc", "altura"]] = data_puno[["gpc", "altura"]]

# %%
