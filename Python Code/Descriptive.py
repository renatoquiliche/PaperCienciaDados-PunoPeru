# %%
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, jaccard_score, cohen_kappa_score
from locale import normalize

#url = "https://github.com/renatoquiliche/PaperCienciaDados-PunoPeru/blob/main/Databases/peru_2019.csv?raw=true"
data_puno = pd.read_csv('..\Databases\peru_2018_2021.csv')
data_puno.columns
data_puno = data_puno.loc[data_puno.dep==21]
data_puno = pd.concat([data_puno.iloc[:,9:-3], data_puno['gpc']], axis=1)

#df = pd.read_csv("Databases\peru_2019.csv")
# %%
def preprocessing(df: pd.DataFrame, umbral: float):
    import pandas as pd
    umbral = 0
    #x = df.iloc[:,6:]
    print("Old data dimension", df.shape)
    # 1
    frequencies = df.mean()
    index_features = frequencies[frequencies > umbral].index
    print("Features removed :", len(frequencies)-len(index_features))

    # 2
    final_data = df.loc[:,index_features]
    descriptive_cat = pd.concat([final_data.mean(), final_data.sum()], axis=1)

    descriptive_cat.to_excel("descriptive_cat.xlsx")

    final_data[["gpc", "altitude"]] = df[["gpc", "altitude"]]
    
    
    print("New data dimension", final_data.shape)
    return final_data, x

final_data, x = preprocessing(df=data_puno, umbral=0.02)

# %%
corr_matrix = final_data.corr(method='spearman')


# %%

import seaborn as sns
import matplotlib.pyplot as plt

#cmap_reversed = plt.cm.get_cmap('Spectral_r')

plt.figure(figsize=(10,10))
#sns.heatmap(corr_matrix, cbar=False
#, cmap='Spectral_r', annot=True, robust=True, fmt="0.1%", center=0
#, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)

sns.heatmap(corr_matrix, cbar=True, cmap='viridis',
            xticklabels=False, yticklabels=False, alpha=0.8)

#plt.savefig("..\Resultados\Graphics\corr_matrix_full.png", dpi=600)

plt.savefig("..\Resultados\Graphics\corr_matrix.png", dpi=600, bbox_inches='tight')
# %%

plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix.values, cbar=True,cmap='viridis')

plt.savefig("..\Resultados\Graphics\corr_matrix_full.png", dpi=600)

# %%
plt.figure(figsize=(10,10))
a = np.zeros((86,86))

for i in range(86):
    for j in range(86):
        if np.abs(corr_matrix[corr_matrix != 1]).iloc[i,j]>=0.6:
            a[i,j]=1

sns.heatmap(a, cbar=True, cmap='viridis')
# %%
a.mean()
# %%
