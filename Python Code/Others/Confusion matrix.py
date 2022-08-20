# %%

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
# %%
plt.figure(figsize=(10, 7))
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font)

cf_matrix= [[45,  5],
       [ 5, 45]]

labels = ['True Negatives','False Positives','False Negatives','True Positives']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',
            xticklabels=['Model No','Model Yes'], yticklabels=['Data No','Data Yes'])

plt.savefig('Resultados\Confusion Matrix.png', dpi=600)
# %%
