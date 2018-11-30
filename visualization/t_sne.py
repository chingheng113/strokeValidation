from utils import data_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr('', 'is')
# remove duplicate
# bi_df = bi_df.drop_duplicates()
mrs_df = mrs_df.loc[bi_df.index]
#
t_sne = TSNE(n_components=2, perplexity=30).fit_transform(bi_df)
result = np.concatenate([t_sne, mrs_df], axis=1)
df_result = pd.DataFrame(result, columns=['x', 'y']+list(mrs_df.columns.values))
data_utils.save_dataframe_to_csv(df_result, 'tSNE')
label = mrs_df.values.ravel()
n_class = np.unique(label).shape[0]
plt.figure()
plt.scatter(df_result.ix[:,0], df_result.ix[:,1], c=label, s=8, cmap=plt.cm.get_cmap("jet", n_class))
plt.colorbar(ticks=range(n_class))
plt.title('t-SNE 2D visualization: ')
plt.show()