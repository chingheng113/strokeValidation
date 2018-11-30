import numpy as np
from utils import data_utils
from sklearn import metrics
from sklearn import decomposition
from sklearn.preprocessing import scale
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
# https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html

def pca_reduction(data):
    pca = decomposition.PCA(n_components=2)
    scaled_data = scale(data)
    pca.fit(scaled_data)
    transformed_data = pca.transform(scaled_data)
    df_pca = pd.DataFrame(transformed_data, columns=['pca_1', 'pca_2'], index=data.index)
    return df_pca


def hdbscan_validation(X):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=11, allow_single_cluster=True).fit(X)
    s = clusterer.outlier_scores_
    threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
    outlier_inx = np.where(clusterer.outlier_scores_ > threshold)[0]
    outliers = X.iloc[outlier_inx]
    # sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
    return outliers


def plot_hdbscan(X, outliers):
    plt.scatter(X[['pca_1']], X[['pca_2']], c='red',
                cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.scatter(outliers[['pca_1']], outliers[['pca_2']], s=50, linewidth=0, c='black', alpha=1)
    plt.show()


if __name__ == '__main__':
    mrs = 5
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_reduced = pca_reduction(bi_df_unique)
    outliers = hdbscan_validation(bi_df_reduced)
    plot_hdbscan(bi_df_reduced, outliers)
    print('done')