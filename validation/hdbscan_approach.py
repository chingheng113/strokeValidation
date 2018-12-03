import numpy as np
from utils import data_utils
from sklearn import metrics
from sklearn.preprocessing import scale
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
# https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html


def hdbscan_validation(X, mSample):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mSample, allow_single_cluster=True).fit(X)
    return clusterer


def outlier_detection(X, clusterer):
    # threshold methods
    threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9) # the larger the score the more outlier-like the point.
    outlier_inx = np.where(clusterer.outlier_scores_ > threshold)[0]
    outliers = X.iloc[outlier_inx]
    plot_outlier_distribution(clusterer)

    # default methods
    # X['cluster'] = clusterer.labels_
    # outliers = X[X['cluster'] == -1]
    return outliers


def plot_outlier_distribution(clusterer):
    sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
    plt.show()


def plot_hdbscan(X, outliers):
    plt.scatter(X[['pca_1']], X[['pca_2']], c='red',
                cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.scatter(outliers[['pca_1']], outliers[['pca_2']], s=50, linewidth=0, c='black', alpha=1)
    plt.show()


if __name__ == '__main__':
    mrs = 0
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_reduced = data_utils.pca_reduction(bi_df_unique)
    mSample = int(round(bi_df_reduced.shape[0] * 0.05, 0))
    clusterer = hdbscan_validation(bi_df_reduced, 11)
    outliers = outlier_detection(bi_df_reduced, clusterer)
    data_utils.label_data(bi_df_unique, id_df, outliers, 'hdbscan_outlier')
    plot_hdbscan(bi_df_reduced, outliers)
    print('done')