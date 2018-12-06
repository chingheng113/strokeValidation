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


def make_score_label(X, clusterer, cutoff):
    s_label = pd.DataFrame(index=X.index, columns=['label'])
    threshold = pd.Series(clusterer.outlier_scores_).quantile(cutoff) # the larger the score the more outlier-like the point.
    outlier_inx = np.where(clusterer.outlier_scores_ > threshold)[0]
    s_label.iloc[outlier_inx] = -1
    s_label.fillna(0, inplace=True)
    return s_label.values


def plot_outlier_distribution(clusterer):
    sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
    plt.show()


def plot_hdbscan(X, outliers_inx, n):
    plt.title("HDBSCAN (mRS="+str(n)+")")
    plt.scatter(X[['pca_1']], X[['pca_2']], c='red', cmap=plt.cm.nipy_spectral, edgecolor='k', label='Data points')
    os = X.loc[outliers_inx]
    plt.scatter(os[['pca_1']], os[['pca_2']], s=50, linewidth=0, c='black', alpha=1, label='outliers')
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()


if __name__ == '__main__':
    mrs = 4
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_pca = data_utils.pca_reduction(bi_df)
    bi_df_pca_unique = bi_df_pca.drop_duplicates()

    # mSample = int(round(bi_df.shape[0] * 0.05, 0))
    clusterer = hdbscan_validation(bi_df_pca_unique, 11)

    # plot_outlier_distribution(clusterer)
    score_label = make_score_label(bi_df_pca_unique, clusterer, 0.9)

    # data_labeled_all, data_labeled_unique = data_utils.label_data(bi_df, bi_df_pca_unique, clusterer.labels_)
    data_labeled_all, data_labeled_unique = data_utils.label_data(bi_df, bi_df_pca_unique, score_label)

    outliers_unique, outliers_all = data_utils.outlier_filter(data_labeled_all, data_labeled_unique)

    plot_hdbscan(bi_df_pca_unique, outliers_unique.index, mrs)
    print('done')