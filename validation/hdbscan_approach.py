import numpy as np
from utils import data_utils
from sklearn import metrics
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import math
# https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html


def hdbscan_validation(X, mClust, mSample):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mClust, min_samples=mSample, prediction_data=True).fit(X)
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, clusterer.labels_))
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
    da = X[~X.index.isin(outliers_inx)]
    plt.scatter(da[['pca_1']], da[['pca_2']], c='red', cmap=plt.cm.nipy_spectral, alpha=0.3, edgecolor='k', label='Data points')
    os = X.loc[outliers_inx]
    plt.scatter(os[['pca_1']], os[['pca_2']], s=50, linewidth=0, c='black', alpha=0.3, label='Outliers')
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]


def predict_new_points(clusterer, mrs):
    test_bi_pca_all = data_utils.get_nih_test_transformed(mrs)

    labels = test_bi_pca_all[['label']]
    test_bi_pca = test_bi_pca_all.drop(['label'], axis=1)

    # see what happened
    ot = test_bi_pca_all[test_bi_pca_all['label'] == -1]
    plt.scatter(ot[['pca_1']], ot[['pca_2']], s=50, linewidth=0, c='yellow', alpha=1, label='Test outliers')
    noot = test_bi_pca_all[test_bi_pca_all['label'] != -1]
    plt.scatter(noot[['pca_1']], noot[['pca_2']], s=50, linewidth=0, c='blue', alpha=1, label='Test data points')
    legend = plt.legend(loc='upper left')
    legend.legendHandles[2]._sizes = [30]
    legend.legendHandles[3]._sizes = [40]
    #
    test_labels, strengths = hdbscan.approximate_predict(clusterer, test_bi_pca)
    test_labels [test_labels > -1] = 0
    sensitivity, specificity, accuracy = data_utils.show_performance(labels, test_labels)
    return sensitivity, specificity, accuracy


if __name__ == '__main__':
    mrs = 4
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_pca, pca = data_utils.pca_reduction(bi_df)
    bi_df_pca_unique = bi_df_pca.drop_duplicates()

    # mSample = int(round(bi_df_pca_unique.shape[0] * 0.05, 0))
    mSample = math.floor(math.log(bi_df_pca_unique.shape[0], 10))
    clusterer = hdbscan_validation(bi_df_pca_unique, 3*mSample, 3*mSample)

    # plot_outlier_distribution(clusterer)
    # score_label = make_score_label(bi_df_pca_unique, clusterer, 0.9)

    data_labeled_all, data_labeled_unique = data_utils.label_data(bi_df, bi_df_pca_unique, clusterer.labels_)
    # data_labeled_all, data_labeled_unique = data_utils.label_data(bi_df, bi_df_pca_unique, score_label)

    outliers_unique, outliers_all = data_utils.outlier_filter(data_labeled_all, data_labeled_unique)

    plot_hdbscan(bi_df_pca, outliers_all.index, mrs)

    print(predict_new_points(clusterer, mrs))

    plt.show()
    print(bi_df_pca_unique.shape[0])
    print(mSample*3)
    print('done')