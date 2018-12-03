import numpy as np
from utils import data_utils
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import scale
import pandas as pd


def dbscan_validation(X, eps, mSample):
    # cosine
    db = DBSCAN(eps=eps, min_samples=mSample, metric='euclidean').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    print("Calinski-Harabaz Index: %0.3f" % metrics.calinski_harabaz_score(X, labels))
    return core_samples_mask, n_clusters_, labels


def outlier_detection(X, labels):
    X['clust'] = labels
    outliers = X[X['clust'] == -1]
    print(outliers.shape[0])
    return outliers


def dbscan_plot(num, X, core_samples_mask, n_clusters_, labels):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=7)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)

    plt.title(
        'mRS = '+ str(num)+'\n'+
        'Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def dbscan_plot2(X, outliers):
    plt.scatter(X[['pca_1']], X[['pca_2']], c='red',
                cmap=plt.cm.nipy_spectral, edgecolor='k')

    class_member_mask = (labels == -1)
    outliers = X.iloc[class_member_mask]
    plt.scatter(outliers[['pca_1']], outliers[['pca_2']], s=50, linewidth=0, c='black', alpha=1)
    plt.show()


if __name__ == '__main__':
    mrs = 5
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_reduced = data_utils.pca_reduction(bi_df_unique)
    # mSample = round(bi_df.shape[0]/10, 0)
    core_samples_mask, n_clusters, labels = dbscan_validation(bi_df_reduced, 0.55, 11)
    outlier = outlier_detection(bi_df_reduced, labels)
    data_utils.label_data(bi_df_unique, id_df, outlier, 'dbscan_outlier')
    # dbscan_plot(mrs, bi_df_reduced.values, core_samples_mask, n_clusters, labels)
    dbscan_plot2(bi_df_reduced, labels)
    print('done')