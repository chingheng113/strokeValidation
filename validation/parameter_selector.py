from sklearn import metrics
from utils import data_utils
from sklearn.neighbors import LocalOutlierFactor
import hdbscan


def hdbscan_see(X, test_bi_pca, labels, min_samples):
    for mSample in min_samples:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mSample, prediction_data=True).fit(X)

        test_labels, strengths = hdbscan.approximate_predict(clusterer, test_bi_pca)
        test_labels[test_labels > -1] = 0
        print(mSample)
        print(data_utils.show_performance(labels, test_labels))
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, clusterer.labels_))
        print('--')


def lof_see(X, test_bi_pca, labels, min_samples):
    for mSample in min_samples:
        clf = LocalOutlierFactor(n_neighbors=mSample, contamination=0.1)
        clf_predict = LocalOutlierFactor(n_neighbors=mSample, contamination=0.1, novelty=True).fit(
            bi_df_pca_unique)
        test_labels = clf_predict._predict(test_bi_pca)
        test_labels[test_labels > -1] = 0
        print(mSample)
        print(data_utils.show_performance(labels, test_labels))
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, clf.fit_predict(X)))
        print('--')


if __name__ == '__main__':
    mrs = 3
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_pca, pca = data_utils.pca_reduction(bi_df)
    bi_df_pca_unique = bi_df_pca.drop_duplicates()
    # test data
    test_bi_pca_all = data_utils.get_nih_test_transformed(mrs)
    labels = test_bi_pca_all[['label']]
    test_bi_pca = test_bi_pca_all.drop(['label'], axis=1)

    hdb_samples =[3, 4, 5, 6, 7, 8, 9, 10, 11]
    lof_samples =[10, 11, 12, 13, 14, 15, 16]
    # hdbscan_see(bi_df_pca_unique, test_bi_pca, labels, hdb_samples)
    lof_see(bi_df_pca_unique, test_bi_pca, labels, hdb_samples)