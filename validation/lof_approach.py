import numpy as np
import pandas as pd
from utils import data_utils
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
from sklearn import metrics


# def lof_validation(X, outliers_fraction):
#     clf = LocalOutlierFactor(n_neighbors=11, contamination=outliers_fraction)
#     return clf


def make_score_label(X, clf, cutoff):
    s_label = pd.DataFrame(index=X.index, columns=['label'])
    clf.fit_predict(X)
    X_scores = clf.negative_outlier_factor_
    threshold = pd.Series(X_scores).quantile(cutoff) # the larger the score the more outlier-like the point.
    outlier_inx = np.where(X_scores > threshold)[0]
    s_label.iloc[outlier_inx] = -1
    s_label.fillna(0, inplace=True)
    return s_label.values


def plot_lof(X, outliers_inx, n):
    plt.title("Local Outlier Factor (mRS="+str(n)+")")
    da = X[~X.index.isin(outliers_inx)]
    plt.scatter(da[['pca_1']], da[['pca_2']], c='red', cmap=plt.cm.nipy_spectral, alpha=0.3, edgecolor='k', label='Data points')
    os = X.loc[outliers_inx]
    plt.scatter(os[['pca_1']], os[['pca_2']], s=50, linewidth=0, c='black', alpha=0.3, label='Outliers')
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]


def predict_new_points(test_dataset, clf, mrs):
    test_bi_pca_all = data_utils.get_test_transformed(test_dataset, mrs)
    # test_bi_pca_all = test_bi_pca_all.iloc[65:70,]
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
    test_labels = clf._predict(test_bi_pca)
    test_labels [test_labels > -1] = 0
    sensitivity, specificity, accuracy = data_utils.show_performance(labels, test_labels)
    return sensitivity, specificity, accuracy


def plot_outlier_distribution(clf):
    sns.distplot(clf.negative_outlier_factor_[np.isfinite(clf.negative_outlier_factor_)], rug=True)
    plt.show()


if __name__ == '__main__':
    mrs = 0
    test_dataset = 'alias'
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_pca, pca = data_utils.pca_reduction(bi_df)
    bi_df_pca_unique = bi_df_pca.drop_duplicates()

    k_neighbors = 3
    outliers_fraction = 0.13
    clf = LocalOutlierFactor(n_neighbors=k_neighbors, contamination=outliers_fraction)
    labels = clf.fit_predict(bi_df_pca_unique)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(bi_df_pca_unique, labels))

    # label = make_score_label(bi_df_pca_unique, clf, 0.2)
    # plot_outlier_distribution(clf)
    data_labeled_all, data_labeled_unique = data_utils.label_data(bi_df, bi_df_pca_unique, labels)
    outliers_unique, outliers_all = data_utils.outlier_filter(data_labeled_all, data_labeled_unique)
    plot_lof(bi_df_pca_unique, outliers_unique.index, mrs)

    clf_predict = LocalOutlierFactor(n_neighbors=k_neighbors, contamination=outliers_fraction, novelty=True).fit(bi_df_pca_unique.drop(['label'], axis=1))
    print(predict_new_points(test_dataset, clf_predict, mrs))
    plt.show()
    print('done')