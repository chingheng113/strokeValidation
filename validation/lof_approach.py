import numpy as np
import pandas as pd
from utils import data_utils
import matplotlib.pyplot as plt
from matplotlib import mlab
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns


def make_score_label(X, clf, cutoff):
    s_label = pd.DataFrame(index=X.index, columns=['label'])
    clf.fit_predict(X)
    X_scores = -clf.negative_outlier_factor_
    outlier_inx = np.where(X_scores > cutoff)[0]
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


def predict_new_points(bi_df_pca_unique, test_dataset, mrs, cutoff):
    test_bi_pca_all = data_utils.get_test_transformed(test_dataset, mrs)
    true_labels = test_bi_pca_all[['label']]
    test_bi_pca = test_bi_pca_all.drop(['label'], axis=1)
    train_bi_pca = bi_df_pca_unique.drop(['label'], axis=1)

    # see what happened
    ot = test_bi_pca_all[test_bi_pca_all['label'] == -1]
    plt.scatter(ot[['pca_1']], ot[['pca_2']], s=50, linewidth=0, c='yellow', alpha=1, label='Test outliers')
    noot = test_bi_pca_all[test_bi_pca_all['label'] != -1]
    plt.scatter(noot[['pca_1']], noot[['pca_2']], s=50, linewidth=0, c='blue', alpha=1, label='Test data points')
    legend = plt.legend(loc='upper left')
    legend.legendHandles[2]._sizes = [30]
    legend.legendHandles[3]._sizes = [40]
    #


    clf_predict = LocalOutlierFactor(n_neighbors=k_neighbors, contamination=outliers_fraction, novelty=True)
    clf_predict.fit(train_bi_pca)
    outlier_scores = -clf_predict.score_samples(test_bi_pca)
    test_labels = (outlier_scores > cutoff).astype(int)*-1
    sensitivity, specificity, accuracy = data_utils.show_performance(true_labels, test_labels)
    return sensitivity, specificity, accuracy


def score_cutoff(X, clf):
    clf.fit_predict(X)
    X_scores = np.sort(-clf.negative_outlier_factor_)

    # Percentile values plot
    # p = np.array([0, 25, 50, 75, 90, 95])
    # perc = mlab.prctile(X_scores, p=p)
    # plt.plot(X_scores)
    # plt.plot((len(X_scores)-1) * p/100., perc, 'ro') # Place red dots on the percentiles
    # plt.xticks((len(X_scores)-1) * p/100., map(str, p)) # Set tick locations and labels
    # plt.xlabel('Percentile')
    # plt.ylabel('Outlier Scores')
    # plt.title('mRS-3')
    #
    threshold = pd.Series(X_scores).quantile(0.85)
    # 0:95, 1:90, 2:90, 3:90, 4:90, 5:85
    plt.annotate(
        'Elbow point',
        xy=(1104, threshold), arrowprops=dict(arrowstyle='->'), xytext=(900, threshold+0.09))
    print(threshold)
    return threshold


if __name__ == '__main__':
    mrs = 5
    test_dataset = 'tnk'
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_pca, pca = data_utils.pca_reduction(bi_df)
    bi_df_pca_unique = bi_df_pca.drop_duplicates()

    k_neighbors = 10
    outliers_fraction = 0.01
    # Training
    clf = LocalOutlierFactor(n_neighbors=k_neighbors, contamination=outliers_fraction)
    # determine cutoff
    cutoff = score_cutoff(bi_df_pca_unique, clf)
    #
    labels = make_score_label(bi_df_pca_unique, clf, cutoff)
    data_labeled_all, data_labeled_unique = data_utils.label_data(bi_df, bi_df_pca_unique, labels)
    outliers_unique, outliers_all = data_utils.outlier_filter(data_labeled_all, data_labeled_unique)
    plot_lof(bi_df_pca_unique, outliers_unique.index, mrs)

    print(predict_new_points(bi_df_pca_unique, test_dataset, mrs, cutoff))
    plt.show()
    print('done')