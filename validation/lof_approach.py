import numpy as np
import pandas as pd
from utils import data_utils
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
from scipy import stats


def lof_validation(X, outliers_fraction):
    clf = LocalOutlierFactor(n_neighbors=11, contamination=outliers_fraction)
    return clf


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
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c='red', cmap=plt.cm.nipy_spectral, edgecolor='k', label='Data points')
    os = X.loc[outliers_inx]
    plt.scatter(os.iloc[:, 0], os.iloc[:, 1], s=50, linewidth=0, c='black', alpha=1, label='outliers')
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()


def plot_outlier_distribution(clf):
    sns.distplot(clf.negative_outlier_factor_[np.isfinite(clf.negative_outlier_factor_)], rug=True)
    plt.show()


if __name__ == '__main__':
    mrs = 5
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_pca = data_utils.pca_reduction(bi_df)
    bi_df_pca_unique = bi_df_pca.drop_duplicates()

    clf = lof_validation(bi_df_pca_unique, 0.1)
    label = clf.fit_predict(bi_df_pca_unique)
    # label = make_score_label(bi_df_pca_unique, clf, 0.2)
    # plot_outlier_distribution(clf)
    data_labeled_all, data_labeled_unique = data_utils.label_data(bi_df, bi_df_pca_unique, label)
    outliers_unique, outliers_all = data_utils.outlier_filter(data_labeled_all, data_labeled_unique)

    plot_lof(bi_df_pca_unique, outliers_unique.index, mrs)

    print('done')