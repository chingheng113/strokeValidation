import numpy as np
import pandas as pd
from utils import data_utils
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


def lof_validation(X):
    outliers_fraction = 0.1
    clf = LocalOutlierFactor(n_neighbors=11, contamination=outliers_fraction)
    y_pred = clf.fit_predict(X)
    X_scores = clf.negative_outlier_factor_
    # threshold methods
    threshold = pd.Series(X_scores).quantile(0.1) # while outliers tend to have a larger LOF score.
    outlier_inx = np.where(X_scores < threshold)[0]
    outliers = X.iloc[outlier_inx]

    # default methods
    # labels = clf.fit_predict(X)
    # outliers = X[labels==-1]

    return clf, outliers


def plot_lof(X, outliers_inx):
    plt.title("Local Outlier Factor (LOF)")
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c='red', cmap=plt.cm.nipy_spectral, edgecolor='k', label='Data points')
    os = X.loc[outliers_inx]
    plt.scatter(os.iloc[:, 0], os.iloc[:, 1], s=50, linewidth=0, c='black', alpha=1, label='outliers')
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()


if __name__ == '__main__':
    mrs = 5
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    bi_df_unique = bi_df.drop_duplicates()
    bi_df_pca = data_utils.pca_reduction(bi_df)

    clf, outliers = lof_validation(bi_df_unique)

    data_utils.label_data(bi_df_unique, id_df, outliers, 'lof_outlier')
    plot_lof(bi_df_pca, outliers.index)
    print('done')