import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from utils import data_utils
from sklearn.preprocessing import scale, StandardScaler
from pandas.core.frame import DataFrame


def do_pca(data, labels):
    pca = decomposition.PCA(n_components=10)
    # Using the correlation matrix is equivalent to standardizing each of the variables (to mean 0 and standard deviation 1).
    # we want to use covariance matrix so don't scale the data
    # data = scale(data)
    pca.fit(data)
    # Variance explains
    va = pca.explained_variance_ratio_
    print(va)
    # Cumulative Variance explains
    va1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print(va1)
    '''
    plt.plot(va1)
    '''
    transformed_data = pca.transform(data)
    # data_utils.save_dataframe_to_csv(DataFrame(transformed_data), 'pca')

    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='green',
                cmap=plt.cm.nipy_spectral, s=50, marker='.', label='TSR data points')


def plot_nih(mrs):
    test_bi_pca_all = data_utils.get_test_transformed('nih', mrs)
    ot = test_bi_pca_all[test_bi_pca_all['label'] == -1]
    plt.scatter(ot[['pca_1']], ot[['pca_2']], linewidth=0, c='red', marker='^', label='testing outliers')
    noot = test_bi_pca_all[test_bi_pca_all['label'] != -1]
    plt.scatter(noot[['pca_1']], noot[['pca_2']], linewidth=0, c='blue', marker='X', label='testing normal data')


def plot_alias(mrs):
    test_bi_pca_all = data_utils.get_test_transformed('alias', mrs)
    ot = test_bi_pca_all[test_bi_pca_all['label'] == -1]
    plt.scatter(ot[['pca_1']], ot[['pca_2']], linewidth=0, c='red', marker='^')
    noot = test_bi_pca_all[test_bi_pca_all['label'] != -1]
    plt.scatter(noot[['pca_1']], noot[['pca_2']], linewidth=0, c='blue', marker='X')


def plot_fast(mrs):
    test_bi_pca_all = data_utils.get_test_transformed('fast', mrs)
    ot = test_bi_pca_all[test_bi_pca_all['label'] == -1]
    plt.scatter(ot[['pca_1']], ot[['pca_2']], linewidth=0,c='red', marker='^')
    noot = test_bi_pca_all[test_bi_pca_all['label'] != -1]
    plt.scatter(noot[['pca_1']], noot[['pca_2']], linewidth=0, c='blue', marker='X')


def plot_tnk(mrs):
    test_bi_pca_all = data_utils.get_test_transformed('tnk', mrs)
    ot = test_bi_pca_all[test_bi_pca_all['label'] == -1]
    plt.scatter(ot[['pca_1']], ot[['pca_2']], linewidth=0, c='red', marker='^')
    noot = test_bi_pca_all[test_bi_pca_all['label'] != -1]
    plt.scatter(noot[['pca_1']], noot[['pca_2']], linewidth=0, c='blue', marker='X')


if __name__ == '__main__':
    mrs = 5
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'all')
    # bi_df = bi_df.drop_duplicates()
    # mrs_df = mrs_df.loc[bi_df.index]
    do_pca(bi_df, mrs_df)
    plot_nih(mrs)
    plot_alias(mrs)
    plot_fast(mrs)
    plot_tnk(mrs)

    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]


    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title('mRS = '+ str(mrs))
    plt.show()