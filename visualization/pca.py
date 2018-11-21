import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from utils import data_utils


def do_pca(data, labels):
    pca = decomposition.PCA(n_components=2)
    pca.fit(data)
    print(pca.explained_variance_ratio_)
    first_pca = pca.components_[0]
    second_pca = pca.components_[1]
    transformed_data = pca.transform(data)
    # for ii, jj in zip(transformed_data, data):
    #     plt.scatter(first_pca[0]*ii[0], first_pca[1]*ii[0], color='r')
    #     plt.scatter(second_pca[0]*ii[1], second_pca[1]*ii[1], color='b')
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels.values.ravel(),
                cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.show()


if __name__ == '__main__':
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(5)
    # bi_df = data_utils.scale(bi_df)
    do_pca(nih_df, mrs_df)