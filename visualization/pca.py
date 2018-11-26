import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from utils import data_utils
from sklearn.preprocessing import scale


def do_pca(data, labels):
    pca = decomposition.PCA(n_components=10)
    data = scale(data)
    pca.fit(data)
    va = pca.explained_variance_ratio_
    print(va)
    #Cumulative Variance explains
    va1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print(va1)
    # plt.plot(va1)

    transformed_data = pca.transform(data)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels.values.ravel(),
                cmap=plt.cm.nipy_spectral, edgecolor='k', alpha=0.5)
    plt.xlabel('component 1')
    plt.ylabel('ylabelcomponent 2')
    plt.show()


if __name__ == '__main__':
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(4, 'is')
    # bi_df = data_utils.scale(bi_df)
    do_pca(bi_df, mrs_df)