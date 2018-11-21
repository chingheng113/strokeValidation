import numpy as np
from utils import data_utils, figure_plot
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import DistanceMetric
import ddbscan
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def dbscan_validation(X):
    mSample = round(X.shape[0]/10, 0)
    # cosine
    db = DBSCAN(eps=4, min_samples=mSample, metric='euclidean').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    return core_samples_mask, n_clusters_, labels


def ddbscan_validation(X):
    # https://github.com/cloudwalkio/ddbscan
    mSample = round(X.shape[0]/10, 0)
    scan = ddbscan.DDBSCAN(10, mSample)
    # Add points to model
    for index, row in X.iterrows():
        scan.add_point(point=row.tolist(), count=1, desc="")

    # Compute clusters
    scan.compute()
    print(X.shape[0])
    print('Clusters found and its members points index:')
    cluster_number = 0
    for cluster in scan.clusters:
        print ('=== Cluster %d ===' % cluster_number)
        print('Size:', len(cluster))
        print ('Cluster points index: %s' % list(cluster))
        cluster_number += 1

    # print ('\nCluster assigned to each point:')
    # for i in range(len(scan.points)):
    #     print ('=== Point: %s ===' % scan.points[i])
    #     print ('Cluster: %2d' % scan.points_data[i].cluster,)
    #     # If a point cluster is -1, it's an anomaly
    #     if scan.points_data[i].cluster == -1:
    #         print ('\t <== Anomaly found!')
    #     else:
    #         print()


if __name__ == '__main__':
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr()
    mrs_target = 0
    # bi_mrs = bi_df[mrs_df['discharged_mrs'] == mrs_target]
    # bi_mrs = data_utils.scale(bi_mrs)
    core_samples_mask, n_clusters, labels = dbscan_validation(bi_df)
    figure_plot.dbscan_plot(bi_df.values, core_samples_mask, n_clusters, labels)
    print('done')