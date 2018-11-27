from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from utils import data_utils

id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(5, 'is')
# a minimum minPts can be derived from the number of dimensions D in the data set, as minPts ≥ D + 1
minPts = bi_df.shape[1]+1
# The value for ε can then be chosen by using a k-distance graph, plotting the distance to the k = minPts-1
k = minPts-1
nbrs = NearestNeighbors(n_neighbors=k).fit(bi_df)
distances, indices = nbrs.kneighbors(bi_df)
distanceDec = sorted(distances[:,k-1], reverse=True)
plt.plot(list(range(1, bi_df.shape[0]+1)), distanceDec)
plt.show()

