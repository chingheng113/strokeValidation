from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from utils import data_utils

id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr()
mrs_target = 3
bi_mrs = bi_df[mrs_df['discharged_mrs'] == mrs_target]
# a minimum minPts can be derived from the number of dimensions D in the data set, as minPts ≥ D + 1
minPts = bi_mrs.shape[1]+1
# The value for ε can then be chosen by using a k-distance graph, plotting the distance to the k = minPts-1
k = minPts-1
nbrs = NearestNeighbors(n_neighbors=k).fit(bi_mrs)
distances, indices = nbrs.kneighbors(bi_mrs)
distanceDec = sorted(distances[:,k-1], reverse=True)
plt.plot(list(range(1, bi_mrs.shape[0]+1)), distanceDec)
plt.show()

