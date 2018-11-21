from sklearn.neighbors import DistanceMetric
import try2

X = [[1, 2], [3, 5]]
dist = DistanceMetric.get_metric('euclidean')
try2.csv_save([0, 1, 2, 3])
print(dist.pairwise(X))
