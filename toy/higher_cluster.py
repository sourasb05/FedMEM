import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import rbf_kernel


# Generate sample data
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Spectral Clustering to obtain similarity matrix and reduce dimensionality
n_clusters = 2  # initial number of clusters for spectral clustering
sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
sc.fit(X)
labels = sc.labels_

# Use RBF kernel to construct similarity matrix
similarity_matrix = rbf_kernel(X, gamma=1.0)

# Hierarchical Clustering on the similarity matrix
Z = linkage(similarity_matrix, 'ward')
print(Z)
# Plotting the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
