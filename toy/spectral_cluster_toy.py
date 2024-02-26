import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Create a synthetic dataset with two clusters
X, y = make_blobs(n_samples=150, centers=2, random_state=42, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("Synthetic Dataset")
plt.show()