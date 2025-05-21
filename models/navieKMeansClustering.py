import numpy as np
import random

class naiveKMeansClustering:
    def __init__(self, data: np.ndarray, maxCluster: int):
        self.data = data
        self.maxCluster = maxCluster
        self.centroids = np.array([random.choice(self.data) for _ in range(maxCluster)])
    
    def train(self, epochs: int, log:bool = False):
        for _ in range(epochs):
            # Assign each data point to the nearest centroid
            clusters = {i: [] for i in range(self.maxCluster)}
            for point in self.data:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(point)
            
            # Update centroids
            for i in range(self.maxCluster):
                if clusters[i]:  # Avoid division by zero
                    self.centroids[i] = np.mean(clusters[i], axis=0)
            
            if i % (epochs // 10) == 0 and i != 0 and log:
                print("-", end="", flush=True)
    
    def predict(self, point: np.ndarray):
        distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
        return self.centroids[np.argmin(distances)]
    
    def get_centroids(self):
        return self.centroids