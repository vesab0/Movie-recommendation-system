import numpy as np


class KMeans:
    def __init__(self, n_clusters=50, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = {}
    
    def _initialize_centroids(self, vectors_matrix):
        n_samples = vectors_matrix.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return vectors_matrix[indices].copy()
    
    def _assign_clusters(self, vectors_matrix):
        sq_norms_X = np.sum(vectors_matrix ** 2, axis=1, keepdims=True)  # (n, 1)
        sq_norms_C = np.sum(self.centroids ** 2, axis=1)                 # (k,)
        dot = vectors_matrix @ self.centroids.T                          # (n, k)
        distances = sq_norms_X + sq_norms_C - 2 * dot                   # (n, k)

        labels = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)

        return labels, min_distances
    
    def _update_centroids(self, vectors_matrix, labels):
        new_centroids = np.zeros_like(self.centroids)
        
        for i in range(self.n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                new_centroids[i] = np.mean(vectors_matrix[mask], axis=0)
            else:
                new_centroids[i] = vectors_matrix[np.random.choice(vectors_matrix.shape[0])]
        
        return new_centroids
    
    def fit(self, vectors_dict):
        print(f"Fitting K-Means with K={self.n_clusters}...")
        
        movie_ids = list(vectors_dict.keys())
        vectors_matrix = np.stack(list(vectors_dict.values()))
        n_samples, n_features = vectors_matrix.shape
        
        print(f"  Data: {n_samples} movies, {n_features} features")
        
        self.centroids = self._initialize_centroids(vectors_matrix)
        
        for iteration in range(self.max_iters):
            labels_array, distances = self._assign_clusters(vectors_matrix)
            new_centroids = self._update_centroids(vectors_matrix, labels_array)
            
            centroid_shift = np.max(np.sum((new_centroids - self.centroids) ** 2, axis=1))
            self.centroids = new_centroids
            
            if iteration % 10 == 0:
                inertia = np.sum(distances)
                print(f"  Iteration {iteration}: inertia={inertia:.2f}, shift={centroid_shift:.6f}")
            
            if centroid_shift < self.tol:
                print(f"  Converged at iteration {iteration}")
                break
        
        for i, mid in enumerate(movie_ids):
            self.labels[mid] = int(labels_array[i])
        
        cluster_sizes = {}
        for c in self.labels.values():
            cluster_sizes[c] = cluster_sizes.get(c, 0) + 1
        
        print(f"  Cluster sizes: min={min(cluster_sizes.values())}, "
              f"max={max(cluster_sizes.values())}, "
              f"avg={np.mean(list(cluster_sizes.values())):.1f}")
        
        return self.labels
    
    def predict(self, vector):
        distances = np.sum((self.centroids - vector) ** 2, axis=1)
        return int(np.argmin(distances))
    
    def get_cluster_movies(self, cluster_id):
        return [mid for mid, cid in self.labels.items() if cid == cluster_id]