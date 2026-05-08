from distance_functions import cosine_distance, cosine_similarity
import numpy as np


class ContentKNN:
    def __init__(self, vectors, profiles, distance_fn=cosine_distance, kmeans=None, search_radius=0):
        self.vectors = vectors
        self.profiles = profiles
        self.distance_fn = distance_fn
        self.kmeans = kmeans
        self.search_radius = search_radius
        
        self._nonzero_indices = {}
        for movie_id, vec in vectors.items():
            self._nonzero_indices[movie_id] = set(np.nonzero(vec)[0])
    
    def _get_search_space(self, movie_id):
        """Determine which movies to compare against."""
        if self.kmeans is None:
            return self.vectors.items()
        
        query_cluster = self.kmeans.labels.get(movie_id)
        if query_cluster is None:
            return self.vectors.items()
        
        clusters_to_search = {query_cluster}
        
        if self.search_radius > 0:
            query_centroid = self.kmeans.centroids[query_cluster]
            distances = np.sum((self.kmeans.centroids - query_centroid) ** 2, axis=1)
            nearest = np.argsort(distances)[1:self.search_radius + 1]
            clusters_to_search.update(nearest)
        
        for other_id, vec in self.vectors.items():
            if self.kmeans.labels.get(other_id) in clusters_to_search:
                yield other_id, vec
    
    def find_neighbors(self, movie_id, k=10, exclude_same=True):
        if movie_id not in self.vectors:
            raise ValueError(f"Movie ID {movie_id} not found in vectors")
        
        input_vector = self.vectors[movie_id]
        input_nonzero = self._nonzero_indices[movie_id]
        distances = []
        
        for other_id, other_vector in self._get_search_space(movie_id):
            if exclude_same and other_id == movie_id:
                continue
            
            other_nonzero = self._nonzero_indices[other_id]
            if not (input_nonzero & other_nonzero):
                continue
            
            dist = self.distance_fn(input_vector, other_vector)
            distances.append((other_id, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def recommend(self, movie_id, k=10):
        neighbors = self.find_neighbors(movie_id, k=k)
        
        recommendations = []
        for neighbor_id, dist in neighbors:
            profile = self.profiles.get(neighbor_id, {})
            similarity = 1.0 - dist if self.distance_fn == cosine_distance else None
            
            recommendations.append({
                'movie_id': neighbor_id,
                'title': profile.get('title', 'Unknown'),
                'distance': dist,
                'similarity': similarity,
                'genres': [g['name'] for g in profile.get('genres', [])],
                'overview': profile.get('overview', '')[:100] + '...' if profile.get('overview') else '',
            })
        
        return recommendations


if __name__ == "__main__":
    import cacher as cache_manager
    from clusterer import get_clusters
    
    print("Loading data and vectors...")
    profiles, ratings_by_movie, ratings_by_user, vectors = cache_manager.load_everything()
    
    print("\nBuilding clusters...")
    kmeans = get_clusters(vectors, n_clusters=50)
    
    print("\nInitializing Content KNN with clustering...")
    knn = ContentKNN(vectors, profiles, distance_fn=cosine_distance, kmeans=kmeans, search_radius=1)
    
    test_movie_id = 862
    test_movie = profiles[test_movie_id]
    
    print(f"\n{'=' * 60}")
    print(f"CONTENT-BASED RECOMMENDATIONS")
    print(f"{'=' * 60}")
    print(f"Input movie: {test_movie['title']} ({test_movie_id})")
    print(f"Genres: {[g['name'] for g in test_movie['genres']]}")
    print(f"Cluster: {kmeans.labels.get(test_movie_id, 'N/A')}")
    print(f"\nTop 10 similar movies:\n")
    
    recs = knn.recommend(test_movie_id, k=10)
    
    for i, rec in enumerate(recs, 1):
        sim_pct = rec['similarity'] * 100 if rec['similarity'] is not None else 0
        cluster = kmeans.labels.get(rec['movie_id'], 'N/A')
        print(f"{i:2d}. {rec['title']:<35} | Sim: {sim_pct:5.1f}% | Cluster: {cluster} | {', '.join(rec['genres'][:3])}")
    
    print(f"\n{'=' * 60}")
    print("Content KNN test complete.")