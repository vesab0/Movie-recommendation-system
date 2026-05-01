from distance_functions import cosine_distance, cosine_similarity
import numpy as np

class ContentKNN:
    def __init__(self, vectors, profiles, distance_fn=cosine_distance):
        self.vectors = vectors
        self.profiles = profiles
        self.distance_fn = distance_fn
        
        self._nonzero_indices = {}
        for movie_id, vec in vectors.items():
            self._nonzero_indices[movie_id] = set(np.nonzero(vec)[0])
    
    def find_neighbors(self, movie_id, k=10, exclude_same=True):
        if movie_id not in self.vectors:
            raise ValueError(f"Movie ID {movie_id} not found in vectors")
        
        input_vector = self.vectors[movie_id]
        input_nonzero = self._nonzero_indices[movie_id]
        distances = []
        
        for other_id, other_vector in self.vectors.items():
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
    
    print("Loading data and vectors...")
    profiles, ratings_by_movie, ratings_by_user, vectors = cache_manager.load_everything()
    
    print("\nInitializing Content KNN...")
    knn = ContentKNN(vectors, profiles, distance_fn=cosine_distance)
    
    test_movie_id = 862
    test_movie = profiles[test_movie_id]
    
    print(f"\n{'=' * 60}")
    print(f"CONTENT-BASED RECOMMENDATIONS")
    print(f"{'=' * 60}")
    print(f"Input movie: {test_movie['title']} ({test_movie_id})")
    print(f"Genres: {[g['name'] for g in test_movie['genres']]}")
    print(f"Collection: {test_movie['belongs_to_collection']['name'] if test_movie['belongs_to_collection'] else 'None'}")
    print(f"\nTop 10 similar movies:\n")
    
    recs = knn.recommend(test_movie_id, k=10)
    
    for i, rec in enumerate(recs, 1):
        sim_pct = rec['similarity'] * 100 if rec['similarity'] is not None else 0
        print(f"{i:2d}. {rec['title']:<35} | Sim: {sim_pct:5.1f}% | {', '.join(rec['genres'][:3])}")
    
    print(f"\n{'=' * 60}")
    print("Content KNN test complete.")