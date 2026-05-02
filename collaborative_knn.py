from distance_functions import cosine_similarity, cosine_distance
from config import MIN_COMMON_USERS


class CollaborativeKNN:

    def __init__(self, ratings_by_movie, ratings_by_user, profiles, min_common_users=MIN_COMMON_USERS):
        self.ratings_by_movie = ratings_by_movie
        self.ratings_by_user = ratings_by_user
        self.profiles = profiles
        self.min_common_users = min_common_users
        
        self._movie_ratings_dict = {}
        for movie_id, ratings in ratings_by_movie.items():
            self._movie_ratings_dict[movie_id] = {user_id: rating for user_id, rating, _ in ratings}
    
    def _pearson_correlation(self, movie_id_1, movie_id_2):

        ratings_1 = self._movie_ratings_dict.get(movie_id_1, {})
        ratings_2 = self._movie_ratings_dict.get(movie_id_2, {})
        
        common_users = set(ratings_1.keys()) & set(ratings_2.keys())
        
        if len(common_users) < self.min_common_users:
            return None
        
        x = [ratings_1[u] for u in common_users]
        y = [ratings_2[u] for u in common_users]
        
        n = len(common_users)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x_sq = sum(v * v for v in x)
        sum_y_sq = sum(v * v for v in y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        
        numerator = sum_xy - (sum_x * sum_y / n)
        denominator = ((sum_x_sq - sum_x**2 / n) * (sum_y_sq - sum_y**2 / n)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def find_neighbors(self, movie_id, k=10, exclude_same=True):

        if movie_id not in self.ratings_by_movie:
            raise ValueError(f"Movie ID {movie_id} not found in ratings")
        
        similarities = []
        
        for other_id in self.ratings_by_movie.keys():
            if exclude_same and other_id == movie_id:
                continue
            
            sim = self._pearson_correlation(movie_id, other_id)
            
            if sim is None or sim <= 0:
                continue
            if other_id not in self.profiles or not self.profiles[other_id].get('title'):
                continue
            
            if sim > 0:
                similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def recommend(self, movie_id, k=10):

        neighbors = self.find_neighbors(movie_id, k=k)
        
        recommendations = []
        for neighbor_id, sim in neighbors:
            profile = self.profiles.get(neighbor_id, {})
            
            recommendations.append({
                'movie_id': neighbor_id,
                'title': profile.get('title', 'Unknown'),
                'similarity': sim,
                'genres': [g['name'] for g in profile.get('genres', [])],
                'overview': profile.get('overview', '')[:100] + '...' if profile.get('overview') else '',
            })
        
        return recommendations


if __name__ == "__main__":
    import cacher as cache_manager
    
    print("Loading data...")
    profiles, ratings_by_movie, ratings_by_user, vectors = cache_manager.load_everything()
    
    print("\nInitializing Collaborative KNN...")
    knn = CollaborativeKNN(ratings_by_movie, ratings_by_user, profiles, min_common_users=40)
    
    test_movie_id = 862
    test_movie = profiles[test_movie_id]
    
    print(f"\n{'=' * 60}")
    print(f"COLLABORATIVE RECOMMENDATIONS")
    print(f"{'=' * 60}")
    print(f"Input movie: {test_movie['title']} ({test_movie_id})")
    print(f"Rating count: {test_movie['rating_count']}")
    print(f"\nTop 10 movies with similar rating patterns:\n")
    
    recs = knn.recommend(test_movie_id, k=10)
    
    for i, rec in enumerate(recs, 1):
        sim_pct = rec['similarity'] * 100
        print(f"{i:2d}. {rec['title']:<35} | Sim: {sim_pct:5.1f}% | {', '.join(rec['genres'][:3])}")
    
    print(f"\n{'=' * 60}")
    print("Collaborative KNN test complete.")