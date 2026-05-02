from hybrid_recommender import HybridRecommender
from content_knn import ContentKNN
from collaborative_knn import CollaborativeKNN
import cacher as cache_manager
from config import RRF_K, PER_MOVIE_K, RECOMMENDATIONS_K


class MultiMovieRecommender:

    def __init__(self, hybrid_recommender):
        self.hybrid = hybrid_recommender

    _RRF_K = RRF_K

    def recommend(self, movie_ids, k=RECOMMENDATIONS_K, per_movie_k=PER_MOVIE_K):
        final_scores = {}

        for mid in movie_ids:
            if mid not in self.hybrid.content_knn.vectors:
                continue

            try:
                recs = self.hybrid.recommend(mid, k=per_movie_k)
            except Exception as e:
                print(f"Error recommending for movie ID {mid}: {e}")
                continue

            for rank, rec in enumerate(recs):
                cid = rec['movie_id']
                final_scores[cid] = final_scores.get(cid, 0.0) + 1.0 / (self._RRF_K + rank + 1)
        
        for mid in movie_ids:
            final_scores.pop(mid, None)
        
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:k]
        
        recommendations = []
        for cid, score in top_k:
            profile = self.hybrid.content_knn.profiles.get(cid, {})
            recommendations.append({
                'movie_id': cid,
                'title': profile.get('title', 'Unknown'),
                'final_score': score,
                'genres': [g['name'] for g in profile.get('genres', [])],
            })
        
        return recommendations


if __name__ == "__main__":
    print("Loading data...")
    profiles, ratings_by_movie, ratings_by_user, vectors = cache_manager.load_everything()
    
    content_knn = ContentKNN(vectors, profiles)
    collab_knn = CollaborativeKNN(ratings_by_movie, ratings_by_user, profiles, min_common_users=40)
    hybrid = HybridRecommender(content_knn, collab_knn)
    
    multi = MultiMovieRecommender(hybrid)
    
    liked_movies = [862, 13, 364] 
    
    print(f"\n{'=' * 60}")
    print(f"MULTI-MOVIE RECOMMENDATIONS")
    print(f"{'=' * 60}")
    print(f"Input movies: {[profiles.get(mid, {}).get('title', 'Unknown') for mid in liked_movies]}")
    print(f"\nTop 10 recommendations:\n")
    
    recs = multi.recommend(liked_movies, k=10)
    
    for i, rec in enumerate(recs, 1):
        print(f"{i:2d}. {rec['title']:<35} | Score: {rec['final_score']:.3f} | {', '.join(rec['genres'][:3])}")
    
    print(f"\n{'=' * 60}")