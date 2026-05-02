import numpy as np
from content_knn import ContentKNN
from collaborative_knn import CollaborativeKNN
from distance_functions import cosine_distance, cosine_similarity
import vectorizer
from config import (
    CONTENT_WEIGHT, COLLAB_WEIGHT,
    HYBRID_CONTENT_K, HYBRID_COLLAB_K,
    COLLAB_CONFIDENCE_SCALE, CONSENSUS_BONUS_WEIGHT,
)

class HybridRecommender:

    def __init__(self, content_knn, collaborative_knn, content_weight=CONTENT_WEIGHT, collab_weight=COLLAB_WEIGHT):
        self.content_knn = content_knn
        self.collaborative_knn = collaborative_knn
        self.content_weight = content_weight
        self.collab_weight = collab_weight

    def recommend(self, movie_id, k=10, content_k=HYBRID_CONTENT_K, collab_k=HYBRID_COLLAB_K):
        content_recs = self.content_knn.recommend(movie_id, k=content_k)
        collab_recs = self.collaborative_knn.recommend(movie_id, k=collab_k)

        # Raw similarities — cosine distance in [0,1] for non-negative vectors,
        # so similarity = 1 - distance is already in [0,1]. No normalization needed.
        content_scores = {r['movie_id']: 1.0 - r['distance'] for r in content_recs}
        collab_scores = {r['movie_id']: max(0.0, r['similarity']) for r in collab_recs}

        # Adaptive collab weight: scale down for movies with few ratings
        rating_count = len(self.collaborative_knn.ratings_by_movie.get(movie_id, {}))
        confidence = min(1.0, rating_count / COLLAB_CONFIDENCE_SCALE)
        eff_collab_w = self.collab_weight * confidence
        eff_content_w = 1.0 - eff_collab_w

        # --- Merge ---
        all_movie_ids = set(content_scores.keys()) | set(collab_scores.keys())

        final_scores = {}
        for mid in all_movie_ids:
            c_score = content_scores.get(mid, 0.0) * eff_content_w
            l_score = collab_scores.get(mid, 0.0) * eff_collab_w

            if mid in content_scores and mid in collab_scores:
                consensus_bonus = CONSENSUS_BONUS_WEIGHT * (content_scores[mid] + collab_scores[mid]) / 2
            else:
                consensus_bonus = 0.0

            final_scores[mid] = c_score + l_score + consensus_bonus
        
        # --- Rank and return ---
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:k]
        
        recommendations = []
        for mid, score in top_k:
            profile = self.content_knn.profiles.get(mid, {})
            from_content = mid in content_scores
            from_collab = mid in collab_scores
            
            recommendations.append({
                'movie_id': mid,
                'title': profile.get('title', 'Unknown'),
                'final_score': score,
                'from_content': from_content,
                'from_collab': from_collab,
                'genres': [g['name'] for g in profile.get('genres', [])],
            })
        
        return recommendations
    
if __name__ == "__main__":
    import cacher as cache_manager
    from content_knn import ContentKNN
    from collaborative_knn import CollaborativeKNN

    print("Loading data...")
    profiles, ratings_by_movie, ratings_by_user, vectors = cache_manager.load_everything()
    
    print("\nInitializing recommenders...")
    content_knn = ContentKNN(vectors, profiles)
    collab_knn = CollaborativeKNN(ratings_by_movie, ratings_by_user, profiles, min_common_users=40)

    hybrid = HybridRecommender(content_knn, collab_knn, content_weight=0.7, collab_weight=0.3)

    test_movie_id = 862
    test_movie = profiles[test_movie_id]


    
    print(f"\n{'=' * 60}")
    print(f"HYBRID RECOMMENDATIONS")
    print(f"{'=' * 60}")
    print(f"Input movie: {test_movie['title']} ({test_movie_id})")
    print(f"Genres: {[g['name'] for g in test_movie['genres']]}")
    print(f"Rating count: {test_movie['rating_count']}")
    print(f"\nTop 10 hybrid recommendations:\n")
    
    recs = hybrid.recommend(test_movie_id, k=10)
    
    for i, rec in enumerate(recs, 1):
        sources = []
        if rec['from_content']:
            sources.append("C")
        if rec['from_collab']:
            sources.append("R")
        
        source_str = "+".join(sources) if len(sources) == 2 else sources[0]
        
        print(f"{i:2d}. {rec['title']:<35} | Score: {rec['final_score']:.3f} | [{source_str}] | {', '.join(rec['genres'][:3])}")
    
    print(f"\n{'=' * 60}")
    print("Hybrid recommender test complete.")