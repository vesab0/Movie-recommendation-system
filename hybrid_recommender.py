import numpy as np
from content_knn import ContentKNN
from collaborative_knn import CollaborativeKNN
from distance_functions import cosine_distance, cosine_similarity
import vectorizer

class HybridRecommender:
    
    def __init__(self, content_knn, collaborative_knn, content_weight=0.85, collab_weight=0.15):
        self.content_knn = content_knn
        self.collaborative_knn = collaborative_knn
        self.content_weight = content_weight
        self.collab_weight = collab_weight
    
    def recommend(self, movie_id, k=10, content_k=20, collab_k=20):
        content_recs = self.content_knn.recommend(movie_id, k=content_k)
        collab_recs = self.collaborative_knn.recommend(movie_id, k=collab_k)
        
        # --- Normalize content scores ---
        content_scores = {}
        if content_recs:
            # Convert distance to similarity
            raw_sims = [(r['movie_id'], 1.0 - (r['distance'] / 2.0)) for r in content_recs]
            min_sim = min(s for _, s in raw_sims)
            max_sim = max(s for _, s in raw_sims)
            sim_range = max_sim - min_sim if max_sim > min_sim else 1.0
            
            for mid, sim in raw_sims:
                normalized = (sim - min_sim) / sim_range
                content_scores[mid] = normalized
        
        # --- Normalize collaborative scores ---
        collab_scores = {}
        if collab_recs:
            raw_sims = [(r['movie_id'], max(0.0, r['similarity'])) for r in collab_recs]
            min_sim = min(s for _, s in raw_sims)
            max_sim = max(s for _, s in raw_sims)
            sim_range = max_sim - min_sim if max_sim > min_sim else 1.0
            
            for mid, sim in raw_sims:
                normalized = (sim - min_sim) / sim_range
                collab_scores[mid] = normalized
        
        # --- Merge ---
        all_movie_ids = set(content_scores.keys()) | set(collab_scores.keys())
        
        final_scores = {}
        for mid in all_movie_ids:
            c_score = content_scores.get(mid, 0.0) * self.content_weight
            l_score = collab_scores.get(mid, 0.0) * self.collab_weight
            
            # Consensus boost: both paths agree
            if mid in content_scores and mid in collab_scores:
                consensus_bonus = 0.1 * (content_scores[mid] + collab_scores[mid]) / 2
            else:
                consensus_bonus = 0.0

            final_scores[mid] = c_score + l_score + consensus_bonus
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