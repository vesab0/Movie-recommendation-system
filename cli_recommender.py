from hybrid_recommender import HybridRecommender
from content_knn import ContentKNN
from collaborative_knn import CollaborativeKNN
import cacher as cache_manager


def find_movie_by_title(profiles, query):

    query_lower = query.lower().strip()
    matches = []
    
    for movie_id, profile in profiles.items():
        title = profile.get('title', '')
        original_title = profile.get('original_title', '')
        
        if not title:
            continue
        
        if query_lower == title.lower():
            return [(movie_id, title, profile.get('release_date', 'N/A'))]
        
        if query_lower in title.lower() or query_lower in original_title.lower():
            matches.append((
                movie_id,
                title,
                profile.get('release_date', 'N/A'),
                profile.get('vote_average', 0)
            ))
    
    matches.sort(key=lambda x: x[3], reverse=True)
    return [(m[0], m[1], m[2]) for m in matches[:10]]


def main():
    print("Loading data and building recommenders...")
    profiles, ratings_by_movie, ratings_by_user, vectors = cache_manager.load_everything()
    
    content_knn = ContentKNN(vectors, profiles)
    collab_knn = CollaborativeKNN(ratings_by_movie, ratings_by_user, profiles, min_common_users=40)
    hybrid = HybridRecommender(content_knn, collab_knn, content_weight=0.85, collab_weight=0.15)
    
    print(f"\nLoaded {len(profiles)} movies. Ready for queries.\n")
    
    while True:
        query = input("Enter movie title (or 'quit' to exit): ").strip()
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        matches = find_movie_by_title(profiles, query)
        
        if not matches:
            print(f"No movies found matching '{query}'. Try again.\n")
            continue
        
        selected_id = None
        if len(matches) == 1:
            selected_id, title, year = matches[0]
            print(f"Found: {title} ({year})")
        else:
            print(f"\nFound {len(matches)} matches:")
            for i, (mid, title, year) in enumerate(matches, 1):
                print(f"  {i}. {title} ({year}) [ID: {mid}]")
            
            choice = input("\nPick number (or 0 to search again): ").strip()
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(matches):
                    print("Cancelled.\n")
                    continue
                selected_id, title, year = matches[idx]
            except ValueError:
                print("Invalid input.\n")
                continue
        
        print(f"\n{'=' * 60}")
        print(f"Recommendations for: {title}")
        print(f"{'=' * 60}")
        
        recs = hybrid.recommend(selected_id, k=10)
        
        if not recs:
            print("No recommendations found.")
            continue
        
        for i, rec in enumerate(recs, 1):
            sources = []
            if rec['from_content']:
                sources.append("C")
            if rec['from_collab']:
                sources.append("R")
            source_str = "+".join(sources) if len(sources) == 2 else sources[0]
            
            print(f"{i:2d}. {rec['title']:<35} | Score: {rec['final_score']:.3f} | [{source_str}] | {', '.join(rec['genres'][:3])}")
        
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()