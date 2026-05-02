from hybrid_recommender import HybridRecommender
from content_knn import ContentKNN
from collaborative_knn import CollaborativeKNN
from multi_movie_recommender import MultiMovieRecommender
import cacher as cache_manager
from config import MIN_COMMON_USERS, CONTENT_WEIGHT, COLLAB_WEIGHT, RECOMMENDATIONS_K, SEARCH_MAX_RESULTS


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
    return [(m[0], m[1], m[2]) for m in matches[:SEARCH_MAX_RESULTS]]


def pick_movie(profiles):
    while True:
        query = input("Movie title (or 'done' to finish, 'cancel' to abort): ").strip()
        
        if query.lower() in ('done', 'd'):
            return 'done', None
        
        if query.lower() in ('cancel', 'c', 'abort'):
            return 'cancel', None
        
        if not query:
            continue
        
        matches = find_movie_by_title(profiles, query)
        
        if not matches:
            print(f"  No movies found matching '{query}'. Try again.")
            continue
        
        if len(matches) == 1:
            mid, title, year = matches[0]
            print(f"  -> Selected: {title} ({year})")
            return mid, title
        
        print(f"\n  Found {len(matches)} matches:")
        for i, (mid, title, year) in enumerate(matches, 1):
            print(f"    {i}. {title} ({year})")
        
        choice = input("\n  Pick number (or 0 to search again): ").strip()
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(matches):
                print("  Cancelled, search again.")
                continue
            mid, title, year = matches[idx]
            print(f"  -> Selected: {title} ({year})")
            return mid, title
        except ValueError:
            print("  Invalid input, search again.")
            continue


def main():
    print("Loading data and building recommenders...")
    profiles, ratings_by_movie, ratings_by_user, vectors = cache_manager.load_everything()
    
    content_knn = ContentKNN(vectors, profiles)
    collab_knn = CollaborativeKNN(ratings_by_movie, ratings_by_user, profiles, min_common_users=MIN_COMMON_USERS)
    hybrid = HybridRecommender(content_knn, collab_knn, content_weight=CONTENT_WEIGHT, collab_weight=COLLAB_WEIGHT)
    multi = MultiMovieRecommender(hybrid)
    
    print(f"\nLoaded {len(profiles)} movies. Ready.\n")
    
    while True:
        # Collect movies from user
        selected_movies = []
        
        print("Add movies you like (type 'done' when finished, 'cancel' to start over):")
        
        while True:
            if not selected_movies:
                prompt = "First movie: "
            else:
                prompt = f"Next movie ({len(selected_movies)} so far): "
            
            print(prompt, end="")
            mid, title = pick_movie(profiles)
            
            if mid == 'cancel':
                selected_movies = []
                print("Cleared. Starting over.\n")
                continue
            
            if mid == 'done':
                break
            
            # Avoid duplicates
            if mid in [m[0] for m in selected_movies]:
                print(f"  '{title}' already in list. Skipping.")
                continue
            
            selected_movies.append((mid, title))
            print(f"  Added '{title}' to your list.\n")
        
        if not selected_movies:
            print("No movies selected. Try again or type 'quit' to exit.\n")
            continue
        
        # Show collected movies
        print(f"\n{'=' * 60}")
        print(f"YOUR MOVIES ({len(selected_movies)}):")
        for i, (mid, title) in enumerate(selected_movies, 1):
            print(f"  {i}. {title}")
        
        # Get recommendations
        movie_ids = [mid for mid, _ in selected_movies]
        
        print(f"\n{'=' * 60}")
        print(f"RECOMMENDATIONS")
        print(f"{'=' * 60}")
        
        recs = multi.recommend(movie_ids, k=RECOMMENDATIONS_K)
        
        if not recs:
            print("No recommendations found.")
        else:
            for i, rec in enumerate(recs, 1):
                print(f"{i:2d}. {rec['title']:<35} | Score: {rec['final_score']:.3f} | {', '.join(rec['genres'][:3])}")
        
        print(f"{'=' * 60}\n")
        
        # Ask to continue or quit
        again = input("Search again? (y/n): ").strip().lower()
        if again not in ('y', 'yes'):
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()