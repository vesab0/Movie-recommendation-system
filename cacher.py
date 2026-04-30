import pickle
import os

import loader
from vectorizer import ContentVectorizer


CACHE_FILENAME = "cached-data/movie_cache.pkl"
VECTORS_CACHE_FILENAME = "cached-data/vectors_cache.pkl"


def get_cache_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def save_cache(data, cache_path):
    with open(cache_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Cache saved to {cache_path}")


def load_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as file:
        data = pickle.load(file)
    print(f"Cache loaded from {cache_path}")
    return data


def clear_cache():
    cache_path = get_cache_path(CACHE_FILENAME)
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Cache cleared: {cache_path}")
    else:
        print(f"No cache found at {cache_path}")


def clear_vectors():
    cache_path = get_cache_path(VECTORS_CACHE_FILENAME)
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Vectors cache cleared: {cache_path}")
    else:
        print(f"No vectors cache found at {cache_path}")


def clear_all():
    clear_cache()
    clear_vectors()


def load_all_data_with_cache(
    credits_path="data/credits.csv",
    keywords_path="data/keywords.csv",
    links_path="data/links.csv",
    movies_path="data/movies_metadata.csv",
    ratings_path="data/ratings.csv",
    force_rebuild=False
):
    """
    Loads raw data from cache or CSVs.
    Returns: profiles, ratings_by_movie, ratings_by_user
    """
    cache_path = get_cache_path(CACHE_FILENAME)
    
    if not force_rebuild:
        cached = load_cache(cache_path)
        if cached is not None:
            profiles, ratings_by_movie, ratings_by_user = cached
            print(f"  -> Loaded {len(profiles)} profiles from cache")
            return profiles, ratings_by_movie, ratings_by_user
    
    print("Cache miss. Building from CSV files...\n")
    profiles, ratings_by_movie, ratings_by_user = loader.load_all_data(
        credits_path, keywords_path, links_path, movies_path, ratings_path
    )
    
    save_cache((profiles, ratings_by_movie, ratings_by_user), cache_path)
    return profiles, ratings_by_movie, ratings_by_user


def get_vectors_with_cache(profiles, force_rebuild=False):
    """
    Loads vectors from cache, or builds them from profiles.
    Returns: {movieId: vector}
    """
    cache_path = get_cache_path(VECTORS_CACHE_FILENAME)
    
    if not force_rebuild:
        vectors = load_cache(cache_path)
        if vectors is not None:
            print(f"  -> Loaded {len(vectors)} vectors from cache")
            return vectors
    
    print("Vector cache miss. Building vectors...\n")
    
    vectorizer = ContentVectorizer()
    vectorizer.fit(profiles)
    vectors = vectorizer.vectorize_all(profiles)
    
    save_cache(vectors, cache_path)
    return vectors


def load_everything(
    credits_path="data/credits.csv",
    keywords_path="data/keywords.csv",
    links_path="data/links.csv",
    movies_path="data/movies_metadata.csv",
    ratings_path="data/ratings.csv",
    force_rebuild=False
):
    profiles, ratings_by_movie, ratings_by_user = load_all_data_with_cache(
        credits_path, keywords_path, links_path, movies_path, ratings_path,
        force_rebuild=force_rebuild
    )
    
    vectors = get_vectors_with_cache(profiles, force_rebuild=force_rebuild)
    
    return profiles, ratings_by_movie, ratings_by_user, vectors


if __name__ == "__main__":
    import sys
    
    force_rebuild = "--rebuild" in sys.argv
    
    if force_rebuild:
        print("Forcing rebuild from CSVs and vectors...\n")
        clear_all()
    
    profiles, ratings_by_movie, ratings_by_user, vectors = load_everything(
        force_rebuild=force_rebuild
    )
    
    print(f"\nLoaded {len(profiles)} movie profiles")
    print(f"Ratings by movie entries: {len(ratings_by_movie)}")
    print(f"Ratings by user entries: {len(ratings_by_user)}")
    print(f"Vectors ready: {len(vectors)}")
    print(f"Vector length: {len(next(iter(vectors.values())))}")