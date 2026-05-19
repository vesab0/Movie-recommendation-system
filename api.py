"""
FastAPI wrapper for the hybrid movie recommender.
Run with: uvicorn api:app --port 8001 --reload

Accepts TMDB IDs (the public IDs used by cinema apps) and maps them
to internal MovieLens IDs via the links dataset built into profiles.
"""

import sys
import os
from threading import Lock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import numpy as np

import cacher
from config import LINKS_PATH, MOVIES_PATH, CLUSTER_SEARCH_RADIUS
import loader
from content_knn import ContentKNN
from collaborative_knn import CollaborativeKNN
from hybrid_recommender import HybridRecommender
from multi_movie_recommender import MultiMovieRecommender
from vectorizer import ContentVectorizer

app = FastAPI(title="Cinema Movie Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5000",
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _load_catalog() -> dict[int, dict]:
    print("Loading lightweight movie catalog...")
    movies = loader.load_movies_metadata(MOVIES_PATH)
    links = loader.load_links(LINKS_PATH)

    tmdb_by_movie_id = {
        row["movieId"]: row.get("tmdbId")
        for row in links
        if row.get("movieId") is not None
    }

    catalog: dict[int, dict] = {}
    for movie in movies:
        movie_id = movie.get("id")
        if not movie_id or not movie.get("title"):
            continue

        catalog[movie_id] = {
            "title": movie.get("title", "Unknown"),
            "genres": movie.get("genres", []),
            "release_date": movie.get("release_date", ""),
            "vote_count": movie.get("vote_count", 0),
            "vote_average": movie.get("vote_average", 0),
            "poster_path": movie.get("poster_path") or "",
            "tmdbId": tmdb_by_movie_id.get(movie_id),
        }

    print(f"Catalog ready. {len(catalog)} movies loaded.")
    return catalog


profiles = _load_catalog()
ratings_by_movie: dict[int, list] = {}
ratings_by_user: dict[int, list] = {}
vectors: dict[int, list] = {}
kmeans = None
content_knn = None
collab_knn = None
hybrid = None
multi = None
vectorizer: Optional[ContentVectorizer] = None
_recommender_lock = Lock()


def _ensure_recommenders():
    global ratings_by_movie, ratings_by_user, vectors, kmeans, content_knn, collab_knn, hybrid, multi, vectorizer

    if hybrid is not None and multi is not None:
        return

    with _recommender_lock:
        if hybrid is not None and multi is not None:
            return

        print("Loading full recommender model on demand...")
        profiles_full, ratings_by_movie_full, ratings_by_user_full, vectors_full, kmeans_full = cacher.load_everything()

        ratings_by_movie = ratings_by_movie_full
        ratings_by_user = ratings_by_user_full
        vectors = vectors_full
        kmeans = kmeans_full

        # Rebuild vectorizer vocabulary from profiles so we can inject new movies later.
        # fit() only builds dicts — it's fast and produces the exact same vocabulary as
        # the one used when vectors_full was originally created.
        _vect = ContentVectorizer()
        _vect.fit(profiles_full)
        vectorizer = _vect

        content_knn = ContentKNN(vectors_full, profiles_full, kmeans=kmeans_full, search_radius=CLUSTER_SEARCH_RADIUS)
        collab_knn = CollaborativeKNN(ratings_by_movie_full, ratings_by_user_full, profiles_full)
        hybrid = HybridRecommender(content_knn, collab_knn)
        multi = MultiMovieRecommender(hybrid)

        for movie_id, profile in profiles_full.items():
            profiles[movie_id] = {
                "title": profile.get("title", "Unknown"),
                "genres": profile.get("genres", []),
                "release_date": profile.get("release_date", ""),
                "vote_count": profile.get("vote_count", 0),
                "vote_average": profile.get("vote_average", 0),
                "poster_path": profile.get("poster_path") or "",
                "tmdbId": profile.get("tmdbId"),
            }

# Build TMDB ID → MovieLens ID index
tmdb_to_movielens: dict[int, int] = {
    int(p["tmdbId"]): mid
    for mid, p in profiles.items()
    if p.get("tmdbId") and str(p["tmdbId"]).isdigit()
}

print(f"Recommender ready. {len(profiles)} movies loaded, {len(tmdb_to_movielens)} have TMDB IDs.")


class SimilarByTmdbRequest(BaseModel):
    tmdb_id: int
    top_k: int = 20


class SimilarByTmdbMultiRequest(BaseModel):
    tmdb_ids: list[int]
    top_k: int = 20


class RecommendationResult(BaseModel):
    movie_id: int
    title: str
    final_score: float
    from_content: Optional[bool] = None
    from_collab: Optional[bool] = None
    genres: list[str]
    tmdb_id: Optional[int] = None


def _enrich_with_tmdb(recs: list[dict]) -> list[RecommendationResult]:
    """Add tmdbId to each recommendation result from the profiles data."""
    results = []
    for r in recs:
        mid = r["movie_id"]
        profile = profiles.get(mid, {})
        tmdb_raw = profile.get("tmdbId")
        tmdb_id = int(tmdb_raw) if tmdb_raw and str(tmdb_raw).isdigit() else None
        results.append(RecommendationResult(
            movie_id=mid,
            title=r.get("title", "Unknown"),
            final_score=r.get("final_score", 0.0),
            from_content=r.get("from_content"),
            from_collab=r.get("from_collab"),
            genres=r.get("genres", []),
            tmdb_id=tmdb_id,
        ))
    return results


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _profile_to_card(mid: int, p: dict) -> dict:
    tmdb_raw = p.get("tmdbId")
    tmdb_id = int(tmdb_raw) if tmdb_raw and str(tmdb_raw).isdigit() else None
    poster_path = p.get("poster_path") or ""
    return {
        "movieLensId": mid,
        "tmdbId": tmdb_id,
        "title": p.get("title", "Unknown"),
        "genres": [g["name"] for g in p.get("genres", [])],
        "releaseDate": p.get("release_date", ""),
        "voteAverage": p.get("vote_average", 0),
        "posterPath": poster_path,
        "posterUrl": f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else "",
    }


@app.get("/health")
def health():
    return {"status": "ok", "movies_loaded": len(profiles)}


@app.get("/browse")
def browse(limit: int = 40, offset: int = 0):
    """Return top movies by vote count × average rating (quality proxy)."""
    scored = [
        (mid, p, p.get("vote_count", 0) * p.get("vote_average", 0))
        for mid, p in profiles.items()
        if p.get("poster_path")  # only movies with a poster
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    page = scored[offset : offset + limit]
    return [_profile_to_card(mid, p) for mid, p, _ in page]


@app.get("/search")
def search(q: str, limit: int = 20):
    """Search movies in the dataset by title (case-insensitive substring)."""
    q_lower = q.strip().lower()
    if not q_lower:
        return []
    results = [
        (mid, p)
        for mid, p in profiles.items()
        if q_lower in p.get("title", "").lower()
    ]
    # Sort: exact match first, then by popularity
    results.sort(
        key=lambda x: (
            not x[1].get("title", "").lower().startswith(q_lower),
            -(x[1].get("vote_count", 0) * x[1].get("vote_average", 0)),
        )
    )
    return [_profile_to_card(mid, p) for mid, p in results[:limit]]


@app.post("/similar/by-tmdb", response_model=list[RecommendationResult])
def get_similar_by_tmdb(req: SimilarByTmdbRequest):
    """Find movies similar to the one identified by its TMDB ID."""
    _ensure_recommenders()
    ml_id = tmdb_to_movielens.get(req.tmdb_id)
    if ml_id is None:
        raise HTTPException(status_code=404, detail=f"TMDB ID {req.tmdb_id} not found in dataset")
    recs = hybrid.recommend(ml_id, k=req.top_k)
    return _enrich_with_tmdb(recs)


@app.post("/similar/by-tmdb-multi", response_model=list[RecommendationResult])
def get_similar_by_tmdb_multi(req: SimilarByTmdbMultiRequest):
    """Find movies similar to a list of movies identified by their TMDB IDs (uses RRF fusion)."""
    _ensure_recommenders()
    ml_ids = [tmdb_to_movielens[t] for t in req.tmdb_ids if t in tmdb_to_movielens]
    if not ml_ids:
        raise HTTPException(status_code=404, detail="None of the provided TMDB IDs found in dataset")
    recs = multi.recommend(ml_ids, k=req.top_k)
    return _enrich_with_tmdb(recs)


class InjectMovieRequest(BaseModel):
    title: str
    genres: list[str] = []
    keywords: list[str] = []
    cast: list[str] = []
    director: Optional[str] = None
    overview: Optional[str] = None
    collection: Optional[str] = None
    runtime: Optional[float] = None
    release_date: Optional[str] = None
    tmdb_id: Optional[int] = None


class InjectMovieResponse(BaseModel):
    ml_id: int
    cluster_id: int


class SimilarByMlIdRequest(BaseModel):
    ml_id: int
    top_k: int = 20


@app.post("/inject-movie", response_model=InjectMovieResponse)
def inject_movie(req: InjectMovieRequest):
    """
    Inject a new movie into the live ML model without a full rebuild.
    Vectorizes the provided metadata using the existing vocabulary, assigns it to the
    nearest cluster, and registers it in all in-memory recommender structures.
    Unknown genres/keywords/actors receive zero weight (graceful degradation).
    """
    _ensure_recommenders()

    # Assign a new internal ID that won't collide with existing Kaggle IDs
    new_id = max(vectors.keys()) + 1

    # Build a profile dict in the same shape as Kaggle profiles
    profile: dict = {
        "id": new_id,
        "title": req.title,
        "overview": req.overview or "",
        "release_date": req.release_date or "",
        "runtime": req.runtime or 0,
        "budget": 0,
        "revenue": 0,
        "vote_average": 0,
        "vote_count": 0,
        "imdb_id": None,
        "tmdbId": req.tmdb_id,
        "poster_path": "",
        "genres": [{"name": g} for g in req.genres],
        "keywords": [{"name": k} for k in req.keywords],
        "cast": [{"name": n, "order": i} for i, n in enumerate(req.cast)],
        "crew": ([{"name": req.director, "job": "Director"}] if req.director else []),
        "production_companies": [],
        "production_countries": [],
        "spoken_languages": [],
        "belongs_to_collection": ({"name": req.collection} if req.collection else None),
    }

    # Normalize genres/cast/director against the actual vocabulary (case-insensitive).
    # The vectorizer does exact string matching, so "fantasy" ≠ "Fantasy".
    genre_vocab_lower   = {g.lower(): g for g in vectorizer.genre_to_index}
    cast_vocab_lower    = {a.lower(): a for a in vectorizer.cast_to_index}
    crew_vocab_lower    = {c.lower(): c for c in vectorizer.crew_job_to_index}

    profile["genres"] = [
        {"name": genre_vocab_lower[g.lower()]}
        for g in req.genres if g.lower() in genre_vocab_lower
    ]
    profile["cast"] = [
        {"name": cast_vocab_lower[n.lower()], "order": i}
        for i, n in enumerate(req.cast) if n.lower() in cast_vocab_lower
    ]
    if req.director:
        crew_key = f"director:{req.director.lower()}"
        if crew_key in crew_vocab_lower:
            profile["crew"] = [{"name": crew_vocab_lower[crew_key].split(":", 1)[1], "job": "Director"}]

    matched_genres    = [g["name"] for g in profile["genres"]]
    matched_cast      = [a["name"] for a in profile["cast"]]
    matched_crew      = [m["name"] for m in profile["crew"]]
    unmatched_genres  = [g for g in req.genres  if g.lower() not in genre_vocab_lower]
    unmatched_cast    = [n for n in req.cast     if n.lower() not in cast_vocab_lower]
    print(f"[inject] '{req.title}': matched genres={matched_genres} cast={matched_cast} crew={matched_crew}")
    if unmatched_genres or unmatched_cast:
        print(f"[inject] '{req.title}': NOT IN VOCAB genres={unmatched_genres} cast={unmatched_cast}")

    # Vectorize using the existing vocabulary
    vector = vectorizer.vectorize(profile)
    nonzero_count = int(np.count_nonzero(vector))
    print(f"[inject] '{req.title}': {nonzero_count} non-zero vector dimensions")

    # Assign to nearest existing cluster — no refit needed
    cluster_id = int(kmeans.predict(vector))

    # --- Update all in-memory structures ---
    vectors[new_id] = vector
    profiles[new_id] = {
        "title": req.title,
        "genres": profile["genres"],
        "release_date": req.release_date or "",
        "vote_count": 0,
        "vote_average": 0,
        "poster_path": "",
        "tmdbId": req.tmdb_id,
    }

    # ContentKNN
    content_knn.vectors[new_id] = vector
    content_knn.profiles[new_id] = profile
    content_knn._nonzero_indices[new_id] = set(np.nonzero(vector)[0])
    content_knn.kmeans.labels[new_id] = cluster_id

    # CollaborativeKNN — new movie starts cold (no ratings)
    collab_knn.ratings_by_movie[new_id] = []
    collab_knn._movie_ratings_dict[new_id] = {}
    collab_knn.profiles[new_id] = profile

    # TMDB ID index
    if req.tmdb_id is not None:
        tmdb_to_movielens[req.tmdb_id] = new_id

    return InjectMovieResponse(ml_id=new_id, cluster_id=cluster_id)


@app.post("/similar/by-ml-id", response_model=list[RecommendationResult])
def get_similar_by_ml_id(req: SimilarByMlIdRequest):
    """Find movies similar to one identified by its internal ML ID (used for injected movies)."""
    _ensure_recommenders()
    if req.ml_id not in vectors:
        raise HTTPException(status_code=404, detail=f"ML ID {req.ml_id} not found")
    recs = hybrid.recommend(req.ml_id, k=req.top_k)
    return _enrich_with_tmdb(recs)
