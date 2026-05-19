(# Movie Predictor — FastAPI Service)

This repository contains a FastAPI wrapper around a hybrid movie recommender. It exposes HTTP endpoints for browsing the built-in catalog, searching titles, requesting recommendations (by TMDB or internal ML ID) and injecting a new movie into the live model without a full rebuild.

**Key features:**
- Lightweight browse and search endpoints with poster enrichment (TMDB public search)
- Hybrid recommendations combining content- and collaborative-based signals
- On-demand model loading to keep startup fast and memory usage modest
- Ability to inject a new movie into the running model for quick experimentation

Requirements
------------
- Python 3.10+
- See `requirements.txt` for Python packages used by the project.

Quick start (local)
-------------------
1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI app (development):

```bash
uvicorn api:app --port 8001 --reload
```

The API will be available at http://127.0.0.1:8001

Configuration & data
--------------------
Configuration constants and dataset paths are defined in `config.py`. By default the service expects the Kaggle-derived CSV files in the repository `data/` directory:

- `data/movies_metadata.csv`
- `data/links.csv`
- `data/credits.csv`
- `data/keywords.csv`
- `data/ratings.csv`

Cached artifacts (vector cache, poster cache) are stored under `cached-data/` by default. See `config.py` for cache filenames and cluster settings.

API endpoints
-------------
- `GET /health` — basic health and movies-loaded inspection
- `GET /browse?limit=40&offset=0` — return top movies by popularity-quality proxy
- `GET /search?q=term&limit=20` — case-insensitive title substring search
- `POST /similar/by-tmdb` — body: `{ "tmdb_id": 278, "top_k": 20 }` — get recommendations for a TMDB movie
- `POST /similar/by-tmdb-multi` — body: `{ "tmdb_ids": [278, 238], "top_k": 20 }` — fused recommendations
- `POST /similar/by-ml-id` — body: `{ "ml_id": 1000, "top_k": 20 }` — recommend by internal ML id
- `POST /inject-movie` — inject a new movie into the live model. See the `InjectMovieRequest` schema in `api.py` for required fields.

Examples
--------
Health check:

```bash
curl http://127.0.0.1:8001/health
```

Search example:

```bash
curl "http://127.0.0.1:8001/search?q=dark+knight&limit=10"
```

Recommend by TMDB:

```bash
curl -X POST -H "Content-Type: application/json" \
	-d '{"tmdb_id":238, "top_k":10}' \
	http://127.0.0.1:8001/similar/by-tmdb
```

Development notes
-----------------
- The service lazily loads heavier ML artifacts on first recommendation request; the initial browse/search endpoints operate from a lightweight catalog.
- Poster URLs are retrieved using TMDB public search (no API key required); results are cached in `cached-data/poster_cache.json`.
- The vectorizer vocabulary is reconstructed from the full profiles when the recommender is loaded, which allows `inject-movie` to map new metadata into the existing feature space.

Docker
------
This repository includes a `Dockerfile` for building a containerized service. A typical build and run sequence:

```bash
docker build -t movie-predictor .
docker run --rm -p 8001:8001 movie-predictor
```

If you mount local `data/` and `cached-data/` directories into the container, the service will use those files directly.

Where to look next
------------------
- `api.py` — endpoint definitions and request/response models
- `loader.py` and `cacher.py` — dataset loading and caching logic
- `vectorizer.py`, `content_knn.py`, `collaborative_knn.py` — core ML components