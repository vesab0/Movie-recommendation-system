VECTORIZER_WEIGHTS = {
    'numerical':  0.5,
    'collection': 5.0,
    'genres':     2.0,
    'keywords':   3.0,
    'cast':       2.5,
    'crew':       4.0,
    'companies':  0.3,
    'countries':  0.2,
    'language':   0.3,
    'binary':     0.3,
}

MIN_KEYWORD_COUNT = 3 
MIN_COMPANY_COUNT = 3
MIN_CAST_COUNT    = 2
MIN_CREW_COUNT    = 2

CAST_LIMIT = 5
KEY_CREW_JOBS = {'Director', 'Writer', 'Screenplay', 'Producer', 'Original Music Composer'}
NUMERICAL_FIELDS = ['budget', 'revenue', 'runtime']

MIN_COMMON_USERS = 40 

CONTENT_WEIGHT = 0.85
COLLAB_WEIGHT  = 0.15

HYBRID_CONTENT_K = 20  
HYBRID_COLLAB_K  = 20

COLLAB_CONFIDENCE_SCALE = 300

CONSENSUS_BONUS_WEIGHT = 0.05 

RRF_K       = 60
PER_MOVIE_K = 25

RECOMMENDATIONS_K   = 10
SEARCH_MAX_RESULTS  = 10

CACHE_FILENAME         = "cached-data/movie_cache.pkl"
VECTORS_CACHE_FILENAME = "cached-data/vectors_cache.pkl"

CREDITS_PATH  = "data/credits.csv"
KEYWORDS_PATH = "data/keywords.csv"
LINKS_PATH    = "data/links.csv"
MOVIES_PATH   = "data/movies_metadata.csv"
RATINGS_PATH  = "data/ratings.csv"

CLUSTERS_CACHE_FILENAME = "clusters_cache.pkl"
N_CLUSTERS = 50
CLUSTER_SEARCH_RADIUS = 1
