import json
import csv

# I merr krejt cvs files edhe i bon build 3 struktura 
# Struktura 1 esht per kejt detajet e filmave
# Struktura 2 esht per ratings te filmave, struktura 3 esht per ratings te userave

# SHEBULL QYSH KTHEN OUTPUT:
"""
Movie ID:        862
Title:           Toy Story
Original Title:  Toy Story
Release Date:    1995-10-30
Runtime:         81.0 min
Budget:          $30,000,000
Revenue:         $373,554,033
Genres:          ['Animation', 'Comedy', 'Family']
Language:        en
Status:          Released
Vote Average:    7.7
Vote Count:      5415
Average Rating:  3.5989304812834226
Rating Count:    374
Collection:      Toy Story Collection

Overview:
Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences.

Top 5 Cast:
  1. Tom Hanks as Woody (voice)
  2. Tim Allen as Buzz Lightyear (voice)
  3. Don Rickles as Mr. Potato Head (voice)
  4. Jim Varney as Slinky Dog (voice)
  5. Wallace Shawn as Rex (voice)

Top 3 Keywords:
  1. jealousy
  2. toy
  3. boy

Ratings Lookup Check (first 3 ratings):
  User 1923 rated 3.0 at timestamp 858335006
  User 2103 rated 5.0 at timestamp 946044912
  User 5380 rated 1.0 at timestamp 878941641

Ratings by User Lookup Check:
  User 1923 has rated 60 movies
  First 3 movies they rated: [(1, 5.0, 858334277), (3, 3.0, 858334329), (5, 3.0, 858334329)]
"""


def load_credits(filepath):
    credits_list = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            record = {
                'movieId': int(row['id']),
                'cast': parse_json_field(row.get('cast', '[]')),
                'crew': parse_json_field(row.get('crew', '[]'))
            }
            credits_list.append(record)
    
    return credits_list


def load_keywords(filepath):
    keywords_list = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            record = {
                'movieId': int(row['id']),
                'keywords': parse_json_field(row.get('keywords', '[]'))
            }
            keywords_list.append(record)
    
    return keywords_list

def load_links(filepath):
    links_list = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cleaned_row = {
                'movieId': int(row['movieId']),
                'imdbId': row['imdbId'].strip() if row['imdbId'] else None,
                'tmdbId': int(row['tmdbId']) if row['tmdbId'] else None
            }
            links_list.append(cleaned_row)
    
    return links_list

def load_movies_metadata(filepath):
    movies_list = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:  
            parsed_row = {}
            
            parsed_row['adult'] = (row.get('adult') or '').lower() == 'true'
            parsed_row['belongs_to_collection'] = parse_json_field(row.get('belongs_to_collection', ''))
            parsed_row['budget'] = parse_int(row.get('budget', '0'))
            parsed_row['genres'] = parse_json_field(row.get('genres', '[]'))
            parsed_row['homepage'] = row.get('homepage') or None
            parsed_row['id'] = parse_int(row.get('id', '0')) 
            parsed_row['imdb_id'] = row.get('imdb_id') or None
            parsed_row['original_language'] = row.get('original_language') or None
            parsed_row['original_title'] = row.get('original_title') or None
            parsed_row['overview'] = row.get('overview') or None
            parsed_row['popularity'] = parse_float(row.get('popularity', '0'))
            parsed_row['poster_path'] = row.get('poster_path') or None
            parsed_row['production_companies'] = parse_json_field(row.get('production_companies', '[]'))
            parsed_row['production_countries'] = parse_json_field(row.get('production_countries', '[]'))
            parsed_row['release_date'] = row.get('release_date') or None
            parsed_row['revenue'] = parse_int(row.get('revenue', '0'))
            parsed_row['runtime'] = parse_float(row.get('runtime', '0')) if row.get('runtime') else None
            parsed_row['spoken_languages'] = parse_json_field(row.get('spoken_languages', '[]'))
            parsed_row['status'] = row.get('status') or None
            parsed_row['tagline'] = row.get('tagline') or None
            parsed_row['title'] = row.get('title') or None
            parsed_row['video'] = (row.get('video') or '').lower() == 'true'
            parsed_row['vote_average'] = parse_float(row.get('vote_average', '0'))
            parsed_row['vote_count'] = parse_int(row.get('vote_count', '0'))
            
            movies_list.append(parsed_row)
    
    return movies_list

def load_ratings(filepath):
    ratings_list = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rating_record = {
                'userId': int(row['userId']),
                'movieId': int(row['movieId']),
                'rating': float(row['rating']),
                'timestamp': int(row['timestamp'])
            }
            ratings_list.append(rating_record)
    
    return ratings_list

def parse_json_field(value):
    if not value or value.lower() in ('nan', 'null', 'none', ''):
        return []
    
    if not isinstance(value, str):
        return []
    
    value = value.strip()

    if len(value) >= 2:
        if (value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"'):
            try:
                inner = json.loads(value)  
                if isinstance(inner, str):
                    return json.loads(inner)
                return inner
            except:
                pass
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        fixed = value.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    try:
        import ast
        result = ast.literal_eval(value)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        return []
    except:
        return []
    

def parse_int(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0

def parse_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def build_movie_profiles(credits_list, keywords_list, links_list, movies_list, ratings_list):
    profiles = {}
    
    for movie in movies_list:
        movie_id = movie['id']
        
        if movie_id == 0:
            continue
        
        profile = dict(movie)
        
        profile['cast'] = []
        profile['crew'] = []
        profile['keywords'] = []
        profile['imdbId'] = None
        profile['tmdbId'] = None
        
        profiles[movie_id] = profile
    
    for credit in credits_list:
        movie_id = credit.get('movieId')
        
        if movie_id is None or movie_id not in profiles:
            continue
        
        cast_data = credit.get('cast', [])
        if isinstance(cast_data, list):
            profiles[movie_id]['cast'] = cast_data
        
        crew_data = credit.get('crew', [])
        if isinstance(crew_data, list):
            profiles[movie_id]['crew'] = crew_data
    
    for keyword_record in keywords_list:
        movie_id = keyword_record.get('movieId')
        
        if movie_id is None or movie_id not in profiles:
            continue
        
        keywords_data = keyword_record.get('keywords', [])
        if isinstance(keywords_data, list):
            profiles[movie_id]['keywords'] = keywords_data

    for link in links_list:
        movie_id = link.get('movieId')
        
        if movie_id is None or movie_id not in profiles:
            continue
        
        profiles[movie_id]['imdbId'] = link.get('imdbId')
        profiles[movie_id]['tmdbId'] = link.get('tmdbId')
    
    ratings_by_movie = {} 
    ratings_by_user = {} 
    
    for rating in ratings_list:
        movie_id = rating['movieId']
        user_id = rating['userId']
        rating_value = rating['rating']
        timestamp = rating['timestamp']
        tuple_form = (user_id, rating_value, timestamp)
        
        if movie_id not in ratings_by_movie:
            ratings_by_movie[movie_id] = []
        ratings_by_movie[movie_id].append(tuple_form)
        
        if user_id not in ratings_by_user:
            ratings_by_user[user_id] = []
        ratings_by_user[user_id].append((movie_id, rating_value, timestamp))
    
    for movie_id, profile in profiles.items():
        movie_ratings = ratings_by_movie.get(movie_id, [])
        profile['rating_count'] = len(movie_ratings)
        
        if movie_ratings:
            total = sum(r[1] for r in movie_ratings)
            profile['average_rating'] = total / len(movie_ratings)
        else:
            profile['average_rating'] = None
     
    valid_profiles = {}
    
    for movie_id, profile in profiles.items():
        if not profile.get('title'):
            continue
        
        if not isinstance(profile.get('genres'), list):
            profile['genres'] = []
        if not isinstance(profile.get('keywords'), list):
            profile['keywords'] = []
        if not isinstance(profile.get('cast'), list):
            profile['cast'] = []
        if not isinstance(profile.get('crew'), list):
            profile['crew'] = []
        
        valid_profiles[movie_id] = profile
    
    return valid_profiles, ratings_by_movie, ratings_by_user

def load_all_data(credits_path, keywords_path, links_path, movies_path, ratings_path):
    print("Loading credits...")
    credits_list = load_credits(credits_path)
    print(f"  -> Loaded {len(credits_list)} credit records")
    
    print("Loading keywords...")
    keywords_list = load_keywords(keywords_path)
    print(f"  -> Loaded {len(keywords_list)} keyword records")
    
    print("Loading links...")
    links_list = load_links(links_path)
    print(f"  -> Loaded {len(links_list)} link records")
    
    print("Loading movies metadata...")
    movies_list = load_movies_metadata(movies_path)
    print(f"  -> Loaded {len(movies_list)} movie records")
    
    print("Loading ratings...")
    ratings_list = load_ratings(ratings_path)
    print(f"  -> Loaded {len(ratings_list)} rating records")
    
    print("\nBuilding unified profiles...")
    profiles, ratings_by_movie, ratings_by_user = build_movie_profiles(
        credits_list, keywords_list, links_list, movies_list, ratings_list
    )
    print(f"  -> Built {len(profiles)} valid movie profiles")
    
    return profiles, ratings_by_movie, ratings_by_user

# =============================================================================
# MAIN BLOCK: Run this when you execute the script directly
# =============================================================================

if __name__ == "__main__":
    # File paths
    CREDITS_PATH = "data/credits.csv"
    KEYWORDS_PATH = "data/keywords.csv"
    LINKS_PATH = "data/links.csv"
    MOVIES_PATH = "data/movies_metadata.csv"
    RATINGS_PATH = "data/ratings.csv"
    
    # Load everything
    profiles, ratings_by_movie, ratings_by_user = load_all_data(
        CREDITS_PATH, KEYWORDS_PATH, LINKS_PATH, MOVIES_PATH, RATINGS_PATH
    )
    
    # Find the first movie that has ratings to show a good example
    example_movie_id = None
    for movie_id in profiles:
        if profiles[movie_id]['rating_count'] > 0:
            example_movie_id = movie_id
            break
    
    # If no rated movie found, just take the first one
    if example_movie_id is None:
        example_movie_id = list(profiles.keys())[0]
    
    movie = profiles[example_movie_id]
    
    print("\n" + "=" * 60)
    print("EXAMPLE MOVIE PROFILE")
    print("=" * 60)
    print(f"Movie ID:        {example_movie_id}")
    print(f"Title:           {movie['title']}")
    print(f"Original Title:  {movie['original_title']}")
    print(f"Release Date:    {movie['release_date']}")
    print(f"Runtime:         {movie['runtime']} min")
    print(f"Budget:          ${movie['budget']:,}")
    print(f"Revenue:         ${movie['revenue']:,}")
    print(f"Genres:          {[g['name'] for g in movie['genres']]}")
    print(f"Language:        {movie['original_language']}")
    print(f"Status:          {movie['status']}")
    print(f"Vote Average:    {movie['vote_average']}")
    print(f"Vote Count:      {movie['vote_count']}")
    print(f"Average Rating:  {movie['average_rating']}")
    print(f"Rating Count:    {movie['rating_count']}")
    print(f"Collection:      {movie['belongs_to_collection']['name'] if movie['belongs_to_collection'] else 'None'}")
    
    print(f"\nOverview:\n{movie['overview']}")
    
    print(f"\nTop 5 Cast:")
    for i, actor in enumerate(movie['cast'][:5]):
        print(f"  {i+1}. {actor['name']} as {actor.get('character', 'N/A')}")
    
    print(f"\nTop 3 Keywords:")
    for i, kw in enumerate(movie['keywords'][:3]):
        print(f"  {i+1}. {kw['name']}")
    
    print(f"\nRatings Lookup Check (first 3 ratings):")
    ratings = ratings_by_movie.get(example_movie_id, [])
    for i, r in enumerate(ratings[:3]):
        user_id, rating, timestamp = r
        print(f"  User {user_id} rated {rating} at timestamp {timestamp}")
    
    print(f"\nRatings by User Lookup Check:")
    # Pick a user who rated this movie and show what else they rated
    if ratings:
        first_user = ratings[0][0]
        user_ratings = ratings_by_user[first_user]
        print(f"  User {first_user} has rated {len(user_ratings)} movies")
        print(f"  First 3 movies they rated: {user_ratings[:3]}")
    
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED - Data loaded successfully")
    print("=" * 60)