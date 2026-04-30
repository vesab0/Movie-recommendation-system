import math
import os

class ContentVectorizer:

    def __init__(self):
        self.genre_to_index = {}
        self.keyword_to_index = {}
        self.company_to_index = {}
        self.country_to_index = {}
        self.language_to_index = {}
        self.cast_to_index = {} 
        self.crew_job_to_index = {}
        self.collection_to_index = {}
        
        self.weights = {
            'numerical': 1.0,      
            'collection': 5.0,       
            'genres': 3.0,           
            'keywords': 2.0,         
            'cast': 2.0,             
            'crew': 2.5,             
            'companies': 0.5,        
            'countries': 0.3,        
            'language': 0.5,         
            'binary': 0.5,           
        }
        
        self.numerical_mins = {}
        self.numerical_maxs = {}
        
        self.vector_size = 0
        
        self.numerical_fields = [
            'budget', 'revenue', 'runtime', 'popularity',
            'vote_average', 'vote_count'
        ]

    def fit(self, profiles):

        print("Fitting vectorizer: building vocabularies...")
        
        all_genres = set()
        all_keywords = set()
        all_companies = set()
        all_countries = set()
        all_languages = set()
        all_cast = set()
        all_crew_jobs = set()
        all_collections = set()
        
        numerical_values = {field: [] for field in self.numerical_fields}
        
        for movie_id, profile in profiles.items():
            for g in profile.get('genres', []):
                if isinstance(g, dict) and 'name' in g:
                    all_genres.add(g['name'])
            
            for k in profile.get('keywords', []):
                if isinstance(k, dict) and 'name' in k:
                    all_keywords.add(k['name'])
            
            for c in profile.get('production_companies', []):
                if isinstance(c, dict) and 'name' in c:
                    all_companies.add(c['name'])
            
            for c in profile.get('production_countries', []):
                if isinstance(c, dict) and 'name' in c:
                    all_countries.add(c['name'])
            
            for l in profile.get('spoken_languages', []):
                if isinstance(l, dict) and 'name' in l:
                    all_languages.add(l['name'])
        
            for actor in profile.get('cast', [])[:5]:
                if isinstance(actor, dict) and 'name' in actor:
                    all_cast.add(actor['name'])
            
            key_jobs = {'Director', 'Writer', 'Screenplay', 'Producer', 'Original Music Composer'}
            for member in profile.get('crew', []):
                if isinstance(member, dict) and 'job' in member and 'name' in member:
                    if member['job'] in key_jobs:
                        all_crew_jobs.add(f"{member['job']}:{member['name']}")
            
            collection = profile.get('belongs_to_collection')
            if isinstance(collection, dict) and 'name' in collection:
                all_collections.add(collection['name'])
            
            for field in self.numerical_fields:
                val = profile.get(field)
                if val is not None and val != 0:
                    numerical_values[field].append(val)
        
        self.genre_to_index = {g: i for i, g in enumerate(sorted(all_genres))}
        self.keyword_to_index = {k: i for i, k in enumerate(sorted(all_keywords))}
        self.company_to_index = {c: i for i, c in enumerate(sorted(all_companies))}
        self.country_to_index = {c: i for i, c in enumerate(sorted(all_countries))}
        self.language_to_index = {l: i for i, l in enumerate(sorted(all_languages))}
        self.cast_to_index = {a: i for i, a in enumerate(sorted(all_cast))}
        self.crew_job_to_index = {j: i for i, j in enumerate(sorted(all_crew_jobs))}
        self.collection_to_index = {c: i for i, c in enumerate(sorted(all_collections))}
        
        for field in self.numerical_fields:
            values = numerical_values[field]
            if values:
                self.numerical_mins[field] = min(values)
                self.numerical_maxs[field] = max(values)
            else:
                self.numerical_mins[field] = 0
                self.numerical_maxs[field] = 1 
        
        self.vector_size = (
            len(self.numerical_fields) +           
            2 +                                   
            1 +                                
            len(self.genre_to_index) +            
            len(self.keyword_to_index) +           
            len(self.company_to_index) +         
            len(self.country_to_index) +         
            len(self.language_to_index) +     
            len(self.cast_to_index) +        
            len(self.crew_job_to_index) +       
            len(self.collection_to_index)     
        )
        
        print(f"  -> Genres: {len(self.genre_to_index)}")
        print(f"  -> Keywords: {len(self.keyword_to_index)}")
        print(f"  -> Companies: {len(self.company_to_index)}")
        print(f"  -> Countries: {len(self.country_to_index)}")
        print(f"  -> Languages: {len(self.language_to_index)}")
        print(f"  -> Cast members: {len(self.cast_to_index)}")
        print(f"  -> Crew entries: {len(self.crew_job_to_index)}")
        print(f"  -> Collections: {len(self.collection_to_index)}")
        print(f"  -> Total vector dimensions: {self.vector_size}")
    
    def vectorize(self, profile):
        vector = [0.0] * self.vector_size
        idx = 0 
        
        for field in self.numerical_fields:
            raw_value = profile.get(field, 0)
            if raw_value is None:
                raw_value = 0
            min_val = self.numerical_mins[field]
            max_val = self.numerical_maxs[field]
            
            if max_val > min_val:
                normalized = (raw_value - min_val) / (max_val - min_val)
            else:
                normalized = 0.0

            vector[idx] = normalized * self.weights['numerical']
            idx += 1
        
        vector[idx] = (1.0 if profile.get('adult') else 0.0) * self.weights['binary']
        idx += 1

        vector[idx] = (1.0 if profile.get('video') else 0.0) * self.weights['binary']
        idx += 1
        
        has_collection = 1.0 if profile.get('belongs_to_collection') else 0.0
        vector[idx] = has_collection * self.weights['collection']
        idx += 1
        
        movie_genres = set()
        for g in profile.get('genres', []):
            if isinstance(g, dict) and 'name' in g:
                movie_genres.add(g['name'])
        
        for genre, genre_idx in self.genre_to_index.items():
            if genre in movie_genres:
                vector[idx + genre_idx] = 1.0 * self.weights['genres']
        idx += len(self.genre_to_index)
        
        movie_keywords = set()
        for k in profile.get('keywords', []):
            if isinstance(k, dict) and 'name' in k:
                movie_keywords.add(k['name'])
        
        for keyword, kw_idx in self.keyword_to_index.items():
            if keyword in movie_keywords:
                vector[idx + kw_idx] = 1.0 * self.weights['keywords']
        idx += len(self.keyword_to_index)
        
        movie_companies = set()
        for c in profile.get('production_companies', []):
            if isinstance(c, dict) and 'name' in c:
                movie_companies.add(c['name'])
        
        for company, comp_idx in self.company_to_index.items():
            if company in movie_companies:
                vector[idx + comp_idx] = 1.0 * self.weights['companies']
        idx += len(self.company_to_index)
        
        movie_countries = set()
        for c in profile.get('production_countries', []):
            if isinstance(c, dict) and 'name' in c:
                movie_countries.add(c['name'])
        
        for country, country_idx in self.country_to_index.items():
            if country in movie_countries:
                vector[idx + country_idx] = 1.0 * self.weights['countries']
        idx += len(self.country_to_index)
        
        movie_languages = set()
        for l in profile.get('spoken_languages', []):
            if isinstance(l, dict) and 'name' in l:
                movie_languages.add(l['name'])
        
        orig_lang = profile.get('original_language')
        if orig_lang:
            movie_languages.add(orig_lang)
        
        for language, lang_idx in self.language_to_index.items():
            if language in movie_languages:
                vector[idx + lang_idx] = 1.0 * self.weights['language']
        idx += len(self.language_to_index)
        
        movie_cast = set()
        for actor in profile.get('cast', [])[:5]:
            if isinstance(actor, dict) and 'name' in actor:
                movie_cast.add(actor['name'])
        
        for actor, cast_idx in self.cast_to_index.items():
            if actor in movie_cast:
                vector[idx + cast_idx] = 1.0 * self.weights['cast']
        idx += len(self.cast_to_index)
        
        key_jobs = {'Director', 'Writer', 'Screenplay', 'Producer', 'Original Music Composer'}
        movie_crew = set()
        for member in profile.get('crew', []):
            if isinstance(member, dict) and 'job' in member and 'name' in member:
                if member['job'] in key_jobs:
                    movie_crew.add(f"{member['job']}:{member['name']}")
        
        for crew_entry, crew_idx in self.crew_job_to_index.items():
            if crew_entry in movie_crew:
                vector[idx + crew_idx] = 1.0 * self.weights['crew']
        idx += len(self.crew_job_to_index)
        
        collection = profile.get('belongs_to_collection')
        collection_name = None
        if isinstance(collection, dict) and 'name' in collection:
            collection_name = collection['name']
        
        for coll, coll_idx in self.collection_to_index.items():
            if collection_name == coll:
                vector[idx + coll_idx] = 1.0 * self.weights['collection']
        idx += len(self.collection_to_index)
        
        return vector
 
    def vectorize_all(self, profiles):

        print("Vectorizing all movies...")
        vectors = {}
        
        for movie_id, profile in profiles.items():
            vectors[movie_id] = self.vectorize(profile)
        
        print(f"  -> Vectorized {len(vectors)} movies")
        return vectors


if __name__ == "__main__":

    import cacher as cache_manager
    import distance_functions as dist_funcs
    
    # Load data (from cache if available)
    profiles, ratings_by_movie, ratings_by_user = cache_manager.load_all_data_with_cache()
    
    # Create and fit the vectorizer
    vectorizer = ContentVectorizer()
    vectorizer.fit(profiles)
    
    # Vectorize all movies
    vectors = vectorizer.vectorize_all(profiles)
    
    # Test: pick a movie and show its vector stats
    example_id = 862  # Toy Story
    if example_id in vectors:
        vec = vectors[example_id]
        non_zero = sum(1 for x in vec if x != 0.0)
        print(f"\nExample vector for movieId {example_id} ({profiles[example_id]['title']}):")
        print(f"  Vector length: {len(vec)}")
        print(f"  Non-zero dimensions: {non_zero}")
        print(f"  Sparsity: {100 * (1 - non_zero/len(vec)):.1f}%")
        print(f"\n  Active features (non-zero):")
        
        # Numerical
        numerical_start = 0
        for i, field in enumerate(vectorizer.numerical_fields):
            if vec[numerical_start + i] > 0:
                print(f"    {field}: {vec[numerical_start + i]:.4f}")
        
        # Genres
        genre_start = len(vectorizer.numerical_fields) + 3  # after numerical + binary + has_collection
        for genre, idx in vectorizer.genre_to_index.items():
            val = vec[genre_start + idx]
            if val > 0:
                print(f"    Genre '{genre}': {val:.1f}")
        
        # Collection
        coll_start = genre_start + len(vectorizer.genre_to_index) + len(vectorizer.keyword_to_index) + \
                     len(vectorizer.company_to_index) + len(vectorizer.country_to_index) + \
                     len(vectorizer.language_to_index) + len(vectorizer.cast_to_index) + \
                     len(vectorizer.crew_job_to_index)
        for coll, idx in vectorizer.collection_to_index.items():
            val = vec[coll_start + idx]
            if val > 0:
                print(f"    Collection '{coll}': {val:.1f}")
        
        # Quick distance test to another movie
        other_id = 863  # Usually another movie
        if other_id in vectors:
            dist = dist_funcs.cosine_distance(vec, vectors[other_id])
            sim = dist_funcs.cosine_similarity(vec, vectors[other_id])
            print(f"\n  Distance to movieId {other_id} ({profiles.get(other_id, {}).get('title', 'N/A')}):")
            print(f"    Cosine similarity: {sim:.4f}")
            print(f"    Cosine distance: {dist:.4f}")
    
    print("\nStep 2 complete. Vectors ready for KNN.")