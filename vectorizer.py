import numpy as np


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
            'numerical': 0.5,
            'collection': 5.0,
            'genres': 2.0,
            'keywords': 3.0,
            'cast': 2.5,
            'crew': 4.0,
            'companies': 0.3,
            'countries': 0.2,
            'language': 0.3,
            'binary': 0.3,
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
        keyword_counts = {}
        company_counts = {}
        all_countries = set()
        all_languages = set()
        cast_counts = {}
        crew_counts = {}
        all_collections = set()
        
        numerical_values = {field: [] for field in self.numerical_fields}
        
        for movie_id, profile in profiles.items():
            for g in profile.get('genres', []):
                if isinstance(g, dict) and 'name' in g:
                    all_genres.add(g['name'])
            
            for k in profile.get('keywords', []):
                if isinstance(k, dict) and 'name' in k:
                    keyword_counts[k['name']] = keyword_counts.get(k['name'], 0) + 1
            
            for c in profile.get('production_companies', []):
                if isinstance(c, dict) and 'name' in c:
                    company_counts[c['name']] = company_counts.get(c['name'], 0) + 1
            
            for c in profile.get('production_countries', []):
                if isinstance(c, dict) and 'name' in c:
                    all_countries.add(c['name'])
            
            for l in profile.get('spoken_languages', []):
                if isinstance(l, dict) and 'name' in l:
                    all_languages.add(l['name'])
            
            for actor in profile.get('cast', [])[:5]:
                if isinstance(actor, dict) and 'name' in actor:
                    cast_counts[actor['name']] = cast_counts.get(actor['name'], 0) + 1
            
            key_jobs = {'Director', 'Writer', 'Screenplay', 'Producer', 'Original Music Composer'}
            for member in profile.get('crew', []):
                if isinstance(member, dict) and 'job' in member and 'name' in member:
                    if member['job'] in key_jobs:
                        entry = f"{member['job']}:{member['name']}"
                        crew_counts[entry] = crew_counts.get(entry, 0) + 1
            
            collection = profile.get('belongs_to_collection')
            if isinstance(collection, dict) and 'name' in collection:
                all_collections.add(collection['name'])
            
            for field in self.numerical_fields:
                val = profile.get(field)
                if val is not None and val != 0:
                    numerical_values[field].append(val)
        
        MIN_KEYWORD_COUNT = 3
        MIN_COMPANY_COUNT = 3
        MIN_CAST_COUNT = 2
        MIN_CREW_COUNT = 2
        
        all_keywords = {k for k, count in keyword_counts.items() if count >= MIN_KEYWORD_COUNT}
        all_companies = {c for c, count in company_counts.items() if count >= MIN_COMPANY_COUNT}
        all_cast = {a for a, count in cast_counts.items() if count >= MIN_CAST_COUNT}
        all_crew_jobs = {j for j, count in crew_counts.items() if count >= MIN_CREW_COUNT}
        
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
            2 + 1 +
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
        print(f"  -> Keywords: {len(self.keyword_to_index)} (filtered from {len(keyword_counts)})")
        print(f"  -> Companies: {len(self.company_to_index)} (filtered from {len(company_counts)})")
        print(f"  -> Countries: {len(self.country_to_index)}")
        print(f"  -> Languages: {len(self.language_to_index)}")
        print(f"  -> Cast members: {len(self.cast_to_index)} (filtered from {len(cast_counts)})")
        print(f"  -> Crew entries: {len(self.crew_job_to_index)} (filtered from {len(crew_counts)})")
        print(f"  -> Collections: {len(self.collection_to_index)}")
        print(f"  -> Total vector dimensions: {self.vector_size}")
    
    def vectorize(self, profile):
        vector = np.zeros(self.vector_size, dtype=np.float32)
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
    
    profiles, ratings_by_movie, ratings_by_user = cache_manager.load_all_data_with_cache()
    
    vectorizer = ContentVectorizer()
    vectorizer.fit(profiles)
    vectors = vectorizer.vectorize_all(profiles)
    
    example_id = 862
    if example_id in vectors:
        vec = vectors[example_id]
        non_zero = np.count_nonzero(vec)
        print(f"\nExample vector for movieId {example_id} ({profiles[example_id]['title']}):")
        print(f"  Vector length: {len(vec)}")
        print(f"  Non-zero dimensions: {non_zero}")
        print(f"  Sparsity: {100 * (1 - non_zero/len(vec)):.1f}%")
    
    print(f"\nTotal vectors ready: {len(vectors)}")
    print("Step 2 complete.")